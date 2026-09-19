#include <opencv2/opencv.hpp>

#include "cvproject/mesh_loader.hpp"
#include "cvproject/pose_utils.hpp"
#include "cvproject/reference_db.hpp"
#include "cvproject/viewpoint_selector.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

struct ClickState {
    cv::Mat baseImage;
    cv::Mat image;
    std::array<std::optional<cv::Point2f>, 8> points;
    int currentCorner = 0;
};

struct PoseObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int frameId = 0;
    gtsam::Pose3 measuredPose;
    gtsam::Pose3 initialPose;
};

void redrawClickImage(ClickState& state) {
    state.image = state.baseImage.clone();

    for (size_t i = 0; i < state.points.size(); ++i) {
        if (!state.points[i].has_value()) {
            continue;
        }

        const cv::Point2f point = state.points[i].value();
        cv::circle(state.image, point, 5, cv::Scalar(0, 255, 255), -1);
        cv::putText(state.image, std::to_string(i + 1), point + cv::Point2f(8.0f, -8.0f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }

    cv::putText(state.image, "Selected 3D corner: " + std::to_string(state.currentCorner + 1),
                cv::Point(30, 45), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
}

void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    auto* state = static_cast<ClickState*>(userdata);
    state->points[state->currentCorner] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));

    redrawClickImage(*state);
    cv::imshow("Click visible box corners", state->image);
}

cv::Point toPixel(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

void drawProjectedBox(cv::Mat& image,
                      const std::vector<cv::Point3f>& boxCorners,
                      const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      const cv::Mat& cameraMatrix,
                      const cv::Mat& distCoeffs) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(boxCorners, rvec, tvec, cameraMatrix, distCoeffs, projected);

    const std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    for (const auto& edge : edges) {
        cv::line(image, toPixel(projected[edge.first]), toPixel(projected[edge.second]),
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    for (size_t i = 0; i < projected.size(); ++i) {
        cv::circle(image, toPixel(projected[i]), 4, cv::Scalar(0, 255, 255), -1);
        cv::putText(image, std::to_string(i + 1), toPixel(projected[i] + cv::Point2f(5.0f, -5.0f)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }
}

void drawAxis(cv::Mat& image,
              const cv::Mat& rvec,
              const cv::Mat& tvec,
              const cv::Mat& cameraMatrix,
              const cv::Mat& distCoeffs,
              float axisLength) {
    const std::vector<cv::Point3f> axisPoints = {
        {0.0f, 0.0f, 0.0f},
        {axisLength, 0.0f, 0.0f},
        {0.0f, axisLength, 0.0f},
        {0.0f, 0.0f, axisLength}
    };

    std::vector<cv::Point2f> projected;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, projected);

    cv::line(image, toPixel(projected[0]), toPixel(projected[1]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    cv::line(image, toPixel(projected[0]), toPixel(projected[2]), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    cv::line(image, toPixel(projected[0]), toPixel(projected[3]), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);

    cv::putText(image, "X", toPixel(projected[1]), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "Y", toPixel(projected[2]), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "Z", toPixel(projected[3]), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
}

bool collectManualCorrespondences(const cv::Mat& frame,
                                  const std::vector<cv::Point3f>& boxCorners,
                                  std::vector<cv::Point3f>& objectPoints,
                                  std::vector<cv::Point2f>& imagePoints,
                                  const std::string& prompt) {
    std::cout << prompt << std::endl;
    std::cout << "Press number 1-8 to select a 3D corner, then click its 2D image location." << std::endl;
    std::cout << "Use at least 4 visible corners. More corners usually gives a better pose." << std::endl;
    std::cout << "Keys: 1-8 = select corner, c = clear selected corner, Enter = solve, Esc = quit" << std::endl;

    ClickState clickState;
    clickState.baseImage = frame.clone();
    redrawClickImage(clickState);

    cv::namedWindow("Click visible box corners", cv::WINDOW_NORMAL);
    cv::resizeWindow("Click visible box corners", 1280, 720);
    cv::setMouseCallback("Click visible box corners", onMouse, &clickState);
    cv::imshow("Click visible box corners", clickState.image);

    while (true) {
        const int key = cv::waitKey(20);
        if (key == 27) {
            return false;
        }

        if (key >= '1' && key <= '8') {
            clickState.currentCorner = key - '1';
            redrawClickImage(clickState);
            cv::imshow("Click visible box corners", clickState.image);
        }

        if (key == 'c' || key == 'C') {
            clickState.points[clickState.currentCorner].reset();
            redrawClickImage(clickState);
            cv::imshow("Click visible box corners", clickState.image);
        }

        if (key == 13 || key == 10) {
            break;
        }
    }

    objectPoints.clear();
    imagePoints.clear();
    for (size_t i = 0; i < clickState.points.size(); ++i) {
        if (!clickState.points[i].has_value()) {
            continue;
        }

        objectPoints.push_back(boxCorners[i]);
        imagePoints.push_back(clickState.points[i].value());
    }

    if (imagePoints.size() < 4) {
        std::cerr << "Need at least 4 clicked corners, got " << imagePoints.size() << std::endl;
        return false;
    }

    return true;
}

void saveManualCorrespondences(const std::filesystem::path& path,
                               const std::vector<cv::Point3f>& allCorners,
                               const std::vector<cv::Point3f>& objectPoints,
                               const std::vector<cv::Point2f>& imagePoints) {
    std::filesystem::create_directories(path.parent_path());

    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    fs << "points" << "[";
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        int cornerId = -1;
        for (size_t j = 0; j < allCorners.size(); ++j) {
            if (cv::norm(objectPoints[i] - allCorners[j]) < 1e-3f) {
                cornerId = static_cast<int>(j);
                break;
            }
        }

        fs << "{"
           << "id" << cornerId + 1
           << "x" << imagePoints[i].x
           << "y" << imagePoints[i].y
           << "}";
    }
    fs << "]";
}

bool loadManualCorrespondences(const std::filesystem::path& path,
                               const std::vector<cv::Point3f>& allCorners,
                               std::vector<cv::Point3f>& objectPoints,
                               std::vector<cv::Point2f>& imagePoints) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    objectPoints.clear();
    imagePoints.clear();

    cv::FileNode points = fs["points"];
    for (const cv::FileNode& pointNode : points) {
        const int id = static_cast<int>(pointNode["id"]) - 1;
        const float x = static_cast<float>(pointNode["x"]);
        const float y = static_cast<float>(pointNode["y"]);
        if (id < 0 || id >= static_cast<int>(allCorners.size())) {
            continue;
        }

        objectPoints.push_back(allCorners[id]);
        imagePoints.emplace_back(x, y);
    }

    return imagePoints.size() >= 4;
}

gtsam::Pose3 cvPoseToGtsamPose(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);

    gtsam::Matrix3 R;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            R(row, col) = rotation.at<double>(row, col);
        }
    }

    return gtsam::Pose3(
        gtsam::Rot3(R),
        gtsam::Point3(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2))
    );
}

void gtsamPoseToCvPose(const gtsam::Pose3& pose, cv::Mat& rvec, cv::Mat& tvec) {
    cv::Mat rotation(3, 3, CV_64F);
    const gtsam::Matrix3 R = pose.rotation().matrix();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            rotation.at<double>(row, col) = R(row, col);
        }
    }

    cv::Rodrigues(rotation, rvec);
    rvec = rvec.reshape(1, 3).clone();
    tvec = (cv::Mat_<double>(3, 1) << pose.x(), pose.y(), pose.z());
}

class GtsamPoseSmoother {
public:
    explicit GtsamPoseSmoother(const cvproject::ReferenceTrackingConfig& config)
        : windowSize_(config.gtsamWindowSize) {
        gtsam::Vector6 measurementSigmas;
        measurementSigmas << config.gtsamMeasurementRotSigma,
                              config.gtsamMeasurementRotSigma,
                              config.gtsamMeasurementRotSigma,
                              config.gtsamMeasurementTransSigmaMm,
                              config.gtsamMeasurementTransSigmaMm,
                              config.gtsamMeasurementTransSigmaMm;
        gtsam::Vector6 motionSigmas;
        motionSigmas << config.gtsamMotionRotSigma,
                        config.gtsamMotionRotSigma,
                        config.gtsamMotionRotSigma,
                        config.gtsamMotionTransSigmaMm,
                        config.gtsamMotionTransSigmaMm,
                        config.gtsamMotionTransSigmaMm;

        const auto measurementBase = gtsam::noiseModel::Diagonal::Sigmas(measurementSigmas);
        measurementNoise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(config.gtsamHuberK),
            measurementBase
        );
        motionNoise_ = gtsam::noiseModel::Diagonal::Sigmas(motionSigmas);
    }

    gtsam::Pose3 update(int frameId, const gtsam::Pose3& measuredPose, bool hasFreshMeasurement) {
        if (observations_.empty()) {
            observations_.push_back({frameId, measuredPose, measuredPose});
            return measuredPose;
        }

        const gtsam::Pose3 initialPose = hasFreshMeasurement ? measuredPose : observations_.back().initialPose;
        observations_.push_back({frameId, measuredPose, initialPose});
        while (observations_.size() > static_cast<size_t>(windowSize_)) {
            observations_.erase(observations_.begin());
        }

        if (observations_.size() < 2) {
            return observations_.back().initialPose;
        }

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;

        const gtsam::Symbol firstKey('x', 0);
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            firstKey, observations_.front().initialPose, measurementNoise_));

        for (size_t i = 0; i < observations_.size(); ++i) {
            const gtsam::Symbol key('x', static_cast<uint64_t>(i));
            initial.insert(key, observations_[i].initialPose);
            graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                key, observations_[i].measuredPose, measurementNoise_));

            if (i > 0) {
                const gtsam::Symbol previousKey('x', static_cast<uint64_t>(i - 1));
                graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    previousKey, key, gtsam::Pose3(), motionNoise_));
            }
        }

        gtsam::LevenbergMarquardtParams params;
        params.setVerbosityLM("ERROR");
        params.setMaxIterations(20);

        try {
            const gtsam::Values result =
                gtsam::LevenbergMarquardtOptimizer(graph, initial, params).optimize();
            const gtsam::Pose3 optimizedPose =
                result.at<gtsam::Pose3>(gtsam::Symbol('x', observations_.size() - 1));
            observations_.back().initialPose = optimizedPose;
            return optimizedPose;
        } catch (const std::exception& e) {
            std::cerr << "GTSAM optimization failed: " << e.what() << std::endl;
            return observations_.back().initialPose;
        }
    }

private:
    int windowSize_ = 15;
    gtsam::SharedNoiseModel measurementNoise_;
    gtsam::SharedNoiseModel motionNoise_;
    std::vector<PoseObservation, Eigen::aligned_allocator<PoseObservation>> observations_;
};

struct StableTrackedCorner {
    int id = -1;
    cv::Point2f initialPoint;
    cv::Point2f point;
    bool valid = false;
    int lostCount = 0;
};

struct StableFaceTrackResult {
    bool ok = false;
    cv::Mat rvec;
    cv::Mat tvec;
    double reprojectionError = -1.0;
    int trackedPoints = 0;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
};

int findCornerId(const std::vector<cv::Point3f>& boxCorners, const cv::Point3f& point) {
    for (size_t i = 0; i < boxCorners.size(); ++i) {
        if (cv::norm(boxCorners[i] - point) < 1e-3f) {
            return static_cast<int>(i) + 1;
        }
    }
    return -1;
}

class StableFacePoseTracker {
public:
    bool reset(const cv::Mat& frame,
               const std::vector<cv::Point3f>& boxCorners,
               const std::vector<cv::Point3f>& objectPoints,
               const std::vector<cv::Point2f>& imagePoints) {
        corners_.clear();
        boxCorners_ = boxCorners;
        cv::cvtColor(frame, previousGray_, cv::COLOR_BGR2GRAY);

        for (size_t i = 0; i < objectPoints.size(); ++i) {
            const int id = findCornerId(boxCorners, objectPoints[i]);
            if (id < 0) {
                continue;
            }
            StableTrackedCorner corner;
            corner.id = id;
            corner.initialPoint = imagePoints[i];
            corner.point = imagePoints[i];
            corner.valid = true;
            corners_[id] = corner;
        }

        initialized_ = hasCorner(1) && hasCorner(2) && hasCorner(5) && hasCorner(6);
        return initialized_;
    }

    bool isReady() const {
        return initialized_;
    }

    StableFaceTrackResult update(const cv::Mat& frame,
                                 const cv::Mat& cameraMatrix,
                                 const cv::Mat& distCoeffs,
                                 const cv::Mat& poseGuessRvec,
                                 const cv::Mat& poseGuessTvec,
                                 double maxReprojectionError) {
        StableFaceTrackResult result;
        if (!initialized_ || previousGray_.empty()) {
            return result;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        trackStableAnchors(gray);
        recoverCorner5FromFrontFace();
        previousGray_ = gray;

        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        if (!collectFrontFaceCorrespondences(objectPoints, imagePoints)) {
            return result;
        }

        if (poseGuessRvec.empty() || poseGuessTvec.empty()) {
            return result;
        }

        cv::Mat candidateRvec = poseGuessRvec.clone();
        cv::Mat candidateTvec = poseGuessTvec.clone();
        try {
            if (!cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                              candidateRvec, candidateTvec, true, cv::SOLVEPNP_ITERATIVE)) {
                return result;
            }
            cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                                 candidateRvec, candidateTvec);
        } catch (const cv::Exception&) {
            return result;
        }

        const double reprojectionError = cvproject::computeMeanReprojectionError(
            objectPoints, imagePoints, candidateRvec, candidateTvec, cameraMatrix, distCoeffs);
        if (reprojectionError > maxReprojectionError) {
            return result;
        }

        result.ok = true;
        result.rvec = candidateRvec;
        result.tvec = candidateTvec;
        result.reprojectionError = reprojectionError;
        result.trackedPoints = static_cast<int>(imagePoints.size());
        result.objectPoints = objectPoints;
        result.imagePoints = imagePoints;
        return result;
    }

private:
    bool hasCorner(int id) const {
        const auto it = corners_.find(id);
        return it != corners_.end() && it->second.valid;
    }

    bool getCornerPoint(int id, cv::Point2f& point) const {
        const auto it = corners_.find(id);
        if (it == corners_.end() || !it->second.valid) {
            return false;
        }
        point = it->second.point;
        return true;
    }

    void trackStableAnchors(const cv::Mat& gray) {
        const std::array<int, 3> stableIds = {1, 2, 6};
        std::vector<int> activeIds;
        std::vector<cv::Point2f> previousPoints;
        for (int id : stableIds) {
            const auto it = corners_.find(id);
            if (it == corners_.end() || !it->second.valid) {
                continue;
            }
            activeIds.push_back(id);
            previousPoints.push_back(it->second.point);
        }

        if (previousPoints.empty()) {
            return;
        }

        std::vector<cv::Point2f> nextPoints;
        std::vector<cv::Point2f> backPoints;
        std::vector<unsigned char> statusForward;
        std::vector<unsigned char> statusBackward;
        std::vector<float> errorForward;
        std::vector<float> errorBackward;

        const cv::Size winSize(31, 31);
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(previousGray_, gray, previousPoints, nextPoints,
                                 statusForward, errorForward, winSize, 4, criteria);
        cv::calcOpticalFlowPyrLK(gray, previousGray_, nextPoints, backPoints,
                                 statusBackward, errorBackward, winSize, 4, criteria);

        for (size_t i = 0; i < activeIds.size(); ++i) {
            StableTrackedCorner& corner = corners_[activeIds[i]];
            const bool inside =
                nextPoints[i].x >= 0.0f && nextPoints[i].x < static_cast<float>(gray.cols) &&
                nextPoints[i].y >= 0.0f && nextPoints[i].y < static_cast<float>(gray.rows);
            const double fbError = cv::norm(previousPoints[i] - backPoints[i]);
            if (statusForward[i] != 0 && statusBackward[i] != 0 && inside && fbError <= 2.5) {
                corner.point = nextPoints[i];
                corner.lostCount = 0;
                corner.valid = true;
            } else {
                ++corner.lostCount;
                if (corner.lostCount > 5) {
                    corner.valid = false;
                }
            }
        }
    }

    void recoverCorner5FromFrontFace() {
        cv::Point2f current1;
        cv::Point2f current2;
        cv::Point2f current6;
        if (!getCornerPoint(1, current1) || !getCornerPoint(2, current2) ||
            !getCornerPoint(6, current6) || !hasCorner(5)) {
            return;
        }

        const std::vector<cv::Point2f> source = {
            corners_[1].initialPoint,
            corners_[2].initialPoint,
            corners_[6].initialPoint
        };
        const std::vector<cv::Point2f> target = {current1, current2, current6};
        const cv::Mat affine = cv::getAffineTransform(source, target);

        std::vector<cv::Point2f> source5 = {corners_[5].initialPoint};
        std::vector<cv::Point2f> predicted5;
        cv::transform(source5, predicted5, affine);
        corners_[5].point = predicted5.front();
        corners_[5].valid = true;
        corners_[5].lostCount = 0;
    }

    bool collectFrontFaceCorrespondences(std::vector<cv::Point3f>& objectPoints,
                                         std::vector<cv::Point2f>& imagePoints) const {
        objectPoints.clear();
        imagePoints.clear();
        const std::array<int, 4> ids = {1, 2, 5, 6};
        for (int id : ids) {
            const auto it = corners_.find(id);
            if (it == corners_.end() || !it->second.valid ||
                id < 1 || id > static_cast<int>(boxCorners_.size())) {
                return false;
            }
            objectPoints.push_back(boxCorners_[id - 1]);
            imagePoints.push_back(it->second.point);
        }
        return true;
    }

    bool initialized_ = false;
    cv::Mat previousGray_;
    std::vector<cv::Point3f> boxCorners_;
    std::map<int, StableTrackedCorner> corners_;
};

bool solvePoseFromManualInput(const cv::Mat& frame,
                              const std::vector<cv::Point3f>& boxCorners,
                              const cv::Mat& cameraMatrix,
                              const cv::Mat& distCoeffs,
                              std::vector<cv::Point3f>& objectPoints,
                              std::vector<cv::Point2f>& imagePoints,
                              cv::Mat& rvec,
                              cv::Mat& tvec,
                              const std::string& prompt) {
    if (!collectManualCorrespondences(frame, boxCorners, objectPoints, imagePoints, prompt)) {
        return false;
    }

    return cvproject::solvePosePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                                   rvec, tvec, false, true);
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path videoPath = projectRoot / "data/raw/videos/input.mp4";
    const std::filesystem::path calibPath = projectRoot / "config/camera_calib.yml";
    const std::filesystem::path objectConfigPath = projectRoot / "config/object_01.yml";
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path referenceKeypointsPath = referenceDir / "reference_keypoints.yml";
    const std::filesystem::path referencePosesPath = referenceDir / "reference_poses.yml";
    const std::filesystem::path resultPath = projectRoot / "data/processed/ar/pnp_first_frame.jpg";
    const std::filesystem::path arVideoPath = projectRoot / "data/processed/ar/output_ar.mp4";
    const std::filesystem::path poseCsvPath = projectRoot / "data/processed/poses/poses.csv";
    const std::filesystem::path manualKeypointsPath = projectRoot / "data/processed/matches/manual_keypoints_frame000.yml";
    const std::filesystem::path transferredKeypointsPath = projectRoot / "data/processed/matches/transferred_keypoints.yml";

    cv::FileStorage fs(calibPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open calibration file: " << calibPath.string() << std::endl;
        return 1;
    }

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["dist_coeffs"] >> distCoeffs;
    fs.release();

    if (cameraMatrix.empty() || distCoeffs.empty()) {
        std::cerr << "Invalid calibration file: missing camera_matrix or dist_coeffs" << std::endl;
        return 1;
    }

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath.string() << std::endl;
        return 1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to read the first frame from: " << videoPath.string() << std::endl;
        return 1;
    }

    cv::FileStorage objectFs(objectConfigPath.string(), cv::FileStorage::READ);
    if (!objectFs.isOpened()) {
        std::cerr << "Failed to open object config: " << objectConfigPath.string() << std::endl;
        return 1;
    }

    std::string meshPathText;
    float widthMm = 0.0f;
    float heightMm = 0.0f;
    float depthMm = 0.0f;
    objectFs["mesh_path"] >> meshPathText;
    objectFs["width_mm"] >> widthMm;
    objectFs["height_mm"] >> heightMm;
    objectFs["depth_mm"] >> depthMm;
    objectFs.release();

    const std::filesystem::path meshPath = projectRoot / meshPathText;
    const cvproject::Mesh mesh = cvproject::loadObjMesh(meshPath);
    const float meshScale = cvproject::inferObjToMillimeterScale(mesh, widthMm, heightMm, depthMm);
    const cvproject::BoundingBox bbox = cvproject::computeBoundingBox(mesh, meshScale);
    cvproject::printMeshSummary(mesh, bbox);
    const std::vector<cv::Point3f> boxCorners = bbox.corners;

    std::cout << "3D corner index order:" << std::endl;
    for (size_t i = 0; i < boxCorners.size(); ++i) {
        std::cout << (i + 1) << " [" << boxCorners[i].x << ", "
                  << boxCorners[i].y << ", " << boxCorners[i].z << "]" << std::endl;
    }

    std::vector<cv::Point3f> selectedObjectPoints;
    std::vector<cv::Point2f> selectedImagePoints;
    cv::Mat rvec;
    cv::Mat tvec;
    bool poseInitialized = false;

    if (loadManualCorrespondences(transferredKeypointsPath, boxCorners, selectedObjectPoints, selectedImagePoints)) {
        std::cout << "Loaded transferred keypoints from: " << transferredKeypointsPath.string() << std::endl;
        if (!cvproject::hasNonCoplanarObjectPoints(selectedObjectPoints)) {
            cv::Mat transferredPreview = frame.clone();
            for (size_t i = 0; i < selectedImagePoints.size(); ++i) {
                cv::circle(transferredPreview, selectedImagePoints[i], 5, cv::Scalar(0, 255, 255), -1);
            }
            cv::putText(transferredPreview,
                        "Transferred points are coplanar. Press r for manual 6D initialization.",
                        cv::Point(30, 45), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            cv::namedWindow("Transferred Keypoint Hints", cv::WINDOW_NORMAL);
            cv::resizeWindow("Transferred Keypoint Hints", 1280, 720);
            cv::imshow("Transferred Keypoint Hints", transferredPreview);
            cv::waitKey(1000);
            cv::destroyWindow("Transferred Keypoint Hints");
            selectedObjectPoints.clear();
            selectedImagePoints.clear();
            std::cout << "Transferred keypoints are on one plane, so they are used as hints only." << std::endl;
            std::cout << "Falling back to saved/manual non-coplanar correspondences for full 6D pose." << std::endl;
        }
    }

    if (!selectedImagePoints.empty()) {
        if (cvproject::solvePosePnP(selectedObjectPoints, selectedImagePoints, cameraMatrix, distCoeffs,
                                    rvec, tvec, false, true)) {
            cv::Mat candidate = frame.clone();
            drawProjectedBox(candidate, boxCorners, rvec, tvec, cameraMatrix, distCoeffs);
            drawAxis(candidate, rvec, tvec, cameraMatrix, distCoeffs, 50.0f);
            const double candidateError = cvproject::computeMeanReprojectionError(
                selectedObjectPoints, selectedImagePoints, rvec, tvec, cameraMatrix, distCoeffs);
            cv::putText(candidate, "Transferred PnP candidate | y accept | r manual",
                        cv::Point(30, 45), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
            cv::putText(candidate, "Reproj err: " + std::to_string(candidateError) + " px",
                        cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);

            cv::namedWindow("Transferred PnP Candidate", cv::WINDOW_NORMAL);
            cv::resizeWindow("Transferred PnP Candidate", 1280, 720);
            cv::imshow("Transferred PnP Candidate", candidate);
            const int key = cv::waitKey(0);
            cv::destroyWindow("Transferred PnP Candidate");
            if (key == 'y' || key == 'Y') {
                poseInitialized = true;
                std::cout << "Accepted transferred keypoint initialization." << std::endl;
            } else {
                selectedObjectPoints.clear();
                selectedImagePoints.clear();
                std::cout << "Transferred initialization rejected. Falling back to manual/saved keypoints." << std::endl;
            }
        } else {
            selectedObjectPoints.clear();
            selectedImagePoints.clear();
            std::cout << "Transferred keypoint solvePnP failed. Falling back to manual/saved keypoints." << std::endl;
        }
    }

    const bool loadedManual = !poseInitialized && loadManualCorrespondences(
        manualKeypointsPath, boxCorners, selectedObjectPoints, selectedImagePoints);
    if (!poseInitialized && loadedManual) {
        std::cout << "Loaded manual correspondences from: " << manualKeypointsPath.string() << std::endl;
        std::cout << "Press r to re-annotate, or any other key to use the saved correspondences." << std::endl;
        cv::namedWindow("Saved Correspondences Preview", cv::WINDOW_NORMAL);
        cv::resizeWindow("Saved Correspondences Preview", 1280, 720);
        cv::Mat preview = frame.clone();
        for (size_t i = 0; i < selectedImagePoints.size(); ++i) {
            cv::circle(preview, selectedImagePoints[i], 5, cv::Scalar(0, 255, 255), -1);
        }
        cv::imshow("Saved Correspondences Preview", preview);
        const int key = cv::waitKey(1000);
        cv::destroyWindow("Saved Correspondences Preview");
        if (key == 'r' || key == 'R') {
            selectedObjectPoints.clear();
            selectedImagePoints.clear();
        }
    }

    if (!poseInitialized && selectedImagePoints.empty()) {
        if (!collectManualCorrespondences(frame, boxCorners, selectedObjectPoints, selectedImagePoints,
                                          "Initial manual PnP annotation.")) {
            return 1;
        }
        saveManualCorrespondences(manualKeypointsPath, boxCorners, selectedObjectPoints, selectedImagePoints);
        std::cout << "Saved manual correspondences to: " << manualKeypointsPath.string() << std::endl;
    } else if (!poseInitialized) {
        std::cout << "Using saved manual correspondences." << std::endl;
    }

    if (!poseInitialized) {
        if (!cvproject::solvePosePnP(
            selectedObjectPoints,
            selectedImagePoints,
            cameraMatrix,
            distCoeffs,
            rvec,
            tvec,
            false,
            true
        )) {
            std::cerr << "solvePnP failed." << std::endl;
            return 1;
        }
    }

    cv::Mat result = frame.clone();
    drawProjectedBox(result, boxCorners, rvec, tvec, cameraMatrix, distCoeffs);
    drawAxis(result, rvec, tvec, cameraMatrix, distCoeffs, 50.0f);

    double meanError = cvproject::computeMeanReprojectionError(
        selectedObjectPoints, selectedImagePoints, rvec, tvec, cameraMatrix, distCoeffs);

    std::cout << "solvePnP succeeded." << std::endl;
    std::cout << "rvec:\n" << rvec << std::endl;
    std::cout << "tvec:\n" << tvec << std::endl;
    std::cout << "Mean reprojection error: " << meanError << " px" << std::endl;

    cv::putText(result, "Mean reprojection error: " + std::to_string(meanError) + " px",
                cv::Point(30, 45), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

    std::filesystem::create_directories(resultPath.parent_path());
    cv::imwrite(resultPath.string(), result);
    std::cout << "Saved result to: " << resultPath.string() << std::endl;

    cv::imshow("PnP Result", result);
    cv::waitKey(500);

    std::filesystem::create_directories(arVideoPath.parent_path());
    std::filesystem::create_directories(poseCsvPath.parent_path());

    const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    cv::VideoWriter writer(
        arVideoPath.string(),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        frame.size()
    );

    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << arVideoPath.string() << std::endl;
        return 1;
    }

    std::ofstream poseCsv(poseCsvPath);
    if (!poseCsv.is_open()) {
        std::cerr << "Failed to open pose csv: " << poseCsvPath.string() << std::endl;
        return 1;
    }

    poseCsv << "frame_id,timestamp_sec,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,mean_reprojection_error,tracked_points,manual_relocalized,reference_relocalized,stable_face_tracked,lost_count,reference_name,reference_inliers,reference_good_matches,reference_reproj_error,gtsam_enabled\n";

    cvproject::ReferenceTrackingConfig trackingConfig;
    cv::Ptr<cv::ORB> trackingOrb = cv::ORB::create(trackingConfig.orbFeatures);
    std::vector<cvproject::ReferenceView> referenceViews =
        cvproject::loadReferenceViews(referenceDir, referenceKeypointsPath, trackingOrb);

    std::vector<cvproject::ReferencePoseEntry> referencePoses;
    if (!cvproject::loadReferencePoses(referencePosesPath, referencePoses)) {
        std::cout << "reference_poses.yml not found. Run prepare_reference_poses first." << std::endl;
        referencePoses = cvproject::computeReferencePoses(
            referenceViews, boxCorners, cameraMatrix, distCoeffs);
        cvproject::saveReferencePoses(referencePosesPath, referencePoses);
        std::cout << "Generated reference poses at: " << referencePosesPath.string() << std::endl;
    }
    cvproject::attachReferencePoses(referenceViews, referencePoses);

    cvproject::ViewpointSelector viewpointSelector(trackingConfig);
    std::cout << "Loaded " << referenceViews.size() << " annotated reference views for tracking." << std::endl;
    if (referenceViews.empty()) {
        std::cerr << "No usable reference views were loaded. Tracking will rely on manual initialization only." << std::endl;
    }

    cv::Mat currentRvec = rvec.clone();
    cv::Mat currentTvec = tvec.clone();
    StableFacePoseTracker stableFaceTracker;
    const bool stableFaceTrackerReady = stableFaceTracker.reset(
        frame, boxCorners, selectedObjectPoints, selectedImagePoints);
    if (stableFaceTrackerReady) {
        std::cout << "Stable face tracker initialized from corners 1/2/5/6." << std::endl;
    } else {
        std::cout << "Stable face tracker not initialized. Need corners 1/2/5/6 in manual initialization." << std::endl;
    }
    GtsamPoseSmoother poseSmoother(trackingConfig);
    double lastAcceptedError = meanError;
    int lostCount = 0;
    int framesSinceRelocalize = 0;
    std::optional<std::string> lockedReference;

    int frameId = 0;
    while (true) {
        bool manualRelocalizedThisFrame = false;
        bool referenceRelocalizedThisFrame = false;
        std::string acceptedReferenceName = lockedReference.value_or("none");
        int acceptedReferenceInliers = 0;
        int acceptedReferenceGoodMatches = 0;
        double acceptedReferenceReprojError = -1.0;
        bool hasFreshPoseMeasurement = frameId == 0;
        bool stableFaceTrackedThisFrame = false;
        cv::Mat measuredRvec = currentRvec.clone();
        cv::Mat measuredTvec = currentTvec.clone();

        double holdPoseError = cvproject::computeMeanReprojectionError(
            selectedObjectPoints, selectedImagePoints, currentRvec, currentTvec, cameraMatrix, distCoeffs);
        if (frameId > 0 && stableFaceTracker.isReady()) {
            const StableFaceTrackResult stablePose = stableFaceTracker.update(
                frame,
                cameraMatrix,
                distCoeffs,
                currentRvec,
                currentTvec,
                trackingConfig.maxReprojectionError
            );

            if (stablePose.ok) {
                measuredRvec = stablePose.rvec.clone();
                measuredTvec = stablePose.tvec.clone();
                selectedObjectPoints = stablePose.objectPoints;
                selectedImagePoints = stablePose.imagePoints;
                holdPoseError = stablePose.reprojectionError;
                lastAcceptedError = stablePose.reprojectionError;
                lostCount = 0;
                hasFreshPoseMeasurement = true;
                stableFaceTrackedThisFrame = true;
                acceptedReferenceName = "stable_face";
                acceptedReferenceReprojError = stablePose.reprojectionError;
            }
        }

        const bool shouldRelocalize = frameId == 0
            ? false
            : (!referenceViews.empty() &&
               !stableFaceTrackedThisFrame &&
               (lostCount > 0 ||
                framesSinceRelocalize >= trackingConfig.relocalizeIntervalFrames ||
                holdPoseError > trackingConfig.maxReprojectionError + trackingConfig.maxAcceptedErrorIncrease));

        if (shouldRelocalize) {
            const cvproject::ViewpointMatchResult relocalization = viewpointSelector.localizeFrame(
                frame,
                referenceViews,
                trackingOrb,
                boxCorners,
                cameraMatrix,
                distCoeffs,
                lockedReference,
                true,
                currentRvec
            );

            if (relocalization.ok) {
                cv::Mat relocalizedRvec = relocalization.rvec.clone();
                cv::Mat relocalizedTvec = relocalization.tvec.clone();
                const bool rejectJump = cvproject::isPoseJumpTooLarge(
                    currentRvec,
                    currentTvec,
                    relocalizedRvec,
                    relocalizedTvec,
                    trackingConfig.translationJumpThresholdMm,
                    trackingConfig.rotationJumpThresholdRad
                ) && lostCount == 0;
                const bool rejectErrorIncrease =
                    relocalization.reprojectionError >
                    lastAcceptedError + trackingConfig.maxAcceptedErrorIncrease && lostCount == 0;

                if (rejectJump) {
                    ++lostCount;
                    std::cout << "Reference pose rejected at frame " << frameId
                              << " due to pose jump." << std::endl;
                } else if (rejectErrorIncrease) {
                    ++lostCount;
                    std::cout << "Reference pose rejected at frame " << frameId
                              << " because reprojection error increased from "
                              << lastAcceptedError << " to "
                              << relocalization.reprojectionError << " px." << std::endl;
                } else {
                    measuredRvec = relocalizedRvec;
                    measuredTvec = relocalizedTvec;
                    selectedObjectPoints = relocalization.objectPoints;
                    selectedImagePoints = relocalization.imagePoints;
                    stableFaceTracker.reset(frame, boxCorners, selectedObjectPoints, selectedImagePoints);
                    lockedReference = relocalization.referenceName;
                    lostCount = 0;
                    framesSinceRelocalize = 0;
                    hasFreshPoseMeasurement = true;
                    referenceRelocalizedThisFrame = true;
                    acceptedReferenceName = relocalization.referenceName;
                    acceptedReferenceInliers = relocalization.inliers;
                    acceptedReferenceGoodMatches = relocalization.goodMatches;
                    acceptedReferenceReprojError = relocalization.reprojectionError;
                    lastAcceptedError = relocalization.reprojectionError;
                    std::cout << "Reference pose accepted at frame " << frameId
                              << " using " << relocalization.referenceName
                              << " with " << relocalization.inliers << " inliers, "
                              << relocalization.goodMatches << " good matches, reproj "
                              << relocalization.reprojectionError << " px." << std::endl;
                }
            } else {
                ++lostCount;
            }
        } else if (frameId > 0) {
            ++framesSinceRelocalize;
        }

        if (lostCount > trackingConfig.maxLostBeforeManual) {
            std::cout << "Reference tracking lost for " << lostCount
                      << " frames. Manual relocalization required." << std::endl;
            if (!solvePoseFromManualInput(frame, boxCorners, cameraMatrix, distCoeffs,
                                          selectedObjectPoints, selectedImagePoints,
                                          currentRvec, currentTvec,
                                          "Manual relocalization.")) {
                std::cerr << "Manual relocalization solvePnP failed." << std::endl;
                break;
            }
            manualRelocalizedThisFrame = true;
            lostCount = 0;
            framesSinceRelocalize = 0;
            lockedReference.reset();
            measuredRvec = currentRvec.clone();
            measuredTvec = currentTvec.clone();
            hasFreshPoseMeasurement = true;
            stableFaceTracker.reset(frame, boxCorners, selectedObjectPoints, selectedImagePoints);
            lastAcceptedError = cvproject::computeMeanReprojectionError(
                selectedObjectPoints, selectedImagePoints, currentRvec, currentTvec, cameraMatrix, distCoeffs);
        }

        const gtsam::Pose3 measuredPose = cvPoseToGtsamPose(measuredRvec, measuredTvec);
        const gtsam::Pose3 optimizedPose =
            poseSmoother.update(frameId, measuredPose, hasFreshPoseMeasurement);
        gtsamPoseToCvPose(optimizedPose, currentRvec, currentTvec);

        cv::Mat output = frame.clone();
        drawProjectedBox(output, boxCorners, currentRvec, currentTvec, cameraMatrix, distCoeffs);
        drawAxis(output, currentRvec, currentTvec, cameraMatrix, distCoeffs, 50.0f);

        const double timestampSec = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        const double currentError = cvproject::computeMeanReprojectionError(
            selectedObjectPoints, selectedImagePoints, currentRvec, currentTvec, cameraMatrix, distCoeffs);

        cv::putText(output, "Frame: " + std::to_string(frameId),
                    cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, "Reproj err: " + std::to_string(currentError) + " px",
                    cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, "r: manual relocalize | Esc: quit",
                    cv::Point(30, 110), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, "tracking: stable face pose + reference fallback",
                    cv::Point(30, 145), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
        if (stableFaceTrackedThisFrame) {
            cv::putText(output, "stable face pose: 1/2/6 LK + 5 recovered",
                        cv::Point(30, 250), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 0), 2);
        }
        cv::putText(output, "locked ref: " + acceptedReferenceName,
                    cv::Point(30, 180), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, "lost count: " + std::to_string(lostCount),
                    cv::Point(30, 215), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
        if (referenceRelocalizedThisFrame) {
            cv::putText(output, "ref inliers=" + std::to_string(acceptedReferenceInliers),
                        cv::Point(30, stableFaceTrackedThisFrame ? 285 : 250),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
        }

        writer.write(output);
        poseCsv << frameId << ','
                << timestampSec << ','
                << currentRvec.at<double>(0) << ','
                << currentRvec.at<double>(1) << ','
                << currentRvec.at<double>(2) << ','
                << currentTvec.at<double>(0) << ','
                << currentTvec.at<double>(1) << ','
                << currentTvec.at<double>(2) << ','
                << currentError << ','
                << selectedImagePoints.size() << ','
                << (manualRelocalizedThisFrame ? 1 : 0) << ','
                << (referenceRelocalizedThisFrame ? 1 : 0) << ','
                << (stableFaceTrackedThisFrame ? 1 : 0) << ','
                << lostCount << ','
                << acceptedReferenceName << ','
                << acceptedReferenceInliers << ','
                << acceptedReferenceGoodMatches << ','
                << acceptedReferenceReprojError << ','
                << 1 << '\n';

        cv::imshow("AR Pose Tracking", output);
        const int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
        if (key == 'r' || key == 'R') {
            std::cout << "Manual relocalization requested at frame " << frameId << "." << std::endl;
            if (!solvePoseFromManualInput(frame, boxCorners, cameraMatrix, distCoeffs,
                                          selectedObjectPoints, selectedImagePoints,
                                          currentRvec, currentTvec,
                                          "Manual relocalization requested by user.")) {
                std::cerr << "Manual relocalization failed or was canceled." << std::endl;
                break;
            }
            lostCount = 0;
            framesSinceRelocalize = 0;
            lockedReference.reset();
            measuredRvec = currentRvec.clone();
            measuredTvec = currentTvec.clone();
            stableFaceTracker.reset(frame, boxCorners, selectedObjectPoints, selectedImagePoints);
            lastAcceptedError = cvproject::computeMeanReprojectionError(
                selectedObjectPoints, selectedImagePoints, currentRvec, currentTvec, cameraMatrix, distCoeffs);
        }

        if (!cap.read(frame)) {
            break;
        }
        ++frameId;
    }

    writer.release();
    poseCsv.close();

    std::cout << "Saved AR video to: " << arVideoPath.string() << std::endl;
    std::cout << "Saved pose CSV to: " << poseCsvPath.string() << std::endl;

    return 0;
}
