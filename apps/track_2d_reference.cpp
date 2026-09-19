#include <opencv2/opencv.hpp>

#include "cvproject/mesh_loader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct ReferencePoint {
    int id = -1;
    cv::Point2f point;
};

struct ReferenceView {
    std::string name;
    cv::Mat imageGray;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<ReferencePoint> annotatedPoints;
};

struct TrackingConfig {
    int orbFeatures = 2000;
    int minGoodMatches = 12;
    int minHomographyInliers = 25;
    double ratioTest = 0.75;
    double homographyRansacThreshold = 4.0;
    double maxReprojectionError = 8.0;
};

struct FramePose {
    bool valid = false;
    cv::Mat rvec;
    cv::Mat tvec;
    std::string referenceName;
    int inliers = 0;
    double reprojError = 0.0;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
};

cv::Point toPixel(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

double computeMeanReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Mat& rvec,
                                    const cv::Mat& tvec,
                                    const cv::Mat& cameraMatrix,
                                    const cv::Mat& distCoeffs) {
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, reprojected);

    double meanError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i) {
        meanError += cv::norm(imagePoints[i] - reprojected[i]);
    }
    return meanError / static_cast<double>(imagePoints.size());
}

bool hasNonCoplanarObjectPoints(const std::vector<cv::Point3f>& points) {
    if (points.size() < 4) {
        return false;
    }

    const cv::Point3f origin = points[0];
    for (size_t i = 1; i + 2 < points.size(); ++i) {
        const cv::Point3f a = points[i] - origin;
        for (size_t j = i + 1; j + 1 < points.size(); ++j) {
            const cv::Point3f b = points[j] - origin;
            const cv::Point3f normal = a.cross(b);
            if (cv::norm(normal) < 1e-4f) {
                continue;
            }

            for (size_t k = j + 1; k < points.size(); ++k) {
                const cv::Point3f c = points[k] - origin;
                if (std::abs(normal.dot(c)) > 1e-2f) {
                    return true;
                }
            }
        }
    }

    return false;
}

void drawProjectedBox(cv::Mat& image,
                      const std::vector<cv::Point3f>& boxCorners,
                      const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      const cv::Mat& cameraMatrix,
                      const cv::Mat& distCoeffs,
                      const cv::Scalar& color) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(boxCorners, rvec, tvec, cameraMatrix, distCoeffs, projected);

    const std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    for (const auto& edge : edges) {
        cv::line(image, toPixel(projected[edge.first]), toPixel(projected[edge.second]),
                 color, 2, cv::LINE_AA);
    }
}

void drawTrackedPoints(cv::Mat& image, const std::vector<cv::Point2f>& points) {
    for (size_t i = 0; i < points.size(); ++i) {
        cv::circle(image, toPixel(points[i]), 5, cv::Scalar(0, 255, 255), -1);
    }
}

std::vector<std::filesystem::path> listReferenceImages(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::exists(dir)) {
        return paths;
    }

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string lowerFilename = entry.path().filename().string();
        std::transform(lowerFilename.begin(), lowerFilename.end(), lowerFilename.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowerFilename.find("preview") != std::string::npos ||
            lowerFilename.find("keypoints") != std::string::npos ||
            lowerFilename.find("transferred") != std::string::npos) {
            continue;
        }

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

bool loadReferenceKeypointsForImage(const std::filesystem::path& path,
                                    const std::string& targetImageName,
                                    std::vector<ReferencePoint>& points) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    points.clear();
    cv::FileNode references = fs["references"];
    if (!references.empty()) {
        for (const cv::FileNode& refNode : references) {
            std::string imageName;
            refNode["image"] >> imageName;
            if (imageName != targetImageName) {
                continue;
            }

            for (const cv::FileNode& node : refNode["points"]) {
                ReferencePoint point;
                point.id = static_cast<int>(node["id"]);
                point.point.x = static_cast<float>(node["x"]);
                point.point.y = static_cast<float>(node["y"]);
                points.push_back(point);
            }
            return !points.empty();
        }
        return false;
    }

    return false;
}

std::vector<ReferenceView> loadReferenceViews(const std::filesystem::path& referenceDir,
                                              const std::filesystem::path& referenceKeypointsPath,
                                              const cv::Ptr<cv::ORB>& orb) {
    std::vector<ReferenceView> views;
    for (const std::filesystem::path& refPath : listReferenceImages(referenceDir)) {
        cv::Mat refImage = cv::imread(refPath.string(), cv::IMREAD_COLOR);
        if (refImage.empty()) {
            continue;
        }

        ReferenceView view;
        view.name = refPath.filename().string();
        if (!loadReferenceKeypointsForImage(referenceKeypointsPath, view.name, view.annotatedPoints)) {
            continue;
        }

        cv::cvtColor(refImage, view.imageGray, cv::COLOR_BGR2GRAY);
        orb->detectAndCompute(view.imageGray, cv::noArray(), view.keypoints, view.descriptors);
        if (view.descriptors.empty()) {
            continue;
        }

        views.push_back(view);
    }
    return views;
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
    for (const cv::FileNode& pointNode : fs["points"]) {
        const int id = static_cast<int>(pointNode["id"]) - 1;
        if (id < 0 || id >= static_cast<int>(allCorners.size())) {
            continue;
        }
        objectPoints.push_back(allCorners[id]);
        imagePoints.emplace_back(static_cast<float>(pointNode["x"]), static_cast<float>(pointNode["y"]));
    }

    return imagePoints.size() >= 4;
}

FramePose estimatePoseFromReferences(const cv::Mat& frame,
                                     const std::vector<ReferenceView>& referenceViews,
                                     const cv::Ptr<cv::ORB>& orb,
                                     const TrackingConfig& config,
                                     const std::vector<cv::Point3f>& boxCorners,
                                     const cv::Mat& cameraMatrix,
                                     const cv::Mat& distCoeffs) {
    FramePose bestPose;

    cv::Mat frameGray;
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb->detectAndCompute(frameGray, cv::noArray(), frameKeypoints, frameDescriptors);
    if (frameDescriptors.empty()) {
        return bestPose;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    for (const ReferenceView& referenceView : referenceViews) {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(referenceView.descriptors, frameDescriptors, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const std::vector<cv::DMatch>& pair : knnMatches) {
            if (pair.size() < 2) {
                continue;
            }
            if (pair[0].distance < config.ratioTest * pair[1].distance) {
                goodMatches.push_back(pair[0]);
            }
        }

        if (goodMatches.size() < static_cast<size_t>(config.minGoodMatches)) {
            continue;
        }

        std::vector<cv::Point2f> refPoints;
        std::vector<cv::Point2f> framePoints;
        for (const cv::DMatch& match : goodMatches) {
            refPoints.push_back(referenceView.keypoints[match.queryIdx].pt);
            framePoints.push_back(frameKeypoints[match.trainIdx].pt);
        }

        cv::Mat inlierMask;
        cv::Mat homography = cv::findHomography(
            refPoints, framePoints, cv::RANSAC, config.homographyRansacThreshold, inlierMask);
        if (homography.empty() || inlierMask.empty()) {
            continue;
        }

        int inliers = 0;
        for (int i = 0; i < inlierMask.rows; ++i) {
            if (inlierMask.at<unsigned char>(i, 0)) {
                ++inliers;
            }
        }
        if (inliers < config.minHomographyInliers || inliers <= bestPose.inliers) {
            continue;
        }

        std::vector<cv::Point2f> sourcePoints;
        std::vector<cv::Point3f> objectPoints;
        for (const ReferencePoint& point : referenceView.annotatedPoints) {
            const int cornerIndex = point.id - 1;
            if (cornerIndex < 0 || cornerIndex >= static_cast<int>(boxCorners.size())) {
                continue;
            }
            sourcePoints.push_back(point.point);
            objectPoints.push_back(boxCorners[cornerIndex]);
        }

        if (sourcePoints.size() < 4 || !hasNonCoplanarObjectPoints(objectPoints)) {
            continue;
        }

        std::vector<cv::Point2f> transferredPoints;
        cv::perspectiveTransform(sourcePoints, transferredPoints, homography);

        cv::Mat rvec;
        cv::Mat tvec;
        const int pnpMethod = transferredPoints.size() >= 6 ? cv::SOLVEPNP_ITERATIVE : cv::SOLVEPNP_EPNP;
        if (!cv::solvePnP(objectPoints, transferredPoints, cameraMatrix, distCoeffs,
                          rvec, tvec, false, pnpMethod)) {
            continue;
        }

        const double reprojError = computeMeanReprojectionError(
            objectPoints, transferredPoints, rvec, tvec, cameraMatrix, distCoeffs);
        if (reprojError > config.maxReprojectionError) {
            continue;
        }

        bestPose.valid = true;
        bestPose.referenceName = referenceView.name;
        bestPose.inliers = inliers;
        bestPose.reprojError = reprojError;
        bestPose.rvec = rvec;
        bestPose.tvec = tvec;
        bestPose.objectPoints = objectPoints;
        bestPose.imagePoints = transferredPoints;
    }

    return bestPose;
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path videoPath = projectRoot / "data/raw/videos/input.mp4";
    const std::filesystem::path calibPath = projectRoot / "config/camera_calib.yml";
    const std::filesystem::path objectConfigPath = projectRoot / "config/object_01.yml";
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path referenceKeypointsPath = referenceDir / "reference_keypoints.yml";
    const std::filesystem::path manualKeypointsPath = projectRoot / "data/processed/matches/manual_keypoints_frame000.yml";
    const std::filesystem::path outputVideoPath = projectRoot / "data/processed/ar/output_2d_reference_track.mp4";
    const std::filesystem::path outputCsvPath = projectRoot / "data/processed/validation/track_2d_reference.csv";

    cv::FileStorage calibFs(calibPath.string(), cv::FileStorage::READ);
    if (!calibFs.isOpened()) {
        std::cerr << "Failed to open calibration file: " << calibPath.string() << std::endl;
        return 1;
    }

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    calibFs["camera_matrix"] >> cameraMatrix;
    calibFs["dist_coeffs"] >> distCoeffs;
    calibFs.release();

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

    const cvproject::Mesh mesh = cvproject::loadObjMesh(projectRoot / meshPathText);
    const float meshScale = cvproject::inferObjToMillimeterScale(mesh, widthMm, heightMm, depthMm);
    const cvproject::BoundingBox bbox = cvproject::computeBoundingBox(mesh, meshScale);
    const std::vector<cv::Point3f> boxCorners = bbox.corners;

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath.string() << std::endl;
        return 1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to read first frame." << std::endl;
        return 1;
    }

    TrackingConfig config;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config.orbFeatures);
    const std::vector<ReferenceView> referenceViews =
        loadReferenceViews(referenceDir, referenceKeypointsPath, orb);
    if (referenceViews.empty()) {
        std::cerr << "No annotated reference views loaded." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << referenceViews.size()
              << " reference views. Pure 2D homography + PnP tracking (no GTSAM)." << std::endl;

    std::vector<cv::Point3f> initObjectPoints;
    std::vector<cv::Point2f> initImagePoints;
    if (!loadManualCorrespondences(manualKeypointsPath, boxCorners, initObjectPoints, initImagePoints)) {
        std::cerr << "Missing manual first-frame keypoints: " << manualKeypointsPath.string() << std::endl;
        std::cerr << "Run pose_estimation or annotate frame 0 first." << std::endl;
        return 1;
    }

    cv::Mat currentRvec;
    cv::Mat currentTvec;
    const int initPnpMethod = initImagePoints.size() >= 6 ? cv::SOLVEPNP_ITERATIVE : cv::SOLVEPNP_EPNP;
    if (!cv::solvePnP(initObjectPoints, initImagePoints, cameraMatrix, distCoeffs,
                      currentRvec, currentTvec, false, initPnpMethod)) {
        std::cerr << "Initial solvePnP failed on manual keypoints." << std::endl;
        return 1;
    }

    std::vector<cv::Point3f> activeObjectPoints = initObjectPoints;
    std::vector<cv::Point2f> activeImagePoints = initImagePoints;
    std::string activeReference = "manual_init";

    std::filesystem::create_directories(outputVideoPath.parent_path());
    std::filesystem::create_directories(outputCsvPath.parent_path());

    const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    cv::VideoWriter writer(outputVideoPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, frame.size());
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << outputVideoPath.string() << std::endl;
        return 1;
    }

    std::ofstream csv(outputCsvPath);
    csv << "frame_id,pose_valid,reference,inliers,reproj_error,held_last_pose\n";

    int frameId = 0;
    int heldCount = 0;
    int lastInliers = -1;
    while (true) {
        bool heldLastPose = false;
        if (frameId > 0) {
            const FramePose pose = estimatePoseFromReferences(
                frame, referenceViews, orb, config, boxCorners, cameraMatrix, distCoeffs);
            if (pose.valid) {
                currentRvec = pose.rvec.clone();
                currentTvec = pose.tvec.clone();
                activeObjectPoints = pose.objectPoints;
                activeImagePoints = pose.imagePoints;
                activeReference = pose.referenceName;
                lastInliers = pose.inliers;
                heldCount = 0;
            } else {
                ++heldCount;
                heldLastPose = true;
            }
        }

        cv::Mat output = frame.clone();
        drawProjectedBox(output, boxCorners, currentRvec, currentTvec, cameraMatrix, distCoeffs,
                         heldLastPose ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0));
        drawTrackedPoints(output, activeImagePoints);

        const double reprojError = computeMeanReprojectionError(
            activeObjectPoints, activeImagePoints, currentRvec, currentTvec, cameraMatrix, distCoeffs);

        cv::putText(output, "2D reference track | frame " + std::to_string(frameId),
                    cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, "ref: " + activeReference + " | reproj " + std::to_string(reprojError) + " px",
                    cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, heldLastPose ? "LOST: holding last pose" : "tracking fresh pose",
                    cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    heldLastPose ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0), 2);

        writer.write(output);
        csv << frameId << ','
            << (heldLastPose ? 0 : 1) << ','
            << activeReference << ','
            << lastInliers << ','
            << reprojError << ','
            << (heldLastPose ? 1 : 0) << '\n';

        cv::imshow("2D Reference Tracking", output);
        if (cv::waitKey(1) == 27) {
            break;
        }

        if (!cap.read(frame)) {
            break;
        }
        ++frameId;
    }

    writer.release();
    csv.close();

    std::cout << "Saved 2D tracking video to: " << outputVideoPath.string() << std::endl;
    std::cout << "Saved CSV log to: " << outputCsvPath.string() << std::endl;
    std::cout << "If box is stable here but flies in pose_estimation, the issue is likely GTSAM/jump logic." << std::endl;
    std::cout << "If box still flies here, check annotation with validate_reference_annotations." << std::endl;

    return 0;
}
