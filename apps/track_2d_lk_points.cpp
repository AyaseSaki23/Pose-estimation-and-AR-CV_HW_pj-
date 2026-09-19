#include <opencv2/opencv.hpp>

#include "cvproject/mesh_loader.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

struct TrackedPoint {
    int id = -1;
    cv::Point2f point;
    bool valid = true;
    bool recovered = false;
    float lastFbError = -1.0f;
    int lostCount = 0;
};

struct RecoveryRule {
    int targetId = -1;
    std::vector<int> anchorIds;
};

cv::Point toPixel(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

bool loadInitialPoints(const std::filesystem::path& path, std::vector<TrackedPoint>& points) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    points.clear();
    for (const cv::FileNode& node : fs["points"]) {
        TrackedPoint point;
        point.id = static_cast<int>(node["id"]);
        point.point.x = static_cast<float>(node["x"]);
        point.point.y = static_cast<float>(node["y"]);
        point.valid = true;
        points.push_back(point);
    }

    return !points.empty();
}

int findPointIndexById(const std::vector<TrackedPoint>& points, int id) {
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].id == id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<RecoveryRule> buildRecoveryRules(const std::vector<TrackedPoint>& points) {
    const std::vector<RecoveryRule> candidates = {
        {5, {1, 2, 6}}
    };

    std::vector<RecoveryRule> rules;
    for (const RecoveryRule& candidate : candidates) {
        if (findPointIndexById(points, candidate.targetId) < 0) {
            continue;
        }

        bool hasAllAnchors = true;
        for (int anchorId : candidate.anchorIds) {
            if (findPointIndexById(points, anchorId) < 0) {
                hasAllAnchors = false;
                break;
            }
        }
        if (hasAllAnchors) {
            rules.push_back(candidate);
        }
    }
    return rules;
}

bool getTrackedPointById(const std::vector<TrackedPoint>& points, int id, cv::Point2f& point) {
    const int index = findPointIndexById(points, id);
    if (index < 0 || !points[index].valid) {
        return false;
    }
    point = points[index].point;
    return true;
}

bool setTrackedPointById(std::vector<TrackedPoint>& points,
                         int id,
                         const cv::Point2f& point,
                         bool recovered) {
    const int index = findPointIndexById(points, id);
    if (index < 0) {
        return false;
    }

    points[index].point = point;
    points[index].valid = true;
    points[index].recovered = recovered;
    points[index].lastFbError = -1.0f;
    points[index].lostCount = 0;
    return true;
}

bool collectCorrespondencesForIds(const std::vector<TrackedPoint>& points,
                                  const std::vector<cv::Point3f>& boxCorners,
                                  const std::vector<int>& ids,
                                  std::vector<cv::Point3f>& objectPoints,
                                  std::vector<cv::Point2f>& imagePoints) {
    objectPoints.clear();
    imagePoints.clear();

    for (int id : ids) {
        const int cornerIndex = id - 1;
        const int trackedIndex = findPointIndexById(points, id);
        if (cornerIndex < 0 || cornerIndex >= static_cast<int>(boxCorners.size()) ||
            trackedIndex < 0 || !points[trackedIndex].valid) {
            return false;
        }
        objectPoints.push_back(boxCorners[cornerIndex]);
        imagePoints.push_back(points[trackedIndex].point);
    }

    return objectPoints.size() == ids.size();
}

bool solvePoseFromTrackedIds(const std::vector<TrackedPoint>& points,
                             const std::vector<cv::Point3f>& boxCorners,
                             const std::vector<int>& ids,
                             const cv::Mat& cameraMatrix,
                             const cv::Mat& distCoeffs,
                             cv::Mat& rvec,
                             cv::Mat& tvec,
                             bool useExtrinsicGuess) {
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    if (!collectCorrespondencesForIds(points, boxCorners, ids, objectPoints, imagePoints)) {
        return false;
    }

    try {
        return cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                            rvec, tvec, useExtrinsicGuess, cv::SOLVEPNP_ITERATIVE);
    } catch (const cv::Exception&) {
        return false;
    }
}

void recoverBackTopPointsFromPose(std::vector<TrackedPoint>& points,
                                  const std::vector<cv::Point3f>& boxCorners,
                                  const cv::Mat& cameraMatrix,
                                  const cv::Mat& distCoeffs,
                                  const cv::Mat& rvec,
                                  const cv::Mat& tvec) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(boxCorners, rvec, tvec, cameraMatrix, distCoeffs, projected);
    if (projected.size() < 8) {
        return;
    }

    setTrackedPointById(points, 7, projected[6], true);
    setTrackedPointById(points, 8, projected[7], true);
}

bool predictFromAnchors(const std::vector<TrackedPoint>& initialPoints,
                        const std::vector<TrackedPoint>& currentPoints,
                        const RecoveryRule& rule,
                        cv::Point2f& predicted) {
    const int targetIndex = findPointIndexById(currentPoints, rule.targetId);
    if (targetIndex < 0) {
        return false;
    }

    std::vector<cv::Point2f> sourceAnchors;
    std::vector<cv::Point2f> currentAnchors;
    for (int anchorId : rule.anchorIds) {
        const int sourceIndex = findPointIndexById(initialPoints, anchorId);
        const int currentIndex = findPointIndexById(currentPoints, anchorId);
        if (sourceIndex < 0 || currentIndex < 0 || !currentPoints[currentIndex].valid) {
            return false;
        }
        sourceAnchors.push_back(initialPoints[sourceIndex].point);
        currentAnchors.push_back(currentPoints[currentIndex].point);
    }

    const cv::Mat affine = cv::getAffineTransform(sourceAnchors, currentAnchors);
    if (affine.empty()) {
        return false;
    }

    std::vector<cv::Point2f> sourceTarget = {initialPoints[targetIndex].point};
    std::vector<cv::Point2f> predictedTarget;
    cv::transform(sourceTarget, predictedTarget, affine);
    predicted = predictedTarget.front();
    return true;
}

void recoverByFaceGeometry(const std::vector<TrackedPoint>& initialPoints,
                           const std::vector<RecoveryRule>& rules,
                           std::vector<TrackedPoint>& points,
                           const cv::Size& imageSize) {
    const float maxAllowedDeviationPx = 28.0f;

    for (const RecoveryRule& rule : rules) {
        const int targetIndex = findPointIndexById(points, rule.targetId);
        if (targetIndex < 0) {
            continue;
        }

        cv::Point2f predicted;
        if (!predictFromAnchors(initialPoints, points, rule, predicted)) {
            continue;
        }

        const bool inside =
            predicted.x >= 0.0f && predicted.x < static_cast<float>(imageSize.width) &&
            predicted.y >= 0.0f && predicted.y < static_cast<float>(imageSize.height);
        if (!inside) {
            continue;
        }

        TrackedPoint& target = points[targetIndex];
        const float deviation = static_cast<float>(cv::norm(target.point - predicted));
        if (!target.valid || target.lostCount > 0 || deviation > maxAllowedDeviationPx) {
            target.point = predicted;
            target.valid = true;
            target.recovered = true;
            target.lastFbError = -1.0f;
            target.lostCount = 0;
        }
    }
}

void drawBoxEdges(cv::Mat& image, const std::vector<TrackedPoint>& points) {
    std::map<int, cv::Point2f> validPoints;
    for (const TrackedPoint& point : points) {
        if (point.valid) {
            validPoints[point.id] = point.point;
        }
    }

    const std::vector<std::pair<int, int>> edges = {
        {1, 2}, {2, 3}, {3, 4}, {4, 1},
        {5, 6}, {6, 7}, {7, 8}, {8, 5},
        {1, 5}, {2, 6}, {3, 7}, {4, 8}
    };

    for (const auto& edge : edges) {
        const auto first = validPoints.find(edge.first);
        const auto second = validPoints.find(edge.second);
        if (first == validPoints.end() || second == validPoints.end()) {
            continue;
        }
        cv::line(image, toPixel(first->second), toPixel(second->second),
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}

void drawTrackedPoints(cv::Mat& image, const std::vector<TrackedPoint>& points) {
    for (const TrackedPoint& point : points) {
        const cv::Scalar color = !point.valid ? cv::Scalar(0, 0, 255) :
                                 point.recovered ? cv::Scalar(255, 255, 0) :
                                                   cv::Scalar(0, 255, 255);
        cv::circle(image, toPixel(point.point), 6, color, -1, cv::LINE_AA);
        cv::circle(image, toPixel(point.point), 9,
                   point.valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::putText(image, std::to_string(point.id), toPixel(point.point + cv::Point2f(8.0f, -8.0f)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv::LINE_AA);
    }
}

int countValid(const std::vector<TrackedPoint>& points) {
    int count = 0;
    for (const TrackedPoint& point : points) {
        if (point.valid) {
            ++count;
        }
    }
    return count;
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path videoPath = projectRoot / "data/raw/videos/input.mp4";
    const std::filesystem::path calibPath = projectRoot / "config/camera_calib.yml";
    const std::filesystem::path objectConfigPath = projectRoot / "config/object_01.yml";
    const std::filesystem::path manualKeypointsPath =
        projectRoot / "data/processed/matches/manual_keypoints_frame000.yml";
    const std::filesystem::path outputVideoPath =
        projectRoot / "data/processed/validation/output_2d_lk_points.mp4";
    const std::filesystem::path outputCsvPath =
        projectRoot / "data/processed/validation/track_2d_lk_points.csv";

    std::vector<TrackedPoint> points;
    if (!loadInitialPoints(manualKeypointsPath, points)) {
        std::cerr << "Failed to load first-frame points: " << manualKeypointsPath.string() << std::endl;
        return 1;
    }
    const std::vector<TrackedPoint> initialPoints = points;
    const std::vector<RecoveryRule> recoveryRules = buildRecoveryRules(points);

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
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Failed to read first frame." << std::endl;
        return 1;
    }

    cv::Mat prevGray;
    cv::cvtColor(frame, prevGray, cv::COLOR_BGR2GRAY);

    std::filesystem::create_directories(outputVideoPath.parent_path());
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    cv::VideoWriter writer(outputVideoPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, frame.size());
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << outputVideoPath.string() << std::endl;
        return 1;
    }

    std::ofstream csv(outputCsvPath);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV: " << outputCsvPath.string() << std::endl;
        return 1;
    }
    csv << "frame_id,id,x,y,valid,recovered,forward_backward_error,lost_count\n";

    const cv::Size winSize(31, 31);
    const int maxLevel = 4;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    const float maxForwardBackwardErrorPx = 2.5f;
    const int maxLostCount = 5;
    const std::vector<int> frontFaceIds = {1, 2, 5, 6};
    cv::Mat currentRvec;
    cv::Mat currentTvec;
    bool hasPose = solvePoseFromTrackedIds(points, boxCorners, frontFaceIds,
                                           cameraMatrix, distCoeffs,
                                           currentRvec, currentTvec, false);

    int frameId = 0;
    while (true) {
        if (frameId > 0) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            for (TrackedPoint& point : points) {
                point.recovered = false;
                point.lastFbError = -1.0f;
            }

            std::vector<int> activeIndices;
            std::vector<cv::Point2f> prevPoints;
            for (size_t i = 0; i < points.size(); ++i) {
                if (!points[i].valid) {
                    continue;
                }
                activeIndices.push_back(static_cast<int>(i));
                prevPoints.push_back(points[i].point);
            }

            if (!prevPoints.empty()) {
                std::vector<cv::Point2f> nextPoints;
                std::vector<cv::Point2f> backPoints;
                std::vector<unsigned char> statusForward;
                std::vector<unsigned char> statusBackward;
                std::vector<float> errorForward;
                std::vector<float> errorBackward;

                cv::calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints,
                                         statusForward, errorForward, winSize, maxLevel, criteria);
                cv::calcOpticalFlowPyrLK(gray, prevGray, nextPoints, backPoints,
                                         statusBackward, errorBackward, winSize, maxLevel, criteria);

                for (size_t i = 0; i < activeIndices.size(); ++i) {
                    TrackedPoint& point = points[activeIndices[i]];
                    const bool inside =
                        nextPoints[i].x >= 0.0f && nextPoints[i].x < static_cast<float>(frame.cols) &&
                        nextPoints[i].y >= 0.0f && nextPoints[i].y < static_cast<float>(frame.rows);
                    const float fbError = static_cast<float>(cv::norm(prevPoints[i] - backPoints[i]));
                    point.lastFbError = fbError;
                    if (statusForward[i] != 0 && statusBackward[i] != 0 &&
                        inside && fbError <= maxForwardBackwardErrorPx) {
                        point.point = nextPoints[i];
                        point.lostCount = 0;
                    } else {
                        ++point.lostCount;
                        if (point.lostCount > maxLostCount) {
                            point.valid = false;
                        }
                    }
                }
            }

            recoverByFaceGeometry(initialPoints, recoveryRules, points, frame.size());

            cv::Point2f predicted5;
            if (!recoveryRules.empty() &&
                predictFromAnchors(initialPoints, points, recoveryRules.front(), predicted5)) {
                setTrackedPointById(points, 5, predicted5, true);
            }

            cv::Mat solvedRvec = hasPose ? currentRvec.clone() : cv::Mat();
            cv::Mat solvedTvec = hasPose ? currentTvec.clone() : cv::Mat();
            const bool poseOk = solvePoseFromTrackedIds(
                points, boxCorners, frontFaceIds, cameraMatrix, distCoeffs,
                solvedRvec, solvedTvec, hasPose);
            if (poseOk) {
                currentRvec = solvedRvec;
                currentTvec = solvedTvec;
                hasPose = true;
                recoverBackTopPointsFromPose(points, boxCorners, cameraMatrix, distCoeffs,
                                             currentRvec, currentTvec);
            }
            prevGray = gray;
        }

        cv::Mat output = frame.clone();
        drawBoxEdges(output, points);
        drawTrackedPoints(output, points);

        cv::putText(output, "2D LK point tracking | frame " + std::to_string(frameId),
                    cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(output,
                    "valid points: " + std::to_string(countValid(points)) +
                        "/" + std::to_string(points.size()) +
                        " | 7/8 projected from 1/2/5/6 pose",
                    cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, "Esc: quit",
                    cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

        writer.write(output);
        for (const TrackedPoint& point : points) {
            csv << frameId << ','
                << point.id << ','
                << point.point.x << ','
                << point.point.y << ','
                << (point.valid ? 1 : 0) << ','
                << (point.recovered ? 1 : 0) << ','
                << point.lastFbError << ','
                << point.lostCount << '\n';
        }

        cv::imshow("2D LK Point Tracking", output);
        const int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }

        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        ++frameId;
    }

    writer.release();
    csv.close();

    std::cout << "Saved LK tracking video to: " << outputVideoPath.string() << std::endl;
    std::cout << "Saved LK CSV to: " << outputCsvPath.string() << std::endl;
    std::cout << "Cyan points were recovered from same-face geometry when LK drifted or lost a weak corner." << std::endl;

    return 0;
}
