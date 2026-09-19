#include <opencv2/opencv.hpp>

#include "cvproject/mesh_loader.hpp"
#include "cvproject/pose_utils.hpp"
#include "cvproject/reference_db.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

struct FaceDefinition {
    std::string name;
    std::array<int, 4> ids;
};

struct FacePoseResult {
    bool ok = false;
    std::string referenceName;
    std::string faceName;
    int goodMatches = 0;
    int inliers = 0;
    double reprojectionError = -1.0;
    cv::Mat rvec;
    cv::Mat tvec;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
};

const std::array<FaceDefinition, 4> kFaces = {{
    {"front", {1, 2, 5, 6}},
    {"right", {2, 3, 6, 7}},
    {"left", {1, 4, 5, 8}},
    {"top", {5, 6, 7, 8}}
}};

constexpr int kMaxStalePoseDrawFrames = 10;

cv::Point toPixel(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

std::string toLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool isFaceAllowedForReference(const std::string& referenceName, const std::string& faceName) {
    const std::string lowerName = toLower(referenceName);
    if (lowerName.find("ref_front") != std::string::npos) {
        return faceName == "front";
    }
    if (lowerName.find("ref_right") != std::string::npos) {
        return faceName == "right";
    }
    if (lowerName.find("ref_left") != std::string::npos) {
        return faceName == "left";
    }
    if (lowerName.find("ref_top") != std::string::npos) {
        return faceName == "top";
    }
    return true;
}

cv::Mat buildReferenceObjectMask(const cv::Size& size,
                                 const std::vector<cvproject::ReferencePoint>& points) {
    cv::Mat mask(size, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point2f> point2f;
    for (const cvproject::ReferencePoint& point : points) {
        point2f.push_back(point.point);
    }
    if (point2f.size() < 4) {
        return mask;
    }

    std::vector<cv::Point2f> hull2f;
    cv::convexHull(point2f, hull2f);
    std::vector<cv::Point> hull;
    for (const cv::Point2f& point : hull2f) {
        hull.push_back(toPixel(point));
    }

    cv::fillConvexPoly(mask, hull, cv::Scalar(255), cv::LINE_AA);
    const int dilation = std::max(9, std::min(size.width, size.height) / 35);
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(dilation * 2 + 1, dilation * 2 + 1));
    cv::dilate(mask, mask, kernel);
    return mask;
}

void restrictReferenceFeaturesToObject(std::vector<cvproject::ReferenceView>& references,
                                       const cv::Ptr<cv::ORB>& orb) {
    for (cvproject::ReferenceView& reference : references) {
        const cv::Mat mask = buildReferenceObjectMask(reference.imageGray.size(), reference.annotatedPoints);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(reference.imageGray, mask, keypoints, descriptors);
        if (!descriptors.empty() && keypoints.size() >= 12) {
            reference.keypoints = std::move(keypoints);
            reference.descriptors = descriptors;
        }
    }
}

bool hasReferenceFace(const std::vector<cvproject::ReferencePoint>& points,
                      const FaceDefinition& face,
                      std::vector<cv::Point2f>& sourcePoints,
                      std::vector<int>& ids) {
    std::map<int, cv::Point2f> pointById;
    for (const cvproject::ReferencePoint& point : points) {
        pointById[point.id] = point.point;
    }

    sourcePoints.clear();
    ids.clear();
    for (int id : face.ids) {
        const auto it = pointById.find(id);
        if (it == pointById.end()) {
            return false;
        }
        sourcePoints.push_back(it->second);
        ids.push_back(id);
    }
    return true;
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

    for (size_t i = 0; i < projected.size(); ++i) {
        cv::circle(image, toPixel(projected[i]), 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
        cv::putText(image, std::to_string(i + 1),
                    toPixel(projected[i] + cv::Point2f(5.0f, -5.0f)),
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
}

FacePoseResult estimateRotationPose(
    const cv::Mat& frame,
    const std::vector<cvproject::ReferenceView>& referenceViews,
    const cv::Ptr<cv::ORB>& orb,
    const std::vector<cv::Point3f>& boxCorners,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const std::optional<cv::Mat>& previousRvec,
    const std::optional<cv::Mat>& previousTvec) {
    FacePoseResult best;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb->detectAndCompute(gray, cv::noArray(), frameKeypoints, frameDescriptors);
    if (frameDescriptors.empty()) {
        return best;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    for (const cvproject::ReferenceView& reference : referenceViews) {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(reference.descriptors, frameDescriptors, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const std::vector<cv::DMatch>& pair : knnMatches) {
            if (pair.size() < 2) {
                continue;
            }
            if (pair[0].distance < 0.75f * pair[1].distance) {
                goodMatches.push_back(pair[0]);
            }
        }
        if (goodMatches.size() < 18) {
            continue;
        }

        std::vector<cv::Point2f> refPoints;
        std::vector<cv::Point2f> framePoints;
        for (const cv::DMatch& match : goodMatches) {
            refPoints.push_back(reference.keypoints[match.queryIdx].pt);
            framePoints.push_back(frameKeypoints[match.trainIdx].pt);
        }

        cv::Mat inlierMask;
        cv::Mat homography = cv::findHomography(refPoints, framePoints, cv::RANSAC, 4.0, inlierMask);
        if (homography.empty() || inlierMask.empty()) {
            continue;
        }

        int inliers = 0;
        for (int i = 0; i < inlierMask.rows; ++i) {
            if (inlierMask.at<unsigned char>(i, 0) != 0) {
                ++inliers;
            }
        }
        if (inliers < 25) {
            continue;
        }

        for (const FaceDefinition& face : kFaces) {
            if (!isFaceAllowedForReference(reference.name, face.name)) {
                continue;
            }

            std::vector<cv::Point2f> sourceFacePoints;
            std::vector<int> ids;
            if (!hasReferenceFace(reference.annotatedPoints, face, sourceFacePoints, ids)) {
                continue;
            }

            std::vector<cv::Point2f> transferredFacePoints;
            cv::perspectiveTransform(sourceFacePoints, transferredFacePoints, homography);

            std::vector<cv::Point3f> objectPoints;
            for (int id : ids) {
                objectPoints.push_back(boxCorners[id - 1]);
            }

            cv::Mat rvec = previousRvec.has_value() ? previousRvec.value().clone() : cv::Mat();
            cv::Mat tvec = previousTvec.has_value() ? previousTvec.value().clone() : cv::Mat();
            const bool useGuess = previousRvec.has_value() && previousTvec.has_value();

            try {
                if (!cv::solvePnP(objectPoints, transferredFacePoints, cameraMatrix, distCoeffs,
                                  rvec, tvec, useGuess, cv::SOLVEPNP_ITERATIVE)) {
                    continue;
                }
                cv::solvePnPRefineLM(objectPoints, transferredFacePoints, cameraMatrix, distCoeffs,
                                     rvec, tvec);
            } catch (const cv::Exception&) {
                continue;
            }

            const double reprojError = cvproject::computeMeanReprojectionError(
                objectPoints, transferredFacePoints, rvec, tvec, cameraMatrix, distCoeffs);
            if (reprojError > 10.0) {
                continue;
            }
            if (useGuess && cvproject::isPoseJumpTooLarge(
                    previousRvec.value(), previousTvec.value(), rvec, tvec, 70.0, 0.75)) {
                continue;
            }

            const double score = static_cast<double>(inliers) + 0.04 * goodMatches.size() - 3.0 * reprojError;
            const double bestScore = best.ok
                ? static_cast<double>(best.inliers) + 0.04 * best.goodMatches - 3.0 * best.reprojectionError
                : -1e9;
            if (!best.ok || score > bestScore) {
                best.ok = true;
                best.referenceName = reference.name;
                best.faceName = face.name;
                best.goodMatches = static_cast<int>(goodMatches.size());
                best.inliers = inliers;
                best.reprojectionError = reprojError;
                best.rvec = rvec.clone();
                best.tvec = tvec.clone();
                best.objectPoints = objectPoints;
                best.imagePoints = transferredFacePoints;
            }
        }
    }

    return best;
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path videoPath = projectRoot / "data/raw/videos/input_rotation.mp4";
    const std::filesystem::path fallbackVideoPath = projectRoot / "data/raw/videos/input.mp4";
    const std::filesystem::path calibPath = projectRoot / "config/camera_calib.yml";
    const std::filesystem::path objectConfigPath = projectRoot / "config/object_01.yml";
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path referenceKeypointsPath = referenceDir / "reference_keypoints.yml";
    const std::filesystem::path outputVideoPath = projectRoot / "data/processed/rotation/output_rotation_pose.mp4";
    const std::filesystem::path outputCsvPath = projectRoot / "data/processed/rotation/poses_rotation.csv";

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

    cvproject::ReferenceTrackingConfig config;
    config.orbFeatures = 3500;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config.orbFeatures);
    std::vector<cvproject::ReferenceView> references =
        cvproject::loadReferenceViews(referenceDir, referenceKeypointsPath, orb);
    if (references.empty()) {
        std::cerr << "No reference views loaded. Add reference images/keypoints first." << std::endl;
        return 1;
    }
    restrictReferenceFeaturesToObject(references, orb);

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cout << "input_rotation.mp4 not found; falling back to input.mp4" << std::endl;
        cap.open(fallbackVideoPath.string());
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open rotation video." << std::endl;
        return 1;
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Failed to read first frame." << std::endl;
        return 1;
    }

    std::filesystem::create_directories(outputVideoPath.parent_path());
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    cv::VideoWriter writer(outputVideoPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, frame.size());
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << outputVideoPath.string() << std::endl;
        return 1;
    }

    std::ofstream csv(outputCsvPath);
    csv << "frame_id,timestamp_sec,pose_ok,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,"
           "reference,face,inliers,good_matches,reprojection_error\n";

    std::optional<cv::Mat> previousRvec;
    std::optional<cv::Mat> previousTvec;
    FacePoseResult lastAccepted;
    int frameId = 0;
    int lostCount = 0;

    while (true) {
        bool drewStalePose = false;
        FacePoseResult pose = estimateRotationPose(
            frame, references, orb, boxCorners, cameraMatrix, distCoeffs, previousRvec, previousTvec);

        if (pose.ok) {
            lastAccepted = pose;
            previousRvec = pose.rvec.clone();
            previousTvec = pose.tvec.clone();
            lostCount = 0;
        } else {
            ++lostCount;
            if (lastAccepted.ok && lostCount <= kMaxStalePoseDrawFrames) {
                pose = lastAccepted;
                drewStalePose = true;
            }
        }

        cv::Mat output = frame.clone();
        if (pose.ok) {
            drawProjectedBox(output, boxCorners, pose.rvec, pose.tvec, cameraMatrix, distCoeffs,
                             lostCount > 0 ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0));
            drawAxis(output, pose.rvec, pose.tvec, cameraMatrix, distCoeffs, 45.0f);
            for (const cv::Point2f& point : pose.imagePoints) {
                cv::circle(output, point, 5, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
            }
        }

        cv::putText(output, "Rotation extension: multi-reference face PnP",
                    cv::Point(24, 36), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
        cv::putText(output,
                    "frame " + std::to_string(frameId) +
                        " | " + (pose.ok
                            ? (drewStalePose ? "stale " : "") + pose.referenceName + " / " + pose.faceName
                            : "lost") +
                        " | lost " + std::to_string(lostCount),
                    cv::Point(24, 72), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    pose.ok ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255), 2);
        if (pose.ok) {
            cv::putText(output,
                        "inliers " + std::to_string(pose.inliers) +
                            " | reproj " + std::to_string(pose.reprojectionError),
                        cv::Point(24, 104), cv::FONT_HERSHEY_SIMPLEX, 0.62,
                        cv::Scalar(0, 255, 255), 2);
        }

        writer.write(output);
        const double timestampSec = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        csv << frameId << ','
            << timestampSec << ','
            << (pose.ok && !drewStalePose ? 1 : 0) << ',';
        if (pose.ok) {
            csv << pose.rvec.at<double>(0) << ','
                << pose.rvec.at<double>(1) << ','
                << pose.rvec.at<double>(2) << ','
                << pose.tvec.at<double>(0) << ','
                << pose.tvec.at<double>(1) << ','
                << pose.tvec.at<double>(2) << ','
                << pose.referenceName << ','
                << pose.faceName << ','
                << pose.inliers << ','
                << pose.goodMatches << ','
                << pose.reprojectionError << '\n';
        } else {
            csv << "0,0,0,0,0,0,none,none,0,0,-1\n";
        }

        cv::imshow("Rotation Pose Extension", output);
        if (cv::waitKey(1) == 27) {
            break;
        }

        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        ++frameId;
    }

    writer.release();
    csv.close();
    std::cout << "Saved rotation pose video to: " << outputVideoPath.string() << std::endl;
    std::cout << "Saved rotation pose CSV to: " << outputCsvPath.string() << std::endl;
    return 0;
}
