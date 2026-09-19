#include "cvproject/gen6d_pipeline.hpp"

#include "cvproject/mesh_loader.hpp"
#include "cvproject/pose_utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/core.hpp>

#include <array>
#include <iostream>
#include <utility>

namespace cvproject {

Gen6DPipeline::Gen6DPipeline(std::filesystem::path projectRoot)
    : projectRoot_(std::move(projectRoot)),
      orb_(cv::ORB::create(3500)),
      viewRetriever_(orb_),
      correspondenceEstimator_(orb_) {}

bool Gen6DPipeline::initialize() {
    std::cout << "[Gen6D-like] initialize pipeline at " << projectRoot_ << '\n';

    if (!loadCameraCalibration()) {
        return false;
    }
    if (!loadObjectConfig()) {
        return false;
    }
    if (!loadReferenceViews()) {
        return false;
    }

    state_ = PoseTrackerState::Initializing;
    return true;
}

int Gen6DPipeline::run() {
    const std::filesystem::path videoPath = selectInputVideoPath();
    if (videoPath.empty()) {
        std::cerr << "[Gen6D-like] no input video found. Tried input.mp4, input_rotation.mp4, 2.mp4.\n";
        return 1;
    }

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "[Gen6D-like] failed to open video: " << videoPath << '\n';
        return 1;
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "[Gen6D-like] failed to read first frame from: " << videoPath << '\n';
        return 1;
    }

    const std::filesystem::path outputDir = projectRoot_ / "data" / "processed" / "gen6d_like";
    const std::filesystem::path outputVideoPath = outputDir / "output_gen6d_like.mp4";
    const std::filesystem::path outputCsvPath = outputDir / "poses_gen6d_like.csv";
    std::filesystem::create_directories(outputDir);

    const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    cv::VideoWriter writer(outputVideoPath.string(),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps,
                           frame.size());
    if (!writer.isOpened()) {
        std::cerr << "[Gen6D-like] failed to open output video: " << outputVideoPath << '\n';
        return 1;
    }

    std::ofstream csv(outputCsvPath);
    if (!csv.is_open()) {
        std::cerr << "[Gen6D-like] failed to open output CSV: " << outputCsvPath << '\n';
        return 1;
    }
    csv << "frame_id,timestamp_sec,state,detection_valid,detection_score,used_fallback,"
           "used_pose_projection,lost_count,"
           "roi_x,roi_y,roi_w,roi_h,top_view,top_view_score,top_view_good_matches,"
           "top_view_inliers,top_view_inlier_ratio,num_view_candidates,correspondence_valid,"
           "correspondence_reference,correspondence_points,correspondence_score,"
           "correspondence_good_matches,correspondence_inliers,pose_valid,rvec_x,rvec_y,rvec_z,"
           "tvec_x,tvec_y,tvec_z,pose_reprojection_error,refinement_score\n";

    std::cout << "[Gen6D-like] reading video: " << videoPath << '\n';
    std::cout << "[Gen6D-like] writing video: " << outputVideoPath << '\n';
    std::cout << "[Gen6D-like] writing CSV: " << outputCsvPath << '\n';

    int frameId = 0;
    while (true) {
        const double timestampSec = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        const ObjectDetection detection = localizer_.localize(frame);
        const std::vector<ViewCandidate> viewCandidates =
            viewRetriever_.retrieveTopK(frame, detection, referenceViews_, 3);
        const CorrespondenceSet correspondences = correspondenceEstimator_.estimate(
            frame, detection, viewCandidates, referenceViews_, boxCorners_);
        const PoseEstimate initialPose = estimateInitialPose(correspondences);
        const PoseEstimate pose = poseRefiner_.refine(
            frame, initialPose, boxCorners_, cameraMatrix_, distCoeffs_);
        state_ = pose.valid
            ? PoseTrackerState::Tracking
            : (!viewCandidates.empty()
                ? PoseTrackerState::Relocalizing
                : (detection.valid ? PoseTrackerState::WeakTracking : PoseTrackerState::Lost));
        if (pose.valid) {
            localizer_.updateFromPose(pose.rvec, pose.tvec, boxCorners_,
                                      cameraMatrix_, distCoeffs_, frame.size());
        }

        cv::Mat output = frame.clone();
        drawOverlay(output, frameId, timestampSec, detection, viewCandidates, correspondences, pose);
        writer.write(output);
        writeCsvRow(csv, frameId, timestampSec, detection, viewCandidates, correspondences, pose);

        cv::imshow("Gen6D-like Pose Estimation", output);
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

    std::cout << "[Gen6D-like] processed " << (frameId + 1) << " frames\n";
    std::cout << "[Gen6D-like] saved video to: " << outputVideoPath << '\n';
    std::cout << "[Gen6D-like] saved CSV to: " << outputCsvPath << '\n';
    return 0;
}

bool Gen6DPipeline::loadCameraCalibration() {
    const std::filesystem::path calibrationPath = projectRoot_ / "config" / "camera_calib.yml";
    cv::FileStorage fs(calibrationPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[Gen6D-like] failed to open camera calibration: "
                  << calibrationPath << '\n';
        return false;
    }
    fs["camera_matrix"] >> cameraMatrix_;
    fs["dist_coeffs"] >> distCoeffs_;
    if (cameraMatrix_.empty() || distCoeffs_.empty()) {
        std::cerr << "[Gen6D-like] camera calibration is missing camera_matrix or dist_coeffs\n";
        return false;
    }

    std::cout << "[Gen6D-like] camera calibration found: " << calibrationPath << '\n';
    return true;
}

bool Gen6DPipeline::loadObjectConfig() {
    const std::filesystem::path objectConfigPath = projectRoot_ / "config" / "object_01.yml";
    cv::FileStorage fs(objectConfigPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[Gen6D-like] failed to open object config: "
                  << objectConfigPath << '\n';
        return false;
    }

    std::string meshPathText;
    float widthMm = 0.0f;
    float heightMm = 0.0f;
    float depthMm = 0.0f;
    fs["mesh_path"] >> meshPathText;
    fs["width_mm"] >> widthMm;
    fs["height_mm"] >> heightMm;
    fs["depth_mm"] >> depthMm;

    if (meshPathText.empty()) {
        std::cerr << "[Gen6D-like] object config is missing mesh_path\n";
        return false;
    }

    const std::filesystem::path meshPath = projectRoot_ / meshPathText;
    const Mesh mesh = loadObjMesh(meshPath);
    if (mesh.vertices.empty()) {
        std::cerr << "[Gen6D-like] failed to load object mesh: " << meshPath << '\n';
        return false;
    }

    const float meshScale = inferObjToMillimeterScale(mesh, widthMm, heightMm, depthMm);
    const BoundingBox bbox = computeBoundingBox(mesh, meshScale);
    boxCorners_ = bbox.corners;
    if (boxCorners_.size() < 8) {
        std::cerr << "[Gen6D-like] object bounding box did not produce 8 corners\n";
        return false;
    }

    std::cout << "[Gen6D-like] object config found: " << objectConfigPath << '\n';
    std::cout << "[Gen6D-like] loaded " << boxCorners_.size() << " object box corners\n";
    return true;
}

bool Gen6DPipeline::loadReferenceViews() {
    const std::filesystem::path referenceDir =
        projectRoot_ / "data" / "objects" / "object_01" / "reference";
    const std::filesystem::path referenceKeypointsPath = referenceDir / "reference_keypoints.yml";

    referenceViews_ = cvproject::loadReferenceViews(referenceDir, referenceKeypointsPath, orb_);
    if (referenceViews_.empty()) {
        std::cerr << "[Gen6D-like] no reference views loaded from: " << referenceDir << '\n';
        return false;
    }

    std::cout << "[Gen6D-like] loaded " << referenceViews_.size()
              << " reference views for retrieval\n";
    return true;
}

std::filesystem::path Gen6DPipeline::selectInputVideoPath() const {
    const std::array<std::filesystem::path, 3> candidates = {
        projectRoot_ / "data" / "raw" / "videos" / "input.mp4",
        projectRoot_ / "data" / "raw" / "videos" / "input_rotation.mp4",
        projectRoot_ / "data" / "raw" / "videos" / "2.mp4",
    };

    for (const std::filesystem::path& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return {};
}

void Gen6DPipeline::drawOverlay(cv::Mat& frame,
                                const int frameId,
                                const double timestampSec,
                                const ObjectDetection& detection,
                                const std::vector<ViewCandidate>& viewCandidates,
                                const CorrespondenceSet& correspondences,
                                const PoseEstimate& pose) const {
    const cv::Scalar textColor(0, 255, 255);
    const cv::Scalar roiColor = detection.valid && !detection.usedFallback
        ? cv::Scalar(255, 180, 0)
        : cv::Scalar(0, 220, 255);

    cv::putText(frame,
                "Gen6D-like pipeline: video loop + object localization placeholder",
                cv::Point(24, 36),
                cv::FONT_HERSHEY_SIMPLEX,
                0.72,
                textColor,
                2,
                cv::LINE_AA);
    cv::putText(frame,
                "frame " + std::to_string(frameId) +
                    " | t " + std::to_string(timestampSec) +
                    " | state " + trackerStateToString(state_),
                cv::Point(24, 72),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                textColor,
                2,
                cv::LINE_AA);

    if (detection.valid) {
        cv::rectangle(frame, detection.roi, roiColor, 2, cv::LINE_AA);
        cv::putText(frame,
                    "ROI score " + std::to_string(detection.score),
                    cv::Point(std::max(12, detection.roi.x + 8),
                              std::max(28, detection.roi.y + 28)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    roiColor,
                    2,
                    cv::LINE_AA);
        cv::putText(frame,
                    detection.usedFallback ? "localizer: full-frame fallback"
                                           : (detection.usedPoseProjection
                                                ? "localizer: pose-projected ROI"
                                                : "localizer: previous ROI prior"),
                    cv::Point(24, 108),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.65,
                    textColor,
                    2,
                    cv::LINE_AA);
        if (!viewCandidates.empty()) {
            const ViewCandidate& top = viewCandidates.front();
            cv::putText(frame,
                        "top view: " + top.referenceName +
                            " | score " + std::to_string(top.score) +
                            " | inliers " + std::to_string(top.homographyInliers),
                        cv::Point(24, 144),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.62,
                        textColor,
                        2,
                        cv::LINE_AA);
            if (viewCandidates.size() > 1) {
                cv::putText(frame,
                            "next: " + viewCandidates[1].referenceName,
                            cv::Point(24, 180),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.58,
                            textColor,
                            2,
                            cv::LINE_AA);
            }
        } else {
            cv::putText(frame,
                        "view retrieval: no candidate",
                        cv::Point(24, 144),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.62,
                        cv::Scalar(0, 0, 255),
                        2,
                        cv::LINE_AA);
        }

        if (correspondences.valid) {
            for (const cv::Point2f& point : correspondences.imagePoints) {
                cv::circle(frame, point, 5, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
            }
            cv::putText(frame,
                        "corr: " + correspondences.referenceName +
                            " | pts " + std::to_string(correspondences.imagePoints.size()),
                        cv::Point(24, 216),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.62,
                        cv::Scalar(255, 255, 0),
                        2,
                        cv::LINE_AA);
        } else {
            cv::putText(frame,
                        "corr: not enough transferred 2D-3D points",
                        cv::Point(24, 216),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.62,
                        cv::Scalar(0, 0, 255),
                        2,
                        cv::LINE_AA);
        }

        if (pose.valid) {
            drawProjectedBox(frame, pose, cv::Scalar(0, 255, 0));
            drawAxis(frame, pose, 45.0f);
            cv::putText(frame,
                        "PnP+refine: reproj " + std::to_string(pose.reprojectionError) +
                            " | edge " + std::to_string(pose.refinementScore),
                        cv::Point(24, 252),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.62,
                        cv::Scalar(0, 255, 0),
                        2,
                        cv::LINE_AA);
        } else {
            cv::putText(frame,
                        "PnP: no valid initial pose",
                        cv::Point(24, 252),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.62,
                        cv::Scalar(0, 0, 255),
                        2,
                        cv::LINE_AA);
        }
    } else {
        cv::putText(frame,
                    "object localization failed",
                    cv::Point(24, 108),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.65,
                    roiColor,
                    2,
                    cv::LINE_AA);
    }
}

void Gen6DPipeline::writeCsvRow(std::ofstream& csv,
                                const int frameId,
                                const double timestampSec,
                                const ObjectDetection& detection,
                                const std::vector<ViewCandidate>& viewCandidates,
                                const CorrespondenceSet& correspondences,
                                const PoseEstimate& pose) const {
    csv << frameId << ','
        << timestampSec << ','
        << trackerStateToString(state_) << ','
        << (detection.valid ? 1 : 0) << ','
        << detection.score << ','
        << (detection.usedFallback ? 1 : 0) << ','
        << (detection.usedPoseProjection ? 1 : 0) << ','
        << detection.lostCount << ','
        << detection.roi.x << ','
        << detection.roi.y << ','
        << detection.roi.width << ','
        << detection.roi.height << ',';

    if (!viewCandidates.empty()) {
        const ViewCandidate& top = viewCandidates.front();
        csv << top.referenceName << ','
            << top.score << ','
            << top.goodMatches << ','
            << top.homographyInliers << ','
            << top.inlierRatio << ',';
    } else {
        csv << "none,0,0,0,0,";
    }

    csv << viewCandidates.size() << ','
        << (correspondences.valid ? 1 : 0) << ','
        << (correspondences.valid ? correspondences.referenceName : "none") << ','
        << correspondences.imagePoints.size() << ','
        << correspondences.score << ','
        << correspondences.goodMatches << ','
        << correspondences.homographyInliers << ','
        << (pose.valid ? 1 : 0) << ',';

    if (pose.valid) {
        csv << pose.rvec.at<double>(0) << ','
            << pose.rvec.at<double>(1) << ','
            << pose.rvec.at<double>(2) << ','
            << pose.tvec.at<double>(0) << ','
            << pose.tvec.at<double>(1) << ','
            << pose.tvec.at<double>(2) << ','
            << pose.reprojectionError << ','
            << pose.refinementScore << '\n';
    } else {
        csv << "0,0,0,0,0,0,-1,0\n";
    }
}

PoseEstimate Gen6DPipeline::estimateInitialPose(const CorrespondenceSet& correspondences) const {
    PoseEstimate pose;
    if (!correspondences.valid ||
        correspondences.objectPoints.size() < 4 ||
        correspondences.objectPoints.size() != correspondences.imagePoints.size()) {
        return pose;
    }

    cv::Mat rvec;
    cv::Mat tvec;
    try {
        if (!solvePosePnP(correspondences.objectPoints,
                          correspondences.imagePoints,
                          cameraMatrix_,
                          distCoeffs_,
                          rvec,
                          tvec,
                          false,
                          true)) {
            return pose;
        }
    } catch (const cv::Exception&) {
        return pose;
    }

    const double reprojectionError = computeMeanReprojectionError(
        correspondences.objectPoints,
        correspondences.imagePoints,
        rvec,
        tvec,
        cameraMatrix_,
        distCoeffs_);
    if (reprojectionError > 18.0) {
        return pose;
    }

    pose.rvec = rvec.clone();
    pose.tvec = tvec.clone();
    pose.reprojectionError = reprojectionError;
    pose.refinementScore = correspondences.score;
    pose.valid = true;
    return pose;
}

void Gen6DPipeline::drawProjectedBox(cv::Mat& frame,
                                     const PoseEstimate& pose,
                                     const cv::Scalar& color) const {
    if (!pose.valid || boxCorners_.size() < 8) {
        return;
    }

    std::vector<cv::Point2f> projected;
    cv::projectPoints(boxCorners_, pose.rvec, pose.tvec, cameraMatrix_, distCoeffs_, projected);
    if (projected.size() < 8) {
        return;
    }

    const std::array<std::pair<int, int>, 12> edges = {{
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    }};

    for (const auto& edge : edges) {
        cv::line(frame,
                 cv::Point(cvRound(projected[edge.first].x), cvRound(projected[edge.first].y)),
                 cv::Point(cvRound(projected[edge.second].x), cvRound(projected[edge.second].y)),
                 color,
                 2,
                 cv::LINE_AA);
    }
}

void Gen6DPipeline::drawAxis(cv::Mat& frame, const PoseEstimate& pose, const float axisLength) const {
    if (!pose.valid) {
        return;
    }

    const std::vector<cv::Point3f> axisPoints = {
        {0.0f, 0.0f, 0.0f},
        {axisLength, 0.0f, 0.0f},
        {0.0f, axisLength, 0.0f},
        {0.0f, 0.0f, axisLength},
    };
    std::vector<cv::Point2f> projected;
    cv::projectPoints(axisPoints, pose.rvec, pose.tvec, cameraMatrix_, distCoeffs_, projected);
    if (projected.size() < 4) {
        return;
    }

    const cv::Point origin(cvRound(projected[0].x), cvRound(projected[0].y));
    cv::line(frame, origin,
             cv::Point(cvRound(projected[1].x), cvRound(projected[1].y)),
             cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    cv::line(frame, origin,
             cv::Point(cvRound(projected[2].x), cvRound(projected[2].y)),
             cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    cv::line(frame, origin,
             cv::Point(cvRound(projected[3].x), cvRound(projected[3].y)),
             cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
}

}  // namespace cvproject
