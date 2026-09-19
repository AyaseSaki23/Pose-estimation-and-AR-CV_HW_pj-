#pragma once

#include "cvproject/correspondence_estimator.hpp"
#include "cvproject/object_localizer.hpp"
#include "cvproject/pose_refiner.hpp"
#include "cvproject/pose_tracker_state.hpp"
#include "cvproject/reference_db.hpp"
#include "cvproject/view_retriever.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace cvproject {

class Gen6DPipeline {
public:
    explicit Gen6DPipeline(std::filesystem::path projectRoot);

    bool initialize();
    int run();

private:
    std::filesystem::path projectRoot_;
    PoseTrackerState state_ = PoseTrackerState::Initializing;
    ObjectLocalizer localizer_;
    cv::Ptr<cv::ORB> orb_;
    ViewRetriever viewRetriever_;
    CorrespondenceEstimator correspondenceEstimator_;
    PoseRefiner poseRefiner_;
    std::vector<ReferenceView> referenceViews_;
    std::vector<cv::Point3f> boxCorners_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;

    bool loadCameraCalibration();
    bool loadObjectConfig();
    bool loadReferenceViews();
    std::filesystem::path selectInputVideoPath() const;
    void drawOverlay(cv::Mat& frame,
                     int frameId,
                     double timestampSec,
                     const ObjectDetection& detection,
                     const std::vector<ViewCandidate>& viewCandidates,
                     const CorrespondenceSet& correspondences,
                     const PoseEstimate& pose) const;
    void writeCsvRow(std::ofstream& csv,
                     int frameId,
                     double timestampSec,
                     const ObjectDetection& detection,
                     const std::vector<ViewCandidate>& viewCandidates,
                     const CorrespondenceSet& correspondences,
                     const PoseEstimate& pose) const;
    PoseEstimate estimateInitialPose(const CorrespondenceSet& correspondences) const;
    void drawProjectedBox(cv::Mat& frame,
                          const PoseEstimate& pose,
                          const cv::Scalar& color) const;
    void drawAxis(cv::Mat& frame, const PoseEstimate& pose, float axisLength) const;
};

}  // namespace cvproject
