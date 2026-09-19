#pragma once

#include "cvproject/reference_db.hpp"

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

namespace cvproject {

struct ViewpointMatchResult {
    bool ok = false;
    std::string referenceName;
    double score = 0.0;
    int inliers = 0;
    int goodMatches = 0;
    double reprojectionError = 0.0;
    double viewSimilarity = 0.0;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    cv::Mat rvec;
    cv::Mat tvec;
};

class ViewpointSelector {
public:
    explicit ViewpointSelector(const ReferenceTrackingConfig& config);

    ViewpointMatchResult localizeFrame(const cv::Mat& frame,
                                       const std::vector<ReferenceView>& referenceViews,
                                       const cv::Ptr<cv::ORB>& orb,
                                       const std::vector<cv::Point3f>& boxCorners,
                                       const cv::Mat& cameraMatrix,
                                       const cv::Mat& distCoeffs,
                                       const std::optional<std::string>& lockedReference,
                                       bool hasPoseHint,
                                       const cv::Mat& poseHintRvec) const;

    std::string selectReferenceWithHysteresis(
        const std::vector<ViewpointMatchResult>& candidates,
        const std::optional<std::string>& lockedReference) const;

private:
    ReferenceTrackingConfig config_;
};

}  // namespace cvproject
