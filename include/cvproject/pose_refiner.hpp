#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace cvproject {

struct PoseEstimate {
    cv::Mat rvec;
    cv::Mat tvec;
    double reprojectionError = -1.0;
    double refinementScore = 0.0;
    bool valid = false;
};

class PoseRefiner {
public:
    PoseEstimate refine(const cv::Mat& frame,
                        const PoseEstimate& initialPose,
                        const std::vector<cv::Point3f>& boxCorners,
                        const cv::Mat& cameraMatrix,
                        const cv::Mat& distCoeffs) const;

private:
    double scoreProjectedEdges(const cv::Mat& edgeDistance,
                               const cv::Size& imageSize,
                               const PoseEstimate& pose,
                               const std::vector<cv::Point3f>& boxCorners,
                               const cv::Mat& cameraMatrix,
                               const cv::Mat& distCoeffs) const;
};

}  // namespace cvproject
