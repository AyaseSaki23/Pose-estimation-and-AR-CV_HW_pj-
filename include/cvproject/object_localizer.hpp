#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace cvproject {

struct ObjectDetection {
    cv::Rect roi;
    double score = 0.0;
    bool valid = false;
    bool usedFallback = false;
    bool usedPoseProjection = false;
    int lostCount = 0;
};

class ObjectLocalizer {
public:
    ObjectDetection localize(const cv::Mat& frame);
    void updateFromPose(const cv::Mat& rvec,
                        const cv::Mat& tvec,
                        const std::vector<cv::Point3f>& boxCorners,
                        const cv::Mat& cameraMatrix,
                        const cv::Mat& distCoeffs,
                        const cv::Size& frameSize);
    void reset();

private:
    cv::Rect previousRoi_;
    int lostCount_ = 0;
    bool hasPreviousRoi_ = false;
    bool previousRoiFromPose_ = false;

    cv::Rect clampRoi(const cv::Rect& roi, const cv::Size& imageSize) const;
    cv::Rect expandRoi(const cv::Rect& roi, const cv::Size& imageSize, double scale) const;
    cv::Rect fullFrameRoi(const cv::Size& imageSize) const;
};

}  // namespace cvproject
