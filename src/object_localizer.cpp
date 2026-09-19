#include "cvproject/object_localizer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace cvproject {

ObjectDetection ObjectLocalizer::localize(const cv::Mat& frame) {
    ObjectDetection detection;
    if (frame.empty()) {
        ++lostCount_;
        detection.lostCount = lostCount_;
        return detection;
    }

    if (hasPreviousRoi_) {
        detection.roi = expandRoi(previousRoi_, frame.size(), 1.35);
        detection.score = previousRoiFromPose_ ? 0.9 : 0.75;
        detection.usedFallback = false;
        detection.usedPoseProjection = previousRoiFromPose_;
    } else {
        detection.roi = fullFrameRoi(frame.size());
        detection.score = 0.35;
        detection.usedFallback = true;
        detection.usedPoseProjection = false;
    }

    detection.valid = true;
    detection.lostCount = lostCount_;

    previousRoi_ = detection.roi;
    hasPreviousRoi_ = true;
    lostCount_ = 0;
    return detection;
}

void ObjectLocalizer::updateFromPose(const cv::Mat& rvec,
                                     const cv::Mat& tvec,
                                     const std::vector<cv::Point3f>& boxCorners,
                                     const cv::Mat& cameraMatrix,
                                     const cv::Mat& distCoeffs,
                                     const cv::Size& frameSize) {
    if (rvec.empty() || tvec.empty() || boxCorners.empty() ||
        cameraMatrix.empty() || frameSize.empty()) {
        return;
    }

    std::vector<cv::Point2f> projected;
    try {
        cv::projectPoints(boxCorners, rvec, tvec, cameraMatrix, distCoeffs, projected);
    } catch (const cv::Exception&) {
        return;
    }
    if (projected.empty()) {
        return;
    }

    std::vector<cv::Point> pixels;
    pixels.reserve(projected.size());
    for (const cv::Point2f& point : projected) {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
            pixels.emplace_back(cvRound(point.x), cvRound(point.y));
        }
    }
    if (pixels.size() < 4) {
        return;
    }

    const cv::Rect projectedRoi = cv::boundingRect(pixels);
    const cv::Rect expanded = expandRoi(projectedRoi, frameSize, 1.8);
    if (expanded.empty()) {
        return;
    }

    previousRoi_ = expanded;
    hasPreviousRoi_ = true;
    previousRoiFromPose_ = true;
    lostCount_ = 0;
}

void ObjectLocalizer::reset() {
    previousRoi_ = {};
    lostCount_ = 0;
    hasPreviousRoi_ = false;
    previousRoiFromPose_ = false;
}

cv::Rect ObjectLocalizer::clampRoi(const cv::Rect& roi, const cv::Size& imageSize) const {
    const cv::Rect imageBounds(0, 0, imageSize.width, imageSize.height);
    return roi & imageBounds;
}

cv::Rect ObjectLocalizer::expandRoi(const cv::Rect& roi,
                                    const cv::Size& imageSize,
                                    const double scale) const {
    if (roi.empty()) {
        return fullFrameRoi(imageSize);
    }

    const cv::Point2f center(
        static_cast<float>(roi.x) + static_cast<float>(roi.width) * 0.5f,
        static_cast<float>(roi.y) + static_cast<float>(roi.height) * 0.5f);
    const int expandedWidth = std::max(1, static_cast<int>(std::round(roi.width * scale)));
    const int expandedHeight = std::max(1, static_cast<int>(std::round(roi.height * scale)));

    const cv::Rect expanded(
        static_cast<int>(std::round(center.x - expandedWidth * 0.5f)),
        static_cast<int>(std::round(center.y - expandedHeight * 0.5f)),
        expandedWidth,
        expandedHeight);

    const cv::Rect clamped = clampRoi(expanded, imageSize);
    return clamped.empty() ? fullFrameRoi(imageSize) : clamped;
}

cv::Rect ObjectLocalizer::fullFrameRoi(const cv::Size& imageSize) const {
    return cv::Rect(0, 0, imageSize.width, imageSize.height);
}

}  // namespace cvproject
