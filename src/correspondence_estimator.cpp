#include "cvproject/correspondence_estimator.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <utility>

namespace cvproject {

CorrespondenceEstimator::CorrespondenceEstimator(cv::Ptr<cv::ORB> orb)
    : orb_(std::move(orb)) {}

CorrespondenceSet CorrespondenceEstimator::estimate(
    const cv::Mat& frame,
    const ObjectDetection& detection,
    const std::vector<ViewCandidate>& viewCandidates,
    const std::vector<ReferenceView>& references,
    const std::vector<cv::Point3f>& boxCorners) const {
    CorrespondenceSet bestSet;
    if (frame.empty() || !detection.valid || viewCandidates.empty() ||
        references.empty() || boxCorners.empty() || orb_.empty()) {
        return bestSet;
    }

    const cv::Rect roi = clampRoi(detection.roi, frame.size());
    if (roi.empty()) {
        return bestSet;
    }
    for (const ViewCandidate& candidate : viewCandidates) {
        if (candidate.referenceIndex < 0 ||
            candidate.referenceIndex >= static_cast<int>(references.size())) {
            continue;
        }

        const ReferenceView& reference = references[static_cast<size_t>(candidate.referenceIndex)];
        if (reference.descriptors.empty() || reference.annotatedPoints.empty()) {
            continue;
        }
        if (candidate.homography.empty()) {
            continue;
        }

        std::vector<cv::Point2f> referenceAnnotatedPoints;
        std::vector<int> cornerIds;
        for (const ReferencePoint& point : reference.annotatedPoints) {
            const int cornerIndex = point.id - 1;
            if (cornerIndex < 0 || cornerIndex >= static_cast<int>(boxCorners.size())) {
                continue;
            }
            referenceAnnotatedPoints.push_back(point.point);
            cornerIds.push_back(cornerIndex);
        }

        if (referenceAnnotatedPoints.size() < 4) {
            continue;
        }

        std::vector<cv::Point2f> transferredPoints;
        cv::perspectiveTransform(referenceAnnotatedPoints, transferredPoints, candidate.homography);

        CorrespondenceSet currentSet;
        currentSet.referenceName = reference.name;
        currentSet.referenceIndex = candidate.referenceIndex;
        currentSet.goodMatches = candidate.goodMatches;
        currentSet.homographyInliers = candidate.homographyInliers;
        for (size_t i = 0; i < transferredPoints.size(); ++i) {
            const cv::Point2f& point = transferredPoints[i];
            if (point.x < 0.0f || point.x >= static_cast<float>(frame.cols) ||
                point.y < 0.0f || point.y >= static_cast<float>(frame.rows)) {
                continue;
            }
            currentSet.objectPoints.push_back(boxCorners[static_cast<size_t>(cornerIds[i])]);
            currentSet.imagePoints.push_back(point);
        }

        currentSet.valid = currentSet.imagePoints.size() >= 4;
        currentSet.score =
            static_cast<double>(currentSet.imagePoints.size()) * 8.0 +
            static_cast<double>(candidate.homographyInliers) +
            0.1 * static_cast<double>(candidate.goodMatches);

        if (currentSet.valid && (!bestSet.valid || currentSet.score > bestSet.score)) {
            bestSet = std::move(currentSet);
        }
    }

    return bestSet;
}

cv::Rect CorrespondenceEstimator::clampRoi(const cv::Rect& roi, const cv::Size& imageSize) const {
    return roi & cv::Rect(0, 0, imageSize.width, imageSize.height);
}

}  // namespace cvproject
