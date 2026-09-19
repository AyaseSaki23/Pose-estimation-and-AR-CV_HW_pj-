#include "cvproject/viewpoint_selector.hpp"

#include "cvproject/pose_utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace cvproject {

namespace {

double computeViewSimilarity(const ReferencePoseEntry& referencePose,
                             const cv::Mat& poseHintRvec) {
    if (!referencePose.valid || referencePose.rvec.empty()) {
        return 0.0;
    }

    const double angleRad = rotationDistanceRad(referencePose.rvec, poseHintRvec);
    return std::exp(-angleRad);
}

double computeMatchScore(const ViewpointMatchResult& result,
                         const ReferenceTrackingConfig& config,
                         const double viewSimilarity) {
    const double inlierScore = static_cast<double>(result.inliers);
    const double matchScore = static_cast<double>(result.goodMatches);
    const double reprojPenalty = result.reprojectionError;
    return inlierScore + 0.05 * matchScore - 2.0 * reprojPenalty +
           config.viewSimilarityWeight * viewSimilarity * inlierScore;
}

}  // namespace

ViewpointSelector::ViewpointSelector(const ReferenceTrackingConfig& config)
    : config_(config) {}

ViewpointMatchResult ViewpointSelector::localizeFrame(
    const cv::Mat& frame,
    const std::vector<ReferenceView>& referenceViews,
    const cv::Ptr<cv::ORB>& orb,
    const std::vector<cv::Point3f>& boxCorners,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const std::optional<std::string>& lockedReference,
    const bool hasPoseHint,
    const cv::Mat& poseHintRvec) const {
    ViewpointMatchResult bestResult;

    if (referenceViews.empty()) {
        return bestResult;
    }

    cv::Mat frameGray;
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb->detectAndCompute(frameGray, cv::noArray(), frameKeypoints, frameDescriptors);
    if (frameDescriptors.empty()) {
        return bestResult;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<ViewpointMatchResult> candidates;

    for (const ReferenceView& referenceView : referenceViews) {
        ViewpointMatchResult candidate;
        candidate.referenceName = referenceView.name;

        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(referenceView.descriptors, frameDescriptors, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const std::vector<cv::DMatch>& pair : knnMatches) {
            if (pair.size() < 2) {
                continue;
            }
            if (pair[0].distance < config_.ratioTest * pair[1].distance) {
                goodMatches.push_back(pair[0]);
            }
        }

        candidate.goodMatches = static_cast<int>(goodMatches.size());
        if (goodMatches.size() < static_cast<size_t>(config_.minGoodMatches)) {
            candidates.push_back(candidate);
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
            refPoints, framePoints, cv::RANSAC, config_.homographyRansacThreshold, inlierMask);
        if (homography.empty() || inlierMask.empty()) {
            candidates.push_back(candidate);
            continue;
        }

        int inliers = 0;
        for (int i = 0; i < inlierMask.rows; ++i) {
            if (inlierMask.at<unsigned char>(i, 0)) {
                ++inliers;
            }
        }
        candidate.inliers = inliers;
        if (inliers < config_.minHomographyInliers) {
            candidates.push_back(candidate);
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
            candidates.push_back(candidate);
            continue;
        }

        std::vector<cv::Point2f> transferredPoints;
        cv::perspectiveTransform(sourcePoints, transferredPoints, homography);

        cv::Mat rvec;
        cv::Mat tvec;
        const bool useGuess = hasPoseHint && referenceView.pose.valid;
        if (useGuess) {
            rvec = referenceView.pose.rvec.clone();
            tvec = referenceView.pose.tvec.clone();
        }

        if (!solvePosePnP(objectPoints, transferredPoints, cameraMatrix, distCoeffs,
                          rvec, tvec, useGuess, true)) {
            candidates.push_back(candidate);
            continue;
        }

        candidate.reprojectionError = computeMeanReprojectionError(
            objectPoints, transferredPoints, rvec, tvec, cameraMatrix, distCoeffs);
        if (candidate.reprojectionError > config_.maxReprojectionError) {
            candidates.push_back(candidate);
            continue;
        }

        candidate.viewSimilarity = hasPoseHint
            ? computeViewSimilarity(referenceView.pose, poseHintRvec)
            : 0.0;
        candidate.score = computeMatchScore(candidate, config_, candidate.viewSimilarity);
        candidate.ok = true;
        candidate.objectPoints = objectPoints;
        candidate.imagePoints = transferredPoints;
        candidate.rvec = rvec;
        candidate.tvec = tvec;
        candidates.push_back(candidate);
    }

    const std::string selectedReference = selectReferenceWithHysteresis(candidates, lockedReference);
    for (const ViewpointMatchResult& candidate : candidates) {
        if (candidate.ok && candidate.referenceName == selectedReference &&
            candidate.score >= bestResult.score) {
            bestResult = candidate;
        }
    }

    return bestResult;
}

std::string ViewpointSelector::selectReferenceWithHysteresis(
    const std::vector<ViewpointMatchResult>& candidates,
    const std::optional<std::string>& lockedReference) const {
    std::string bestReference;
    double bestScore = -1.0;
    for (const ViewpointMatchResult& candidate : candidates) {
        if (!candidate.ok || candidate.score <= bestScore) {
            continue;
        }
        bestScore = candidate.score;
        bestReference = candidate.referenceName;
    }

    if (bestReference.empty()) {
        return bestReference;
    }

    if (!lockedReference.has_value()) {
        return bestReference;
    }

    double lockedScore = -1.0;
    for (const ViewpointMatchResult& candidate : candidates) {
        if (candidate.ok && candidate.referenceName == lockedReference.value()) {
            lockedScore = candidate.score;
            break;
        }
    }

    if (lockedScore >= 0.0 && lockedScore >= bestScore * (1.0 - config_.referenceSwitchMargin)) {
        return lockedReference.value();
    }

    return bestReference;
}

}  // namespace cvproject
