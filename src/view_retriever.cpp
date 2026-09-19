#include "cvproject/view_retriever.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace cvproject {

ViewRetriever::ViewRetriever(cv::Ptr<cv::ORB> orb)
    : orb_(std::move(orb)) {}

std::vector<ViewCandidate> ViewRetriever::retrieveTopK(
    const cv::Mat& frame,
    const ObjectDetection& detection,
    const std::vector<ReferenceView>& references,
    const int topK) const {
    std::vector<ViewCandidate> candidates;
    if (frame.empty() || !detection.valid || references.empty() || topK <= 0 || orb_.empty()) {
        return candidates;
    }

    const cv::Rect roi = clampRoi(detection.roi, frame.size());
    if (roi.empty()) {
        return candidates;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat roiGray = gray(roi);

    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb_->detectAndCompute(roiGray, cv::Mat(), frameKeypoints, frameDescriptors);
    if (frameDescriptors.empty()) {
        return candidates;
    }

    std::vector<int> keypointInRoi(frameKeypoints.size(), 0);
    for (size_t i = 0; i < frameKeypoints.size(); ++i) {
        frameKeypoints[i].pt.x += static_cast<float>(roi.x);
        frameKeypoints[i].pt.y += static_cast<float>(roi.y);
        keypointInRoi[i] = 1;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    for (size_t refIndex = 0; refIndex < references.size(); ++refIndex) {
        const ReferenceView& reference = references[refIndex];
        if (reference.descriptors.empty()) {
            continue;
        }

        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(reference.descriptors, frameDescriptors, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        std::vector<cv::Point2f> refPoints;
        std::vector<cv::Point2f> framePoints;
        for (const std::vector<cv::DMatch>& pair : knnMatches) {
            if (pair.size() < 2) {
                continue;
            }
            const cv::DMatch& best = pair[0];
            const cv::DMatch& second = pair[1];
            if (best.trainIdx < 0 ||
                best.trainIdx >= static_cast<int>(keypointInRoi.size()) ||
                keypointInRoi[static_cast<size_t>(best.trainIdx)] == 0) {
                continue;
            }
            if (best.distance >= static_cast<float>(ratioTest_) * second.distance) {
                continue;
            }

            goodMatches.push_back(best);
            refPoints.push_back(reference.keypoints[best.queryIdx].pt);
            framePoints.push_back(frameKeypoints[best.trainIdx].pt);
        }

        if (static_cast<int>(goodMatches.size()) < minGoodMatches_) {
            continue;
        }

        int inliers = 0;
        cv::Mat homography;
        if (refPoints.size() >= 4) {
            cv::Mat inlierMask;
            homography = cv::findHomography(
                refPoints, framePoints, cv::RANSAC, homographyRansacThreshold_, inlierMask);
            if (!homography.empty() && !inlierMask.empty()) {
                for (int i = 0; i < inlierMask.rows; ++i) {
                    if (inlierMask.at<unsigned char>(i, 0) != 0) {
                        ++inliers;
                    }
                }
            }
        }

        ViewCandidate candidate;
        candidate.referenceName = reference.name;
        candidate.referenceIndex = static_cast<int>(refIndex);
        candidate.homography = homography.clone();
        candidate.goodMatches = static_cast<int>(goodMatches.size());
        candidate.homographyInliers = inliers;
        candidate.inlierRatio = goodMatches.empty()
            ? 0.0
            : static_cast<double>(inliers) / static_cast<double>(goodMatches.size());
        candidate.score =
            static_cast<double>(inliers) +
            0.15 * static_cast<double>(candidate.goodMatches) +
            12.0 * candidate.inlierRatio;
        candidates.push_back(candidate);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const ViewCandidate& a, const ViewCandidate& b) {
                  return a.score > b.score;
              });
    if (candidates.size() > static_cast<size_t>(topK)) {
        candidates.resize(static_cast<size_t>(topK));
    }

    return candidates;
}

cv::Rect ViewRetriever::clampRoi(const cv::Rect& roi, const cv::Size& imageSize) const {
    return roi & cv::Rect(0, 0, imageSize.width, imageSize.height);
}

}  // namespace cvproject
