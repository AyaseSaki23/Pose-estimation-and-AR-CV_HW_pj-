#include "cvproject/pose_refiner.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <limits>

namespace cvproject {

PoseEstimate PoseRefiner::refine(const cv::Mat& frame,
                                 const PoseEstimate& initialPose,
                                 const std::vector<cv::Point3f>& boxCorners,
                                 const cv::Mat& cameraMatrix,
                                 const cv::Mat& distCoeffs) const {
    if (!initialPose.valid || frame.empty() || boxCorners.size() < 8 ||
        cameraMatrix.empty()) {
        return initialPose;
    }

    cv::Mat gray;
    if (frame.channels() == 1) {
        gray = frame;
    } else {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat edges;
    cv::Canny(gray, edges, 80.0, 180.0);

    cv::Mat invertedEdges;
    cv::threshold(edges, invertedEdges, 1, 255, cv::THRESH_BINARY_INV);

    cv::Mat edgeDistance;
    cv::distanceTransform(invertedEdges, edgeDistance, cv::DIST_L2, 3);

    PoseEstimate bestPose = initialPose;
    double bestScore = scoreProjectedEdges(edgeDistance, frame.size(), initialPose,
                                           boxCorners, cameraMatrix, distCoeffs);

    const std::array<double, 3> rotationSteps = {0.0, -0.012, 0.012};
    const std::array<double, 3> translationSteps = {0.0, -2.0, 2.0};

    for (int axis = 0; axis < 3; ++axis) {
        for (const double delta : rotationSteps) {
            if (delta == 0.0) {
                continue;
            }
            PoseEstimate candidate = initialPose;
            candidate.rvec = initialPose.rvec.clone();
            candidate.tvec = initialPose.tvec.clone();
            candidate.rvec.at<double>(axis) += delta;
            const double score = scoreProjectedEdges(edgeDistance, frame.size(), candidate,
                                                     boxCorners, cameraMatrix, distCoeffs);
            if (score < bestScore) {
                bestScore = score;
                bestPose = candidate;
            }
        }
    }

    for (int axis = 0; axis < 3; ++axis) {
        for (const double delta : translationSteps) {
            if (delta == 0.0) {
                continue;
            }
            PoseEstimate candidate = initialPose;
            candidate.rvec = initialPose.rvec.clone();
            candidate.tvec = initialPose.tvec.clone();
            candidate.tvec.at<double>(axis) += delta;
            const double score = scoreProjectedEdges(edgeDistance, frame.size(), candidate,
                                                     boxCorners, cameraMatrix, distCoeffs);
            if (score < bestScore) {
                bestScore = score;
                bestPose = candidate;
            }
        }
    }

    bestPose.valid = true;
    bestPose.reprojectionError = initialPose.reprojectionError;
    bestPose.refinementScore = -bestScore;
    return bestPose;
}

double PoseRefiner::scoreProjectedEdges(const cv::Mat& edgeDistance,
                                        const cv::Size& imageSize,
                                        const PoseEstimate& pose,
                                        const std::vector<cv::Point3f>& boxCorners,
                                        const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs) const {
    if (!pose.valid || edgeDistance.empty()) {
        return std::numeric_limits<double>::max();
    }

    std::vector<cv::Point2f> projected;
    try {
        cv::projectPoints(boxCorners, pose.rvec, pose.tvec, cameraMatrix, distCoeffs, projected);
    } catch (const cv::Exception&) {
        return std::numeric_limits<double>::max();
    }
    if (projected.size() < 8) {
        return std::numeric_limits<double>::max();
    }

    const std::array<std::pair<int, int>, 12> edges = {{
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    }};

    double totalDistance = 0.0;
    int samples = 0;
    for (const auto& edge : edges) {
        const cv::Point2f a = projected[edge.first];
        const cv::Point2f b = projected[edge.second];
        const double length = cv::norm(a - b);
        const int steps = std::max(2, static_cast<int>(length / 4.0));
        for (int i = 0; i <= steps; ++i) {
            const float alpha = static_cast<float>(i) / static_cast<float>(steps);
            const cv::Point2f p = a * (1.0f - alpha) + b * alpha;
            const int x = cvRound(p.x);
            const int y = cvRound(p.y);
            if (x < 0 || x >= imageSize.width || y < 0 || y >= imageSize.height) {
                totalDistance += 30.0;
                ++samples;
                continue;
            }
            totalDistance += edgeDistance.at<float>(y, x);
            ++samples;
        }
    }

    return samples > 0
        ? totalDistance / static_cast<double>(samples)
        : std::numeric_limits<double>::max();
}

}  // namespace cvproject
