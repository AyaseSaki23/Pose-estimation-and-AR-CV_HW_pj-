#include "cvproject/pose_utils.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>

namespace cvproject {

bool hasNonCoplanarObjectPoints(const std::vector<cv::Point3f>& points) {
    if (points.size() < 4) {
        return false;
    }

    const cv::Point3f origin = points[0];
    for (size_t i = 1; i + 2 < points.size(); ++i) {
        const cv::Point3f a = points[i] - origin;
        for (size_t j = i + 1; j + 1 < points.size(); ++j) {
            const cv::Point3f b = points[j] - origin;
            const cv::Point3f normal = a.cross(b);
            if (cv::norm(normal) < 1e-4f) {
                continue;
            }

            for (size_t k = j + 1; k < points.size(); ++k) {
                const cv::Point3f c = points[k] - origin;
                if (std::abs(normal.dot(c)) > 1e-2f) {
                    return true;
                }
            }
        }
    }

    return false;
}

double computeMeanReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Mat& rvec,
                                    const cv::Mat& tvec,
                                    const cv::Mat& cameraMatrix,
                                    const cv::Mat& distCoeffs) {
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, reprojected);

    double meanError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i) {
        meanError += cv::norm(imagePoints[i] - reprojected[i]);
    }

    return meanError / static_cast<double>(imagePoints.size());
}

bool solvePosePnP(const std::vector<cv::Point3f>& objectPoints,
                  const std::vector<cv::Point2f>& imagePoints,
                  const cv::Mat& cameraMatrix,
                  const cv::Mat& distCoeffs,
                  cv::Mat& rvec,
                  cv::Mat& tvec,
                  const bool useExtrinsicGuess,
                  const bool refineWithLm) {
    if (objectPoints.size() < 4 || objectPoints.size() != imagePoints.size()) {
        return false;
    }

    const int pnpMethod = imagePoints.size() >= 6 ? cv::SOLVEPNP_ITERATIVE : cv::SOLVEPNP_EPNP;
    if (!cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                      rvec, tvec, useExtrinsicGuess, pnpMethod)) {
        return false;
    }

    if (refineWithLm) {
        cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    }

    return true;
}

double rotationDistanceRad(const cv::Mat& rvecA, const cv::Mat& rvecB) {
    cv::Mat rotationA;
    cv::Mat rotationB;
    cv::Rodrigues(rvecA, rotationA);
    cv::Rodrigues(rvecB, rotationB);

    cv::Mat relativeRotation = rotationB * rotationA.t();
    cv::Mat relativeRvec;
    cv::Rodrigues(relativeRotation, relativeRvec);
    return cv::norm(relativeRvec);
}

bool isPoseJumpTooLarge(const cv::Mat& previousRvec,
                        const cv::Mat& previousTvec,
                        const cv::Mat& currentRvec,
                        const cv::Mat& currentTvec,
                        const double translationThresholdMm,
                        const double rotationThresholdRad) {
    const double translationJump = cv::norm(currentTvec - previousTvec);
    const double rotationJump = rotationDistanceRad(previousRvec, currentRvec);
    return translationJump > translationThresholdMm || rotationJump > rotationThresholdRad;
}

}  // namespace cvproject
