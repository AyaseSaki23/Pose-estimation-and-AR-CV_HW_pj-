#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace cvproject {

bool hasNonCoplanarObjectPoints(const std::vector<cv::Point3f>& points);

double computeMeanReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Mat& rvec,
                                    const cv::Mat& tvec,
                                    const cv::Mat& cameraMatrix,
                                    const cv::Mat& distCoeffs);

bool solvePosePnP(const std::vector<cv::Point3f>& objectPoints,
                  const std::vector<cv::Point2f>& imagePoints,
                  const cv::Mat& cameraMatrix,
                  const cv::Mat& distCoeffs,
                  cv::Mat& rvec,
                  cv::Mat& tvec,
                  bool useExtrinsicGuess = false,
                  bool refineWithLm = true);

double rotationDistanceRad(const cv::Mat& rvecA, const cv::Mat& rvecB);

bool isPoseJumpTooLarge(const cv::Mat& previousRvec,
                        const cv::Mat& previousTvec,
                        const cv::Mat& currentRvec,
                        const cv::Mat& currentTvec,
                        double translationThresholdMm,
                        double rotationThresholdRad);

}  // namespace cvproject
