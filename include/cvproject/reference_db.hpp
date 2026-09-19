#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace cvproject {

struct ReferencePoint {
    int id = -1;
    cv::Point2f point;
};

struct ReferencePoseEntry {
    std::string imageName;
    cv::Mat rvec;
    cv::Mat tvec;
    double reprojErrorPx = -1.0;
    double viewDirectionX = 0.0;
    double viewDirectionY = 0.0;
    double viewDirectionZ = 0.0;
    bool valid = false;
};

struct ReferenceView {
    std::string name;
    cv::Mat imageGray;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<ReferencePoint> annotatedPoints;
    ReferencePoseEntry pose;
};

struct ReferenceTrackingConfig {
    int orbFeatures = 2000;
    int minGoodMatches = 12;
    int minHomographyInliers = 25;
    double ratioTest = 0.75;
    double homographyRansacThreshold = 4.0;
    double maxReprojectionError = 8.0;
    double maxAcceptedErrorIncrease = 3.0;
    int maxLostBeforeManual = 12;
    double translationJumpThresholdMm = 45.0;
    double rotationJumpThresholdRad = 0.45;
    int relocalizeIntervalFrames = 8;
    double referenceSwitchMargin = 0.12;
    double viewSimilarityWeight = 0.25;
    int gtsamWindowSize = 15;
    double gtsamMeasurementRotSigma = 0.08;
    double gtsamMeasurementTransSigmaMm = 6.0;
    double gtsamMotionRotSigma = 0.12;
    double gtsamMotionTransSigmaMm = 15.0;
    double gtsamHuberK = 1.345;
};

bool loadReferenceKeypointsForImage(const std::filesystem::path& path,
                                    const std::string& targetImageName,
                                    std::vector<ReferencePoint>& points);

std::vector<ReferenceView> loadReferenceViews(const std::filesystem::path& referenceDir,
                                              const std::filesystem::path& referenceKeypointsPath,
                                              const cv::Ptr<cv::ORB>& orb);

bool loadReferencePoses(const std::filesystem::path& path,
                        std::vector<ReferencePoseEntry>& poses);

void saveReferencePoses(const std::filesystem::path& path,
                        const std::vector<ReferencePoseEntry>& poses);

void attachReferencePoses(std::vector<ReferenceView>& views,
                          const std::vector<ReferencePoseEntry>& poses);

std::vector<ReferencePoseEntry> computeReferencePoses(
    const std::vector<ReferenceView>& views,
    const std::vector<cv::Point3f>& boxCorners,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs);

}  // namespace cvproject
