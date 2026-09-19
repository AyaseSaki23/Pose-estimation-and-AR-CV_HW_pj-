#pragma once

#include "cvproject/object_localizer.hpp"
#include "cvproject/reference_db.hpp"
#include "cvproject/view_retriever.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

namespace cvproject {

struct CorrespondenceSet {
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::string referenceName;
    int referenceIndex = -1;
    int goodMatches = 0;
    int homographyInliers = 0;
    double score = 0.0;
    bool valid = false;
};

class CorrespondenceEstimator {
public:
    explicit CorrespondenceEstimator(cv::Ptr<cv::ORB> orb);

    CorrespondenceSet estimate(const cv::Mat& frame,
                               const ObjectDetection& detection,
                               const std::vector<ViewCandidate>& viewCandidates,
                               const std::vector<ReferenceView>& references,
                               const std::vector<cv::Point3f>& boxCorners) const;

private:
    cv::Ptr<cv::ORB> orb_;
    double ratioTest_ = 0.75;
    double homographyRansacThreshold_ = 4.0;

    cv::Rect clampRoi(const cv::Rect& roi, const cv::Size& imageSize) const;
};

}  // namespace cvproject
