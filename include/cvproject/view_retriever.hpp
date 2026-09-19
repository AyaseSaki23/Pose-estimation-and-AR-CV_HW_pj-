#pragma once

#include "cvproject/object_localizer.hpp"
#include "cvproject/reference_db.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

namespace cvproject {

struct ViewCandidate {
    std::string referenceName;
    int referenceIndex = -1;
    cv::Mat homography;
    double score = 0.0;
    int goodMatches = 0;
    int homographyInliers = 0;
    double inlierRatio = 0.0;
};

class ViewRetriever {
public:
    explicit ViewRetriever(cv::Ptr<cv::ORB> orb);

    std::vector<ViewCandidate> retrieveTopK(const cv::Mat& frame,
                                            const ObjectDetection& detection,
                                            const std::vector<ReferenceView>& references,
                                            int topK) const;

private:
    cv::Ptr<cv::ORB> orb_;
    double ratioTest_ = 0.75;
    int minGoodMatches_ = 10;
    double homographyRansacThreshold_ = 4.0;

    cv::Rect clampRoi(const cv::Rect& roi, const cv::Size& imageSize) const;
};

}  // namespace cvproject
