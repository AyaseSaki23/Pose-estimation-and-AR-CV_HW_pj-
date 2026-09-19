#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct MatchStats {
    std::filesystem::path referencePath;
    int refKeypoints = 0;
    int frameKeypoints = 0;
    int rawMatches = 0;
    int goodMatches = 0;
    int inliers = 0;
    std::vector<cv::DMatch> inlierMatches;
    std::vector<cv::KeyPoint> refKeypointsData;
    std::vector<cv::KeyPoint> frameKeypointsData;
    cv::Mat referenceImage;
    cv::Mat homography;
};

struct ReferencePoint {
    int id = -1;
    cv::Point2f point;
};

std::vector<std::filesystem::path> listReferenceImages(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::exists(dir)) {
        return paths;
    }

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string filename = entry.path().filename().string();
        std::string lowerFilename = filename;
        std::transform(lowerFilename.begin(), lowerFilename.end(), lowerFilename.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (lowerFilename.find("preview") != std::string::npos ||
            lowerFilename.find("keypoints") != std::string::npos ||
            lowerFilename.find("transferred") != std::string::npos) {
            continue;
        }

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

MatchStats matchOneReference(const std::filesystem::path& refPath,
                             const cv::Mat& frame,
                             const std::vector<cv::KeyPoint>& frameKeypoints,
                             const cv::Mat& frameDescriptors,
                             const cv::Ptr<cv::ORB>& orb) {
    MatchStats stats;
    stats.referencePath = refPath;
    stats.frameKeypoints = static_cast<int>(frameKeypoints.size());
    stats.frameKeypointsData = frameKeypoints;

    stats.referenceImage = cv::imread(refPath.string(), cv::IMREAD_COLOR);
    if (stats.referenceImage.empty()) {
        return stats;
    }

    cv::Mat refGray;
    cv::cvtColor(stats.referenceImage, refGray, cv::COLOR_BGR2GRAY);
    cv::Mat refDescriptors;
    orb->detectAndCompute(refGray, cv::noArray(), stats.refKeypointsData, refDescriptors);
    stats.refKeypoints = static_cast<int>(stats.refKeypointsData.size());

    if (refDescriptors.empty() || frameDescriptors.empty()) {
        return stats;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(refDescriptors, frameDescriptors, knnMatches, 2);
    stats.rawMatches = static_cast<int>(knnMatches.size());

    std::vector<cv::DMatch> goodMatches;
    for (const std::vector<cv::DMatch>& pair : knnMatches) {
        if (pair.size() < 2) {
            continue;
        }

        if (pair[0].distance < 0.75f * pair[1].distance) {
            goodMatches.push_back(pair[0]);
        }
    }
    stats.goodMatches = static_cast<int>(goodMatches.size());

    if (goodMatches.size() < 4) {
        return stats;
    }

    std::vector<cv::Point2f> refPoints;
    std::vector<cv::Point2f> framePoints;
    for (const cv::DMatch& match : goodMatches) {
        refPoints.push_back(stats.refKeypointsData[match.queryIdx].pt);
        framePoints.push_back(frameKeypoints[match.trainIdx].pt);
    }

    cv::Mat inlierMask;
    stats.homography = cv::findHomography(refPoints, framePoints, cv::RANSAC, 4.0, inlierMask);
    if (inlierMask.empty()) {
        return stats;
    }

    for (int i = 0; i < inlierMask.rows; ++i) {
        if (inlierMask.at<unsigned char>(i, 0)) {
            stats.inlierMatches.push_back(goodMatches[static_cast<size_t>(i)]);
        }
    }
    stats.inliers = static_cast<int>(stats.inlierMatches.size());

    return stats;
}

bool loadReferenceKeypointsForImage(const std::filesystem::path& path,
                            const std::string& targetImageName,
                            std::vector<ReferencePoint>& points) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    points.clear();

    cv::FileNode references = fs["references"];
    if (!references.empty()) {
        for (const cv::FileNode& refNode : references) {
            std::string imageName;
            refNode["image"] >> imageName;
            if (imageName != targetImageName) {
                continue;
            }

            cv::FileNode nodes = refNode["points"];
            for (const cv::FileNode& node : nodes) {
                ReferencePoint point;
                point.id = static_cast<int>(node["id"]);
                point.point.x = static_cast<float>(node["x"]);
                point.point.y = static_cast<float>(node["y"]);
                points.push_back(point);
            }
            return !points.empty();
        }
        return false;
    }

    std::string legacyImageName;
    fs["reference_image"] >> legacyImageName;
    if (legacyImageName != targetImageName) {
        return false;
    }

    cv::FileNode nodes = fs["points"];
    for (const cv::FileNode& node : nodes) {
        ReferencePoint point;
        point.id = static_cast<int>(node["id"]);
        point.point.x = static_cast<float>(node["x"]);
        point.point.y = static_cast<float>(node["y"]);
        points.push_back(point);
    }

    return !points.empty();
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path videoPath = projectRoot / "data/raw/videos/input.mp4";
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path referenceKeypointsPath = referenceDir / "reference_keypoints.yml";
    const std::filesystem::path outputImagePath = projectRoot / "data/processed/matches/ref_match_best.jpg";
    const std::filesystem::path outputCsvPath = projectRoot / "data/processed/matches/ref_match_stats.csv";
    const std::filesystem::path transferImagePath = projectRoot / "data/processed/matches/transferred_keypoints.jpg";
    const std::filesystem::path transferYmlPath = projectRoot / "data/processed/matches/transferred_keypoints.yml";

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath.string() << std::endl;
        return 1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to read first frame from: " << videoPath.string() << std::endl;
        return 1;
    }

    std::vector<std::filesystem::path> references = listReferenceImages(referenceDir);
    if (references.empty()) {
        std::cerr << "No reference images found in: " << referenceDir.string() << std::endl;
        return 1;
    }

    cv::Mat frameGray;
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb->detectAndCompute(frameGray, cv::noArray(), frameKeypoints, frameDescriptors);

    std::vector<MatchStats> allStats;
    for (const std::filesystem::path& refPath : references) {
        MatchStats stats = matchOneReference(refPath, frame, frameKeypoints, frameDescriptors, orb);
        std::cout << refPath.filename().string()
                  << " keypoints(ref/frame)=" << stats.refKeypoints << "/" << stats.frameKeypoints
                  << " raw=" << stats.rawMatches
                  << " good=" << stats.goodMatches
                  << " inliers=" << stats.inliers << std::endl;
        allStats.push_back(std::move(stats));
    }

    auto bestIt = std::max_element(
        allStats.begin(),
        allStats.end(),
        [](const MatchStats& a, const MatchStats& b) {
            return a.inliers < b.inliers;
        }
    );

    std::filesystem::create_directories(outputImagePath.parent_path());

    std::ofstream csv(outputCsvPath);
    csv << "reference_image,keypoints_ref,keypoints_frame,raw_matches,good_matches,inliers\n";
    for (const MatchStats& stats : allStats) {
        csv << stats.referencePath.filename().string() << ','
            << stats.refKeypoints << ','
            << stats.frameKeypoints << ','
            << stats.rawMatches << ','
            << stats.goodMatches << ','
            << stats.inliers << '\n';
    }
    csv.close();

    if (bestIt == allStats.end() || bestIt->referenceImage.empty()) {
        std::cerr << "No usable reference match found." << std::endl;
        return 1;
    }

    cv::Mat matchVisualization;
    cv::drawMatches(
        bestIt->referenceImage,
        bestIt->refKeypointsData,
        frame,
        bestIt->frameKeypointsData,
        bestIt->inlierMatches,
        matchVisualization,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    cv::putText(matchVisualization,
                "Best reference: " + bestIt->referencePath.filename().string() +
                    " | inliers: " + std::to_string(bestIt->inliers),
                cv::Point(30, 45),
                cv::FONT_HERSHEY_SIMPLEX,
                0.9,
                cv::Scalar(0, 255, 255),
                2);

    cv::imwrite(outputImagePath.string(), matchVisualization);

    std::vector<ReferencePoint> annotatedPoints;
    const std::string bestReferenceName = bestIt->referencePath.filename().string();
    if (loadReferenceKeypointsForImage(referenceKeypointsPath, bestReferenceName, annotatedPoints)) {
        if (!bestIt->homography.empty()) {
            std::vector<cv::Point2f> sourcePoints;
            for (const ReferencePoint& point : annotatedPoints) {
                sourcePoints.push_back(point.point);
            }

            std::vector<cv::Point2f> transferredPoints;
            cv::perspectiveTransform(sourcePoints, transferredPoints, bestIt->homography);

            cv::Mat transferVisualization = frame.clone();
            cv::FileStorage transferFs(transferYmlPath.string(), cv::FileStorage::WRITE);
            transferFs << "source_reference" << bestReferenceName;
            transferFs << "points" << "[";
            for (size_t i = 0; i < transferredPoints.size(); ++i) {
                const cv::Point2f point = transferredPoints[i];
                const cv::Point pixel(cvRound(point.x), cvRound(point.y));
                cv::circle(transferVisualization, pixel, 5, cv::Scalar(0, 255, 255), -1);
                cv::putText(transferVisualization, std::to_string(annotatedPoints[i].id),
                            pixel + cv::Point(8, -8),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                transferFs << "{"
                           << "id" << annotatedPoints[i].id
                           << "x" << point.x
                           << "y" << point.y
                           << "}";
            }
            transferFs << "]";
            transferFs.release();

            cv::putText(transferVisualization,
                        "Transferred keypoints from " + bestReferenceName,
                        cv::Point(30, 45),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.85,
                        cv::Scalar(0, 255, 255),
                        2);
            cv::imwrite(transferImagePath.string(), transferVisualization);
            std::cout << "Saved transferred keypoints to: " << transferYmlPath.string() << std::endl;
            std::cout << "Saved transferred keypoint image to: " << transferImagePath.string() << std::endl;
        } else {
            std::cout << "Best reference has no valid homography. Transfer skipped." << std::endl;
        }
    } else {
        std::cout << "No keypoint annotation found for best reference " << bestReferenceName
                  << ". Run annotate_reference for this image first." << std::endl;
    }

    std::cout << "Best reference: " << bestIt->referencePath.filename().string()
              << " with " << bestIt->inliers << " inliers" << std::endl;
    std::cout << "Saved match visualization to: " << outputImagePath.string() << std::endl;
    std::cout << "Saved match stats to: " << outputCsvPath.string() << std::endl;

    cv::namedWindow("Best Reference Match", cv::WINDOW_NORMAL);
    cv::resizeWindow("Best Reference Match", 1280, 720);
    cv::imshow("Best Reference Match", matchVisualization);
    cv::waitKey(0);

    return 0;
}
