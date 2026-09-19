#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct ReferencePoint {
    int id = -1;
    cv::Point2f point;
};

struct ReferenceView {
    std::string name;
    cv::Mat imageGray;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<ReferencePoint> annotatedPoints;
};

struct TrackConfig {
    int orbFeatures = 4000;
    int minGoodMatches = 18;
    int minHomographyInliers = 20;
    double ratioTest = 0.75;
    double ransacThresholdPx = 4.0;
};

struct HomographyTrack {
    bool ok = false;
    std::string referenceName;
    cv::Mat homography;
    int goodMatches = 0;
    int inliers = 0;
    double inlierRatio = 0.0;
    std::vector<ReferencePoint> transferredPoints;
};

cv::Point toPixel(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

std::string lowerString(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::vector<std::filesystem::path> listReferenceImages(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::exists(dir)) {
        return paths;
    }

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filename = lowerString(entry.path().filename().string());
        if (filename.find("preview") != std::string::npos ||
            filename.find("keypoints") != std::string::npos ||
            filename.find("transferred") != std::string::npos ||
            filename.find("reference_poses") != std::string::npos) {
            continue;
        }

        const std::string ext = lowerString(entry.path().extension().string());
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

bool loadReferenceKeypointsForImage(const std::filesystem::path& path,
                                    const std::string& targetImageName,
                                    std::vector<ReferencePoint>& points) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    points.clear();
    const cv::FileNode references = fs["references"];
    for (const cv::FileNode& refNode : references) {
        std::string imageName;
        refNode["image"] >> imageName;
        if (imageName != targetImageName) {
            continue;
        }

        for (const cv::FileNode& node : refNode["points"]) {
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

std::vector<ReferenceView> loadReferenceViews(const std::filesystem::path& referenceDir,
                                              const std::filesystem::path& keypointsPath,
                                              const cv::Ptr<cv::ORB>& orb) {
    std::vector<ReferenceView> views;
    for (const std::filesystem::path& imagePath : listReferenceImages(referenceDir)) {
        cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Skipping unreadable reference image: " << imagePath.string() << std::endl;
            continue;
        }

        ReferenceView view;
        view.name = imagePath.filename().string();
        if (!loadReferenceKeypointsForImage(keypointsPath, view.name, view.annotatedPoints)) {
            std::cerr << "Skipping reference without annotation: " << view.name << std::endl;
            continue;
        }

        cv::cvtColor(image, view.imageGray, cv::COLOR_BGR2GRAY);
        orb->detectAndCompute(view.imageGray, cv::noArray(), view.keypoints, view.descriptors);
        if (view.descriptors.empty()) {
            std::cerr << "Skipping reference without descriptors: " << view.name << std::endl;
            continue;
        }

        views.push_back(view);
    }
    return views;
}

HomographyTrack trackFrame2D(const cv::Mat& frame,
                             const std::vector<ReferenceView>& references,
                             const cv::Ptr<cv::ORB>& orb,
                             const TrackConfig& config) {
    HomographyTrack best;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb->detectAndCompute(gray, cv::noArray(), frameKeypoints, frameDescriptors);
    if (frameDescriptors.empty()) {
        return best;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    for (const ReferenceView& reference : references) {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(reference.descriptors, frameDescriptors, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const std::vector<cv::DMatch>& pair : knnMatches) {
            if (pair.size() < 2) {
                continue;
            }
            if (pair[0].distance < config.ratioTest * pair[1].distance) {
                goodMatches.push_back(pair[0]);
            }
        }

        if (goodMatches.size() < static_cast<size_t>(config.minGoodMatches)) {
            continue;
        }

        std::vector<cv::Point2f> referencePoints;
        std::vector<cv::Point2f> framePoints;
        for (const cv::DMatch& match : goodMatches) {
            referencePoints.push_back(reference.keypoints[match.queryIdx].pt);
            framePoints.push_back(frameKeypoints[match.trainIdx].pt);
        }

        cv::Mat inlierMask;
        cv::Mat homography = cv::findHomography(
            referencePoints, framePoints, cv::RANSAC, config.ransacThresholdPx, inlierMask);
        if (homography.empty() || inlierMask.empty()) {
            continue;
        }

        int inliers = 0;
        for (int i = 0; i < inlierMask.rows; ++i) {
            if (inlierMask.at<unsigned char>(i, 0) != 0) {
                ++inliers;
            }
        }
        if (inliers < config.minHomographyInliers || inliers <= best.inliers) {
            continue;
        }

        std::vector<cv::Point2f> sourceCorners;
        for (const ReferencePoint& point : reference.annotatedPoints) {
            sourceCorners.push_back(point.point);
        }

        std::vector<cv::Point2f> transferredCorners;
        cv::perspectiveTransform(sourceCorners, transferredCorners, homography);

        best.ok = true;
        best.referenceName = reference.name;
        best.homography = homography;
        best.goodMatches = static_cast<int>(goodMatches.size());
        best.inliers = inliers;
        best.inlierRatio = static_cast<double>(inliers) / static_cast<double>(goodMatches.size());
        best.transferredPoints.clear();
        for (size_t i = 0; i < reference.annotatedPoints.size(); ++i) {
            best.transferredPoints.push_back({reference.annotatedPoints[i].id, transferredCorners[i]});
        }
    }

    return best;
}

void drawTransferredAnnotations(cv::Mat& image,
                                const std::vector<ReferencePoint>& points,
                                const cv::Scalar& color) {
    std::map<int, cv::Point2f> byId;
    for (const ReferencePoint& point : points) {
        byId[point.id] = point.point;
    }

    const std::vector<std::pair<int, int>> boxEdges = {
        {1, 2}, {2, 3}, {3, 4}, {4, 1},
        {5, 6}, {6, 7}, {7, 8}, {8, 5},
        {1, 5}, {2, 6}, {3, 7}, {4, 8}
    };

    for (const auto& edge : boxEdges) {
        const auto first = byId.find(edge.first);
        const auto second = byId.find(edge.second);
        if (first == byId.end() || second == byId.end()) {
            continue;
        }
        cv::line(image, toPixel(first->second), toPixel(second->second), color, 2, cv::LINE_AA);
    }

    for (const ReferencePoint& point : points) {
        cv::circle(image, toPixel(point.point), 6, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
        cv::circle(image, toPixel(point.point), 8, color, 2, cv::LINE_AA);
        cv::putText(image, std::to_string(point.id), toPixel(point.point + cv::Point2f(8.0f, -8.0f)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
}

std::string serializePoints(const std::vector<ReferencePoint>& points) {
    std::ostringstream out;
    for (size_t i = 0; i < points.size(); ++i) {
        if (i > 0) {
            out << ';';
        }
        out << points[i].id << ':' << points[i].point.x << ':' << points[i].point.y;
    }
    return out.str();
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path videoPath = projectRoot / "data/raw/videos/input.mp4";
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path keypointsPath = referenceDir / "reference_keypoints.yml";
    const std::filesystem::path outputVideoPath =
        projectRoot / "data/processed/validation/output_2d_reference_diagnostic.mp4";
    const std::filesystem::path outputCsvPath =
        projectRoot / "data/processed/validation/track_2d_reference_diagnostic.csv";

    TrackConfig config;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config.orbFeatures);
    const std::vector<ReferenceView> references =
        loadReferenceViews(referenceDir, keypointsPath, orb);
    if (references.empty()) {
        std::cerr << "No annotated reference views loaded from: " << referenceDir.string() << std::endl;
        return 1;
    }

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath.string() << std::endl;
        return 1;
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Failed to read first frame from: " << videoPath.string() << std::endl;
        return 1;
    }

    std::filesystem::create_directories(outputVideoPath.parent_path());
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    cv::VideoWriter writer(
        outputVideoPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << outputVideoPath.string() << std::endl;
        return 1;
    }

    std::ofstream csv(outputCsvPath);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV: " << outputCsvPath.string() << std::endl;
        return 1;
    }
    csv << "frame_id,track_ok,reference,good_matches,inliers,inlier_ratio,points\n";

    std::cout << "Loaded " << references.size() << " reference views." << std::endl;
    std::cout << "Running pure 2D reference tracking: ORB + homography + transferred annotated corners." << std::endl;

    int frameId = 0;
    HomographyTrack lastTrack;
    while (true) {
        HomographyTrack track = trackFrame2D(frame, references, orb, config);
        const bool holdingLast = !track.ok && lastTrack.ok;
        if (track.ok) {
            lastTrack = track;
        } else if (holdingLast) {
            track = lastTrack;
        }

        cv::Mat output = frame.clone();
        const cv::Scalar drawColor =
            track.ok && !holdingLast ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        if (track.ok || holdingLast) {
            drawTransferredAnnotations(output, track.transferredPoints, drawColor);
        }

        const std::string status = track.ok && !holdingLast ? "fresh homography" :
                                   holdingLast ? "lost: holding last 2D corners" :
                                                 "lost: no reference match";
        cv::putText(output, "2D reference diagnostic | frame " + std::to_string(frameId),
                    cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(output, status,
                    cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.65, drawColor, 2);
        cv::putText(output,
                    "ref: " + (track.ok ? track.referenceName : std::string("none")) +
                        " | inliers: " + std::to_string(track.inliers) +
                        " | good: " + std::to_string(track.goodMatches),
                    cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

        writer.write(output);
        csv << frameId << ','
            << ((track.ok && !holdingLast) ? 1 : 0) << ','
            << (track.ok ? track.referenceName : "none") << ','
            << track.goodMatches << ','
            << track.inliers << ','
            << track.inlierRatio << ','
            << '"' << serializePoints(track.transferredPoints) << '"' << '\n';

        cv::imshow("2D Reference Diagnostic", output);
        const int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }

        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        ++frameId;
    }

    writer.release();
    csv.close();

    std::cout << "Saved diagnostic video to: " << outputVideoPath.string() << std::endl;
    std::cout << "Saved diagnostic CSV to: " << outputCsvPath.string() << std::endl;
    std::cout << "Interpretation: if these yellow corner ids jump away from the object, check reference_keypoints.yml or matching quality." << std::endl;
    std::cout << "If these are stable but pose_estimation flies, the issue is probably PnP/pose filtering/GTSAM rather than 2D annotations." << std::endl;

    return 0;
}
