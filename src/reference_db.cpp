#include "cvproject/reference_db.hpp"

#include "cvproject/pose_utils.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>

namespace cvproject {

namespace {

cv::Vec3d computeViewDirection(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    cv::Mat viewInCamera = -rotation.t() * tvec;
    const double norm = cv::norm(viewInCamera);
    if (norm < 1e-6) {
        return {0.0, 0.0, 1.0};
    }
    viewInCamera /= norm;
    return {viewInCamera.at<double>(0), viewInCamera.at<double>(1), viewInCamera.at<double>(2)};
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

        std::string lowerFilename = entry.path().filename().string();
        std::transform(lowerFilename.begin(), lowerFilename.end(), lowerFilename.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowerFilename.find("preview") != std::string::npos ||
            lowerFilename.find("keypoints") != std::string::npos ||
            lowerFilename.find("transferred") != std::string::npos ||
            lowerFilename.find("reference_poses") != std::string::npos) {
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

}  // namespace

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

    std::string legacyImageName;
    fs["reference_image"] >> legacyImageName;
    if (legacyImageName != targetImageName) {
        return false;
    }

    for (const cv::FileNode& node : fs["points"]) {
        ReferencePoint point;
        point.id = static_cast<int>(node["id"]);
        point.point.x = static_cast<float>(node["x"]);
        point.point.y = static_cast<float>(node["y"]);
        points.push_back(point);
    }

    return !points.empty();
}

std::vector<ReferenceView> loadReferenceViews(const std::filesystem::path& referenceDir,
                                              const std::filesystem::path& referenceKeypointsPath,
                                              const cv::Ptr<cv::ORB>& orb) {
    std::vector<ReferenceView> views;
    for (const std::filesystem::path& refPath : listReferenceImages(referenceDir)) {
        cv::Mat refImage = cv::imread(refPath.string(), cv::IMREAD_COLOR);
        if (refImage.empty()) {
            std::cerr << "Skipping unreadable reference image: " << refPath.string() << std::endl;
            continue;
        }

        ReferenceView view;
        view.name = refPath.filename().string();
        if (!loadReferenceKeypointsForImage(referenceKeypointsPath, view.name, view.annotatedPoints)) {
            std::cerr << "Skipping reference without annotation: " << view.name << std::endl;
            continue;
        }

        cv::cvtColor(refImage, view.imageGray, cv::COLOR_BGR2GRAY);
        orb->detectAndCompute(view.imageGray, cv::noArray(), view.keypoints, view.descriptors);
        if (view.descriptors.empty()) {
            std::cerr << "Skipping reference without ORB descriptors: " << view.name << std::endl;
            continue;
        }

        views.push_back(view);
    }

    return views;
}

bool loadReferencePoses(const std::filesystem::path& path,
                        std::vector<ReferencePoseEntry>& poses) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    poses.clear();
    cv::FileNode references = fs["references"];
    if (references.empty()) {
        return false;
    }

    for (const cv::FileNode& refNode : references) {
        ReferencePoseEntry entry;
        refNode["image"] >> entry.imageName;
        refNode["rvec"] >> entry.rvec;
        refNode["tvec"] >> entry.tvec;
        entry.reprojErrorPx = static_cast<double>(refNode["reproj_error_px"]);
        entry.viewDirectionX = static_cast<double>(refNode["view_dir_x"]);
        entry.viewDirectionY = static_cast<double>(refNode["view_dir_y"]);
        entry.viewDirectionZ = static_cast<double>(refNode["view_dir_z"]);
        entry.valid = static_cast<int>(refNode["valid"]) != 0;
        if (!entry.imageName.empty()) {
            poses.push_back(entry);
        }
    }

    return !poses.empty();
}

void saveReferencePoses(const std::filesystem::path& path,
                        const std::vector<ReferencePoseEntry>& poses) {
    std::filesystem::create_directories(path.parent_path());

    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    fs << "references" << "[";
    for (const ReferencePoseEntry& entry : poses) {
        fs << "{"
           << "image" << entry.imageName
           << "valid" << (entry.valid ? 1 : 0)
           << "rvec" << entry.rvec
           << "tvec" << entry.tvec
           << "reproj_error_px" << entry.reprojErrorPx
           << "view_dir_x" << entry.viewDirectionX
           << "view_dir_y" << entry.viewDirectionY
           << "view_dir_z" << entry.viewDirectionZ
           << "}";
    }
    fs << "]";
}

void attachReferencePoses(std::vector<ReferenceView>& views,
                          const std::vector<ReferencePoseEntry>& poses) {
    for (ReferenceView& view : views) {
        for (const ReferencePoseEntry& pose : poses) {
            if (pose.imageName == view.name) {
                view.pose = pose;
                break;
            }
        }
    }
}

std::vector<ReferencePoseEntry> computeReferencePoses(
    const std::vector<ReferenceView>& views,
    const std::vector<cv::Point3f>& boxCorners,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs) {
    std::vector<ReferencePoseEntry> poses;

    for (const ReferenceView& view : views) {
        ReferencePoseEntry entry;
        entry.imageName = view.name;

        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        for (const ReferencePoint& point : view.annotatedPoints) {
            const int cornerIndex = point.id - 1;
            if (cornerIndex < 0 || cornerIndex >= static_cast<int>(boxCorners.size())) {
                continue;
            }
            objectPoints.push_back(boxCorners[cornerIndex]);
            imagePoints.push_back(point.point);
        }

        if (imagePoints.size() < 4 || !hasNonCoplanarObjectPoints(objectPoints)) {
            entry.valid = false;
            poses.push_back(entry);
            continue;
        }

        cv::Mat rvec;
        cv::Mat tvec;
        if (!solvePosePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, true)) {
            entry.valid = false;
            poses.push_back(entry);
            continue;
        }

        const cv::Vec3d viewDirection = computeViewDirection(rvec, tvec);
        entry.rvec = rvec;
        entry.tvec = tvec;
        entry.reprojErrorPx = computeMeanReprojectionError(
            objectPoints, imagePoints, rvec, tvec, cameraMatrix, distCoeffs);
        entry.viewDirectionX = viewDirection[0];
        entry.viewDirectionY = viewDirection[1];
        entry.viewDirectionZ = viewDirection[2];
        entry.valid = entry.reprojErrorPx <= 8.0;
        poses.push_back(entry);
    }

    return poses;
}

}  // namespace cvproject
