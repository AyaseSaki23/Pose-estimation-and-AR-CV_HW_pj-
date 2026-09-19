#include <opencv2/opencv.hpp>

#include "cvproject/mesh_loader.hpp"
#include "cvproject/pose_utils.hpp"
#include "cvproject/reference_db.hpp"

#include <filesystem>
#include <iostream>

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path calibPath = projectRoot / "config/camera_calib.yml";
    const std::filesystem::path objectConfigPath = projectRoot / "config/object_01.yml";
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path referenceKeypointsPath = referenceDir / "reference_keypoints.yml";
    const std::filesystem::path referencePosesPath = referenceDir / "reference_poses.yml";
    const std::filesystem::path outputDir = projectRoot / "data/processed/validation";

    cv::FileStorage calibFs(calibPath.string(), cv::FileStorage::READ);
    if (!calibFs.isOpened()) {
        std::cerr << "Failed to open calibration file: " << calibPath.string() << std::endl;
        return 1;
    }

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    calibFs["camera_matrix"] >> cameraMatrix;
    calibFs["dist_coeffs"] >> distCoeffs;
    calibFs.release();

    cv::FileStorage objectFs(objectConfigPath.string(), cv::FileStorage::READ);
    if (!objectFs.isOpened()) {
        std::cerr << "Failed to open object config: " << objectConfigPath.string() << std::endl;
        return 1;
    }

    std::string meshPathText;
    float widthMm = 0.0f;
    float heightMm = 0.0f;
    float depthMm = 0.0f;
    objectFs["mesh_path"] >> meshPathText;
    objectFs["width_mm"] >> widthMm;
    objectFs["height_mm"] >> heightMm;
    objectFs["depth_mm"] >> depthMm;
    objectFs.release();

    const cvproject::Mesh mesh = cvproject::loadObjMesh(projectRoot / meshPathText);
    const float meshScale = cvproject::inferObjToMillimeterScale(mesh, widthMm, heightMm, depthMm);
    const cvproject::BoundingBox bbox = cvproject::computeBoundingBox(mesh, meshScale);
    const std::vector<cv::Point3f> boxCorners = bbox.corners;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    const std::vector<cvproject::ReferenceView> views =
        cvproject::loadReferenceViews(referenceDir, referenceKeypointsPath, orb);
    if (views.empty()) {
        std::cerr << "No annotated reference views found." << std::endl;
        return 1;
    }

    const std::vector<cvproject::ReferencePoseEntry> poses =
        cvproject::computeReferencePoses(views, boxCorners, cameraMatrix, distCoeffs);
    cvproject::saveReferencePoses(referencePosesPath, poses);

    std::filesystem::create_directories(outputDir);
    int validCount = 0;
    for (const cvproject::ReferencePoseEntry& entry : poses) {
        std::cout << entry.imageName
                  << " | valid=" << (entry.valid ? "yes" : "no")
                  << " | reproj=" << entry.reprojErrorPx << " px"
                  << " | view_dir=[" << entry.viewDirectionX << ", "
                  << entry.viewDirectionY << ", " << entry.viewDirectionZ << "]"
                  << std::endl;

        if (!entry.valid || entry.rvec.empty()) {
            continue;
        }

        ++validCount;
        cv::Mat image = cv::imread((referenceDir / entry.imageName).string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            continue;
        }

        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        for (const cvproject::ReferenceView& view : views) {
            if (view.name != entry.imageName) {
                continue;
            }
            for (const cvproject::ReferencePoint& point : view.annotatedPoints) {
                const int cornerIndex = point.id - 1;
                if (cornerIndex < 0 || cornerIndex >= static_cast<int>(boxCorners.size())) {
                    continue;
                }
                objectPoints.push_back(boxCorners[cornerIndex]);
                imagePoints.push_back(point.point);
            }
            break;
        }

        std::vector<cv::Point2f> projected;
        cv::projectPoints(objectPoints, entry.rvec, entry.tvec, cameraMatrix, distCoeffs, projected);

        const std::vector<std::pair<int, int>> edges = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7}
        };

        std::vector<cv::Point2f> allProjected;
        cv::projectPoints(boxCorners, entry.rvec, entry.tvec, cameraMatrix, distCoeffs, allProjected);
        for (const auto& edge : edges) {
            cv::line(image, allProjected[edge.first], allProjected[edge.second],
                     cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        for (size_t i = 0; i < imagePoints.size(); ++i) {
            cv::circle(image, imagePoints[i], 6, cv::Scalar(0, 255, 255), -1);
            cv::circle(image, projected[i], 6, cv::Scalar(255, 0, 255), 2);
        }

        cv::putText(image, "reference pose preview | reproj " + std::to_string(entry.reprojErrorPx) + " px",
                    cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::imwrite((outputDir / ("ref_pose_" + entry.imageName)).string(), image);
    }

    std::cout << "Saved reference poses to: " << referencePosesPath.string() << std::endl;
    std::cout << "Valid posed references: " << validCount << " / " << poses.size() << std::endl;
    return validCount > 0 ? 0 : 1;
}
