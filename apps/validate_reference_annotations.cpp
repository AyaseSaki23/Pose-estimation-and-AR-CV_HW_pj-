#include <opencv2/opencv.hpp>

#include "cvproject/mesh_loader.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

struct ReferencePoint {
    int id = -1;
    cv::Point2f point;
};

struct ReferenceAnnotation {
    std::string imageName;
    std::vector<ReferencePoint> points;
};

struct ValidationReport {
    std::string imageName;
    int annotatedCount = 0;
    bool pnpOk = false;
    bool nonCoplanar = false;
    double meanReprojErrorPx = -1.0;
    double maxReprojErrorPx = -1.0;
    std::string notes;
};

cv::Point toPixel(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
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

double computeMaxReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                   const std::vector<cv::Point2f>& imagePoints,
                                   const cv::Mat& rvec,
                                   const cv::Mat& tvec,
                                   const cv::Mat& cameraMatrix,
                                   const cv::Mat& distCoeffs) {
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, reprojected);

    double maxError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i) {
        maxError = std::max(maxError, static_cast<double>(cv::norm(imagePoints[i] - reprojected[i])));
    }
    return maxError;
}

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

std::vector<ReferenceAnnotation> loadAllAnnotations(const std::filesystem::path& path) {
    std::vector<ReferenceAnnotation> annotations;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return annotations;
    }

    cv::FileNode references = fs["references"];
    if (!references.empty()) {
        for (const cv::FileNode& refNode : references) {
            ReferenceAnnotation annotation;
            refNode["image"] >> annotation.imageName;
            for (const cv::FileNode& pointNode : refNode["points"]) {
                ReferencePoint point;
                point.id = static_cast<int>(pointNode["id"]);
                point.point.x = static_cast<float>(pointNode["x"]);
                point.point.y = static_cast<float>(pointNode["y"]);
                annotation.points.push_back(point);
            }
            if (!annotation.imageName.empty()) {
                annotations.push_back(annotation);
            }
        }
    }

    return annotations;
}

std::map<int, cv::Point2f> toPointMap(const std::vector<ReferencePoint>& points) {
    std::map<int, cv::Point2f> map;
    for (const ReferencePoint& point : points) {
        map[point.id] = point.point;
    }
    return map;
}

void drawProjectedBox(cv::Mat& image,
                      const std::vector<cv::Point3f>& boxCorners,
                      const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      const cv::Mat& cameraMatrix,
                      const cv::Mat& distCoeffs) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(boxCorners, rvec, tvec, cameraMatrix, distCoeffs, projected);

    const std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    for (const auto& edge : edges) {
        cv::line(image, toPixel(projected[edge.first]), toPixel(projected[edge.second]),
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    for (size_t i = 0; i < projected.size(); ++i) {
        cv::circle(image, toPixel(projected[i]), 4, cv::Scalar(0, 255, 0), -1);
        cv::putText(image, std::to_string(i + 1), toPixel(projected[i] + cv::Point2f(5.0f, -5.0f)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

ValidationReport validateOneReference(const ReferenceAnnotation& annotation,
                                      const std::filesystem::path& referenceDir,
                                      const std::vector<cv::Point3f>& boxCorners,
                                      const cv::Mat& cameraMatrix,
                                      const cv::Mat& distCoeffs,
                                      const std::filesystem::path& outputDir) {
    ValidationReport report;
    report.imageName = annotation.imageName;
    report.annotatedCount = static_cast<int>(annotation.points.size());

    const std::filesystem::path imagePath = referenceDir / annotation.imageName;
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        report.notes = "image missing";
        return report;
    }

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    for (const ReferencePoint& point : annotation.points) {
        const int cornerIndex = point.id - 1;
        if (cornerIndex < 0 || cornerIndex >= static_cast<int>(boxCorners.size())) {
            continue;
        }
        objectPoints.push_back(boxCorners[cornerIndex]);
        imagePoints.push_back(point.point);
    }

    if (imagePoints.size() < 4) {
        report.notes = "need >= 4 annotated corners";
        return report;
    }

    report.nonCoplanar = hasNonCoplanarObjectPoints(objectPoints);
    if (!report.nonCoplanar) {
        report.notes = "annotated corners are coplanar (cannot solve full 6D pose)";
    }

    cv::Mat rvec;
    cv::Mat tvec;
    const int pnpMethod = imagePoints.size() >= 6 ? cv::SOLVEPNP_ITERATIVE : cv::SOLVEPNP_EPNP;
    report.pnpOk = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                                rvec, tvec, false, pnpMethod);
    if (!report.pnpOk) {
        report.notes = "solvePnP failed";
        return report;
    }

    report.meanReprojErrorPx =
        computeMeanReprojectionError(objectPoints, imagePoints, rvec, tvec, cameraMatrix, distCoeffs);
    report.maxReprojErrorPx =
        computeMaxReprojectionError(objectPoints, imagePoints, rvec, tvec, cameraMatrix, distCoeffs);

    cv::Mat preview = image.clone();
    drawProjectedBox(preview, boxCorners, rvec, tvec, cameraMatrix, distCoeffs);

    for (const ReferencePoint& point : annotation.points) {
        cv::circle(preview, toPixel(point.point), 6, cv::Scalar(0, 255, 255), -1);
        cv::putText(preview, std::to_string(point.id), toPixel(point.point + cv::Point2f(8.0f, -8.0f)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }

    cv::putText(preview,
                "yellow=annotated green=projected | mean err " +
                    std::to_string(report.meanReprojErrorPx) + " px",
                cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(preview,
                report.nonCoplanar ? "non-coplanar: OK for 6D" : "coplanar: tracking will skip",
                cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                report.nonCoplanar ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

    std::filesystem::create_directories(outputDir);
    cv::imwrite((outputDir / ("self_check_" + annotation.imageName)).string(), preview);

    if (report.meanReprojErrorPx > 3.0) {
        report.notes = "high self reprojection error, likely wrong corner ids or clicks";
    } else if (!report.nonCoplanar) {
        report.notes = "coplanar only";
    } else {
        report.notes = "looks consistent";
    }

    return report;
}

double edgeLength3D(const cv::Point3f& a, const cv::Point3f& b) {
    return cv::norm(a - b);
}

double edgeLength2D(const cv::Point2f& a, const cv::Point2f& b) {
    return cv::norm(a - b);
}

void validateCrossReference(const ReferenceAnnotation& source,
                            const ReferenceAnnotation& target,
                            const std::filesystem::path& referenceDir,
                            const cv::Ptr<cv::ORB>& orb,
                            const std::filesystem::path& outputDir) {
    const std::filesystem::path sourcePath = referenceDir / source.imageName;
    const std::filesystem::path targetPath = referenceDir / target.imageName;
    cv::Mat sourceImage = cv::imread(sourcePath.string(), cv::IMREAD_COLOR);
    cv::Mat targetImage = cv::imread(targetPath.string(), cv::IMREAD_COLOR);
    if (sourceImage.empty() || targetImage.empty()) {
        return;
    }

    cv::Mat sourceGray;
    cv::Mat targetGray;
    cv::cvtColor(sourceImage, sourceGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(targetImage, targetGray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> sourceKeypoints;
    std::vector<cv::KeyPoint> targetKeypoints;
    cv::Mat sourceDescriptors;
    cv::Mat targetDescriptors;
    orb->detectAndCompute(sourceGray, cv::noArray(), sourceKeypoints, sourceDescriptors);
    orb->detectAndCompute(targetGray, cv::noArray(), targetKeypoints, targetDescriptors);
    if (sourceDescriptors.empty() || targetDescriptors.empty()) {
        return;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(sourceDescriptors, targetDescriptors, knnMatches, 2);

    std::vector<cv::DMatch> goodMatches;
    for (const std::vector<cv::DMatch>& pair : knnMatches) {
        if (pair.size() < 2) {
            continue;
        }
        if (pair[0].distance < 0.75f * pair[1].distance) {
            goodMatches.push_back(pair[0]);
        }
    }
    if (goodMatches.size() < 8) {
        return;
    }

    std::vector<cv::Point2f> sourcePoints;
    std::vector<cv::Point2f> targetPoints;
    for (const cv::DMatch& match : goodMatches) {
        sourcePoints.push_back(sourceKeypoints[match.queryIdx].pt);
        targetPoints.push_back(targetKeypoints[match.trainIdx].pt);
    }

    cv::Mat inlierMask;
    cv::Mat homography = cv::findHomography(sourcePoints, targetPoints, cv::RANSAC, 4.0, inlierMask);
    if (homography.empty()) {
        return;
    }

    const std::map<int, cv::Point2f> sourceMap = toPointMap(source.points);
    const std::map<int, cv::Point2f> targetMap = toPointMap(target.points);

    cv::Mat preview = targetImage.clone();
    double totalTransferError = 0.0;
    int transferCount = 0;

    for (const auto& [id, sourcePoint] : sourceMap) {
        const auto targetIt = targetMap.find(id);
        if (targetIt == targetMap.end()) {
            continue;
        }

        std::vector<cv::Point2f> src = {sourcePoint};
        std::vector<cv::Point2f> transferred;
        cv::perspectiveTransform(src, transferred, homography);

        const cv::Point2f gt = targetIt->second;
        const cv::Point2f pred = transferred.front();
        const double err = cv::norm(gt - pred);
        totalTransferError += err;
        ++transferCount;

        cv::circle(preview, toPixel(gt), 6, cv::Scalar(0, 255, 255), -1);
        cv::circle(preview, toPixel(pred), 6, cv::Scalar(255, 0, 255), 2);
        cv::line(preview, toPixel(gt), toPixel(pred), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::putText(preview, std::to_string(id), toPixel(gt + cv::Point2f(8.0f, -8.0f)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }

    if (transferCount == 0) {
        return;
    }

    const double meanTransferError = totalTransferError / static_cast<double>(transferCount);
    cv::putText(preview,
                source.imageName + " -> " + target.imageName +
                    " | transfer err " + std::to_string(meanTransferError) + " px",
                cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
    cv::putText(preview, "yellow=target annotation magenta=transferred",
                cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

    const std::string fileName = "cross_" + source.imageName + "_to_" + target.imageName;
    cv::imwrite((outputDir / fileName).string(), preview);
    std::cout << "Cross-check " << source.imageName << " -> " << target.imageName
              << ": mean transfer error = " << meanTransferError << " px ("
              << transferCount << " shared corners)" << std::endl;
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path calibPath = projectRoot / "config/camera_calib.yml";
    const std::filesystem::path objectConfigPath = projectRoot / "config/object_01.yml";
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path keypointsPath = referenceDir / "reference_keypoints.yml";
    const std::filesystem::path outputDir = projectRoot / "data/processed/validation";
    const std::filesystem::path reportPath = outputDir / "annotation_report.csv";

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

    std::cout << "3D corner order (same as annotate_reference 1-8):" << std::endl;
    for (size_t i = 0; i < boxCorners.size(); ++i) {
        std::cout << "  " << (i + 1) << " [" << boxCorners[i].x << ", "
                  << boxCorners[i].y << ", " << boxCorners[i].z << "]" << std::endl;
    }

    const std::vector<ReferenceAnnotation> annotations = loadAllAnnotations(keypointsPath);
    if (annotations.empty()) {
        std::cerr << "No annotations found in: " << keypointsPath.string() << std::endl;
        return 1;
    }

    std::filesystem::create_directories(outputDir);
    std::ofstream report(reportPath);
    report << "image,annotated_count,non_coplanar,pnp_ok,mean_reproj_px,max_reproj_px,notes\n";

    int suspiciousCount = 0;
    for (const ReferenceAnnotation& annotation : annotations) {
        const ValidationReport result = validateOneReference(
            annotation, referenceDir, boxCorners, cameraMatrix, distCoeffs, outputDir);

        report << result.imageName << ','
               << result.annotatedCount << ','
               << (result.nonCoplanar ? 1 : 0) << ','
               << (result.pnpOk ? 1 : 0) << ','
               << result.meanReprojErrorPx << ','
               << result.maxReprojErrorPx << ','
               << result.notes << '\n';

        std::cout << result.imageName
                  << " | annotated=" << result.annotatedCount
                  << " | non_coplanar=" << (result.nonCoplanar ? "yes" : "no")
                  << " | mean reproj=" << result.meanReprojErrorPx << " px"
                  << " | " << result.notes << std::endl;

        if (result.meanReprojErrorPx > 3.0 || !result.nonCoplanar) {
            ++suspiciousCount;
        }
    }
    report.close();

    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    for (size_t i = 0; i < annotations.size(); ++i) {
        for (size_t j = 0; j < annotations.size(); ++j) {
            if (i == j) {
                continue;
            }
            validateCrossReference(annotations[i], annotations[j], referenceDir, orb, outputDir);
        }
    }

    std::cout << "\nSaved self-check previews and cross-check images to: "
              << outputDir.string() << std::endl;
    std::cout << "Saved CSV report to: " << reportPath.string() << std::endl;
    std::cout << "\nHow to interpret:" << std::endl;
    std::cout << "  - self_check_*.jpg: yellow dots should sit on green projected corners." << std::endl;
    std::cout << "  - mean reproj < 2 px on reference image => annotation is self-consistent." << std::endl;
    std::cout << "  - mean reproj > 5 px => likely wrong corner id or click position." << std::endl;
    std::cout << "  - cross_* images: yellow/magenta should overlap; large red gaps => bad annotation or homography." << std::endl;
    std::cout << "Suspicious references: " << suspiciousCount << " / " << annotations.size() << std::endl;

    return 0;
}
