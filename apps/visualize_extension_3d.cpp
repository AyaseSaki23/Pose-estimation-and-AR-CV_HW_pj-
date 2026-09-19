#include <opencv2/opencv.hpp>

#include "cvproject/mesh_loader.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct PoseSample {
    int frameId = 0;
    double timestampSec = 0.0;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Point3d cameraCenter;
    bool stableFaceTracked = false;
    std::string referenceName;
};

struct SceneProjector {
    int width = 1280;
    int height = 720;
    cv::Rect viewport = {0, 0, 1280, 720};
    double focal = 1100.0;
    double distance = 900.0;
    cv::Point3d sceneCenter = {0.0, 0.0, 0.0};
    double sceneScale = 1.0;
    cv::Matx33d viewRotation;

    SceneProjector() {
        const double yaw = -35.0 * CV_PI / 180.0;
        const double pitch = 24.0 * CV_PI / 180.0;
        const cv::Matx33d rotY(
            std::cos(yaw), 0.0, std::sin(yaw),
            0.0, 1.0, 0.0,
            -std::sin(yaw), 0.0, std::cos(yaw)
        );
        const cv::Matx33d rotX(
            1.0, 0.0, 0.0,
            0.0, std::cos(pitch), -std::sin(pitch),
            0.0, std::sin(pitch), std::cos(pitch)
        );
        viewRotation = rotX * rotY;
    }

    cv::Point2i project(const cv::Point3d& point) const {
        const cv::Vec3d p(
            (point.x - sceneCenter.x) * sceneScale,
            (point.y - sceneCenter.y) * sceneScale,
            (point.z - sceneCenter.z) * sceneScale
        );
        const cv::Vec3d camera = viewRotation * p + cv::Vec3d(0.0, 0.0, distance);
        const double z = std::max(1.0, camera[2]);
        return {
            viewport.x + cvRound(viewport.width * 0.5 + focal * camera[0] / z),
            viewport.y + cvRound(viewport.height * 0.58 - focal * camera[1] / z)
        };
    }
};

struct SceneBounds {
    cv::Point3d min = {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    };
    cv::Point3d max = {
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()
    };
};

void includePoint(SceneBounds& bounds, const cv::Point3d& point) {
    bounds.min.x = std::min(bounds.min.x, point.x);
    bounds.min.y = std::min(bounds.min.y, point.y);
    bounds.min.z = std::min(bounds.min.z, point.z);
    bounds.max.x = std::max(bounds.max.x, point.x);
    bounds.max.y = std::max(bounds.max.y, point.y);
    bounds.max.z = std::max(bounds.max.z, point.z);
}

SceneProjector createProjectorForScene(const std::vector<PoseSample>& samples,
                                       const std::vector<cv::Point3f>& boxCorners) {
    SceneBounds bounds;
    for (const cv::Point3f& corner : boxCorners) {
        includePoint(bounds, {corner.x, corner.y, corner.z});
    }

    cv::Point3d cameraMean(0.0, 0.0, 0.0);
    for (const PoseSample& sample : samples) {
        cameraMean.x += sample.cameraCenter.x;
        cameraMean.y += sample.cameraCenter.y;
        cameraMean.z += sample.cameraCenter.z;
    }
    if (!samples.empty()) {
        const double invCount = 1.0 / static_cast<double>(samples.size());
        cameraMean.x *= invCount;
        cameraMean.y *= invCount;
        cameraMean.z *= invCount;
    }

    std::vector<double> cameraDistances;
    cameraDistances.reserve(samples.size());
    for (const PoseSample& sample : samples) {
        cameraDistances.push_back(cv::norm(sample.cameraCenter - cameraMean));
    }
    std::sort(cameraDistances.begin(), cameraDistances.end());
    const double robustDistance = cameraDistances.empty()
        ? std::numeric_limits<double>::max()
        : cameraDistances[static_cast<size_t>(0.95 * static_cast<double>(cameraDistances.size() - 1))];

    for (const PoseSample& sample : samples) {
        const double distance = cv::norm(sample.cameraCenter - cameraMean);
        if (distance <= robustDistance || samples.size() < 20) {
            includePoint(bounds, sample.cameraCenter);
        }
    }

    SceneProjector projector;
    projector.viewport = {0, 0, 870, projector.height};
    projector.sceneCenter = {
        0.5 * (bounds.min.x + bounds.max.x),
        0.5 * (bounds.min.y + bounds.max.y),
        0.5 * (bounds.min.z + bounds.max.z)
    };

    std::vector<cv::Point3d> boundCorners = {
        {bounds.min.x, bounds.min.y, bounds.min.z},
        {bounds.max.x, bounds.min.y, bounds.min.z},
        {bounds.max.x, bounds.max.y, bounds.min.z},
        {bounds.min.x, bounds.max.y, bounds.min.z},
        {bounds.min.x, bounds.min.y, bounds.max.z},
        {bounds.max.x, bounds.min.y, bounds.max.z},
        {bounds.max.x, bounds.max.y, bounds.max.z},
        {bounds.min.x, bounds.max.y, bounds.max.z}
    };

    double maxProjectedRadius = 1.0;
    for (const cv::Point3d& corner : boundCorners) {
        const cv::Vec3d centered(
            corner.x - projector.sceneCenter.x,
            corner.y - projector.sceneCenter.y,
            corner.z - projector.sceneCenter.z
        );
        const cv::Vec3d view = projector.viewRotation * centered;
        maxProjectedRadius = std::max(maxProjectedRadius, std::abs(view[0]));
        maxProjectedRadius = std::max(maxProjectedRadius, std::abs(view[1]));
    }

    const double usablePixels = 0.44 * static_cast<double>(
        std::min(projector.viewport.width, projector.viewport.height));
    const double perspectiveScale = projector.distance / projector.focal;
    projector.sceneScale = usablePixels * perspectiveScale / maxProjectedRadius;
    projector.sceneScale *= 0.85;
    return projector;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> columns;
    std::string current;
    bool inQuotes = false;
    for (char ch : line) {
        if (ch == '"') {
            inQuotes = !inQuotes;
        } else if (ch == ',' && !inQuotes) {
            columns.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    columns.push_back(current);
    return columns;
}

int findColumn(const std::vector<std::string>& header, const std::string& name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool parseDouble(const std::vector<std::string>& row, int index, double& value) {
    if (index < 0 || index >= static_cast<int>(row.size())) {
        return false;
    }
    try {
        value = std::stod(row[index]);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parseInt(const std::vector<std::string>& row, int index, int& value) {
    if (index < 0 || index >= static_cast<int>(row.size())) {
        return false;
    }
    try {
        value = std::stoi(row[index]);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

cv::Point3d cameraCenterFromPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    cv::Mat rvecMat = (cv::Mat_<double>(3, 1) << rvec[0], rvec[1], rvec[2]);
    cv::Mat rotation;
    cv::Rodrigues(rvecMat, rotation);
    cv::Mat t = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
    cv::Mat center = -rotation.t() * t;
    return {
        center.at<double>(0),
        center.at<double>(1),
        center.at<double>(2)
    };
}

std::vector<PoseSample> loadPoseSamples(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open pose CSV: " + path.string());
    }

    std::string line;
    if (!std::getline(input, line)) {
        return {};
    }

    const std::vector<std::string> header = splitCsvLine(line);
    const int frameCol = findColumn(header, "frame_id");
    const int timeCol = findColumn(header, "timestamp_sec");
    const int rvecXCol = findColumn(header, "rvec_x");
    const int rvecYCol = findColumn(header, "rvec_y");
    const int rvecZCol = findColumn(header, "rvec_z");
    const int tvecXCol = findColumn(header, "tvec_x");
    const int tvecYCol = findColumn(header, "tvec_y");
    const int tvecZCol = findColumn(header, "tvec_z");
    const int stableCol = findColumn(header, "stable_face_tracked");
    int referenceCol = findColumn(header, "reference_name");
    if (referenceCol < 0) {
        referenceCol = findColumn(header, "top_view");
    }
    const int poseValidCol = findColumn(header, "pose_valid");

    std::vector<PoseSample> samples;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = splitCsvLine(line);

        int poseValid = 1;
        if (poseValidCol >= 0 && parseInt(row, poseValidCol, poseValid) && poseValid == 0) {
            continue;
        }

        PoseSample sample;
        int frameId = 0;
        double timestamp = 0.0;
        double rx = 0.0;
        double ry = 0.0;
        double rz = 0.0;
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        if (!parseInt(row, frameCol, frameId) ||
            !parseDouble(row, timeCol, timestamp) ||
            !parseDouble(row, rvecXCol, rx) ||
            !parseDouble(row, rvecYCol, ry) ||
            !parseDouble(row, rvecZCol, rz) ||
            !parseDouble(row, tvecXCol, tx) ||
            !parseDouble(row, tvecYCol, ty) ||
            !parseDouble(row, tvecZCol, tz)) {
            continue;
        }

        sample.frameId = frameId;
        sample.timestampSec = timestamp;
        sample.rvec = {rx, ry, rz};
        sample.tvec = {tx, ty, tz};
        sample.cameraCenter = cameraCenterFromPose(sample.rvec, sample.tvec);
        if (stableCol >= 0 && stableCol < static_cast<int>(row.size())) {
            sample.stableFaceTracked = row[stableCol] == "1";
        }
        if (referenceCol >= 0 && referenceCol < static_cast<int>(row.size())) {
            sample.referenceName = row[referenceCol];
        }
        samples.push_back(sample);
    }

    return samples;
}

void drawLine3D(cv::Mat& image,
                const SceneProjector& projector,
                const cv::Point3d& a,
                const cv::Point3d& b,
                const cv::Scalar& color,
                int thickness = 2) {
    cv::line(image, projector.project(a), projector.project(b), color, thickness, cv::LINE_AA);
}

void drawPoint3D(cv::Mat& image,
                 const SceneProjector& projector,
                 const cv::Point3d& point,
                 const cv::Scalar& color,
                 int radius = 4) {
    cv::circle(image, projector.project(point), radius, color, -1, cv::LINE_AA);
}

void drawGroundGrid(cv::Mat& image,
                    const SceneProjector& projector,
                    const std::vector<PoseSample>& samples,
                    const std::vector<cv::Point3f>& corners) {
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();
    double groundY = 0.0;

    for (const cv::Point3f& corner : corners) {
        minX = std::min(minX, static_cast<double>(corner.x));
        maxX = std::max(maxX, static_cast<double>(corner.x));
        minZ = std::min(minZ, static_cast<double>(corner.z));
        maxZ = std::max(maxZ, static_cast<double>(corner.z));
        groundY = std::min(groundY, static_cast<double>(corner.y));
    }
    for (const PoseSample& sample : samples) {
        minX = std::min(minX, sample.cameraCenter.x);
        maxX = std::max(maxX, sample.cameraCenter.x);
        minZ = std::min(minZ, sample.cameraCenter.z);
        maxZ = std::max(maxZ, sample.cameraCenter.z);
    }

    const double margin = 100.0;
    const double step = 50.0;
    minX = std::floor((minX - margin) / step) * step;
    maxX = std::ceil((maxX + margin) / step) * step;
    minZ = std::floor((minZ - margin) / step) * step;
    maxZ = std::ceil((maxZ + margin) / step) * step;

    for (double x = minX; x <= maxX; x += step) {
        const bool major = std::fmod(std::abs(x), 100.0) < 1e-6;
        drawLine3D(image, projector, {x, groundY, minZ}, {x, groundY, maxZ},
                   major ? cv::Scalar(210, 210, 205) : cv::Scalar(226, 226, 222),
                   major ? 2 : 1);
    }
    for (double z = minZ; z <= maxZ; z += step) {
        const bool major = std::fmod(std::abs(z), 100.0) < 1e-6;
        drawLine3D(image, projector, {minX, groundY, z}, {maxX, groundY, z},
                   major ? cv::Scalar(210, 210, 205) : cv::Scalar(226, 226, 222),
                   major ? 2 : 1);
    }
}

void fillPolygonAlpha(cv::Mat& image,
                      const std::vector<cv::Point>& polygon,
                      const cv::Scalar& color,
                      double alpha) {
    cv::Mat overlay = image.clone();
    std::vector<std::vector<cv::Point>> polygons = {polygon};
    cv::fillPoly(overlay, polygons, color, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image);
}

void drawBox(cv::Mat& image,
             const SceneProjector& projector,
             const std::vector<cv::Point3f>& corners) {
    const std::vector<std::vector<int>> faces = {
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {0, 1, 5, 4},
        {1, 2, 6, 5},
        {2, 3, 7, 6},
        {3, 0, 4, 7}
    };
    for (const std::vector<int>& face : faces) {
        std::vector<cv::Point> polygon;
        for (int index : face) {
            polygon.push_back(projector.project({
                corners[index].x,
                corners[index].y,
                corners[index].z
            }));
        }
        fillPolygonAlpha(image, polygon, cv::Scalar(210, 175, 95), 0.18);
    }

    const std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };
    for (const auto& edge : edges) {
        const cv::Point3d a(corners[edge.first].x, corners[edge.first].y, corners[edge.first].z);
        const cv::Point3d b(corners[edge.second].x, corners[edge.second].y, corners[edge.second].z);
        drawLine3D(image, projector, a, b, cv::Scalar(70, 220, 70), 3);
    }

    for (size_t i = 0; i < corners.size(); ++i) {
        const cv::Point2i p = projector.project({corners[i].x, corners[i].y, corners[i].z});
        cv::circle(image, p, 4, cv::Scalar(80, 180, 80), -1, cv::LINE_AA);
        cv::putText(image, std::to_string(i + 1), p + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(55, 130, 55), 1);
    }
}

void drawAxes(cv::Mat& image, const SceneProjector& projector, double length) {
    drawLine3D(image, projector, {0, 0, 0}, {length, 0, 0}, cv::Scalar(60, 60, 240), 3);
    drawLine3D(image, projector, {0, 0, 0}, {0, length, 0}, cv::Scalar(60, 200, 60), 3);
    drawLine3D(image, projector, {0, 0, 0}, {0, 0, length}, cv::Scalar(240, 90, 60), 3);
    cv::putText(image, "X", projector.project({length, 0, 0}) + cv::Point(8, -8),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(60, 60, 240), 2);
    cv::putText(image, "Y", projector.project({0, length, 0}) + cv::Point(8, -8),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(60, 200, 60), 2);
    cv::putText(image, "Z", projector.project({0, 0, length}) + cv::Point(8, -8),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(240, 90, 60), 2);
}

void drawCameraFrustum(cv::Mat& image,
                       const SceneProjector& projector,
                       const PoseSample& sample,
                       double size) {
    cv::Mat rvecMat = (cv::Mat_<double>(3, 1) << sample.rvec[0], sample.rvec[1], sample.rvec[2]);
    cv::Mat rotationCv;
    cv::Rodrigues(rvecMat, rotationCv);
    cv::Matx33d rotation;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            rotation(row, col) = rotationCv.at<double>(row, col);
        }
    }

    const cv::Matx33d cameraToWorld = rotation.t();
    const cv::Vec3d center(sample.cameraCenter.x, sample.cameraCenter.y, sample.cameraCenter.z);
    const std::vector<cv::Vec3d> local = {
        {0.0, 0.0, 0.0},
        {-0.8 * size, -0.55 * size, size},
        {0.8 * size, -0.55 * size, size},
        {0.8 * size, 0.55 * size, size},
        {-0.8 * size, 0.55 * size, size}
    };

    std::vector<cv::Point3d> world;
    for (const cv::Vec3d& point : local) {
        const cv::Vec3d p = center + cameraToWorld * point;
        world.emplace_back(p[0], p[1], p[2]);
    }

    const cv::Scalar color = sample.stableFaceTracked ? cv::Scalar(255, 220, 60) : cv::Scalar(255, 120, 40);
    drawPoint3D(image, projector, world[0], color, 5);
    for (int i = 1; i <= 4; ++i) {
        drawLine3D(image, projector, world[0], world[i], color, 1);
        drawLine3D(image, projector, world[i], world[i == 4 ? 1 : i + 1], color, 1);
    }
}

void drawHistoryFrustums(cv::Mat& image,
                         const SceneProjector& projector,
                         const std::vector<PoseSample>& samples,
                         size_t currentIndex,
                         int windowSize) {
    const size_t first = currentIndex > static_cast<size_t>(windowSize)
        ? currentIndex - static_cast<size_t>(windowSize)
        : 0;
    for (size_t i = first; i < currentIndex; i += 12) {
        drawCameraFrustum(image, projector, samples[i], 20.0);
    }
}

void drawTrajectory(cv::Mat& image,
                    const SceneProjector& projector,
                    const std::vector<PoseSample>& samples,
                    size_t currentIndex,
                    int windowSize) {
    if (samples.empty()) {
        return;
    }

    const size_t first = currentIndex > static_cast<size_t>(windowSize)
        ? currentIndex - static_cast<size_t>(windowSize)
        : 0;
    for (size_t i = first + 1; i <= currentIndex; ++i) {
        const double alpha = static_cast<double>(i - first) /
            static_cast<double>(std::max<size_t>(1, currentIndex - first));
        const cv::Scalar color(40 + 180 * alpha, 160 + 70 * alpha, 255);
        drawLine3D(image, projector, samples[i - 1].cameraCenter, samples[i].cameraCenter, color, 2);
    }
    for (size_t i = first; i <= currentIndex; i += 5) {
        drawPoint3D(image, projector, samples[i].cameraCenter, cv::Scalar(50, 180, 255), 3);
    }
}

cv::Point2i projectTopDown(const cv::Point3d& point,
                           const cv::Rect& panel,
                           const cv::Point2d& center,
                           double scale) {
    return {
        panel.x + cvRound(panel.width * 0.5 + (point.x - center.x) * scale),
        panel.y + cvRound(panel.height * 0.55 - (point.z - center.y) * scale)
    };
}

double percentileValue(std::vector<double> values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::clamp(percentile, 0.0, 1.0);
    const size_t index = static_cast<size_t>(
        std::round(clamped * static_cast<double>(values.size() - 1)));
    return values[index];
}

void drawTopDownPanel(cv::Mat& image,
                      const cv::Rect& panel,
                      const std::vector<PoseSample>& samples,
                      size_t currentIndex,
                      const std::vector<cv::Point3f>& boxCorners,
                      int trajectoryWindow) {
    cv::rectangle(image, panel, cv::Scalar(238, 238, 234), -1, cv::LINE_AA);
    cv::rectangle(image, panel, cv::Scalar(195, 195, 188), 1, cv::LINE_AA);
    cv::putText(image, "Top-down 3D track",
                panel.tl() + cv::Point(18, 34), cv::FONT_HERSHEY_SIMPLEX,
                0.68, cv::Scalar(55, 55, 55), 2);

    const size_t first = currentIndex > static_cast<size_t>(trajectoryWindow)
        ? currentIndex - static_cast<size_t>(trajectoryWindow)
        : 0;
    std::vector<double> xs;
    std::vector<double> zs;
    xs.reserve(currentIndex - first + 1 + boxCorners.size());
    zs.reserve(currentIndex - first + 1 + boxCorners.size());
    for (size_t i = first; i <= currentIndex; ++i) {
        xs.push_back(samples[i].cameraCenter.x);
        zs.push_back(samples[i].cameraCenter.z);
    }
    for (const cv::Point3f& corner : boxCorners) {
        xs.push_back(static_cast<double>(corner.x));
        zs.push_back(static_cast<double>(corner.z));
    }

    double minX = percentileValue(xs, 0.05);
    double maxX = percentileValue(xs, 0.95);
    double minZ = percentileValue(zs, 0.05);
    double maxZ = percentileValue(zs, 0.95);

    const double spanXBeforeMargin = std::max(1.0, maxX - minX);
    const double spanZBeforeMargin = std::max(1.0, maxZ - minZ);
    const double marginX = std::max(30.0, 0.18 * spanXBeforeMargin);
    const double marginZ = std::max(30.0, 0.18 * spanZBeforeMargin);
    minX -= marginX;
    maxX += marginX;
    minZ -= marginZ;
    maxZ += marginZ;

    const cv::Point2d center(0.5 * (minX + maxX), 0.5 * (minZ + maxZ));
    const double spanX = std::max(1.0, maxX - minX);
    const double spanZ = std::max(1.0, maxZ - minZ);
    const double scale = 0.78 * std::min(panel.width / spanX, panel.height / spanZ);

    const double gridStep = 50.0;
    for (double x = std::floor(minX / gridStep) * gridStep; x <= maxX; x += gridStep) {
        const cv::Point2i a = projectTopDown({x, 0.0, minZ}, panel, center, scale);
        const cv::Point2i b = projectTopDown({x, 0.0, maxZ}, panel, center, scale);
        cv::line(image, a, b, cv::Scalar(220, 220, 216), 1, cv::LINE_AA);
    }
    for (double z = std::floor(minZ / gridStep) * gridStep; z <= maxZ; z += gridStep) {
        const cv::Point2i a = projectTopDown({minX, 0.0, z}, panel, center, scale);
        const cv::Point2i b = projectTopDown({maxX, 0.0, z}, panel, center, scale);
        cv::line(image, a, b, cv::Scalar(220, 220, 216), 1, cv::LINE_AA);
    }

    const std::vector<int> footprint = {0, 1, 2, 3};
    std::vector<cv::Point> boxPoly;
    for (int index : footprint) {
        boxPoly.push_back(projectTopDown(
            {boxCorners[index].x, boxCorners[index].y, boxCorners[index].z},
            panel, center, scale));
    }
    fillPolygonAlpha(image, boxPoly, cv::Scalar(145, 210, 155), 0.35);
    cv::polylines(image, boxPoly, true, cv::Scalar(70, 170, 80), 2, cv::LINE_AA);

    for (size_t i = first + 1; i <= currentIndex; ++i) {
        const double alpha = static_cast<double>(i - first) /
            static_cast<double>(std::max<size_t>(1, currentIndex - first));
        const cv::Scalar color(40 + 190 * alpha, 150 + 80 * alpha, 255);
        cv::line(image,
                 projectTopDown(samples[i - 1].cameraCenter, panel, center, scale),
                 projectTopDown(samples[i].cameraCenter, panel, center, scale),
                 color, 2, cv::LINE_AA);
    }

    const cv::Point2i current = projectTopDown(samples[currentIndex].cameraCenter, panel, center, scale);
    cv::circle(image, current, 7, cv::Scalar(35, 95, 245), -1, cv::LINE_AA);
    cv::circle(image, current, 11, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    cv::putText(image, "last " + std::to_string(trajectoryWindow) + " frames",
                panel.tl() + cv::Point(18, panel.height - 22), cv::FONT_HERSHEY_SIMPLEX,
                0.52, cv::Scalar(80, 80, 80), 1);
}

cv::Mat renderFrame(const std::vector<PoseSample>& samples,
                    size_t currentIndex,
                    const std::vector<cv::Point3f>& boxCorners,
                    const SceneProjector& projector,
                    int trajectoryWindow) {
    cv::Mat image(projector.height, projector.width, CV_8UC3, cv::Scalar(245, 245, 242));

    cv::rectangle(image, projector.viewport, cv::Scalar(247, 247, 244), -1, cv::LINE_AA);
    drawGroundGrid(image, projector, samples, boxCorners);
    drawAxes(image, projector, 120.0);
    drawBox(image, projector, boxCorners);
    drawTrajectory(image, projector, samples, currentIndex, trajectoryWindow);
    drawHistoryFrustums(image, projector, samples, currentIndex, trajectoryWindow);
    drawCameraFrustum(image, projector, samples[currentIndex], 35.0);

    const cv::Rect topDownPanel(
        projector.viewport.x + projector.viewport.width + 24,
        94,
        projector.width - (projector.viewport.x + projector.viewport.width + 48),
        projector.height - 170
    );
    drawTopDownPanel(image, topDownPanel, samples, currentIndex, boxCorners, trajectoryWindow);

    cv::putText(image, "Extension 3.1: camera + world model in 3D",
                cv::Point(32, 42), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(40, 40, 40), 2);
    cv::putText(image, "Extension 3.2: last " + std::to_string(trajectoryWindow) + " frame 3D track",
                cv::Point(32, 76), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(70, 70, 70), 2);
    cv::putText(image, "frame " + std::to_string(samples[currentIndex].frameId) +
                    " | ref " + samples[currentIndex].referenceName,
                cv::Point(32, projector.height - 34), cv::FONT_HERSHEY_SIMPLEX,
                0.65, cv::Scalar(70, 70, 70), 2);

    cv::putText(image, "green: object model | orange/blue: camera | yellow: recent track",
                cv::Point(32, projector.height - 68), cv::FONT_HERSHEY_SIMPLEX,
                0.58, cv::Scalar(90, 90, 90), 1);
    return image;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    std::filesystem::path poseCsvPath = projectRoot / "data/processed/poses/poses.csv";
    const std::filesystem::path objectConfigPath = projectRoot / "config/object_01.yml";
    std::filesystem::path outputDir = projectRoot / "data/processed/extension";
    if (argc >= 2) {
        poseCsvPath = std::filesystem::path(argv[1]);
        if (poseCsvPath.is_relative()) {
            poseCsvPath = projectRoot / poseCsvPath;
        }
    }
    if (argc >= 3) {
        outputDir = std::filesystem::path(argv[2]);
        if (outputDir.is_relative()) {
            outputDir = projectRoot / outputDir;
        }
    }
    const std::filesystem::path outputVideoPath = outputDir / "extension_3d_visualization.mp4";
    const std::filesystem::path snapshot31Path = outputDir / "extension_3_1_camera_world_model.jpg";
    const std::filesystem::path snapshot32Path = outputDir / "extension_3_2_last_n_track.jpg";

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

    const std::vector<PoseSample> samples = loadPoseSamples(poseCsvPath);
    if (samples.empty()) {
        std::cerr << "No pose samples loaded from: " << poseCsvPath.string() << std::endl;
        return 1;
    }

    std::filesystem::create_directories(outputDir);

    SceneProjector projector = createProjectorForScene(samples, boxCorners);
    const int trajectoryWindow = 60;
    const double fps = 30.0;
    cv::VideoWriter writer(outputVideoPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(projector.width, projector.height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << outputVideoPath.string() << std::endl;
        return 1;
    }

    for (size_t i = 0; i < samples.size(); ++i) {
        cv::Mat frame = renderFrame(samples, i, boxCorners, projector, trajectoryWindow);
        writer.write(frame);

        if (i == samples.size() / 4) {
            cv::imwrite(snapshot31Path.string(), frame);
        }
        if (i + 1 == samples.size()) {
            cv::imwrite(snapshot32Path.string(), frame);
        }

        cv::imshow("Extension 3D Visualization", frame);
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    writer.release();
    cv::destroyAllWindows();

    std::cout << "Loaded pose samples: " << samples.size() << std::endl;
    std::cout << "Auto scene center: [" << projector.sceneCenter.x << ", "
              << projector.sceneCenter.y << ", " << projector.sceneCenter.z
              << "], scale: " << projector.sceneScale << std::endl;
    std::cout << "Saved extension 3D video to: " << outputVideoPath.string() << std::endl;
    std::cout << "Saved 3.1 snapshot to: " << snapshot31Path.string() << std::endl;
    std::cout << "Saved 3.2 snapshot to: " << snapshot32Path.string() << std::endl;
    return 0;
}
