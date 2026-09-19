#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace cvproject {

struct Mesh {
    std::vector<cv::Point3f> vertices;
    std::vector<std::vector<int>> faces;
};

struct BoundingBox {
    cv::Point3f min;
    cv::Point3f max;
    std::vector<cv::Point3f> corners;
};

Mesh loadObjMesh(const std::filesystem::path& path);
BoundingBox computeBoundingBox(const Mesh& mesh, float scale);
float inferObjToMillimeterScale(const Mesh& mesh, float widthMm, float heightMm, float depthMm);
void printMeshSummary(const Mesh& mesh, const BoundingBox& bbox);

}  // namespace cvproject
