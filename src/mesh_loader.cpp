#include "cvproject/mesh_loader.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cvproject {

namespace {

int parseObjVertexIndex(const std::string& token) {
    const size_t slashPos = token.find('/');
    const std::string indexText = slashPos == std::string::npos ? token : token.substr(0, slashPos);
    return std::stoi(indexText) - 1;
}

}  // namespace

Mesh loadObjMesh(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open OBJ mesh: " + path.string());
    }

    Mesh mesh;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream stream(line);
        std::string tag;
        stream >> tag;

        if (tag == "v") {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            stream >> x >> y >> z;
            mesh.vertices.emplace_back(x, y, z);
        } else if (tag == "f") {
            std::vector<int> face;
            std::string token;
            while (stream >> token) {
                face.push_back(parseObjVertexIndex(token));
            }

            if (face.size() >= 3) {
                mesh.faces.push_back(face);
            }
        }
    }

    if (mesh.vertices.empty()) {
        throw std::runtime_error("OBJ mesh has no vertices: " + path.string());
    }

    return mesh;
}

BoundingBox computeBoundingBox(const Mesh& mesh, float scale) {
    BoundingBox bbox;
    bbox.min = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    bbox.max = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };

    for (const cv::Point3f& vertex : mesh.vertices) {
        const cv::Point3f scaled(vertex.x * scale, vertex.y * scale, vertex.z * scale);
        bbox.min.x = std::min(bbox.min.x, scaled.x);
        bbox.min.y = std::min(bbox.min.y, scaled.y);
        bbox.min.z = std::min(bbox.min.z, scaled.z);
        bbox.max.x = std::max(bbox.max.x, scaled.x);
        bbox.max.y = std::max(bbox.max.y, scaled.y);
        bbox.max.z = std::max(bbox.max.z, scaled.z);
    }

    const float minX = bbox.min.x;
    const float minY = bbox.min.y;
    const float minZ = bbox.min.z;
    const float maxX = bbox.max.x;
    const float maxY = bbox.max.y;
    const float maxZ = bbox.max.z;

    bbox.corners = {
        {minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ},
        {minX, minY, maxZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}
    };

    return bbox;
}

float inferObjToMillimeterScale(const Mesh& mesh, float widthMm, float heightMm, float depthMm) {
    const BoundingBox raw = computeBoundingBox(mesh, 1.0f);
    std::vector<float> rawExtents = {
        raw.max.x - raw.min.x,
        raw.max.y - raw.min.y,
        raw.max.z - raw.min.z
    };
    std::vector<float> targetExtents = {widthMm, heightMm, depthMm};

    std::sort(rawExtents.begin(), rawExtents.end());
    std::sort(targetExtents.begin(), targetExtents.end());

    if (rawExtents.front() <= 0.0f) {
        return 1.0f;
    }

    float scale = 0.0f;
    int valid = 0;
    for (size_t i = 0; i < rawExtents.size(); ++i) {
        if (rawExtents[i] <= 0.0f) {
            continue;
        }
        scale += targetExtents[i] / rawExtents[i];
        ++valid;
    }

    return valid > 0 ? scale / static_cast<float>(valid) : 1.0f;
}

void printMeshSummary(const Mesh& mesh, const BoundingBox& bbox) {
    std::cout << "Mesh vertices: " << mesh.vertices.size() << std::endl;
    std::cout << "Mesh faces: " << mesh.faces.size() << std::endl;
    std::cout << "Bounding box min: [" << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << "]" << std::endl;
    std::cout << "Bounding box max: [" << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << "]" << std::endl;
    std::cout << "Bounding box size: ["
              << bbox.max.x - bbox.min.x << ", "
              << bbox.max.y - bbox.min.y << ", "
              << bbox.max.z - bbox.min.z << "] mm" << std::endl;
}

}  // namespace cvproject
