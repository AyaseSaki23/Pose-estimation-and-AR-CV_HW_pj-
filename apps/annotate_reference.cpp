#include <opencv2/opencv.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct ClickState {
    cv::Mat baseImage;
    cv::Mat image;
    std::array<std::optional<cv::Point2f>, 8> points;
    int currentCorner = 0;
};

struct ReferenceAnnotation {
    std::string imageName;
    std::vector<std::pair<int, cv::Point2f>> points;
};

void redraw(ClickState& state) {
    state.image = state.baseImage.clone();
    for (size_t i = 0; i < state.points.size(); ++i) {
        if (!state.points[i].has_value()) {
            continue;
        }
        const cv::Point2f point = state.points[i].value();
        cv::circle(state.image, point, 5, cv::Scalar(0, 255, 255), -1);
        cv::putText(state.image, std::to_string(i + 1), point + cv::Point2f(8.0f, -8.0f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
    }

    cv::putText(state.image, "Selected corner: " + std::to_string(state.currentCorner + 1),
                cv::Point(30, 45), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0, 255, 255), 2);
}

void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    auto* state = static_cast<ClickState*>(userdata);
    state->points[state->currentCorner] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    redraw(*state);
    cv::imshow("Annotate Reference", state->image);
}

std::vector<ReferenceAnnotation> loadExistingAnnotations(const std::filesystem::path& path) {
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
            cv::FileNode points = refNode["points"];
            for (const cv::FileNode& pointNode : points) {
                const int id = static_cast<int>(pointNode["id"]);
                const float x = static_cast<float>(pointNode["x"]);
                const float y = static_cast<float>(pointNode["y"]);
                annotation.points.emplace_back(id, cv::Point2f(x, y));
            }
            if (!annotation.imageName.empty()) {
                annotations.push_back(annotation);
            }
        }
        return annotations;
    }

    ReferenceAnnotation legacy;
    fs["reference_image"] >> legacy.imageName;
    cv::FileNode points = fs["points"];
    for (const cv::FileNode& pointNode : points) {
        const int id = static_cast<int>(pointNode["id"]);
        const float x = static_cast<float>(pointNode["x"]);
        const float y = static_cast<float>(pointNode["y"]);
        legacy.points.emplace_back(id, cv::Point2f(x, y));
    }
    if (!legacy.imageName.empty()) {
        annotations.push_back(legacy);
    }

    return annotations;
}

void saveAnnotations(const std::filesystem::path& path,
                     std::vector<ReferenceAnnotation>& annotations,
                     const ReferenceAnnotation& updated) {
    bool replaced = false;
    for (ReferenceAnnotation& annotation : annotations) {
        if (annotation.imageName == updated.imageName) {
            annotation = updated;
            replaced = true;
            break;
        }
    }

    if (!replaced) {
        annotations.push_back(updated);
    }

    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    fs << "references" << "[";
    for (const ReferenceAnnotation& annotation : annotations) {
        fs << "{"
           << "image" << annotation.imageName
           << "points" << "[";
        for (const auto& [id, point] : annotation.points) {
            fs << "{"
               << "id" << id
               << "x" << point.x
               << "y" << point.y
               << "}";
        }
        fs << "]"
           << "}";
    }
    fs << "]";
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path referenceDir = projectRoot / "data/objects/object_01/reference";
    const std::filesystem::path outputPath = referenceDir / "reference_keypoints.yml";

    std::string imageName = "ref_oblique_01.jpg";
    if (argc >= 2) {
        imageName = argv[1];
    }

    const std::filesystem::path imagePath = referenceDir / imageName;
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to open reference image: " << imagePath.string() << std::endl;
        return 1;
    }

    std::cout << "Annotating reference image: " << imageName << std::endl;
    std::cout << "Keys: 1-8 select corner, c clear selected, Enter save, Esc quit" << std::endl;

    ClickState state;
    state.baseImage = image.clone();
    redraw(state);

    cv::namedWindow("Annotate Reference", cv::WINDOW_NORMAL);
    cv::resizeWindow("Annotate Reference", 960, 720);
    cv::setMouseCallback("Annotate Reference", onMouse, &state);
    cv::imshow("Annotate Reference", state.image);

    while (true) {
        const int key = cv::waitKey(20);
        if (key == 27) {
            return 0;
        }
        if (key >= '1' && key <= '8') {
            state.currentCorner = key - '1';
            redraw(state);
            cv::imshow("Annotate Reference", state.image);
        }
        if (key == 'c' || key == 'C') {
            state.points[state.currentCorner].reset();
            redraw(state);
            cv::imshow("Annotate Reference", state.image);
        }
        if (key == 13 || key == 10) {
            break;
        }
    }

    int count = 0;
    for (const auto& point : state.points) {
        if (point.has_value()) {
            ++count;
        }
    }

    if (count < 4) {
        std::cerr << "Need at least 4 annotated corners, got " << count << std::endl;
        return 1;
    }

    ReferenceAnnotation updated;
    updated.imageName = imageName;
    for (size_t i = 0; i < state.points.size(); ++i) {
        if (!state.points[i].has_value()) {
            continue;
        }
        updated.points.emplace_back(static_cast<int>(i + 1), state.points[i].value());
    }

    std::vector<ReferenceAnnotation> annotations = loadExistingAnnotations(outputPath);
    saveAnnotations(outputPath, annotations, updated);

    const std::filesystem::path previewPath = referenceDir / ("reference_keypoints_preview_" + imageName);
    cv::imwrite(previewPath.string(), state.image);

    std::cout << "Saved reference keypoints to: " << outputPath.string() << std::endl;
    std::cout << "Saved preview to: " << previewPath.string() << std::endl;
    return 0;
}
