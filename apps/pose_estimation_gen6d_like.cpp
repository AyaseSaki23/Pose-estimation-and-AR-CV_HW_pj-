#include "cvproject/gen6d_pipeline.hpp"

#include <filesystem>
#include <iostream>

#ifndef CVPROJECT_ROOT_DIR
#define CVPROJECT_ROOT_DIR "."
#endif

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;

    std::cout << "[Gen6D-like] pose estimation pipeline scaffold\n";

    cvproject::Gen6DPipeline pipeline(projectRoot);
    if (!pipeline.initialize()) {
        std::cerr << "[Gen6D-like] initialization failed\n";
        return 1;
    }

    return pipeline.run();
}
