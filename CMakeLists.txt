cmake_minimum_required(VERSION 3.10)

# Keep the vcpkg toolchain before project().
if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" VCPKG_ROOT_CMAKE_PATH)
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT_CMAKE_PATH}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else()
    message(WARNING "VCPKG_ROOT is not set. Pass -DCMAKE_TOOLCHAIN_FILE=... if your dependencies are installed through vcpkg.")
endif()

project(CVProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

foreach(CVPROJECT_CONFIG TIMING PROFILING)
    foreach(CVPROJECT_LINKER_KIND EXE SHARED MODULE STATIC)
        set(CMAKE_${CVPROJECT_LINKER_KIND}_LINKER_FLAGS_${CVPROJECT_CONFIG}
            "" CACHE STRING "Linker flags for ${CVPROJECT_CONFIG} builds")
    endforeach()
endforeach()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

if(EXISTS "${PROJECT_SOURCE_DIR}/gtsam/CMakeLists.txt")
    set(GTSAM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(GTSAM_INSTALL_CYTHON_TOOLBOX OFF CACHE BOOL "" FORCE)
    set(GTSAM_USE_BOOST_FEATURES OFF CACHE BOOL "" FORCE)
    set(GTSAM_ENABLE_BOOST_SERIALIZATION OFF CACHE BOOL "" FORCE)
    set(GTSAM_WITH_TBB OFF CACHE BOOL "" FORCE)
    set(GTSAM_COMPILE_OPTIONS_PRIVATE_COMMON "/W3;/GR;/EHsc;/MP;/utf-8" CACHE STRING "" FORCE)
    add_subdirectory(gtsam)
else()
    find_package(GTSAM REQUIRED)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(pose_estimation
    apps/main.cpp
    src/mesh_loader.cpp
    src/pose_utils.cpp
    src/reference_db.cpp
    src/viewpoint_selector.cpp
)
add_executable(calibrate_camera apps/calibrate_camera.cpp)
add_executable(match_reference apps/match_reference.cpp)
add_executable(annotate_reference apps/annotate_reference.cpp)
add_executable(validate_reference_annotations apps/validate_reference_annotations.cpp src/mesh_loader.cpp)
add_executable(track_2d_reference apps/track_2d_reference.cpp src/mesh_loader.cpp)
add_executable(track_2d_reference_diagnostic apps/track_2d_reference_diagnostic.cpp)
add_executable(track_2d_lk_points apps/track_2d_lk_points.cpp src/mesh_loader.cpp)
add_executable(visualize_extension_3d apps/visualize_extension_3d.cpp src/mesh_loader.cpp)
add_executable(pose_estimation_rotation
    apps/pose_estimation_rotation.cpp
    src/mesh_loader.cpp
    src/pose_utils.cpp
    src/reference_db.cpp
)
add_executable(pose_estimation_gen6d_like
    apps/pose_estimation_gen6d_like.cpp
    src/gen6d_pipeline.cpp
    src/object_localizer.cpp
    src/view_retriever.cpp
    src/correspondence_estimator.cpp
    src/pose_refiner.cpp
    src/pose_tracker_state.cpp
    src/reference_db.cpp
    src/pose_utils.cpp
    src/mesh_loader.cpp
)

target_compile_definitions(pose_estimation PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(calibrate_camera PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(match_reference PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(annotate_reference PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(validate_reference_annotations PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(track_2d_reference PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(track_2d_reference_diagnostic PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(track_2d_lk_points PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(visualize_extension_3d PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(pose_estimation_rotation PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(pose_estimation_gen6d_like PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(pose_estimation ${OpenCV_LIBS} gtsam)
target_link_libraries(calibrate_camera ${OpenCV_LIBS})
target_link_libraries(match_reference ${OpenCV_LIBS})
target_link_libraries(annotate_reference ${OpenCV_LIBS})
target_link_libraries(validate_reference_annotations ${OpenCV_LIBS})
target_link_libraries(track_2d_reference ${OpenCV_LIBS})
target_link_libraries(track_2d_reference_diagnostic ${OpenCV_LIBS})
target_link_libraries(track_2d_lk_points ${OpenCV_LIBS})
target_link_libraries(visualize_extension_3d ${OpenCV_LIBS})
target_link_libraries(pose_estimation_rotation ${OpenCV_LIBS})
target_link_libraries(pose_estimation_gen6d_like ${OpenCV_LIBS})
add_executable(prepare_reference_poses apps/prepare_reference_poses.cpp src/mesh_loader.cpp src/pose_utils.cpp src/reference_db.cpp)

target_compile_definitions(prepare_reference_poses PRIVATE
    CVPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(prepare_reference_poses ${OpenCV_LIBS})

if(TARGET gtsam)
    add_custom_command(TARGET pose_estimation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gtsam>
            $<TARGET_FILE_DIR:pose_estimation>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE_DIR:gtsam>/cephes-gtsam$<$<CONFIG:Debug>:Debug>.dll
            $<TARGET_FILE_DIR:pose_estimation>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE_DIR:gtsam>/metis-gtsam$<$<CONFIG:Debug>:Debug>.dll
            $<TARGET_FILE_DIR:pose_estimation>
    )
endif()
