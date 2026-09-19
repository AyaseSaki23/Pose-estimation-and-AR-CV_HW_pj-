#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <vector>

namespace {

std::vector<cv::Point3f> createObjectPoints(const cv::Size& boardSize, float squareSize) {
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<size_t>(boardSize.width * boardSize.height));

    for (int y = 0; y < boardSize.height; ++y) {
        for (int x = 0; x < boardSize.width; ++x) {
            points.emplace_back(x * squareSize, y * squareSize, 0.0f);
        }
    }

    return points;
}

}  // namespace

int main() {
    const std::filesystem::path projectRoot = CVPROJECT_ROOT_DIR;
    const std::filesystem::path videoPath = projectRoot / "data/calibration/calib_video.mp4";
    const std::filesystem::path outputPath = projectRoot / "config/camera_calib.yml";
    const std::filesystem::path figureDir = projectRoot / "docs/figures";
    const std::filesystem::path samplePath = figureDir / "calibration_corners_sample.jpg";

    const cv::Size boardSize(11, 8);
    const float squareSizeMm = 20.0f;
    const int maxValidFrames = 60;

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open calibration video: " << videoPath.string() << std::endl;
        return 1;
    }

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int frameStep = std::max(1, static_cast<int>(std::round(fps)));

    std::cout << "Calibration video: " << videoPath.string() << std::endl;
    std::cout << "FPS: " << fps << ", frames: " << frameCount
              << ", sampling every " << frameStep << " frames" << std::endl;
    std::cout << "Board inner corners: " << boardSize.width << " x " << boardSize.height
              << ", square size: " << squareSizeMm << " mm" << std::endl;

    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    const std::vector<cv::Point3f> singleBoardPoints = createObjectPoints(boardSize, squareSizeMm);

    cv::Mat frame;
    cv::Mat gray;
    cv::Size imageSize;
    bool savedSample = false;

    for (int frameIndex = 0; cap.read(frame); ++frameIndex) {
        if (frameIndex % frameStep != 0) {
            continue;
        }

        imageSize = frame.size();
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        const bool found = cv::findChessboardCorners(
            gray,
            boardSize,
            corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK
        );

        if (!found) {
            continue;
        }

        cv::cornerSubPix(
            gray,
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001)
        );

        imagePoints.push_back(corners);
        objectPoints.push_back(singleBoardPoints);

        std::cout << "Accepted frame " << frameIndex
                  << " (" << imagePoints.size() << "/" << maxValidFrames << ")" << std::endl;

        if (!savedSample) {
            std::filesystem::create_directories(figureDir);
            cv::Mat preview = frame.clone();
            cv::drawChessboardCorners(preview, boardSize, corners, found);
            cv::imwrite(samplePath.string(), preview);
            savedSample = true;
        }

        if (static_cast<int>(imagePoints.size()) >= maxValidFrames) {
            break;
        }
    }

    if (imagePoints.size() < 10) {
        std::cerr << "Only " << imagePoints.size()
                  << " valid chessboard frames were found. Need at least 10." << std::endl;
        std::cerr << "Check board size, focus, motion blur, and whether the full board is visible." << std::endl;
        return 2;
    }

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    const double rms = cv::calibrateCamera(
        objectPoints,
        imagePoints,
        imageSize,
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs
    );

    std::filesystem::create_directories(outputPath.parent_path());
    cv::FileStorage fs(outputPath.string(), cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size_mm" << squareSizeMm;
    fs << "valid_frame_count" << static_cast<int>(imagePoints.size());
    fs << "reprojection_error" << rms;
    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs.release();

    std::cout << "Calibration finished." << std::endl;
    std::cout << "Valid frames: " << imagePoints.size() << std::endl;
    std::cout << "RMS reprojection error: " << rms << std::endl;
    std::cout << "Camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients:\n" << distCoeffs << std::endl;
    std::cout << "Saved calibration to: " << outputPath.string() << std::endl;
    std::cout << "Saved corner sample to: " << samplePath.string() << std::endl;

    return 0;
}
