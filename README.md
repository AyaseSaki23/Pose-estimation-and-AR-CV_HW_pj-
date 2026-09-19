# 6D Object Pose Estimation and Augmented Reality

这是一个基于 C++17、OpenCV 和 GTSAM 的 6D 物体位姿估计项目。输入一段视频和一个已知尺寸的三维盒子模型，系统估计物体相对相机的旋转和平移，再把三维包围盒、坐标轴和相机轨迹叠加回画面里。

这个项目最初的思路比较直接：先做相机标定，首帧人工点几个角点，用 PnP 求出初始位姿，然后用 LK 光流继续跟。真正开始做之后发现，物体只要明显旋转，原本能看到的正面角点就会离开画面，盒子也很容易飞出去。于是后面补了 reference relocalization、GTSAM 平滑，最后把旋转场景的处理重新整理成一个 Gen6D-like 的模块化流程。

> 一句话：输入已知 3D box 和视频，输出 `rvec/tvec`、AR 结果视频，以及三维相机轨迹。

## 它做了什么

```text
相机标定
-> 读取 3D 模型并生成 8 个 box corner
-> 首帧人工标注建立 3D-2D correspondence
-> PnP 求解初始位姿
-> LK 光流跟踪
-> Reference relocalization
-> GTSAM 位姿平滑
-> Gen6D-like 旋转鲁棒流程
-> 三维相机轨迹可视化
```

需要说明的是，这里的 Gen6D-like 不是复现深度学习版 Gen6D。它借用了 Detector、Selector、Refiner 的模块化思想，但实际使用的是传统视觉方法：pose-projected ROI、ORB + RANSAC homography、显式 2D-3D correspondence、PnP，以及 Canny + distance transform 的边缘细化。

## 结果概览

在 1235 帧输入上统计到的主要结果：

| 指标 | 数值 |
| --- | ---: |
| 处理帧数 | 1235 |
| 有效 PnP 位姿 | 1235 / 1235 |
| 使用 pose-projected ROI 的帧数 | 1234 / 1235 |
| 平均重投影误差 | 2.33 px |
| 最大重投影误差 | 7.13 px |
| 平均 2D-3D 对应点数 | 5.49 |
| 平均 top-view inliers | 1443.05 |

扩展任务会把位姿结果放到三维空间里，展示完整相机轨迹和最近 120 帧轨迹。

## 目录结构

```text
apps/                        各个可执行程序入口
src/                         核心算法实现
include/cvproject/           头文件
config/                      相机和物体配置
scripts/                     辅助脚本
data/objects/object_01/      物体 mesh、reference 图和标注
```

## 构建和运行

```powershell
cmake -S . -B build
cmake --build build --config Debug
```

Debug 构建完成后，可执行文件通常位于 `build/Debug/`：

```powershell
.\build\Debug\pose_estimation.exe
.\build\Debug\pose_estimation_rotation.exe
.\build\Debug\pose_estimation_gen6d_like.exe
.\build\Debug\visualize_extension_3d.exe
```

两点需要提前知道：

- 如果依赖由 vcpkg 提供，构建前需要把 `VCPKG_ROOT` 设置为本机 vcpkg 根目录；也可以通过 `-DCMAKE_TOOLCHAIN_FILE=...` 显式指定 toolchain。
- 如果仓库中有 `gtsam/`，CMake 会把它作为子目录编译；否则需要提前安装 GTSAM。

## 数据约定

原始视频没有放在仓库里。运行前需要按代码里的路径准备好数据：

```text
data/calibration/calib_video.mp4
data/raw/videos/input.mp4
data/objects/object_01/...
```

运行结果默认写到 `data/processed/`。

## 已知限制

这个项目不是实时算法。Gen6D-like 每帧仍然包含 ORB 匹配、RANSAC、PnP、Canny、distance transform 和局部姿态搜索，整体更接近 offline 处理。

低纹理侧面仍然是最难处理的情况。ROI 和边缘细化能减少错误 reference 带来的飞框，但并不能保证每个旋转角度都稳定。

首帧初始化目前依赖人工标注，这是为了保证初始位姿足够可靠。
