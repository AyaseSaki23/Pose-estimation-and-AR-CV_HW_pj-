# 项目算法思路整合

本文档从算法流程角度整合项目的核心思路，说明各模块为什么存在、如何衔接，以及整体方法如何从基础 PnP 跟踪逐步扩展到 Gen6D-like 流程。

## 1. 核心问题

项目目标是从视频中估计一个已知三维物体相对于相机的 6D 位姿，并将三维包围盒和坐标轴投影回图像，实现 AR 可视化。

6D 位姿包含：

```text
旋转 rvec
平移 tvec
```

对于已知三维物体，位姿估计可以转化为一个核心问题：

```text
如何在当前图像中找到物体 3D 点对应的 2D 图像点？
```

只要建立了足够可靠的 3D-2D correspondence，就可以使用 PnP 求解物体相对于相机的位姿。

整体思路可以概括为：

```text
相机标定
-> 读取物体 3D 模型
-> 建立 3D-2D 对应关系
-> PnP 求解位姿
-> AR 投影可视化
-> 后续帧跟踪、重定位和平滑
```

## 2. 相机标定

相机标定是所有位姿估计和 AR 投影的基础。

输入：

```text
棋盘格标定视频
```

输出：

```text
相机内参矩阵 K
畸变参数 distCoeffs
```

算法步骤：

1. 从标定视频中抽取若干帧。
2. 检测棋盘格内角点。
3. 使用 `cornerSubPix` 优化角点到亚像素精度。
4. 调用 OpenCV `calibrateCamera`。
5. 保存相机内参和畸变参数。

作用：

- PnP 需要相机内参。
- `projectPoints` 需要相机内参。
- AR 包围盒和坐标轴投影需要相机内参。

当前标定结果重投影误差约为 `0.1964 px`，说明标定质量较稳定。

## 3. 物体模型与 3D 角点

目标物体近似为长方体盒子，因此项目将物体表示为一个带真实尺寸的三维包围盒。

输入：

```text
box.obj
object_01.yml
```

其中 `object_01.yml` 记录物体真实尺寸：

```text
width_mm: 85
height_mm: 113
depth_mm: 63
```

算法步骤：

1. 读取 `.obj` mesh。
2. 计算 mesh 的 bounding box。
3. 根据真实尺寸推断模型缩放比例。
4. 得到以毫米为单位的 8 个 3D 角点。

这 8 个角点是后续所有 PnP、人工标注、reference 标注和 AR 绘制的统一几何基础。

## 4. 首帧人工初始化

第一帧没有先验位姿，也不知道物体在图像中的具体位置，因此项目使用人工标注完成初始化。

输入：

```text
第一帧图像
物体 8 个 3D 角点
人工点击的 2D 角点
相机内参
```

算法步骤：

1. 用户在第一帧中点击可见角点。
2. 系统保存角点编号和图像坐标。
3. 将物体 3D 角点与图像 2D 点配对。
4. 调用 PnP 求解初始 `rvec/tvec`。
5. 将三维包围盒和坐标轴投影回图像，检查初始化是否合理。

人工初始化的意义：

- 提供第一帧可靠位姿。
- 为后续光流跟踪和 GTSAM 平滑提供初始状态。
- 避免一开始就依赖不稳定的全图自动检测。

## 5. PnP 位姿求解

PnP 是项目中反复复用的核心模块。

输入：

```text
3D object points
2D image points
camera matrix
distortion coefficients
```

输出：

```text
rvec
tvec
```

数学意义：

PnP 要找到一个旋转和平移，使得三维点经过相机投影后尽量接近观测到的二维点。

项目实现策略：

1. 检查 3D 点和 2D 点数量是否一致。
2. 至少需要 4 组 correspondence。
3. 点数较多时使用 `SOLVEPNP_ITERATIVE`。
4. 点数较少时使用 `SOLVEPNP_EPNP`。
5. 求解后使用 `solvePnPRefineLM` 做 LM 优化。
6. 计算平均重投影误差作为质量评价指标。

重投影误差的作用：

- 判断 PnP 是否可靠。
- 判断 reference 重定位是否可接受。
- 判断 Gen6D-like 结果是否有效。
- 用于报告和结果统计。

## 6. 基础视频跟踪：LK 光流 + PnP

首帧初始化完成后，后续帧不能每帧人工标注，因此项目使用 LK 光流跟踪角点。

核心思路：

```text
上一帧角点位置
-> LK 光流跟踪到当前帧
-> 当前帧 2D 角点
-> 与固定 3D 角点配对
-> PnP 求当前帧位姿
```

算法步骤：

1. 选取首帧中稳定可见的角点。
2. 使用 `calcOpticalFlowPyrLK` 跟踪角点到下一帧。
3. 做前向-反向一致性检查，过滤漂移点。
4. 根据当前 2D 点和对应 3D 点调用 PnP。
5. 计算重投影误差。
6. 绘制 AR 包围盒和坐标轴。

适用场景：

- 物体主要发生平移。
- 正面角点持续可见。
- 运动幅度较平滑。

局限：

- 物体大幅旋转时，原来的正面角点会离开视野。
- 遮挡或模糊会导致角点漂移。
- 光流只依赖局部连续性，无法从严重丢失中自动恢复。

## 7. Reference Relocalization

为了解决光流跟丢的问题，项目加入 reference 图像重定位。

核心思路：

```text
当前帧
-> 与 reference 图像匹配
-> 估计 homography
-> 将 reference 角点转移到当前帧
-> 重新 PnP
```

输入：

```text
当前视频帧
reference 图像数据库
reference_keypoints.yml
物体 3D 角点
相机内参
```

算法步骤：

1. 加载多张 reference 图像。
2. 对当前帧和 reference 图像提取 ORB 特征。
3. 使用描述子匹配和 ratio test 筛选匹配点。
4. 使用 RANSAC homography 过滤错误匹配。
5. 将 reference 图像上的人工角点通过 homography 转移到当前帧。
6. 根据转移后的 2D 点和物体 3D 点调用 PnP。
7. 检查重投影误差和位姿跳变。

意义：

- 当光流失败时，可以重新建立 3D-2D correspondence。
- 系统具备一定自动恢复能力。

局限：

- ORB 依赖局部纹理。
- 低纹理侧面可能匹配点不足。
- reference 选错时，homography 仍可能产生错误角点。

## 8. GTSAM 位姿平滑

逐帧 PnP 结果可能会抖动，因此项目使用 GTSAM 对连续帧位姿进行平滑。

核心思路：

```text
PnP 结果作为测量约束
相邻帧位姿变化作为平滑约束
在滑动窗口内优化位姿
```

图优化结构：

- 每一帧位姿是一个变量。
- PnP 估计结果作为 `PriorFactor`。
- 相邻帧之间加入 `BetweenFactor`。
- 使用 robust noise 减小异常测量的影响。
- 使用 Levenberg-Marquardt 优化。

作用：

- 减少 AR 包围盒抖动。
- 让相邻帧位姿变化更加连续。
- 提高视觉展示稳定性。

注意：

GTSAM 不负责检测物体，也不负责建立对应点。它只在已经有每帧位姿测量的基础上做平滑优化。

## 9. 旋转场景 baseline

基础 LK 光流方法在旋转场景中会失败，因为原来可见的正面角点可能离开视野。

为此，项目尝试了多 reference 旋转 baseline。

核心思路：

```text
准备多个视角 reference
-> 当前帧与所有 reference 匹配
-> 选择最佳 reference
-> 转移角点
-> PnP 求位姿
```

优点：

- 不再只依赖正面角点。
- 可以处理一定程度的视角变化。

问题：

- 低纹理侧面 ORB 特征较少。
- reference 可能选错。
- 单面四点 PnP 约束较弱。
- 角点一旦转移错误，位姿会明显跳变或飞框。

这个 baseline 的主要作用是暴露问题，并引出后续 Gen6D-like 改进。

## 10. Gen6D-like 改进流程

Gen6D-like 是项目后期的主要改进方案。

它不是完整复现深度学习版 Gen6D，而是借鉴 Gen6D 的模块化框架，用传统计算机视觉方法实现类似处理流程。

整体结构：

```text
Object Localizer
-> View Retriever
-> Correspondence Estimator
-> PnP Initial Pose
-> Pose Refiner
```

### 10.1 Object Localizer

作用：

确定当前帧中物体的大致搜索区域。

策略：

- 第一帧没有位姿先验，使用 full-frame fallback。
- 后续帧使用上一帧 pose 将 3D 包围盒投影到图像。
- 根据投影框生成 ROI。

意义：

- 避免每帧全图匹配。
- 减少背景特征干扰。
- 提高 reference 检索稳定性。

### 10.2 View Retriever

作用：

从 reference 数据库中选择当前帧最相似的视角。

策略：

- 在 ROI 中提取 ORB 特征。
- 与各 reference 图像匹配。
- 使用 RANSAC homography 做几何验证。
- 根据 good matches、inliers 和 inlier ratio 打分。
- 选出 top reference。

对应 Gen6D 思想：

这一模块类似 Gen6D 中的 selector，只是项目中使用传统 ORB 特征和几何验证代替 learned descriptor。

### 10.3 Correspondence Estimator

作用：

显式构造当前帧的 2D-3D correspondence。

策略：

- 使用 top reference 的 homography。
- 将 reference 图像中的角点转移到当前帧。
- 将转移得到的 2D 点与物体 3D 角点配对。

输出：

```text
objectPoints
imagePoints
```

这些点直接作为 PnP 输入。

### 10.4 PnP Initial Pose

作用：

根据 correspondence 求初始位姿。

策略：

- 调用统一的 `solvePosePnP`。
- 计算平均重投影误差。
- 如果误差过大，则认为该帧 pose 无效。

### 10.5 Pose Refiner

作用：

在 PnP 初始位姿附近进一步细化，使投影框更贴合图像边缘。

策略：

1. 对当前帧做 Canny 边缘检测。
2. 计算 distance transform。
3. 在初始 pose 附近做局部搜索。
4. 选择投影边缘到图像边缘距离更小的 pose。

意义：

- 缓解 homography 角点转移的小偏差。
- 让 3D 包围盒与真实物体边缘更贴合。
- 提高旋转场景稳定性。

## 11. 三维轨迹可视化扩展

除了在视频中绘制 AR 包围盒，项目还将位姿结果转换到三维空间中展示。

输入：

```text
poses.csv
poses_rotation.csv
poses_gen6d_like.csv
```

算法步骤：

1. 读取每帧 `rvec/tvec`。
2. 将旋转向量转换为旋转矩阵。
3. 根据物体到相机的位姿计算相机中心。
4. 绘制物体坐标系。
5. 绘制相机视锥。
6. 绘制完整相机轨迹。
7. 绘制最近 n 帧轨迹。

作用：

- 从三维角度观察相机运动。
- 判断位姿是否平滑。
- 观察是否存在跳变。

## 12. 方法演进关系

整个项目的方法演进可以概括为：

```text
相机标定
-> 首帧人工标注
-> PnP 初始位姿
-> LK 光流跟踪
-> Reference Relocalization
-> GTSAM 平滑
-> 旋转场景 baseline
-> Gen6D-like 改进
-> 三维轨迹可视化
```

每一步解决的问题如下：

```text
相机标定：获得投影模型
物体模型：获得 3D 几何点
人工标注：解决首帧初始化
PnP：根据 3D-2D 对应点求位姿
LK 光流：解决连续平移场景跟踪
Reference 重定位：解决光流跟丢后的恢复
GTSAM：解决逐帧位姿抖动
旋转 baseline：尝试处理多视角旋转
Gen6D-like：提升旋转场景下的稳定性
3D 可视化：展示相机轨迹和位姿连续性
```

## 13. 总结

项目的核心不是单一算法，而是一条围绕 3D-2D correspondence 构建的完整位姿估计链路。

基础流程依赖人工首帧标注、LK 光流和 PnP，适合平移场景；reference relocalization 增加了跟踪丢失后的恢复能力；GTSAM 提升了位姿连续性；旋转 baseline 暴露了低纹理、多视角匹配和单面 PnP 的问题；Gen6D-like 流程进一步通过 ROI、视角检索、显式 correspondence 和边缘 refine，提高了旋转场景下的稳定性。

