# 第6章：传统感知到深度学习感知

## 章节概要

自动驾驶感知技术在2016-2020年间经历了从传统计算机视觉到深度学习的范式转变。本章深入剖析这一转变过程中的关键技术演进、架构创新和工程实践，重点关注2D到3D感知的跃迁、多任务学习架构的兴起，以及深度估计技术如何弥合纯视觉与激光雷达的性能鸿沟。

## 6.1 从2D检测到3D感知

### 6.1.1 传统CV时代的局限性 (Pre-2016)

#### 经典方法回顾

传统计算机视觉方法在自动驾驶早期扮演了重要角色，但存在根本性局限：

```
传统CV Pipeline (2010-2015)
┌─────────────┐    ┌──────────────┐    ┌───────────┐
│  特征提取    │ -> │  特征描述     │ -> │  分类器    │
│  HOG/SIFT   │    │  BoW/Fisher  │    │  SVM/RF   │
└─────────────┘    └──────────────┘    └───────────┘
      ↓                    ↓                  ↓
  手工设计            维度诅咒           泛化能力差
```

**HOG (Histogram of Oriented Gradients) 行人检测**
- Dalal & Triggs (2005) 的经典方法
- 滑动窗口 + 梯度直方图
- 检测速度: ~1 FPS (640×480)
- 检测精度: ~80% AP on INRIA dataset
- 致命缺陷: 无法处理遮挡、形变、光照变化

实际工程实现中的细节:
```
HOG特征提取流程:
1. 梯度计算: Sobel算子获取Gx, Gy
2. Cell划分: 8×8像素为一个cell
3. 直方图: 9个bin统计梯度方向
4. Block归一化: 2×2 cells组成block, L2-norm
5. 特征向量: 15×7 blocks × 36维 = 3780维
```

**Haar Cascades 车辆检测**
- Viola-Jones框架的延伸
- 积分图加速计算
- 实时性好但精度低
- MobileEye早期EyeQ芯片的主要算法

Haar特征的局限性:
- 仅使用矩形特征，表达能力有限
- 对旋转和尺度变化敏感
- 级联分类器训练耗时(数周)
- 误检率高，需要大量后处理

**立体视觉深度估计**
- 基于SGBM (Semi-Global Block Matching)
- 计算复杂度O(WHD), D为视差搜索范围
- 对标定精度极度敏感
- 纹理缺失区域失效

SGBM算法核心:
```
能量函数: E(D) = ΣE_data(p,Dp) + ΣE_smooth(Dp,Dq)
其中:
- E_data: 匹配代价(Census/SAD)
- E_smooth: 平滑约束
- 8方向动态规划聚合
```

**光流法运动估计**

早期ADAS系统大量使用光流进行:
- 自身运动估计(Visual Odometry)
- 移动物体检测
- 碰撞时间(TTC)预测

Lucas-Kanade光流:
```
假设: 局部区域内光流恒定
求解: [Ix Iy][u v]' = -It
问题: 孔径问题、光照变化、大位移
```

实际应用案例:
- 2014 Mercedes S-Class: 立体视觉+光流
- 2015 Audi Q7: 单目视觉+SfM
- 早期Tesla AP1.0: MobileEye EyeQ3混合方案

#### MobileEye EyeQ时代 (2014-2017)

MobileEye在深度学习来临前的统治地位源于其精心设计的专用架构：

| EyeQ版本 | 年份 | 算力 | 核心算法 | 客户 |
|---------|------|------|----------|------|
| EyeQ2 | 2010 | 2.5 GOPS | Haar+HOG | BMW, GM |
| EyeQ3 | 2014 | 256 GOPS | 混合CNN | Tesla AP1 |
| EyeQ4 | 2018 | 2.5 TOPS | 深度CNN | 日产ProPilot |
| EyeQ5 | 2021 | 24 TOPS | 全CNN+REM | BMW, NIO |

**EyeQ3架构深度剖析**

EyeQ3的突破在于引入了早期CNN，但仍保留大量传统CV：

```
EyeQ3 处理流水线:
┌──────────────────────────────────────┐
│  Image Preprocessing                  │
│  - 去马赛克(Bayer->RGB)               │
│  - 降噪、HDR合成                      │
└────────────┬─────────────────────────┘
             ↓
     ┌───────┴────────┐
     ↓                ↓
┌────────────┐  ┌──────────────┐
│ 传统CV分支  │  │  CNN分支      │
│ - 车道线    │  │ - 车辆检测    │
│ - 路沿      │  │ - 行人检测    │  
│ - 标志牌    │  │ - 交通灯      │
└────────────┘  └──────────────┘
     ↓                ↓
     └───────┬────────┘
             ↓
      融合与跟踪模块
```

关键技术细节:
- 车道线: 基于Hough变换的多项式拟合
  - RANSAC鲁棒估计
  - 3次多项式: y = ax³ + bx² + cx + d
  - 时序平滑: 卡尔曼滤波
- 目标跟踪: Kalman滤波 + Hungarian匹配
  - 状态向量: [x, y, vx, vy, w, h]
  - 数据关联: IoU + 外观特征
- 空闲空间: 基于v-disparity的地面估计
  - Stixel World表示
  - 动态规划优化

**REM (Road Experience Management)**

MobileEye独特的众包地图策略:
```
车端采集 -> 特征提取 -> 上传云端 -> 地图构建
   ↓           ↓           ↓          ↓
摄像头图像  稀疏特征    10KB/km    HD Map
```

特点:
- 轻量级: 仅传输语义特征，不传图像
- 实时更新: 众包模式，百万级车辆
- 低成本: 无需专业采集车

### 6.1.2 深度学习2D检测革命 (2016-2018)

这一时期见证了深度学习彻底改变目标检测的过程，从学术突破到工程落地仅用了2年时间。

#### YOLO系列：实时检测的突破

```
YOLO演进时间线
2016.6  YOLOv1  45 FPS  63.4% mAP  端到端训练
   ↓
2016.12 YOLOv2  67 FPS  76.8% mAP  Anchor boxes
   ↓  
2018.4  YOLOv3  65 FPS  82.5% mAP  多尺度预测
   ↓
2020.4  YOLOv4  65 FPS  87.2% mAP  CSPNet backbone
```

**YOLOv1的革命性创新**

将检测转化为回归问题:
```
输入: 448×448图像
   ↓
划分: 7×7网格
   ↓  
输出: 每个网格预测:
      - 2个bbox: (x,y,w,h,confidence)
      - 20类概率 (PASCAL VOC)
   ↓
NMS后处理
```

局限性:
- 每个网格只能检测2个物体
- 小物体检测效果差
- 定位精度不如two-stage

**YOLOv2/v3关键改进**

YOLOv2 (YOLO9000):
- Anchor boxes: 使用k-means聚类得到5个anchors
- Batch Normalization: 每层后加BN, mAP提升2%
- Multi-scale training: 随机选择输入尺度
- Darknet-19: 更深的backbone

YOLOv3重大升级:
```
多尺度检测结构:
    Darknet-53
         ↓
    ┌────┬─────┐  
    ↓    ↓     ↓
  52×52 26×26 13×13  <- 三个尺度
    ↓    ↓     ↓
  小物体 中物体 大物体
```

**YOLOv3在自动驾驶中的应用**

*Tesla Autopilot 2.0 (2017)*:
- 8摄像头YOLOv3变体
- 每个摄像头独立处理
- 主前视高分辨率(1280×960)
- 周视低分辨率(640×480)
- HW2.5: NVIDIA Drive PX2

*百度Apollo 2.0 (2018)*:
- YOLOv3 + Deep SORT跟踪
- 融合激光雷达点云
- 关键改进: 
  - Darknet-53 backbone适配车载算力
  - FPN多尺度检测小目标
  - 9个anchor覆盖车辆尺度范围
  - 自定义类别: 车、人、自行车、交通锥

车载部署优化:
```python
# TensorRT优化
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.STRICT_TYPES)
# 量化感知训练(QAT)
# INT8推理，速度提升3×
```

#### Two-Stage方法的精度优势

Faster R-CNN系列在需要高精度的场景仍有优势：

```
Two-Stage Pipeline
┌────────┐    ┌─────────┐    ┌──────────┐
│ RPN    │ -> │ RoI Pool│ -> │ R-CNN    │
│ 候选区  │    │ 特征提取 │    │ 精细分类  │
└────────┘    └─────────┘    └──────────┘
    ↓              ↓              ↓
1000个候选     7×7特征图     类别+bbox
```

**Faster R-CNN核心创新**

RPN (Region Proposal Network):
```
共享卷积特征
      ↓
  3×3滑动窗口
      ↓
  9个anchors
  (3 scales × 3 ratios)
      ↓
  2分类 + 4回归
```

RoI Pooling问题:
- 量化误差大
- 小物体特征丢失

RoI Align解决方案:
- 双线性插值
- 保持空间对齐
- mAP提升~1%

**FPN (Feature Pyramid Networks)**

```
自底向上路径          自顶向下路径
                        P5 ←──┐
C5 (stride=32) ─────→       ↓
                        P4 ←──┼──┐ 
C4 (stride=16) ─────→       ↓   ↓
                        P3 ←──┼──┼─┐
C3 (stride=8)  ─────→       ↓   ↓ ↓
                        P2 ←──┘   ↓ ↓
C2 (stride=4)  ─────→           ↓ ↓
                              检测头
```

优势:
- 多尺度特征融合
- 小物体检测大幅提升
- 计算成本增加少

**Cascade R-CNN在L4系统的应用**

级联结构:
```
RPN -> H1(IoU=0.5) -> H2(IoU=0.6) -> H3(IoU=0.7)
         ↓              ↓              ↓
      粗检测          精细化         更精细
```

*Waymo (2018-2019)*:
- Cascade R-CNN处理稀疏激光雷达
- 点云体素化后输入CNN
- IoU阈值递增: 0.5 -> 0.6 -> 0.7
- 小目标检测提升: +3.2% AP on KITTI
- 边界框精度提升: +5% localization

*Cruise (2019)*:
- 多传感器Cascade R-CNN
- Camera + LiDAR early fusion
- 每个stage不同模态特征

### 6.1.3 从2D到3D的关键跨越 (2018-2020)

3D感知是自动驾驶的核心挑战，从2D到3D的跨越涉及深度估计、坐标转换、遮挡处理等多个难题。

#### 单目3D检测的探索

单目3D检测是计算机视觉的"圣杯"问题之一，需要从单张图像恢复完整的3D信息。

**核心挑战**
1. 尺度模糊性: 远处的大车vs近处的小车
2. 深度估计: 缺少立体基线
3. 遮挡问题: 部分可见物体的3D框
4. 姿态估计: 物体朝向的准确预测

**M3D-RPN (2019)**

创新的2D-3D锚点设计:
```
深度感知锚点生成
┌─────────────────────────────┐
│  2D中心 (u,v)                │
│     ↓ 相机内参K              │
│  3D射线方向                   │
│     ↓ 统计先验               │
│  3D中心 (X,Y,Z)              │
│     ↓ 尺寸先验               │
│  3D框 (l,w,h,θ)              │
└─────────────────────────────┘

关键约束:
1. 2D-3D一致性: 3D框投影=2D检测框
2. 深度有序性: 前后关系约束
3. 地面约束: 车辆贴地假设
```

深度推理策略:
- 利用物体类别的平均尺寸
- 根据2D框大小推断深度
- 公式: depth = (f × height_3d) / height_2d

性能:
- KITTI Easy: 20.27% AP3D
- KITTI Moderate: 17.06% AP3D  
- KITTI Hard: 15.21% AP3D

**SMOKE (2020) - 无需2D检测的3D预测**

革命性的单阶段设计:
```
架构流程:
Image → DLA-34 Backbone → Keypoint Heatmap
                        ↓
                   3D Centers (投影点)
                        ↓
                   3D Attributes
                   - depth
                   - dimensions (l,w,h)
                   - orientation
```

关键创新:
- 直接回归3D中心投影点(不是2D框中心!)
- 关键见解: 3D中心投影≠2D bbox中心
- 消除2D检测的误差累积
- 端到端可微分训练

损失函数设计:
```python
L_total = L_cls + λ₁L_reg + λ₂L_off
其中:
- L_cls: focal loss for keypoints
- L_reg: L1 loss for 3D attributes
- L_off: sub-pixel offset
```

性能指标:
- KITTI moderate: 14.03% AP3D
- 推理速度: 30ms on 1080Ti
- 内存占用: 仅需2GB GPU内存

**MonoDIS (2019)**

解耦的3D检测:
```
2D Detection → 2D-3D Lifting → 3D Refinement
     ↓              ↓                ↓
  2D boxes    Initial 3D      Refined 3D
              w/ uncertainty
```

不确定性建模:
- Aleatoric uncertainty: 数据噪声
- Epistemic uncertainty: 模型不确定性
- 用于后处理的置信度加权

#### 伪激光雷达方法

**Pseudo-LiDAR (2019)**

将深度图转换为点云表示的革命性思路：

```
Pipeline:
图像对 -> 深度估计 -> 3D点云 -> PointNet检测
        PSMNet     坐标变换    现成3D检测器

关键创新: 表示形式比深度精度更重要
性能提升: 单目3D检测 +100% AP3D
```

**核心洞察**

为什么表示形式如此重要？
```
前视图(图像空间)的问题:
┌────────────────┐
│ ▪▪▪▪▪▪▪▪▪▪▪▪  │ <- 远处物体压缩
│  ▪▪▪▪▪▪▪▪▪    │
│   ▪▪▪▪▪▪▪     │ <- 透视畸变
│    ██████      │ <- 近处物体
└────────────────┘

鸟瞰图(BEV空间)的优势:
┌────────────────┐
│ · · · · · · ·  │ <- 均匀分布
│ · · ■ · · · ·  │ <- 无畸变
│ · · · · ■ · ·  │ <- 保持相对位置
│ · · · · · · ·  │
└────────────────┘
```

**技术实现细节**

坐标变换公式:
```python
def image_to_lidar(depth, intrinsics, extrinsics):
    # 生成像素网格
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    # 反投影到相机坐标系
    z = depth
    x = (i - intrinsics[0,2]) * z / intrinsics[0,0]
    y = (j - intrinsics[1,2]) * z / intrinsics[1,1]
    
    # 相机坐标到激光雷达坐标
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    pts_lidar = pts_cam @ extrinsics.T
    
    return pts_lidar[:,:,:3]
```

**Pseudo-LiDAR++改进**

多模态深度增强:
```
立体图像 ────┐
            ├→ Depth CNN → 初始深度
稀疏LiDAR ────┘              ↓
                     Depth Completion
                            ↓
                      精细化深度图
                            ↓
                     伪激光雷达点云
```

关键技术:
1. **深度补全网络**
   - 4线激光雷达提供稀疏但准确的深度
   - CNN学习从稀疏到稠密的映射
   - 性能提升: +15% 3D AP

2. **Foreground Point Segmentation**
   - 使用2D检测mask过滤背景点
   - 减少前景/背景混淆
   - 边界准确度提升30%

3. **Cost Volume Filtering**
   ```
   立体匹配代价体 → 3D CNN滤波 → 置信度图
                                    ↓
                              加权点云生成
   ```

性能对比:
| 方法 | 输入 | 3D AP (Moderate) |
|-----|------|------------------|
| Pseudo-LiDAR | 单目 | 28.31% |
| Pseudo-LiDAR | 立体 | 42.43% |
| Pseudo-LiDAR++ | 立体+4线 | 51.75% |
| 真实64线LiDAR | LiDAR | 82.58% |

### 6.1.4 多视角3D感知 (2019-2020)

多视角3D感知标志着从单相机到环视系统的重要转变，为后续的BEV感知奠定基础。

#### LSS: BEV感知的先驱

**Lift-Splat-Shoot (2020)**

LSS开创性地提出了可微分的视角转换方法：

```
详细的三步转换过程:

1. Lift: 图像特征 -> 3D frustum特征
   ┌─────────────────────────────────┐
   │ 对每个像素(u,v):                 │
   │ - 提取特征向量 f(u,v)            │
   │ - 预测深度分布 D=[d₁...dₙ]      │
   │ - 预测深度概率 α=[α₁...αₙ]      │
   │ - 生成3D点: Σαᵢ·f⊗(u,v,dᵢ)     │
   └─────────────────────────────────┘
   
2. Splat: 3D特征 -> BEV网格
   ┌─────────────────────────────────┐
   │ Voxel Pooling:                  │
   │ - 将3D点分配到BEV网格           │
   │ - 每个网格累加特征               │
   │ - 200×200×1 BEV特征图           │
   └─────────────────────────────────┘
   
3. Shoot: BEV特征 -> 下游任务
   ┌─────────────────────────────────┐
   │ 任务头:                         │
   │ - 目标检测: CenterPoint style   │
   │ - 语义分割: U-Net decoder       │
   │ - 运动规划: Cost volume         │
   └─────────────────────────────────┘
```

**关键技术创新**

1. **离散深度分布**
   ```python
   # 不是预测单一深度值
   depth_logits = model(image)  # [B,D,H,W]
   depth_prob = softmax(depth_logits, dim=1)
   
   # D个离散深度假设
   depth_bins = [4.0, 8.0, 12.0, ..., 45.0]  # 米
   ```

2. **外积构造3D特征**
   ```python
   # 图像特征: [C, H, W]
   # 深度概率: [D, H, W]
   # 3D特征: [C, D, H, W]
   feat_3d = feat_2d.unsqueeze(1) * depth_prob.unsqueeze(0)
   ```

3. **Voxel Pooling**
   ```python
   # 累加到BEV网格
   for each 3d_point:
       bev_x, bev_y = world_to_bev(point.xyz)
       bev_features[bev_x, bev_y] += point.features
   ```

性能指标:
- nuScenes Detection: 32.1 NDS
- BEV Segmentation: 29.5 mIoU
- 6相机推理: 35 FPS on 2080Ti

关键突破:
- 深度分布而非单一深度值(鲁棒性↑)
- 可微分的视角转换(端到端训练)
- 统一的多任务表示(效率↑)

#### DETR3D: Query-based 3D检测

DETR3D将Transformer引入3D检测，开创了query-based的新范式：

```
完整架构流程:

Multi-view Images (6 cameras)
        ↓
  ResNet-101 + FPN
        ↓
  Image Features [6, C, H, W]
        ↓
  ┌──────────────────────┐
  │  3D-2D Transform     │
  │  - 3D ref points     │
  │  - Camera projection │
  │  - Feature sampling  │
  └──────────────────────┘
        ↓
Transformer Decoder (6 layers)
        ↓
  Object Queries (900)
        ↓
  3D Boxes + Classes
```

**核心创新**

1. **3D Reference Points**
   ```python
   # 可学习的3D参考点
   ref_points = nn.Parameter(torch.randn(900, 3))
   
   # 投影到各相机
   for cam in cameras:
       uv = project_3d_to_2d(ref_points, cam.intrinsic, cam.extrinsic)
       features = bilinear_sample(cam.features, uv)
   ```

2. **Multi-Scale 3D位置编码**
   ```
   3D PE = sin(2πk·[x,y,z]/λ)
   其中λ从小到大，捕获不同尺度
   ```

3. **Set-to-Set Loss**
   - Hungarian匹配
   - 分类loss + L1 box loss + GIoU loss

优势:
- 无需密集深度估计
- 全局感受野和关系建模
- 端到端优化
- 自然处理遮挡

性能:
- nuScenes test: 41.2 NDS, 34.9 mAP
- 推理速度: 27 FPS

**BEVFormer延伸**

BEVFormer在DETR3D基础上引入时序和BEV queries：

```
Temporal Self-Attention
    ↓
Spatial Cross-Attention  
    ↓
BEV Queries [200×200]
    ↓
统一BEV特征
```

创新:
- BEV queries作为统一表示
- 时序信息融合
- 可变形注意力加速

## 6.2 多任务学习与特征共享

### 6.2.1 多任务学习动机

#### 计算资源约束

车载平台算力限制下的权衡：

| 平台 | 算力 | 功耗 | 单任务模型数 | 多任务收益 |
|------|------|------|-------------|------------|
| Xavier | 30 TOPS | 30W | 3-4个 | 基准 |
| Orin | 254 TOPS | 60W | 8-10个 | 40%算力节省 |
| J5 | 128 TOPS | 35W | 5-6个 | 50%算力节省 |

#### 任务相关性分析

```
感知任务相关性矩阵
        检测  分割  深度  车道线
检测    1.0   0.7   0.8   0.5
分割    0.7   1.0   0.6   0.7  
深度    0.8   0.6   1.0   0.4
车道线  0.5   0.7   0.4   1.0

高相关性(>0.7)任务适合共享
```

### 6.2.2 早期多任务架构 (2018-2019)

#### MultiNet (2018)

```
共享编码器架构
           Input
             ↓
     Shared Encoder
          /  |  \
    检测头  分割头  深度头
       ↓     ↓      ↓
    Boxes  Mask   Depth
```

特点：
- 简单的hard parameter sharing
- 任务间无交互
- 梯度冲突问题严重

#### DLT-Net (2019)

引入可学习的任务权重：

```python
# 动态任务权重
task_weights = {
    'detection': 1.0,
    'drivable': 0.7 * (1 + 0.3*cos(epoch/max_epoch*π)),
    'lane': 0.5
}
```

### 6.2.3 注意力机制与任务交互 (2019-2020)

#### PAD-Net架构

```
任务交互模块
Detection Features ←→ Distillation ←→ Segmentation Features
         ↓                ↓                    ↓
    Det Output      Shared Feature       Seg Output
```

关键创新：
- Task-specific attention masks
- Cross-task feature distillation  
- 性能提升: +2.3% mAP, +1.8% mIoU

#### YOLOP (2021)

```
高效的三任务模型
┌──────────────────────────────┐
│     Shared Backbone          │
│      CSPDarknet              │
└──────────┬───────────────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
 Neck(FPN)    Seg Decoder
    ↓             ↓
 Det Head    DA&LL Head
    ↓          ↓    ↓
 Boxes    Drivable Lane

推理速度: 40ms@320×180 (Jetson TX2)
精度: 89.2 mAP, 91.5 mIoU drivable, 70.3 IoU lane
```

### 6.2.4 端到端多任务优化

#### 梯度平衡策略

**GradNorm (2018)**
```python
# 自适应任务权重
L_total = Σ w_i(t) * L_i
w_i(t+1) = w_i(t) * exp(α * (G_i/G_avg - 1))
# G_i: 任务i的梯度范数
```

**Uncertainty Weighting**
```python
# 基于不确定性的权重
L_total = Σ (1/2σ_i²) * L_i + log(σ_i)
# σ_i: 可学习的任务不确定性
```

#### 多任务学习收益分析

| 方法 | 检测mAP | 分割mIoU | 深度RMSE | FPS | 内存 |
|------|---------|----------|----------|-----|------|
| 3个独立模型 | 78.2 | 89.3 | 4.82 | 12 | 3.2GB |
| Hard Sharing | 76.5 | 87.1 | 5.13 | 35 | 1.1GB |
| YOLOP | 77.8 | 88.9 | - | 40 | 0.9GB |
| HybridNets | 77.3 | 85.8 | 5.54 | 32 | 1.2GB |

## 6.3 深度估计与伪激光雷达

### 6.3.1 单目深度估计演进

#### 监督学习方法 (2016-2018)

**DORN (Deep Ordinal Regression Network)**

```
深度离散化策略
连续深度 -> 序数标签 -> 分类问题
[0,80m] -> 80个bins -> Softmax

关键创新:
- Spacing-Increasing Discretization
- 近处密集，远处稀疏
- KITTI: 4.46m RMSE
```

**BTS (2019)**

```
架构创新:
Encoder   Decoder
ResNet → Upsampling
   ↓        ↑
 Skip → Local Planar
       Guidance

LPG层: 平面假设指导上采样
性能: KITTI 2.21m RMSE
```

#### 自监督深度估计 (2017-2020)

**Monodepth2 (2019)**

```
自监督损失设计:
L_total = L_photo + L_smooth + L_consistency

L_photo: 光度一致性
L_smooth: 深度平滑
L_consistency: 左右一致性

训练数据: 仅需单目视频
性能: 接近监督方法80%
```

**关键技术突破**

1. **遮挡处理**
```python
# Minimum reprojection loss
L_photo = min(||I_t - I'_t→t-1||, ||I_t - I'_t→t+1||)
```

2. **移动物体处理**
```python
# Auto-masking
mask = (L_photo < L_identity)
```

3. **尺度一致性**
- 通过已知相机高度恢复绝对尺度
- 或使用车速信息作为尺度监督

### 6.3.2 立体深度估计

#### 传统立体匹配回顾

```
经典Pipeline:
左图 ──┐
      ├→ 特征提取 → 代价计算 → 代价聚合 → 视差优化
右图 ──┘
        ↓           ↓          ↓          ↓
     Census    SAD/SSD      SGM       左右检查
```

问题：
- 纹理缺失区域失效
- 遮挡边界错误
- 计算复杂度高

#### 深度学习立体匹配

**PSMNet (2018)**

```
金字塔立体匹配网络
┌─────────────────────────┐
│   Spatial Pyramid       │
│   Pooling Module        │
└───────┬─────────────────┘
        ↓
   Cost Volume (D×H×W)
        ↓
   3D CNN Aggregation
        ↓
   Disparity Regression
```

创新点：
- 空间金字塔池化扩大感受野
- 3D卷积代价聚合
- 亚像素精度: soft argmin
- KITTI 2015: 1.86 pixel error

**GA-Net (2019)**

```
引导聚合网络
Semi-Global → Learnable
   ↓            ↓
 GA Layer: 可学习的聚合方向
 
性能提升但计算量大
推理: 600ms @ 2080Ti
```

### 6.3.3 伪激光雷达技术深度剖析

#### 核心思想与创新

**表示形式的重要性**

```
深度图表示 vs 点云表示

深度图 (前视图)          点云 (鸟瞰图)
┌────────────┐         ┌────────────┐
│░░░░████░░░░│         │  ∙∙∙∙∙∙∙   │
│░░████████░░│   →     │ ∙∙∙∙∙∙∙∙∙  │
│████████████│         │∙∙∙∙∙∙∙∙∙∙∙ │
└────────────┘         └────────────┘
   透视畸变                均匀分布
   
关键洞察: 
- 3D检测器对点云表示的归纳偏置
- BEV视角避免透视畸变
```

#### 技术实现细节

**坐标转换**
```python
# 深度图到点云
def depth_to_pointcloud(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(range(w), range(h))
    
    # 反投影到3D
    z = depth
    x = (u - K[0,2]) * z / K[0,0]  
    y = (v - K[1,2]) * z / K[1,1]
    
    # 相机坐标系到激光雷达坐标系
    points = np.stack([z, -x, -y], axis=-1)
    return points.reshape(-1, 3)
```

**性能对比**

| 方法 | 输入 | 3D AP (Mod) | 推理时间 |
|------|------|-------------|----------|
| M3D-RPN | 单目 | 14.76 | 160ms |
| Pseudo-LiDAR | 单目 | 28.31 | 400ms |
| Pseudo-LiDAR | 立体 | 42.43 | 450ms |
| PointPillars | 64线LiDAR | 82.58 | 40ms |

#### 改进与优化

**PL++关键改进**

1. **深度补全网络**
```
稀疏LiDAR + 深度图 → 稠密深度
4线激光雷达指导立体匹配
性能提升: +15% 3D AP
```

2. **前景分割**
```
Instance mask指导深度估计
避免前景/背景深度混淆
边界准确度提升30%
```

3. **时序融合**
```python
# 多帧深度融合
depth_t = α*depth_t + (1-α)*warp(depth_t-1)
# 提升远距离深度稳定性
```

### 6.3.4 深度估计在量产中的应用

#### Tesla FSD的深度估计策略

**HydraNet架构 (2019-2021)**

```
8摄像头输入
     ↓
Shared Backbone
     ↓
摄像头专属头部 × 8
     ↓
深度+检测+分割
     
关键技术:
- 相机间深度一致性约束
- 运动立体增强单目深度
- 实时性: 36 FPS @ HW3.0
```

#### 中国方案实践

**小鹏XPILOT深度方案**

```
三支路融合:
1. 单目深度网络 (全场景)
2. 环视立体匹配 (停车场景)
3. 结构约束 (车道线/路沿)
     ↓
  统一深度图
```

**地平线深度估计加速**

```
量化策略:
FP32 → INT8
- 深度回归头保持FP16
- Backbone INT8量化
- 性能损失 <2%
- 速度提升 3.5×
```

## 6.4 工程化实践与挑战

### 6.4.1 数据工程

#### 深度真值获取

**激光雷达投影**
```python
# LiDAR点云投影到图像获取深度真值
def project_lidar_to_image(points, T_cam_lidar, K):
    # 坐标变换
    points_cam = T_cam_lidar @ points.T
    
    # 投影
    uv = K @ points_cam[:3]
    uv = uv[:2] / uv[2]
    
    # 深度图
    depth_map = scatter_max(points_cam[2], uv)
    return depth_map
```

问题与解决：
- 稀疏性: 插值或深度补全网络
- 时间同步: 硬件触发 + 软件补偿
- 标定精度: 在线标定算法

#### 困难样本挖掘

```
困难场景识别:
- 强光/逆光: HDR增强
- 雨雾天气: 去雾网络
- 夜晚场景: 多曝光融合
- 动态物体: 运动分割

自动化挖掘pipeline:
1. 模型推理
2. 不确定性估计
3. 人工复核
4. 重新训练
```

### 6.4.2 模型部署优化

#### TensorRT优化

```python
# FP16推理优化
config.set_flag(trt.BuilderFlag.FP16)

# 动态批处理
profile.set_shape("input", 
    min=(1,3,384,1280),
    opt=(4,3,384,1280), 
    max=(8,3,384,1280))

# Layer融合
- Conv+BN+ReLU → CBR
- 多个1×1卷积 → Group Conv
```

性能提升:
- FP32→FP16: 1.8× 加速
- 算子融合: 1.3× 加速
- 动态批处理: 1.2× 吞吐提升

#### 多任务调度

```
任务优先级调度:
┌─────────────────────────┐
│  高优先级 (10ms)         │
│  - 前向碰撞检测          │
│  - 紧急制动              │
├─────────────────────────┤
│  中优先级 (33ms)         │
│  - 3D检测                │
│  - 车道线检测            │
├─────────────────────────┤
│  低优先级 (100ms)        │
│  - 语义分割              │
│  - 停车位检测            │
└─────────────────────────┘
```

### 6.4.3 系统集成挑战

#### 传感器时空同步

```
时间戳对齐:
Camera: 30Hz ──┐
              ├→ 统一时间轴 (100Hz)
LiDAR: 10Hz ───┘

同步策略:
- PTP时钟同步 (<1ms误差)
- 触发器硬同步
- 软件插值补偿
```

#### 坐标系统一

```
坐标系转换链:
像素坐标系 → 相机坐标系 → 车体坐标系 → 世界坐标系
   (u,v)      (x,y,z)_cam   (x,y,z)_ego   (lat,lon,alt)
     ↓            ↓              ↓              ↓
   内参K      外参T_ego_cam   定位系统      高精地图
```

## 6.5 性能评估与对标

### 6.5.1 评价指标体系

#### 2D检测指标
- mAP@IoU=0.5: 主流指标
- mAP@IoU=0.5:0.95: 更严格
- FPS: 实时性要求 >20

#### 3D检测指标
- 3D AP: 3D IoU阈值 (0.7 car, 0.5 ped)
- BEV AP: 鸟瞰图IoU
- AOS: 考虑朝向的AP

#### 深度估计指标
- RMSE: 均方根误差
- REL: 相对误差
- δ<1.25: 准确度阈值

### 6.5.2 主流方案对比 (2020年技术水平)

| 公司/方案 | 2D mAP | 3D AP | 深度RMSE | FPS | 硬件平台 |
|-----------|--------|-------|----------|-----|----------|
| Tesla FSD | 92.3 | - | ~3.5m | 36 | HW3.0 |
| MobileEye | 89.7 | 35.2 | 4.2m | 100 | EyeQ5 |
| 小鹏 | 88.5 | 31.8 | 4.8m | 30 | Xavier |
| 地平线 | 87.2 | 28.5 | 5.1m | 25 | J3 |
| Apollo | 90.1 | 41.3* | 3.8m* | 20 | GPU+LiDAR |

*注: Apollo使用激光雷达融合

## 6.6 本章小结

2016-2020年是自动驾驶感知技术的关键转型期。深度学习不仅取代了传统CV方法，更重要的是开启了端到端学习的新范式。从2D到3D感知的跨越、多任务学习的兴起、伪激光雷达的创新，这些技术突破为后续的BEV感知和端到端驾驶奠定了基础。

关键启示：
1. **表示学习的重要性**: 伪激光雷达证明了表示形式比传感器本身更关键
2. **多任务协同**: 共享特征不仅节省算力，还能提升性能
3. **数据驱动**: 大规模数据和自监督学习降低了标注成本
4. **系统思维**: 感知不是孤立模块，需要与下游任务协同设计

下一章我们将深入探讨BEV感知革命，看看这些基础技术如何演化成统一的3D感知框架。