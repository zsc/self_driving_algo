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

**Haar Cascades 车辆检测**
- Viola-Jones框架的延伸
- 积分图加速计算
- 实时性好但精度低
- MobileEye早期EyeQ芯片的主要算法

**立体视觉深度估计**
- 基于SGBM (Semi-Global Block Matching)
- 计算复杂度O(WHD), D为视差搜索范围
- 对标定精度极度敏感
- 纹理缺失区域失效

#### MobileEye EyeQ时代 (2014-2017)

MobileEye在深度学习来临前的统治地位源于其精心设计的专用架构：

| EyeQ版本 | 年份 | 算力 | 核心算法 | 客户 |
|---------|------|------|----------|------|
| EyeQ2 | 2010 | 2.5 GOPS | Haar+HOG | BMW, GM |
| EyeQ3 | 2014 | 256 GOPS | 混合CNN | Tesla AP1 |
| EyeQ4 | 2018 | 2.5 TOPS | 深度CNN | 日产ProPilot |

EyeQ3的突破在于引入了早期CNN，但仍保留大量传统CV：
- 车道线: 基于Hough变换的多项式拟合
- 目标跟踪: Kalman滤波 + Hungarian匹配
- 空闲空间: 基于v-disparity的地面估计

### 6.1.2 深度学习2D检测革命 (2016-2018)

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

**YOLOv3在自动驾驶中的应用**
- Tesla Autopilot 2.0 (2017): 8摄像头YOLOv3变体
- 百度Apollo 2.0 (2018): YOLOv3 + 跟踪
- 关键改进: 
  - Darknet-53 backbone适配车载算力
  - FPN多尺度检测小目标
  - 9个anchor覆盖车辆尺度范围

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

**Cascade R-CNN在L4系统的应用**
- Waymo (2018): Cascade R-CNN处理稀疏激光雷达
- IoU阈值递增: 0.5 -> 0.6 -> 0.7
- 小目标检测提升: +3.2% AP on KITTI

### 6.1.3 从2D到3D的关键跨越 (2018-2020)

#### 单目3D检测的探索

**M3D-RPN (2019)**
```
深度感知锚点设计
┌─────────────────────────────┐
│  2D中心 (u,v)                │
│     ↓                        │
│  3D中心 (X,Y,Z)              │
│     ↓                        │
│  3D框 (l,w,h,θ)              │
└─────────────────────────────┘
关键: 2D-3D一致性约束
```

**SMOKE (2020) - 无需2D检测的3D预测**
- 直接回归3D中心投影点
- 关键见解: 3D中心投影≠2D bbox中心
- KITTI moderate: 14.03% AP3D
- 推理速度: 30ms on 1080Ti

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

**Pseudo-LiDAR++改进**
- 立体深度网络与稀疏激光雷达联合训练
- Depth Completion Network填补深度空洞
- 性能接近64线激光雷达的30%

### 6.1.4 多视角3D感知 (2019-2020)

#### LSS: BEV感知的先驱

**Lift-Splat-Shoot (2020)**

```
三步转换:
1. Lift: 图像特征 -> 3D frustum特征
   每个像素预测深度分布 D(d|x)
   
2. Splat: 3D特征 -> BEV网格
   Pillar pooling聚合
   
3. Shoot: BEV特征 -> 下游任务
   检测/分割/规划
```

关键技术突破：
- 深度分布而非单一深度值
- 可微分的视角转换
- 端到端训练

#### DETR3D: Query-based 3D检测

```
架构:
Multi-view Images -> CNN Features -> Transformer
                                          ↓
                                    3D Queries
                                          ↓
                                    3D Boxes

创新点:
- 3D reference points采样图像特征
- 无需密集深度估计
- 全局感受野
```

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