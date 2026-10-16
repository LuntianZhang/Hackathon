# 白色纸片标定与体积估计

视频中放置实测 **10 × 2 cm** 的白色长方形纸片，保持平整、静止。尽量让纸片和物体出现在同一场景的多个角度，纸片不要被遮挡或出画。纸片无需与所有物体处于同一高度；它标定整个重建坐标的尺度。

应用视频流程自动执行标定。三维页点击物体显示约多少 mL、最大跨度 cm，以及标定状态；“查看纸片识别”展示检测轮廓。未找到足够完整视角或几何校验不通过时，不输出 mL。物品名称可人工指定，不在营养目录的物品不会生成虚构营养。

## 实现

1. 在去畸变的相机图像上检测低饱和度白色四边形，排除圆形、裁切边缘及过小区域。
2. 使用真实内参和 10 × 2 cm 长方形运行 OpenCV PnP，得到纸片中心相对于相机的厘米位移。
3. 结合 COLMAP / Splatfacto 导出的相机旋转与位置，用确定性的 RANSAC 拟合同一个静止纸片中心：`P_cm = s * C_model + R_c2w * t_cm`。至少 6 个有效视角，并要求足够的相机平移基线。
4. 对齐无文字身份的长方形角点，进行多视角三角化；独立核对长/短边尺度、中心拟合及重投影误差。默认尺度差阈值 18%，960 px 长边下重投影 RMS 上限 4 px。
5. 对语义实例的可靠高斯中心做离群过滤和三维凸包，使用 `V_cm3 = V_model * s^3`。1 cm³ 对应 1 mL。**不会相加高斯核的椭球体积。**

纸片标定绑定模型哈希和相机几何哈希；语义更新时沿用尺度，重新计算物体体积。手工物体跨度标定仍兼容。重新训练或相机几何变化时不能沿用旧尺度。

复用 [OpenCV 相机标定与 PnP API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)，本机实际验证 OpenCV 4.10（重建环境）及 5.0（识别环境）。无需新增权重或训练识别模型。已有双环境包含 OpenCV 与 SciPy；新机器使用项目的 `sfm` 和 `measurement` 依赖。

## 配置与接口

`configs/pansense.yaml`：

```yaml
paper_reference:
  enabled: true
  length_cm: 10
  width_cm: 2
```

CLI：

```powershell
$env:PYTHONUTF8='0'
$env:PYTHONIOENCODING='utf-8'
& $gsPython -m food3d paper-scale --bundle data/outputs/objects3_paper_bundle_01 --length-cm 10 --width-cm 2
& $gsPython -m food3d volume --bundle data/outputs/objects3_paper_bundle_01
```

PWA 的 `/api/meals/{id}` 返回 `volume_summary` 和每个 `items[].volume`；`GET /api/meals/{id}/scale-reference` 返回纸片诊断图。后台计算凸包，HTTP 请求只读取已发布结果。

输出在 bundle 的 `measurements/`：

- `paper_reference.json`：检测四角、通过/拒绝状态、原因及几何校验。
- `paper_detections.jpg`：纸片轮廓诊断。
- `calibration.json`：仅校验通过后保存厘米尺度；与模型/相机绑定。
- `volumes.json`：各语义实例的模型单位体积；CLI/API 按有效标定换算 mL。

## 物品3.mp4 实际验证

输入 `data/raw/videos/物品3.mp4`，27.22 秒。62/62 个筛选视角注册，SfM 平均重投影误差约 1.05 px。Splatfacto 7000 步，SAM 2 使用首帧的人工点选名称（耳机盒、笔、手表）传播，纸片不作为测量物体。

26 个纸片视角通过校验，尺度 **21.402762 cm / 模型单位**，独立短边检查约 **2.106 cm**，重投影 RMS **3.21 px**。

| 实例 | 凸包体积粗估 | 最大跨度粗估 |
|---|---:|---:|
| 耳机盒 | 68.3 mL | 6.62 cm |
| 笔 | 21.1 mL | 15.23 cm |
| 手表 | 49.9 mL | 6.54 cm |

这些数值**没有实测体积真值验证**。短边约 5.3% 的一致性差异不能直接当作体积误差；尺度误差会按三次方影响体积。凸包会包含凹陷和弯曲空隙，反光、遮挡、物体底部不可见及语义缺失也会影响估算。视频含真实物品，但人工名称不代表自动物品分类已验证。跨视角 SAM 2 一致性 IoU 0.662，物体点击一致性 0.667，背景不返回物体名称 1.0。

目录：`data/scenes/objects3_paper_01`、`data/outputs/objects3_paper_training_01`、`objects3_paper_seg_01`、`objects3_paper_bundle_01`、`objects3_paper_evaluation_01`。大文件被 Git 忽略。

完整复现（同机环境，输出目录应未存在；新一次运行请更换目录名）：

```powershell
& $recognitionPythonExe -m food3d extract --video 'data/raw/videos/物品3.mp4' --scene data/scenes/objects3_paper_01 --config configs/objects3_paper.yaml
& $recognitionPythonExe -m food3d colmap --scene data/scenes/objects3_paper_01 --config configs/objects3_paper.yaml
& $recognitionPythonExe -m food3d segment --scene data/scenes/objects3_paper_01 --output data/outputs/objects3_paper_seg_01 --reference-frame 000000 --assistance configs/objects3_instances.json
& $gsPython -m food3d.reconstruction.train_worker --scene data/scenes/objects3_paper_01 --output data/outputs/objects3_paper_training_01 --iterations 7000 --config configs/objects3_paper.yaml
$paperCheckpoint=Get-ChildItem data/outputs/objects3_paper_training_01 -Filter step-000006999.ckpt -Recurse | Select-Object -First 1
$paperTransform=Join-Path $paperCheckpoint.Directory.Parent.FullName dataparser_transforms.json
& $gsPython -m food3d export --checkpoint $paperCheckpoint.FullName --scene data/scenes/objects3_paper_01 --output data/outputs/objects3_paper_bundle_01 --transform $paperTransform
& $gsPython -m food3d paper-scale --bundle data/outputs/objects3_paper_bundle_01
& $gsPython -m food3d fuse --bundle data/outputs/objects3_paper_bundle_01 --segmentation data/outputs/objects3_paper_seg_01 --max-views 50 --width 480
& $gsPython -m food3d volume --bundle data/outputs/objects3_paper_bundle_01
& $gsPython -m food3d evaluate --bundle data/outputs/objects3_paper_bundle_01 --output data/outputs/objects3_paper_evaluation_01 --width 480
& $recognitionPythonExe tools/import_bundle_to_app.py --bundle data/outputs/objects3_paper_bundle_01 --video 'data/raw/videos/物品3.mp4' --title '物品3 · 纸片体积验证'
```

`configs/objects3_instances.json` 的点坐标只适用于这个视频首帧，其他视频必须重新标注。CLI 仍提供人工参考帧标注；PWA 视频入口目前提供自动或文字辅助识别，没有视频逐帧点框编辑器。导入工具只创建未保存草稿，不计入日记。
