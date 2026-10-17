# Food3D POC

工程根目录为 **`E:\study\2_黑客松\app`**。数据、Python 环境、模型缓存和工具的位置由 [configs/paths.json](configs/paths.json) 配置。目录职责与启动方法见 [运行与路径配置](docs/运行与路径配置.md)。

当前功能、手机扫描、定位板测量、密度估重与代码目录说明见 [项目功能与文件架构介绍](docs/项目功能与文件架构介绍.md)（2026-10-17 更新）。

已新增 **盘感 PanSense P0 手机网页应用（PWA）**，涵盖拍照/相册导入、识别纠错、营养估算、真实三维餐盘、饮食日记与本机数据管理。运行 `& tools/start_app.ps1` 后打开 http://127.0.0.1:8000/ 。完整使用及手机安装说明见 [app/README.md](app/README.md)，验收记录见 [app/P0_ACCEPTANCE.md](app/P0_ACCEPTANCE.md)。原有 Gradio 演示继续保留。

已运行均匀 5、10、30 张视角的重建数量实验：见 [实验结果与复现](docs/view_count_comparison.md)。报告分别记录仅 N 张图像的独立 COLMAP 成功率，以及复用完整标定位姿的有限视角画质对比，避免把辅助结果混为仅 N 张照片的独立重建。

多视角图片/视频 → COLMAP → Nerfstudio Splatfacto → SAM 2.1 + CLIP → 高斯实例投票 → Gradio 三维点击。

全部模块已实现，在本机 RTX 4060 Laptop 8GB 上执行训练。已运行水果1的 120 张渲染图片和水果2的 41 秒视频，均通过 COLMAP 估计相机，没有使用已知相机或 Blender 工程。当前演示使用水果2视频结果，验证记录见 `docs/fruit2_validation.md`；水果1的结果仍保留，见 `docs/validation.md` 和 `docs/quality_validation.md`。指标衡量对 SAM 2 的一致性，尚未使用人工逐像素真值或实测体积验证。

## 启动已有演示

在项目根目录的 PowerShell 中运行：

```powershell
& tools/start_demo.ps1
```

浏览器打开 http://127.0.0.1:7860 。直接按住三维画面拖动旋转，滚轮缩放；轻点食物查看 Instance ID、名称和体积，选中实例持续高亮。使用“恢复拍摄视角”回到当前下拉框指定的真实拍摄视角。传统参数控件位于折叠的“精细视角设置（可选）”。“修正名称”只修改名称，不改变实例 ID。背景不显示食物名称，证据不足的区域显示未知。

当前水果2场景包含香蕉和两个小圆形水果，分别具有独立 ID；绿色水果的 CLIP 名称“芒果”待核对。默认视角从分割参考帧读取，本视频为 `000015`，环绕初始角度也靠近这台相机。点击读取同一组三维高斯的语义通道渲染结果，不读取二维检测框或当前视角 SAM 掩码。旧水果1场景含两颗苹果、两根香蕉、两颗榴莲，可使用 `tools/start_demo.ps1 -Bundle data/outputs/fruit1_quality_cleaned` 打开。

## 项目结构

```text
app/web/                       网页界面（工程根目录下的应用资源子目录）
app/catalog/                   食物营养参考目录
src/pansense/                  餐食记录、接口、任务与三维交互
configs/                       场景参数（原始/全连接/匹配修复）
src/food3d/
  data/                        视频、照片、COLMAP、坐标、匹配过滤
  reconstruction/              官方 Splatfacto 训练与 NPZ/PLY 导出
  segmentation/                SAM 2 自动初始化、双向传播、多视角 CLIP
  fusion/                      gsplat 可见贡献、深度门控、边缘加权投票
  viewer/                      CUDA 高斯渲染、三维语义点击、Gradio
    orbit.py / drag.js          相机相对旋转与鼠标拖动/滚轮交互
  evaluation.py                保留视角的一致性与点击检查
tools/                         环境安装、演示、流水线与 GPU/相机调试
tests/                         基础几何与接口回归测试
environments/                  版本约束和本机已安装包清单
docs/                          数据接口与验证记录
data/raw/photos/水果1/         原始图片（只读）
data/scenes/fruit1_refined/     最终 COLMAP 相机和图像
data/outputs/fruit1_final/     最终高斯与三维语义
```

## 环境

`.venv` 使用 Python 3.13（COLMAP/SAM 2/CLIP），`.venv-gs` 使用 Python 3.10（Splatfacto/gsplat/Gradio）。两套 PyTorch/CUDA wheel 不可混装。新机器先安装 uv 和 NVIDIA 驱动：

```powershell
& tools/setup_windows.ps1
& $gsPython tools/gpu_smoke.py
```

安装需网络和磁盘空间。使用官方 gsplat Windows 预编译 wheel，无需本地 nvcc/MSVC；SAM 2 关闭可选 C++ 扩展及依赖它的后处理。CLIP/LPIPS 权重首次使用下载至 `.cache/models`。详见 `environments/README.md`。已有环境无需重新安装。

## 逐模块运行

先在工程根目录执行 `. ./tools/project_paths.ps1`，加载配置中的 Python 路径和导入环境。以下示例的 `data/...` 参数由命令入口解析到配置的数据根目录。

输出目录需为空；失败排查保留旧目录，使用新名称。处理新数据时必须删掉配置中的 `reuse_database`，它只用于相同图片。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& $recognitionPythonExe -m food3d doctor --output data/outputs/environment.json
& $recognitionPythonExe -m food3d photos --input 'data/raw/photos/水果1' --scene data/scenes/new_scene --config configs/poc.yaml --source-kind rendered
& $recognitionPythonExe -m food3d colmap --scene data/scenes/new_scene --config configs/poc.yaml
& $recognitionPythonExe -m food3d inspect --scene data/scenes/new_scene
& $recognitionPythonExe -m food3d train --scene data/scenes/new_scene --output data/outputs/new_training --iterations 7000
& $recognitionPythonExe -m food3d segment --scene data/scenes/new_scene --output data/outputs/new_seg --reference-frame 000077
& tools/finish_pipeline.ps1 -Scene data/scenes/new_scene -Training data/outputs/new_training -Segmentation data/outputs/new_seg -Bundle data/outputs/new_bundle -Evaluation data/outputs/new_evaluation
& tools/start_demo.ps1 -Bundle data/outputs/new_bundle
```

参考帧应选全部对象可见的图像 ID。自动初始化失败时，`--prompts` 可提供固定实例 ID 和 points/box；名称可在 UI 修正。CLIP 是候选名称匹配，分数不是校准概率。后处理脚本读取 `latest_training.json` 的检查点路径，依次执行 export、fuse、evaluate；各命令参数见 `--help`。

视频入口需要 FFmpeg/FFprobe 在 PATH：

```powershell
& $recognitionPythonExe -m food3d extract --video data/raw/food.mp4 --scene data/scenes/video_scene --config configs/poc.yaml
```

随后运行相同流程。抽帧保留真实 PTS，照片时间戳为空，按自然文件名排序；SAM 传播要求顺序连续、相邻画面有重叠。

FFmpeg 8.1.1 和 FFprobe 位于项目的 `.tools/ffmpeg/bin`，`configs/fruit2_video.yaml` 通过项目内路径选择这两个工具。视频元数据按 UTF-8 解码，支持中文标题及 emoji。该配置按 2.85 fps 覆盖完整 41.361 秒的水果2视频；更长视频应降低采样率或提高 max_frames，避免只处理开头。

完整视频流程包装命令（新输出名，参考帧必须实际被保留；各阶段已运行，包装脚本未整段重复运行）：

```powershell
& tools/run_video.ps1 -Video 'data/raw/videos/水果2.mp4' -Name fruit2_rerun -Config configs/fruit2_video.yaml -ReferenceFrame 000015 -Iterations 15000
```

包装流程不自动修改当前演示，需检查投影、分割、重建与点击评估后再打开新 bundle。

## 本组重复物体的 SfM 修复

初次 PINHOLE 全连接重建虽注册 120 张，但同外形榴莲的特征被错误关联，横纵焦距发散。原生 Nerfstudio 渲染与导出渲染一致，定位到 SfM 问题，失败输出保留供调试。

从原始图片完整重做本组流程（使用新目录，含两次 SfM；包装脚本尚未完整重复执行）：

```powershell
& tools/reproduce_fruit1.ps1 -RunName fruit1_rerun
```

最终用已有 SAM 2 实例轨迹过滤跨实例匹配，共享 SIMPLE_PINHOLE，初始焦距为图像宽度的 1.2 倍（1152 像素），本次关闭焦距优化。该焦距是估计假设，不是实测标定。过滤不生成或伪造位姿。

```powershell
& $recognitionPythonExe tools/filter_sfm_matches.py --source data/scenes/fruit1_full/colmap/database.db --output data/outputs/new_filtered.db --segmentation data/outputs/fruit1_seg --focal 1152
```

复制 `configs/fruit1_refined.yaml` 并修改 `reuse_database`，在新照片场景运行 colmap。相机更新后 `tools/rebind_segmentation.py` 可复用掩码，但必须逐张检查图像字节和尺寸完全相同，避免误用旧掩码。

## 模型与三维语义

- `gaussians.npz`：位置、wxyz 四元数、物理尺度、透明度、SH 颜色和稳定索引。
- `gaussians.ply`：标准 binary 3DGS PLY，包含 log scale、opacity logits、SH。
- `scene.json`：训练坐标相机，保留源 frame ID、图片对应关系和内参。
- `semantics/semantics.npz`：每个高斯的实例 ID、置信度、支持视角数和投票。
- `semantics/instances.json`：可改名的实例字典。
- `semantics/fusion.json`：投票/保留视角、模型哈希。模型修改必须重新融合。

背景 ID 0，未知 ID 65535。可见性使用 gsplat 的相交高斯与实际前向合成贡献 T×alpha，并加入深度一致性和掩码边缘抑制。投票至少需 3 个支持视角，不对低证据区域强制赋食物标签。40 个视角投票，12 个保留视角检查一致性；SfM 和 SAM 传播使用全部图片，保留视角只对语义投票独立。评估参考为 SAM 2，不是人工真值。

## 调试与测试

```powershell
& $recognitionPythonExe -m pytest -q -p no:cacheprovider --basetemp .cache/test-new
& $gsPython tools/gpu_smoke.py
& $recognitionPythonExe -m food3d synthetic --scene data/scenes/synthetic_new --config configs/poc.yaml
& $recognitionPythonExe -m food3d inspect --scene data/scenes/synthetic_new
```

29 项测试已通过（包含拖动相机、体积估算和中文视频元数据解码）。GPU smoke 检查真实 gsplat 渲染与贡献提取的 alpha 一致性。检查 debug/projection_sheet.png、分割 debug、训练日志和评估 comparison_sheet.png。仅程序退出成功或注册全部图片不代表重建正确。

## 表面与遮挡改进

水果1改进版使用 fruit1_quality_cleaned：前景 PSNR 从 22.81 提升到 24.74 dB，实例 IoU 从 0.853 提升到 0.874，点击保持 69/72。训练与清理后保留 195615 个高斯。详细数据、失败案例及开发视角的限制见 docs/quality_validation.md。

`configs/fruit1_quality.yaml` 降低低透明度高斯删除阈值、增加增殖，并在 12000 步后固定集合继续优化至 15000 步。实际模型参数会保存到 Nerfstudio config.yml 和输入快照。重新运行需使用新输出名：

```powershell
& tools/run_quality.ps1 -Name fruit1_quality_rerun
```

该脚本将 108 个非保留视角用于投票，仍使用原来的 12 个保留视角评估。透明度影响投票权重，支持视角数单独统计一致且具有足够可见贡献的观测，避免细小高斯因透明度较低而永远没有支持视角。每视角贡献低于 0.001 的残余仍不计支持。

先用期望深度融合获取背景证据，再由 tools/cleanup_floaters.py 删除少于 3 视角支持、背景证据超过 98% 且不透明度大于 0.5 的孤立高斯。清理后保持稳定高斯 ID，重算模型哈希与全部语义，使用中位表面深度门控。食物、混合边缘和可靠背景不参与该清理。

`tools/compare_quality.py` 在相同保留视角比较两版，并输出每实例颜色 PSNR、苹果红色像素恢复比例和食物区域未知像素比例。颜色指标用于定位表面缺口，不是几何真值。背景占比很大，因此优先看前景指标。`tools/start_demo.ps1` 优先读取 data/outputs/active_demo.json 指定的已验证模型。

## 三维实例体积粗估

点击食物时显示该 Instance ID 对应的凸包包围体积，切换视角后数值不变。背景和未知点击不返回体积。当前使用可靠高斯中心：语义置信度至少 0.7、至少 3 个支持视角、不透明度至少 0.1；按第六近邻距离剔除孤立点，再调用 SciPy ConvexHull。高斯核重叠且只是外观表示，因此不累加高斯椭球体积。详见 [体积功能说明](docs/volume.md)。

未经标定显示模型单位³，COLMAP/训练的尺度不是厘米。界面展开“尺寸标定”，选中参考食物，输入实测最大两点直线跨度（厘米），即可按统一比例显示 mL（cm³）。清除标定会恢复模型单位，原三维模型保持不变。渲染图片没有真实尺寸时请勿把假设尺寸当作实测结果。

```powershell
# 已有 viewer 环境无需安装；独立使用体积工具需安装 food3d-poc[measurement]
& $gsPython -m food3d volume --bundle data/outputs/fruit1_quality_cleaned
# 仅当实际测得 Instance 2 的最大跨度为 8 cm 时使用：
& $gsPython -m food3d volume --bundle data/outputs/fruit1_quality_cleaned --reference-instance 2 --reference-span-cm 8
```

输出 measurements/volumes.json；实测标定保存到 measurements/calibration.json。测量缓存绑定模型与语义哈希，任一改变都会重算并停用旧标定。当前 6 个实例均可估算；体积真实精度尚未验证。凸包会计入香蕉弯曲空隙，缺失表面和语义遗漏也会造成低估，数值属于包围体积粗估。
