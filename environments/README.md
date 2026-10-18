# 已验证的 Windows 环境

本页保存当时 Windows 机器的环境记录；`E:\study\2_黑客松\app` 是历史工程位置。两套 Python 环境、模型缓存与工具的位置由 `configs/paths.json` 配置。直接运行 Python 命令前，在工程根目录执行 `. ./tools/project_paths.ps1`，或使用启动脚本。GPU 部署与复现步骤见 [PanSense 技术文档](../docs/PanSense_技术文档.md#7-部署与复现计算平台环境和路径)。

Python 运行时位于 `.tools/python`，依赖包位于 `.venv` 和 `.venv-gs`，两套环境均关闭系统包和用户包目录。uv 位于 `.tools/uv/uv.exe`，FFmpeg 与 FFprobe 位于 `.tools/ffmpeg/bin`；项目运行所需的资源目录均在工程内。

历史验证设备为 RTX 4060 Laptop、8 GB 显存；这只是当时的测试来源，不是最低硬件要求。项目采用两套环境避免 PyTorch/NumPy 冲突，其他设备应按[部署章节](../docs/PanSense_技术文档.md#7-部署与复现计算平台环境和路径)逐层检查。

| 用途 | Python | 核心依赖 |
|---|---|---|
| COLMAP / SAM 2 / CLIP | 3.13.3 | torch 2.11.0+cu128，torchvision 0.26.0+cu128，pycolmap 4.2.1，OpenCLIP 2.32.0 |
| Splatfacto / 融合 / 界面 | 3.10.20 | torch 2.4.0+cu124，torchvision 0.19.0+cu124，Nerfstudio 1.1.5，gsplat 1.4.0+pt24cu124，Gradio 5.49.1 |

实际包清单为 recognition-installed.json、reconstruction-installed.json；版本约束为对应 constraints.txt。安装脚本为 tools/setup_windows.ps1（需 uv 和网络，尚未在全新机器验收）。

本机没有 CUDA Toolkit/nvcc 或 MSVC，使用[官方 gsplat Windows cp310 wheel](https://docs.gsplat.studio/whl/pt24cu124/gsplat/)，GPU 渲染已运行通过。fpsample 固定 0.3.3，避免新版源码在中文路径编译失败。

[SAM 2 官方源码](https://github.com/facebookresearch/sam2)在 .cache/upstream/sam2-main；SAM2_BUILD_CUDA=0，关闭可选后处理，自动掩码与双向传播使用 GPU。源码来自 main ZIP，SHA256 ee2376bdd18ee951f616c151f26fefd4339ab6a20402f93772b5cec06952b747。安装脚本首次获取 main；上游变化需重新检查 API，并非固定提交锁。

SAM 2.1 small 权重 SHA256：6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38。gsplat wheel SHA256：b15a96a2792399a2ef09df3a017e7eccd3068b2c3baf852eaeccc5064804cdbd。

CLIP：ViT-B-32 / laion2b_s34b_b79k；缓存 .cache/models。安装两套环境需要充足磁盘空间。

Windows 中文路径：Python 3.10 site.py 按本地编码读取 editable .pth，强制 PYTHONUTF8=1 会启动失败；子进程保持本地编码，用 PYTHONIOENCODING=utf-8 输出日志。Nerfstudio 配置与检查点只加载可信本地训练产物。
