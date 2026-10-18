# 盘感 PanSense P0 · 本机 PWA

> 此文件保留早期 P0 操作记录；目前的任务顺序、定位板、密度估重及版本差异请以 [PanSense 技术文档](../docs/PanSense_技术文档.md) 和当前界面为准。

数据、运行环境和模型缓存的位置由 `configs/paths.json` 配置，详见 [技术文档的部署章节](../docs/PanSense_技术文档.md#7-部署与复现计算平台环境和路径)。

## 记录入口（按 capture-flow 示意页接入）

底部“记录”提供三个独立流程：

- **照片记录**：拍照 / 相册 → 选择分割方式 → 识别、核对 → 保存。“食物清单分割”“名称定位分割”“点击 / 框选分割”分别提供清单辅助、名称定位和人工区域提示；“不使用语义分割”保留照片，直接手工填写食物清单，不调用分割模型。
- **三维记录**：环绕视频 → 选择分割方式与标尺信息 → 重建 → 拖动观察、核对 → 保存。与照片共用上述四个选项；“不使用语义分割”仅训练和导出三维外观，跳过分割、语义融合、食物区域评估及体积估算。取消白纸选项后不执行纸片尺度标定。
- **调整已有餐盘**：显示真实历史餐食与草稿，按餐食时间倒序，包含三维、照片和手工记录。具有原始素材的餐盘可重新分割；其他记录仍可调整名称与食物清单。自定义数据目录中的上传模型也能校验并继续调整。重新识别生成新草稿，原记录保留，确认保存后才计入日记；选择“不使用语义分割”直接返回原餐食调整清单。

历史列表选中餐盘后立即提供三维预览；编辑页默认打开“三维拖动查看”，支持拖动旋转、滚轮缩放、点击食物与恢复拍摄视角。点框方式自动切换到“参考图标注”，提示始终落在固定参考相机照片上；切换视图不会丢失未提交的名称或区域。三维加载失败时仍可使用参考图，并可重试三维预览。

历史选择和编辑页都提供“不更新，直接查看这一餐营养”：打开原餐食并定位到热量、蛋白质、碳水和脂肪卡片，不提交输入、不创建草稿、不触发重新识别。点击“返回餐盘调整”可继续原来的未提交编辑。

点框标注的“当前食物名称”和食物更正/补充弹窗的名称候选来自已保存历史餐食，按首次出现顺序去重，仍允许自由输入新名称。没有历史记录时不填充预置候选；营养参考目录独立保留。

视频点框标注先点击“准备标注参考帧”：仅上传一次，执行抽帧和 COLMAP，显示已注册、去畸变的真实参考帧。点框坐标与后续 SAM 2 的参考帧严格一致，继续时复用相机场景，不重新抽帧或另选参考图。此准备阶段也需要等待相机重建，尚未运行 Splatfacto；离开页面后可从首页未保存草稿继续。

准备接口为 `POST /api/meals/upload?kind=video`（表单 `prepare=true`），完成标注后调用 `POST /api/meals/{id}/video-start`（`revision`、`assistance`、`paper_present`），防止重复启动和旧版本覆盖。公共响应只提供 `video_reference_ready` 和照片 URL，不暴露准备场景路径。

参考照片、三维示例和识别对比已收进折叠的“示例与识别对比工具”，不再与主记录按钮并列。独立示意页仍保留在 `app/mockups/capture-flow.html`。

当前三维任务使用 **20 × 20 cm 或 A3 ArUco 定位板** 校准尺度；通过校验后才可显示实例凸包体积（mL），并可结合密度估计重量和营养。历史纸片标定的读取兼容仍保留。详见 [定位板与体积说明](PAPER_VOLUME.md)及[技术文档](../docs/PanSense_技术文档.md#4-从三维餐盘到体积和营养)。

语义识别已直接接入餐食应用：三维记录和历史餐盘编辑可选择自动识别、食物清单、文字定位、名称 + 点击 / 框选，无需离开应用。独立 `/recognition.html` 保留用于比较不同方法，已有结果也可导入餐食草稿。使用、模型依赖与验证范围见 [ASSISTANCE.md](ASSISTANCE.md)。

本应用按 `app规划.md` 的 A1–A6、B1–B4 和七个页面实现，沿用 `PanSense_手机App交互原型.html` 的配色、字体层级和布局，同时提供 Food3D/Gradio 三维演示入口。

当前交付是可安装的手机网页应用：手机/桌面使用同一套界面，Python 服务在本机运行识别和重建。界面不是静态假数据演示；照片使用真实 SAM 2 + CLIP，视频使用 COLMAP + Splatfacto，三维点击读取同一帧高斯渲染的语义结果。首次相机权限、实体手机安装与拍摄仍需设备验收，见 [技术文档的实验章节](../docs/PanSense_技术文档.md#8-实验结果应该怎样读)。

## 启动

在项目根目录的 PowerShell 中：

```powershell
& tools/start_app.ps1
```

打开 **http://127.0.0.1:8000/**。首次进入选择每日热量参考值或跳过。页面底部提供首页、日记、拍照、我的。

已有 `.venv`、`.venv-gs` 和模型缓存，无需重新下载。若缺少 App 依赖，可在原重建环境中追加，避免更新 Torch/gsplat：

```powershell
. ./tools/project_paths.ps1
uv pip install --python $gsPython -e '.[app,viewer,dev]' -c environments/reconstruction-constraints.txt -c environments/pansense-constraints.txt
```

新机器先按 [environments/README.md](../environments/README.md) 和 `tools/setup_windows.ps1` 配置两套环境；安装脚本尚未在全新机器重复验收。FFmpeg/FFprobe 使用项目内 `.tools/ffmpeg/bin` 中的工具。无需 Node 构建步骤。

等价 Python 入口与隔离验收服务：

```powershell
$env:PYTHONUTF8 = '0'
$env:PYTHONIOENCODING = 'utf-8'
& $gsPython -m pansense --port 8000 --data-dir data/pansense
# 单独的测试数据，不混入正式日记：
& tools/start_app.ps1 -Port 8001 -DataDir data/pansense-validation
```

服务仅绑定 `127.0.0.1`。同机不要同时运行多个 GPU 分析服务；单个 App 内的识别、训练与渲染会串行使用 GPU。关闭运行终端或按 Ctrl+C 停止服务，未完成任务下次显示“分析已中断”，可重新分析。

## 使用流程

1. **拍照**：手机使用系统相机入口；桌面可打开摄像头。也可从相册导入 JPG、PNG、WebP（≤20 MB）。照片会去掉 EXIF 元数据并缩放后保存。
2. **照片识别**：导入前后都可选择食物清单分割、名称定位分割、点击 / 框选分割，或不使用语义分割。清单与名称定位需要填写食物名称，点框方式需要标注区域；不使用分割时直接填写食物清单。从结果页点击“食物语义识别”可再次选择方式。完成分割后可切换原始照片和语义分割图，点击区域读取对应 Instance ID、名称与营养。自动候选需核对，人工提示名称保留。
3. **纠错**：更正名称并选择参考食物、补充遗漏项、移除误识别项。补充项可以在原图上拖框关联二维区域；这不会自动产生新的三维实例。自定义且未匹配目录的名称保留，营养显示“待补充”。
4. **营养分析**：整餐与单项显示估算热量、蛋白质、碳水、脂肪。照片记录可按目录参考份量显示“未称重”；有可靠体积和密度的三维记录可估计本份重量。缺少资料的项目保持待补充，汇总标记未完整。
5. **保存**：填写标题、时间、早餐/午餐/晚餐/加餐和备注，确认后进入日记。保存只更新一条记录，可再次修改基本信息。首页和日记按偏好时区累计已保存的餐食，草稿不计入。
6. **三维餐盘**：导入 3 秒至 3 分钟环绕视频（≤240 MB），食物和桌面保持静止。分析需要数分钟。已有识别清单可先核对和保存，完成后原记录会获得三维资产，手工改名不会被覆盖。
7. **三维交互**：原始照片/3D 切换；按住图像拖动旋转，滚轮缩放，点击食物高亮并查看固定 Instance ID、名称及营养。背景无食物名称，证据不足区域为未知。点击“恢复拍摄视角”回到初始化相机。改变视角不会重新做二维识别。
8. **失败回退**：无 GPU、视频缺少重叠、重建/融合不足或 GPU 正在处理另一餐时，继续用原图与清单保存记录。分析失败可以重试，也可以直接手工记录。

**已有结果与重新标注**：记录页折叠工具中的“导入已有识别结果”会将对比结果用于餐食草稿，保留食物 ID、掩码和已有三维模型。结果页“调整识别 / 重新标注”使用原照片创建新草稿，原餐食不会被覆盖。两者都需要确认保存后才计入日记。

**用语义输入辅助三维餐盘**：在已有模型的结果页或三维页点击“输入食物，辅助此三维餐盘”，输入正确食物清单或选择点框命名，再点击“分割并更新此三维餐盘”。应用使用该模型的参考相机照片，自动执行分割 → 多视角传播 → 高斯语义融合，完成后切换到三维视图；点击显示本次输入的食物名称和 ID。复用原高斯几何生成语义副本，不重新训练，也不覆盖原餐食。已有示例、视频模型及已关联的照片模型均可使用。

普通单张照片没有对应多视角模型时，仅提供二维识别；不会套用其他餐食的模型。当前 30 视角模型的参考照片也可通过“用参考照片体验语义识别”载入，识别后自动进入三维关联。

拍照页“查看已有三维示例”引用原 POC 的 `fruit2_bundle`，明确标记为已有示例。示例创建为未保存草稿，不会自动进入饮食日记。

## 安装到手机或桌面

应用已提供 Web App Manifest、192/512 图标、Service Worker 和独立窗口模式。个人中心“安装应用”在浏览器支持安装事件时发起安装，否则显示平台操作说明。

安装要求与平台操作可对照 [MDN PWA 安装条件](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable)、[Chrome 本地服务端口转发](https://developer.chrome.com/docs/devtools/remote-debugging/local-server)和 [Apple Safari 添加网页应用](https://support.apple.com/zh-cn/guide/iphone/iphea86e5236/ios)。

**桌面**：在支持 PWA 的 Chrome/Edge 打开本机 URL，使用浏览器安装入口。实际安装动作尚未执行。

**Android 本机开发**：手机 USB 连接这台运行服务的电脑，使用官方 Android Platform Tools，在用户自行开启 USB 调试并授权电脑后：

```powershell
adb reverse tcp:8000 tcp:8000
```

手机 Chrome 打开 **http://localhost:8000/**，从菜单选择安装应用/添加到主屏幕。该方式使用 localhost，适合相机及 PWA 开发访问；断开电脑或停止服务后，识别和日记不可用。此 USB 链路与实体设备安装待验证。

**iPhone/iPad**：需要另行部署可信 HTTPS 的服务入口，然后在 Safari 分享菜单“添加到主屏幕”。当前 localhost 版本未配置公网部署、账户与多用户隔离。直接访问电脑局域网 IP 不是本次交付的手机访问方案；浏览器的安全上下文和服务的主机限制也需要相应部署设计。

离线时只保留应用外壳并提示恢复连接；不声称离线识别或离线查看饮食记录。照片、餐食 JSON、三维画面不写入 Service Worker 缓存。

## 工程结构与架构

```text
app/
  web/                     七页 PWA、样式、图标、Manifest、Service Worker
  catalog/foods.json       食物名称、识别词与部分营养资料
  README.md                运行、安装与限制
  PAPER_VOLUME.md          定位板标定与体积说明
src/pansense/
  api.py                   FastAPI、上传、餐食、隐私、图像与三维接口
  store.py                 SQLite 持久化、版本号、按时区汇总
  nutrition.py             名称匹配、参考份量和营养累计
  photo.py                 SAM 2 自动掩码 + 预训练 CLIP
  jobs.py                  串行 GPU 任务、进度、重试、失败回退
  rendering.py             高斯渲染、相机控制、语义点击帧令牌
  meal_semantics.py        餐食内语义预览、已有结果导入、重新识别、三维关联
configs/pansense.yaml       训练步数、融合分辨率及质量门槛
configs/fruit2_video.yaml   FFmpeg、COLMAP、Splatfacto 基础配置
tools/start_app.ps1         启动应用
tools/pansense_smoke.py     对隔离服务运行真实照片/视频验证
tools/build_pansense_catalog.py  从官方 SR28 原始压缩包生成目录
tests/pansense/             存储、接口、纠错、隐私、任务合并和照片筛选
```

PWA → FastAPI → SQLite。照片任务由 `.venv` 子进程加载 SAM 2.1 small + OpenCLIP ViT-B-32。视频抽帧/COLMAP/SAM 传播同样在 `.venv` 执行；Splatfacto 训练、导出、gsplat 融合、评估和高斯渲染在 `.venv-gs` 执行，复用 `src/food3d`。

三维 RGB 与 Instance ID/置信度图来自同一次渲染，服务返回短期帧令牌。点击绑定该令牌和归一化坐标，查对应语义像素，再从记录读当前食物名称及营养；改名保留实例 ID。渲染缓存最多 24 帧，重启/过期时重新打开视角。手工新增项没有自动三维区域，删除食物后点击原区域也不返回已删除名称。

主要接口可访问 **http://127.0.0.1:8000/docs**。关键端点：`/api/meals/upload`、`/api/meals/{id}/foods`、`/save`、`/retry`、`/view`、`/pick`、`/photo-pick`、`/api/summary?date=YYYY-MM-DD`、`/api/privacy/export`。

## 数据位置与隐私

| 内容 | 正式服务位置 |
|---|---|
| 餐食、偏好、营养、名称、时间、备注 | `data/pansense/pansense.sqlite3` |
| 上传照片或视频 | `data/pansense/uploads/<meal-id>/source.*` |
| 单次分析与日志 | `.../analysis_<uuid>/analysis.log` |
| COLMAP 场景与相机 | `.../analysis_<uuid>/scene/` |
| 掩码与实例名称 | `.../analysis_<uuid>/segmentation/` 或 `photo/` |
| 训练检查点 | `.../analysis_<uuid>/training/` |
| 最终高斯、语义、体积 | `.../analysis_<uuid>/bundle/` |

验证使用独立的 `data/pansense-validation`。原 POC `data/outputs/fruit2_bundle` 与水果1资产继续保留。删除应用示例记录不会删除它们。

个人中心可导出记录/偏好 JSON、删除单条餐食或确认删除全部数据。导出不包含照片与三维模型，也不包含本机绝对路径。照片和模型请另行备份相应目录。关闭“保存餐食照片和三维资产”后，新保存的餐食移除其上传/分析文件，仅保留文字和营养；已有记录不会自动删除。

这是单用户本机服务，数据存储在服务电脑；没有云账户同步。普通 SAM 2 / CLIP 识别使用本地模型，首次获取开源权重可能联网；选择大模型识别或背景处理时，关键视角图像会提交给配置的视觉模型服务。全部删除会清理数据库记录与应用持有的上传资产、重置偏好；手工导出的副本不在其清理范围内。详见[技术文档](../docs/PanSense_技术文档.md#34-在照片中给物体分配身份)。

## 验证与调试

40 项测试在 Python 3.10 的实际重建环境通过，包括原 POC 29 项回归。执行时使用未占用的新临时目录：

```powershell
$env:PYTHONUTF8 = '0'
$env:PYTHONIOENCODING = 'utf-8'
& $gsPython -m pytest -q -p no:cacheprovider --basetemp .cache/test-app-new
node --check app/web/app.js
node --check app/web/sw.js
```

隔离验收服务运行时提交真实素材（会创建草稿与新分析目录，不会清除旧资产）：

```powershell
& $gsPython tools/pansense_smoke.py --url http://127.0.0.1:8001 --kind photo --input 'data/scenes/fruit2_video_ready/frames/000015.jpg' --output data/outputs/photo-check.json --wait
& $gsPython tools/pansense_smoke.py --url http://127.0.0.1:8001 --kind video --input 'data/raw/videos/水果2.mp4' --output data/outputs/video-check.json --wait
# 查看已有任务，不重新训练：
& $gsPython tools/pansense_smoke.py --meal-id <meal-id> --output data/outputs/job-check.json --wait
```

查看 `/api/health` 判断服务状态，餐食的 `analysis.stage/progress/error` 判断阶段。诊断留在本机 `analysis.log`；质量报告位于 `evaluation/report.json`。显存不足可关闭原 Gradio/GPU 任务、降低 `configs/pansense.yaml` 的 `fusion_width` 后重启应用并重试。训练和渲染版本必须与现有约束保持一致。

## 已知范围

当前食物目录有 700 项，但部分项目只有名称和识别词，缺少已核对的营养资料。CLIP 名称只是候选匹配，桌面反光或复杂菜肴可能误识别。缺失的资料可通过后台食品查询补充；油、酱汁、烹饪方式和混合配料仍可能未被计入。

照片记录没有实测克重，参考份量应标明“未称重”；三维记录只有在定位板、体积和密度资料都可用时，才会估算重量。单张照片缺少位姿和多视角，不会生成新的可旋转三维模型。

当前已在本机真实运行识别和视频流程，手机尺寸布局已检查。实体手机相机、独立安装、HTTPS 部署和多设备使用仍需后续验收。
