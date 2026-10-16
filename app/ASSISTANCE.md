# 食物辅助识别：三种可比较的版本

三种方式共用原有预训练模型与数据接口，不训练新模型。对比页比较二维分割结果；选择版本后，如果该照片关联了已有多视角重建，会传播选定的实际掩码、生成独立三维语义副本，并提供可拖拽、缩放、点击的模型页面。原有模型不被覆盖。

## 启动与使用

```powershell
# 独立对比数据，不混入现有餐食日记
& tools/start_app.ps1 -Port 8003 -DataDir data/pansense_assistance
```

打开 http://127.0.0.1:8003/recognition.html 。也可从 PWA “记录”页折叠的“示例与识别对比工具”进入对比页，随后选择输入照片。

1. 上传照片，或选择“使用当前三维模型的参考照片”。这张示例来自 30 张独立重建的 `000018.jpg`。
2. 选择“食物清单”“文字定位”或“点击 / 框选”。同一个输入照片会复制到每次独立运行目录，以 SHA-256 记录一致性。
3. 运行；结果显示分割掩码、Instance ID、名称、耗时及拒绝候选数量。点击结果中的食物查看名称，点击背景不会返回食物名。
4. 结果下拉菜单可查看旧运行。“使用这次输入”恢复清单或标注，便于修改后重试。“下载 JSON”保存结果与诊断。“选这一版”保存偏好并为关联场景排队生成三维语义。
5. 生成完成后，选定卡片出现“拖拽查看这一版的 3D 模型”。进入后按住画面旋转、滚轮或按钮缩放，点击食物显示 Instance ID 和名称，背景不返回食物名。可恢复拍摄视角或切换到其他拍摄相机。
6. 已选过的旧结果可点击“生成并查看三维模型”。失败时保留二维结果，并可重试；重复点击不会同时启动同一个结果的多个三维任务。
5. 可额外运行无提示基线。错误清单和不完整清单也建议尝试。

中文常用名称有本地英文映射；未知中文不会偷偷发送给翻译接口，可写成 `自定义名称|English description`，例如 `紫色圆果|purple fruit`。仅填写类别，多个同类对象仍可产生独立 ID。手工指定名称无需英文翻译。

## 三种实现

| 方式 | 定位 / 分割 | 名称来源 | 额外模型 |
|---|---|---|---|
| list | SAM 2 自动掩码 | CLIP 对比候选清单 | 无 |
| grounded | Grounding DINO 文字定位 → SAM 2 框提示 | CLIP 核对候选清单 | 官方 Grounding DINO Tiny |
| manual | 用户归一化点 / 框 → SAM 2 真正分割 | 用户指定；不被 CLIP 覆盖 | 无 |

CLIP 比较时保留背景、盘子、桌面、未知和目录中其他食物作为拒绝竞争项。如果最优类别不在输入清单中，区域会被拒绝；不能因为只输入“面包”，就将所有水果强制改成面包。边缘、低相似度、低分差及重复区域仍会过滤。这里采用保守的原有阈值；没有用概率解释 CLIP 分数，也没有宣称阈值已做统计校准。

Grounding DINO 每个候选独立查询，按分数做跨类别框去重，然后交给 SAM 2。模型使用 Transformers 官方实现，关闭需编译的定制 CUDA 算子，适配本机 Windows。会检查实际后处理 API，兼容 `threshold` 与旧 `box_threshold` 参数。模型可能找不到小物体，CLIP 核对还可能拒绝它；JSON 的 `candidate_diagnostics` 记录定位提示、分类分数和拒绝原因。

固定实例 ID 在选定结果、其传播序列和对应三维副本中一致。三维任务以选定结果保存的实际掩码初始化 SAM 2，不重新运行一次自动检测，也不让 CLIP 改写已选名称。独立对比运行之间不假定 ID 相同。

模型绑定使用当前餐食的实际 bundle、原始场景、参考相机与照片，不再只支持固定的 30 张水果模型。绑定验证标准化照片 SHA-256、源文件、参考帧图像路径和模型哈希；重新识别创建的草稿保留该绑定。旧对比页的 30 张模型参考照片 `000018.jpg` 仍兼容。普通单独上传的其他照片不会关联到这套水果模型，需要先提供对应视频/多视角照片并重建。

三维副本只复制固定高斯几何和相机，另建语义结果：SAM 2 在 30 个视角传播，27 个投票视角参与可见性融合，3 个保留视角检查一致性。三维点击仍读取高斯渲染的语义像素，不能用二维框或掩码叠加代替。保留视角曾参与 RGB 重建训练，因此该指标是语义一致性检查，不是独立新场景准确率。

自定义食物不在营养目录时，`catalog_key=null`，营养保持待补充。

## 依赖与权重

保留原有两套环境。SAM 2、CLIP、Grounding DINO 在 `.venv`；PWA、渲染、Splatfacto 在 `.venv-gs`。不要将新识别依赖安装到重建环境。

```powershell
uv pip install --python $recognitionPythonExe -e '.[grounding]' -c environments/recognition-constraints.txt
& $recognitionPythonExe -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny', revision='a2bb814dd30d776dcf7e30523b00659f4f141c71', cache_dir='.cache/models/huggingface', allow_patterns=['*.json','*.safetensors','vocab.txt'], max_workers=3)"
```

已在本机验证 Transformers 5.15.0、Torch 2.11.0 + CUDA。权重来自 [IDEA-Research 官方模型](https://huggingface.co/IDEA-Research/grounding-dino-tiny)，固定 revision；加载使用本地缓存，不上传照片。当前机器已完成下载。

## 命令行与视频传播

配置示例：

```json
{"mode":"list","food_text":"香蕉、青苹果、李子"}
```

`mode` 改为 `grounded` 即可测试文字定位。第三种方式使用点或框，每个实例唯一 ID。坐标按参考照片归一化到 0–1：

```json
{"mode":"manual","instances":[
  {"instance_id":1,"name":"香蕉","point":[0.55,0.58]},
  {"instance_id":2,"name":"绿色圆果（待核对）","box":[0.43,0.15,0.57,0.42]},
  {"instance_id":3,"name":"紫色圆果（待核对）","box":[0.60,0.05,0.74,0.30]}
]}
```

另有可编辑配置：`configs/assistance_list.json`、`configs/assistance_grounded.json`、`configs/assistance_manual.json`。手工示例坐标仅适用于当前参考照片，其他照片必须重新标记。

```powershell
& $recognitionPythonExe -m pansense.photo --input 'data/outputs/fruit2_view_counts_run01/n30/independent_colmap_scene/images/000018.jpg' --output data/outputs/assist_photo_list --catalog app/catalog/foods.json --assistance configs/assistance_list.json
& $recognitionPythonExe -m food3d segment --scene data/outputs/fruit2_view_counts_run01/n30/independent_colmap_scene --output data/outputs/assist_video_list --reference-frame 000018 --assistance configs/assistance_list.json
# 后续使用原有 fuse 命令；应创建新的 bundle 副本保存不同版本，不能覆盖当前模型。
```

PWA 三维记录支持三种辅助方式与无辅助基线。视频人工点框先点击“准备标注参考帧”，执行抽帧及 COLMAP，在已注册、去畸变的参考帧上标注，再继续 SAM 2 传播、Splatfacto 和融合；后续复用同一场景与参考相机。准备好的视频可从未保存草稿继续。独立对比页三种方式都能直接操作照片。

## 餐食应用内使用

PWA 记录页分为“照片记录”“三维记录”和“调整已有餐盘”。三处统一显示“食物清单分割”“名称定位分割”“点击 / 框选分割”“不使用语义分割”。最后一项真正跳过分割：照片保留原图并手工填写清单，视频只重建三维外观，已有餐盘直接调整原清单。旧版 `auto` 自动分割仍兼容历史任务，但不再作为“不使用”的选项。历史列表显示已保存记录和草稿，不因没有三维模型或缺少分割素材而隐藏；自定义数据目录的上传模型也能校验绑定。点框模式中每个区域独立编号；结果页提供语义叠色图、实例 ID 与区域点击。已有对比结果可通过折叠工具“导入已有识别结果”或对比卡片“用于餐食记录”进入餐食草稿，已有三维关联一同保留。

本次流程接入通过完整 77 项回归及 2 项新增纸片/人工 ID 流程检查（合计 79 个不同测试）。视频准备、继续、版本冲突、参考帧复用和纸片开关使用隔离测试数据及模拟外部命令验证；浏览器检查使用真实历史餐盘和本地视频选择，未因本次界面改版重跑完整 GPU 训练。

已有模型的结果页和三维页均提供“输入食物，辅助此三维餐盘”。提交食物清单、名称定位或点框命名后，应用自动把这次实际分割掩码传播到同一模型的其他视角并融合，完成后自动切换到三维视图。新名称与实例 ID 直接用于高斯语义点击，无需另按“生成三维关联”。保存为新草稿，原餐食与原三维模型保留。

未关联模型的照片仍可“调整识别 / 重新标注”，只生成二维结果。照片、掩码与 API 结果不写入离线缓存；应用外壳更新时重新验证脚本和样式，避免混用旧版。

新增接口：`GET /api/recognition/results`、`GET /api/recognition/reference-photo`、`POST /api/meals/from-recognition`、`POST /api/meals/{id}/recognize`、`GET /api/meals/{id}/semantics`、`POST /api/meals/{id}/semantic-3d`。

## 验证与输出

- 完整回归 69 项通过；新增测试验证任意已有餐盘模型的参考相机绑定、照片/模型不匹配拒绝，以及示例重新分割后自动排队融合并保护原模型。其他检查覆盖导入幂等、固定 ID、原餐食保护、错误标注拒绝、标签图背景过滤、隐私清理及缓存更新。
- 当前参考照片三种方式已在真实预训练模型上运行；这是已有渲染素材，并非实物拍摄验证。候选类别仍需用户确认，不以实例数量作为识别正确率。
- 使用三张相邻视角 `000012 / 000018 / 000024` 验证三种初始化及双向 SAM 2 传播。输出位于 `data/outputs/assistance_propagation_smoke_01/`，每种方式都产出相同实例 ID 集合的逐帧掩码；这不代表长视频遮挡场景的身份已全面验收。
- `data/pansense_assistance/pansense.sqlite3` 保存对比会话、结果元数据及选择；`uploads/<run_id>/analysis_*/photo/` 保存 `recognition.json`、掩码和叠色图，日志在对应 `analysis.log`。
- 原始三维模型与语义结果保持不变。三维选定版通过 30 视角传播和独立副本融合验证，不重新训练 Splatfacto。未进行真实食物标签准确率评估。

新增三维任务输出位于 `uploads/<run_id>/three_d_<uuid>/`：包含选定掩码清单 `seeds.json`、传播 `segmentation/`、几何与语义副本 `bundle/`、保留视角检查 `evaluation/` 和来源 `provenance.json`。`/model.html?run=<run_id>` 使用该结果的真实高斯渲染；后端 GPU 队列和查看器共用互斥锁。

传播验证工具：

```powershell
# 请先等待网页 GPU 分析队列结束；输出目录必须未存在。
& $recognitionPythonExe tools/assistance_smoke.py --scene data/outputs/fruit2_view_counts_run01/n30/independent_colmap_scene --output data/outputs/assistance_propagation_smoke_new --manual configs/assistance_manual.json
```

不要同时让多个本机服务运行 GPU 分析。一个 PWA 服务内的任务串行执行；已有三维查看器可保持打开，但建议分析期间暂不连续渲染。

## 拖拽响应

三维页面按固定 65 ms 间隔采样最新视角，同一时间只发送一次渲染请求；渲染期间仍接收手势，不积压中间视角。鼠标连续移动不会重置等待时间。图片预加载完成后再更换显示帧和对应的点击令牌。

曾出现的卡顿来自鼠标每次移动都重新等待 80 ms，以及渲染后额外等待 150 ms、忙碌时拒绝新拖动。已移除这些等待。本机六次热渲染测得后端请求约 30–36 ms，获取 JPEG 约 5–6 ms；一次连续拖动验证产生 3 次视角更新，旋转后仍能正确点击 Instance 2。65 ms 是调度间隔，不代表保证帧率；当前仍采用本机后端渲染后发送图片，尚非浏览器原生实时渲染。
