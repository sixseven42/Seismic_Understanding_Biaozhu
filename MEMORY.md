# MEMORY — 版本迭代记录

> 记录每次迭代的版本号、改动内容、原因与遗留问题。新版本追加在最上方。

---

## v3.5（2026-09-08）— 建作业支持「道数下限」过滤

**需求**（用户）：建作业时指定一个道数下限，道数少于该值的道集从作业池滤去、不标注。

**内容**：
- `jobmanager.py`：`JobManager.create_job(..., min_traces=0)` 抽道集后按 `g.n_traces >= min_traces` 过滤（`_filter_min_traces`）；`min_traces` 写入 job.json meta，`load_all()` 启动恢复时按同阈值再次过滤，保证重启后池一致；过滤后为空则抛错。0=不过滤（向后兼容）。
- `web_app.py` ①建作业：新增数字框「道数下限 N（道数 < N 的不参与标注，0=不限）」，传入 create_job_ui → create_job；成功提示显示“道集数 n（已滤去 m 个道数 < N）”。
- 进度/任务池 total 均以过滤后的池为准；不影响既有标注/删除/进度。
- 测试：`TestCreateMinTraces`（变道数 sgy 真链过滤 2/8→留 8；0=全保留；阈值过高抛错；恢复仍过滤；`_filter_min_traces` 单测）；全量 **45 项通过**。

**注意**：重启 `python web_app.py` 生效；被滤去的道集不会出现在该作业任务池（本地数据不动）。

---

## v3.4（2026-09-08）— bbox 框选改为「按住拖动实时框」

**需求**（用户）：面波/近炮点强能量噪声的框选目前按对角线两次点击成框；希望改成**鼠标按住、拖动时框实时跟随，松开即确定**。范围：仅现有两个 bbox 特征（面波、近炮点强能量噪声）。

**调研结论**：Gradio 6.26 的图片组件只在真实 `click` 时发 `select`，且实测纯前端 JS **无法驱动其服务端事件**（程序化 button.click、Textbox input/change、合成鼠标事件均不触发后端）→ “松开即落库”需自定义 HTTP 接口。

**内容**：
- 运行方式改为官方 FastAPI 挂载：`gr.mount_gradio_app(FastAPI, demo, path="/", auth=…, css=ANNO_CSS, head=ANNO_JS)` + `uvicorn.run`（原 `demo.launch` 移除）；新增 `POST /api/anno_box`，body `{user,key,x0,y0,x1,y1}`（512×1024 像素），服务端 `apply_drag_box()` 用 `pixel_box_to_data` 写入该用户当前道集 `st["boxes"][key]`（含 traces/samples），保存/校验逻辑不变。
- 前端（`ANNO_JS` head 注入）：在 `#anno_imgcol` 图上叠加透明 canvas；点「▣ 框选」后按住拖动实时画虚线框（颜色用特征 bbox_color），mouseup `fetch` 提交；「✕ 清除」/换道集时复位。移除旧 `cur_img.select` 两点点击；`on_img_select` 删除。
- 其它：`demo.queue()` 开启；`--share` 公网隧道在接口版不启用（提示并仍监听局域网）；框选/清除按钮加 `elem_id`（bb-/cb-前缀）、`who` 标加 `elem_id=who_md` 供 JS 取当前用户；文案改“按住鼠标拖动”。
- 验证：41 项单测通过；接口冒烟（无会话 `ok:false`、未知用户 401、登录 cookie 正常）；`apply_drag_box` 直连成功落库（xyxy 正确、非法 key 拒绝）。注入 css/head 走 mount 官方参数。

**注意**：**重启 `python web_app.py` 生效**；请在浏览器实测拖动画框，若画框层未出现请反馈（供继续排 head 注入）。

---

## v3.3（2026-09-08）— 删除作业（软删除 + 二次确认 + 可恢复）

**需求**（用户）：管理区点「打开/关闭作业」看不到变化（下拉不更新、关闭后仍可看到但无法领取）。建议把该按钮改为**删除作业**：删除后作业变为不可选；**本地结果不删除**。

**用户确认的决策**：软删除 + 管理区提供「已删除作业（本地保留）」可恢复；删除用**二次确认**（首次点→按钮变「再次点击确认删除」，约 3 秒未确认自动复原）；替换原「打开/关闭作业」。

**内容**：
- `jobmanager.py`：作业状态新增 `deleted`；`list_jobs()` **排除已删除**（标注下拉/管理下拉/进度表自动隐藏）；新增 `deleted_jobs()`、`delete_job()`/`restore_job()`（均复用 `set_state` 落盘 job.json）；`claim` 对非 open 返回「作业未开放或已删除」。
- `web_app.py` 管理③区：删除按钮（`delete_job_click` 两段式，`_PENDING_DELETE` 内存态，>3s 由 5s 周期 Timer 自动复原）；新增一行「已删除作业（本地保留，可恢复）」下拉 + `restore_job_click`；删除/恢复即时刷新 mgr_job_dd、del_job_dd、job_dd、进度表。
- 本地 `jobs/<id>` 结果文件一律不删，恢复=把 state 改回 open。
- 测试：`TestDeleteJob`（删除后隐藏且不可领取 / 恢复后可领取 / 目录保留）；全量 **41 项通过**；handler 冒烟（二次确认、删除/恢复、非管理员拒绝）通过。

**注意**：重启 web 服务生效；状态仅内存 + job.json；旧「closed」作业仍可经“删除→恢复”重开（list_jobs 不再提供 toggle）。

---

## v3.2.1（2026-09-08）— 修复：连续「跳过」在两三张之间循环

**现象**（用户）：连点「跳过」时当前张回池、下一张到手，但再点一次又回到刚跳过那张，在两个道集间循环。

**根因**：`JobManager.skip` 的选张逻辑每次都从乱序池**头部**挑“不是刚跳过那张”的第一个可用项；被跳过者往往正处头部，于是 A→B→A→B 打转，永远走不到后面的任务。

**修复**：引入 per-(job,user) 近期跳过队列 `_skipq`：
- `claim` 变为两段式：第 1 轮向该用户发任务时**避开其近期跳过项**；第 2 轮（其余已标/被占/皆为其近期跳过）才回到近期跳过项并消费移除；
- `skip` 先把被跳过的道集放进本用户队列，再走同一 claim 逻辑。

其他用户不受影响；被跳过的道集照常回池、可被他领取；仅剩一张时仍会退回继续做。

**测试**：`TestSkip` 新增 `test_repeated_skips_do_not_oscillate`（4 张连跳 3 次拿到 3 张不同）、`test_skip_then_manual_claim_avoids_recently_skipped`（跳过后再手动领取不会立刻领回）；全量 **39 项通过**。

**注意**：重启 web 服务生效；跳过队列仅存于内存、单进程运行期有效。

---

## v3.2（2026-09-08）— 跳过按钮 + 管理区作业进度实时一览

**需求**（用户）：
1. 标注区新增「跳过」：点击后当前道集**回池**（不写记录、不计完成），后续分给他人；
2. admin 在管理③区能看到**所有作业**的进度一览（已完成多少份、百分比），并**每几秒自动刷新**的进度条。

**用户确认的决策**：跳过 = 回池 + **自动领取下一张**（保留原「归还此张」）；进度展示为「所有作业一览表」而非单作业。

**内容**：
- `jobmanager.py` 新增 `JobManager.skip(job_id, user)`：把 user 当前持有的道集放回池（不写记录），再自动领取下一张；**优先领不是刚跳过的那张**，只剩它一张时退回继续做（避免“有任务却报无任务”）。返回 dict 含 `skipped`。
- `web_app.py`：标注行加「⏭ 跳过此张」按钮 + `skip_current` handler（清空本张未保存选项/框、显示新领取道集、提示已跳过）；`release_current`（归还此张）保留。
- `web_core.py` 新增纯函数 `jobs_progress_html(jobs)`：渲染作业进度表（标题+ID/状态/已完成 x/总数 y/百分比/CSS 进度条），total=0 防除零、标题做 HTML 转义。
- `web_app.py` 管理③区新增 `mgr_progress`（初始渲染一次）+ `gr.Timer(value=5).tick` **每 5 秒自动刷新**（仅 admin 可见区域）。进度 = labels 已标条数 / 池道集总数。
- 测试：`TestSkip`（跳过回池且领另一张 / 仅剩一张退回）、`TestJobsProgressHtml`（数量与百分比 / 100% / 0÷0 / 标题转义）；全量单测 **37 项通过**。

**注意**：重启 web 服务生效；作业标题等字段来自 job.json，进度表随保存自动增长。

---

## v3.1（2026-09-08）— 三类修复：图/标注对齐、标注页整图显示、乱序派发

**背景**：基线为 v3 多用户中央服务（见下节补记）。本次全部针对 Web 标注实际使用问题。

**1）导出图与显示图解耦（images/ ↔ labels.jsonl 一一对应）**
- 起因：UI「显示 / 领取 / 保存后自动领下一张」都会走 `Job.ensure_image`，把**尚未标注**的下一条道集直接渲染进导出目录 `images/<gid>.png` → 出现「标一张却多一张下一道集的图」「有图无记录、有记录缺图」（真实作业里出现过无任何 jsonl 记录的孤立图 `...1399.png`；8 条记录只有 2 张图）。
- 改法：`jobmanager.Job` 新增 `display_image()`：界面显示图只写 `jobs/<id>/.cache/`（不入导出目录；渲染 temp+os.replace 原子写）；`web_app.render_display()` 改走 `display_image`；`ensure_image()` 改为**仅由 `JM.save` 调用**生成导出图。→ 保存一张只出一张本道集导出图，`images/` 恒等于 `labels.jsonl`。
- 配套：已把该作业 7 张缺失导出图从 npy 用同 clip 管线补齐（逐字节一致）、删除孤立 `1399.png`；现 8 记录 = 8 图。
- 测试：`tests/test_jobmanager.py::TestImageExportOnlyOnSave`（显示不写导出图 / 保存只出自己 / 显示图在 .cache）。

**2）标注页整图显示（修 CSS，不用再点「全局/全屏」）**
- 起因：Gradio Image 外壳 `.block` 被内联 `height:640px; overflow:hidden` 锁死；512×1024 竖长图按标注列宽等比显示后高度远超 640，**下半截被裁**（窗口越宽裁得越多），只能点工具栏全屏看全图。
- 改法：`web_app.ANNO_CSS` 改为 `#anno_imgcol > .block{height:100%!important; min-height:0; overflow:hidden}`（压过内联 640px）、`.image-container / .image-frame{height:100%}`、`img{width/height:auto; max-width/height:100%; object-fit:contain; flex 居中}` → 图片区高度 = 标注行高度，**整张始终完整可见**；`<img>` 元素框恰等于所绘图区域（无内层 letterbox），**图上点框的像素坐标映射不受影响**。已用无头 Chrome + DOM 几何实测：宽窗 1680 下图 512×1024 完整显示、底边贴合。

**3）任务派发打乱（缓解标注疲劳）**
- 起因：`claim()` 遍历原始（按数据值升序）`_gathers`，任一标注者长期拿到连续递增的数据段。
- 改法：`Job.__init__` 用 `random.shuffle` 生成派发池 `self._pool`（每次进程内注册/加载随机一次）；`claim()` 只从 `job._pool` 取下一个未标注且未被占用的道集。resume/租约/进度/归还语义不变。
- 测试：`TestClaimShuffledPool`：12 道集由单用户领完 = 全集排列且不再升序。

**注意**：以上改动需**重启 web 服务**生效；全量单测 31 项通过。

---

## v3（当前功能基线，补记）— Web 版重构为多用户中央服务

> 注：MEMORY 自 v2.1 起未记录 v3 大规模重构，现按代码/README 补一份基线，供后续迭代参照。

**架构**：单 Gradio（6.26）进程 `python web_app.py`（默认 0.0.0.0:7860，可用 `启动.bat`），同局域网多人并发标注；**原始 sgy 只留服务机**。核心模块：`web_app.py`（UI+handler）、`web_core.py`（选项/像素↔数据换算纯逻辑）、`jobmanager.py`（作业池/租约/保存协调）、`users.py`+`users.yaml`（账号/角色）、`cloudsync.py`+`cos_config.yaml`（可选 COS 回传）、底层 `segy_reader / gather / preprocess / imaging / labels / storage`。

**账号/角色**：users.yaml 登录时热读（改文件即生效，无需重启，已登录会话不受影响）。默认 `boss/boss123`=admin（建作业、打开/关闭作业、全部重传）、`ann1/ann123`=annotator（只能领标、查看/重开**自己**标过的记录）。Windows 防火墙需放行 7860；**勿多开实例指向同一 jobs/**。

**作业与标注流程**：
- admin「①建作业」：sgy 绝对路径 + 排序键/抽道集键（1-based 字节区间）+ 勾选键值 + clip 分位数（默认 99）→ 服务端抽一次道集 → `jobs/<标题>_<时间戳>/job.json`（含 values/clip/output_dir 等元数据）。
- 标注者「②作业与标注」：选作业 →「领取下一张」即加**租约（TTL 30 分钟，交互自动续期）**，每张同一时刻仅一人持有；「保存并释放」= 写导出图+npy+upsert labels.jsonl、释放租约并**自动领取下一张**；挂机超时自动回池；「归还此张」放弃不落记录；「⧉ 继承最近已标注」回填类别（框不继承）。
- admin「③管理」：全部重传（推 COS）、打开/关闭作业。

**落点与契约**（`jobs/<job_id>/`）：
- `labels.jsonl`：每行一条记录（gather_id/gather_key/value、n_traces、labels 8 特征→label、sentence、regions、image_path、npy_path、sort_keys、extract_key、source_file、annotated_by、timestamp），原子全量重写；
- `images/<gid>.png`：**512×1024 纯数据图，仅在保存时生成**（v3.1 起），`images/` 应恒等于 labels；
- `npy/<gid>.npy`：原始 float32 (道数,采样数)，**只留本地不上云**；
- `.cache/<gid>.png`：界面显示缓存，非结果、不上传；
- `regions`：`{feature_key: {xyxy:[512×1024 图像像素,含端点], traces/samples:[原始数据半开区间]} 或 null}`；显示图与导出图同 clip 管线、内容一致（实测逐字节相同）。bbox 特征：面波、近炮点强能量噪声（标签非「不存在」必须画框）。

**云回传（可选）**：`cos_config.yaml` 置 `enabled:true` 并填密钥后**重启**生效；需 `pip install cos-python-sdk-v5`；对象键 `seismic/<job>/images/<gid>.png`、`seismic/<job>/labels.jsonl`；失败有界重试并写 `upload.log`，可用「全部重传」；未启用自动本地降级。

**测试**：`tests/` 用合成 sgy（`segy_factory.write_sgy`）驱动 unittest；`python -m unittest discover -s tests`，当前 31 项全绿。

**关键运行信息**：中文字体 `fonts/NotoSansCJKsc-Regular.otf` 勿删（出图中文）；导入 matplotlib 前 `MPLCONFIGDIR` 指到可写临时目录（imaging/web_app 已处理）；道头字节号按 1-based；SEG-Y 端序自动检测。

---

## v2.1 — 继承改为「最近已标注」+ 清空旧标注重标

**需求**（用户）：
1. 跳过/未标的道集没有标注结果可继承，「继承上一张」应回找最近一张**已标注**的道集；
   框不要继承
2. 曾应用户要求加过「剔除」按钮（excluded 标记、不计入进度），随后用户确认
   **跳过已够用，剔除功能移除**（不保留代码）
3. 因版本改动（标签体系 v1.9 + regions v2.0），**清空 output/ 旧标注结果重标**：
   948 条记录、790 张图、158 个 npy（约 2.1GB）连同 labels.jsonl.v18.bak 全部删除

**内容**：
- `inherit_previous` 重写：从当前道集向前找最近一条**有 labels 的已存记录**，
  跳过中间未标注道集（提示"跳过中间 N 张"）；只填类别选项，框不继承；
  前面一张已标注都没有时保持当前已选不变
- 按钮文案改为「⧉ 继承最近已标注」
- 测试（test_output/bbox_check/run_test.py）：继承跳过未标注道集 ✔ 框不被继承 ✔
  第一张无可继承 ✔ v2.0 全部框选用例回归 ✔

**注意**：output/ 已是空目录，下次标注从 0 开始；新记录含 regions 字段（v2.0 格式）。

---

## v2.0 — 面波/近炮点强能量噪声 区域框选标注（Web 版）

**需求**（用户）：面波、近炮点强能量噪声除类别外还要标注范围（矩形框），供检测训练。

**用户确认的决策**：每特征**一个**框；标签为「存在/已压制但有残留」时**强制**画框才能保存，
「不存在」时框自动置空；只做 Web 版（Qt 版不动）；「继承上一张」不复制框（框是道集专属）。

**内容**：
- `label_config.yaml`：面波、近炮点强能量噪声加 `bbox: true` + `bbox_color`（红/黄）；
  `labels.py` 的 Feature 解析这两个字段
- 交互（无新依赖）：`gr.Image.select` 两次点击取对角 → 矩形。
  **已查证 gradio 前端源码**（templates/frontend/assets/ImagePreview-*.js）：
  select 的 index 按 naturalWidth/naturalHeight 换算，即**原图像素**坐标（512×1024 空间）
- JSONL 新增 `regions` 字段：`{feature_key: {"xyxy":[像素], "traces":[半开区间],
  "samples":[半开区间]} 或 null}`；traces/samples 可直接切 npy；增强记录经 `{**base_rec}` 继承
- `imaging.py` 新增 `overlay_boxes()`：显示图叠加框+特征名（PIL），**导出训练图保持干净**
- `web_app.py`：Session 增 box_mode/pending_corner/box_partial；保存校验强制画框；
  框随道集切换从记录/暂存恢复
- 测试（test_output/bbox_check/run_test.py，24 项全过）：
  不画框保存被拒 ✔ 画框保存+坐标换算（像素→道/采样）✔ 增强记录继承 ✔
  改标「不存在」框置空且增强同步 ✔ 导航恢复 ✔ 清除 ✔ 导出图无框 ✔

**注意**：948 条旧记录无 regions 字段（缺省视为未画框），重新保存旧道集时按新规则补画。

**踩坑（v2.0.1 修复）**：Gradio 事件返回 `None` 会把输出组件**置空**而非"保持不变"
（点第一个角点后图片消失即此原因）——保持原值须返回 `gr.skip()`。
已统一替换 web_app.py 中所有"不更新图片"的返回路径。

---

## v1.9 — 标签体系删除「线性噪声」「多次波」

**需求**（用户）：标注特征中去除线性噪声、多次波两项。

**用户确认的决策**：已有 948 条记录**同步清理**——删除两个字段并重新生成 sentence。

**内容**：
- `label_config.yaml`：删除两个特征及句子模板中的对应片段，特征数 10 → 8
- 新增 `migrate_remove_features.py`：按当前配置重建每条记录的 labels
  （只保留配置中的键、按配置顺序）并用 `render_sentence` 重新生成 sentence；
  幂等，首次运行自动备份原文件为 `output/labels.jsonl.v18.bak`
- 代码零改动：Web/Qt 界面布局均按 `len(CFG.features)` 动态生成
  （web_app.py 两列分列 `half=(n+1)//2`），README JSONL 示例同步更新
- 执行结果：948 条全部清理（158 基础 + 790 增强数量不变），labels 键剩 8 个，
  sentence 无旧特征词残留；重跑幂等（0 改动）✔；web_app/main_window 加载 8 特征 ✔

---

## v1.8 — 已标数据增强回补 + clip 采样去重修复

**需求**（用户）：把已标注的 158 条按增强方式改造——原单图删除，换成 5 张采样图；
clip 范围：01 开头文件 95~99.9，02/03 开头文件 90~99.9。
用户确认：每个样本的 clip 独立重新随机采样 ✔

**内容**：
- 新增 `backfill_augment.py`：从各记录 npy 读原始数据 → 按文件前缀选 clip 范围 →
  采 5 个 clip 值渲染 5 张增强图 → 每张一条独立记录；删原基础图、基础记录
  image_path 置 null；幂等（重跑先清旧增强）
- **踩坑**：`uniform + round(0.1)` 采样会在同一道集内撞值（如两个 95.5），
  导致增强图互相覆盖（第一次跑出 790 条记录却只有 771 张图）
  → 修复：新增 `preprocess.sample_clip_values()`，0.1 步长网格**无放回采样**，
  同步应用到 backfill_augment.py / web_app.py / main_window.py
- 执行结果：158 基础 + 790 增强 = 948 条记录；790 张图全在；
  clip 范围按文件校验 ✔；每个道集 5 个 clip 互不相同 ✔；无残留基础图 ✔

---

## v1.7 — clip 随机采样增强保存

**需求**（用户）：同一道集不同 clip 得到的图片标签一致；保存时给定 clip 范围
（如 80-100）和张数 N，随机采样 N 张不同 clip 值的图片保存。

**用户确认的决策**：
- 每张增强图 = labels.jsonl 中一条独立记录（labels/sentence 与基础记录相同，
  gather_id 带 `__clip<值>` 后缀，另含 `augmented_from`、`clip_percentile` 字段）
- **只存增强图，不存基础图**；基础记录仍写入（维持断点续标），其 image_path 为 null
- npy 仍只存一份原始数据（增强图随时可由 npy 重新生成）

**内容**：
- 第 2 步界面新增：增强 clip 下限/上限（默认 80/100）、增强张数 N（默认 0=不增强）
- 保存流程（web_app.py / main_window.py 同步实现）：
  N>0 时从 [下限,上限] 均匀随机采 N 个 clip 值（保留 1 位小数）→ 原始数据分别
  clip 渲染 → `images/<gather_id>__clip<值>.png` → 每张一条记录
- 重复保存同一道集：先删除该道集旧增强记录与图片（storage.remove_augmented +
  glob 清理 `__clip*.png`），再写入新采样，防止堆积
- 测试：5 张增强图全部落盘（512×1024、clip 值在界内、labels/sentence 与基础一致）✔
  重复保存清理 ✔（重存后仍恰好 5 条增强记录 + 5 张图）

---

## v1.6 — 保存时同步存原始数据 npy + 已标数据回补

**需求**（用户）：保存时把该道集原始数据存为 npy 便于数据增强；
并回补已标注的 158 条记录。

**用户确认的决策**：
- npy 存**原始数据，不做任何预处理**（float32，(道数, 采样点数)）
- 历史记录排序键：95-96 批 =「95-96, 13-16」；197-200 批 =「197-200, 25-28」

**内容**：
- `web_app.py` / `main_window.py` 保存时同步导出 `npy/<gather_id>.npy`（原始振幅），
  JSONL 记录新增 `npy_path`、`sort_keys`、`extract_key` 三个字段（保证可复现）
- 新增 `backfill_npy.py`：按记录中的 source_file + 排序键映射重抽道集回补 npy，
  含一致性校验（gather_id 与道数必须吻合），幂等可重跑，原子重写 jsonl
- 回补执行：`output/labels.jsonl` **158 条全部成功**，`output/npy/` 共 1.2GB；
  抽检 3 条形状正确 + 首条与实时抽取逐点一致 ✔
- 新保存流程 e2e 测试通过（npy_path/sort_keys/extract_key 落盘）✔

**数据规模备注**：02WL 文件道集 821 道 × 3501 采样；03XCN 287 道 × 3501；
01SB21 528 道 × 4000（端序均自动检测成功）。

---

## v1.5 — 标注区排版优化（左图右栏、按钮沉底）

**需求**（用户）：标注栏太长，看图和点选要来回滚动。

**内容**（web_app.py 第 3 步布局重排）：
- 左侧（约 60% 宽）：当前道集图（高度 640）
- 右侧（约 40% 宽）：句子预览 + **10 个特征分两列**（各 5 个）排布，大幅压缩纵向长度
- 底部一整行按钮：上一张 / 跳过 / 下一张 / 继承上一张 / 保存并下一张
- radios 列表顺序仍与 features 一致（先左列后右列），事件映射不受影响
- 回归测试通过（保存/继承/JSONL 字段）

---

## v1.4 — 多文件合并标注到同一项目目录（防冲突）

**需求**（用户）：删除旧输出；每次标注量有限，希望指定同一输出文件夹时
新旧标注整合到同一个项目中。

**现状梳理**：输出文件夹本就可指定；LabelStore 加载已有 labels.jsonl + upsert
本就支持跨会话合并（断点续标机制）。**真正缺口**：不同 sgy 文件的 gather_id
（`键_值`）相同会互相覆盖记录与图片。

**内容**：
- `gather.py`：`Gather` 新增 `prefix` 字段，gather_id = `{prefix}{键}_{值}`
- `web_app.py` / `main_window.py`：生成道集时 prefix 自动设为
  **来源文件名（去扩展名）+ "__"**，如 `01SB21_DenoiseB4_FFID19037__95-96_1373`；
  图片文件名同步携带前缀
- 清理：删除旧 test_output/、output/、__pycache__/
- 测试（用 300MB 副本模拟第二文件，测后已删）：
  文件A、文件B 标到同一目录 → 两条记录 + 两张图片互不覆盖 ✔；
  重开文件A → 正确识别已标 1/36 续标 ✔

**行为总结**：
- 同一输出目录 + 同一文件 → 跨会话合并、续标、可改标（原已有）
- 同一输出目录 + 不同文件 → 记录与图片按文件名前缀区分，合并共存（本次新增）

---

## v1.3 — 标签体系改版 + 继承上一张功能

**需求与决策**（用户确认）：
1. 标签特征改为 10 项：集合类型、静校正、直达波、线性噪声、面波、异常振幅、
   50Hz工业噪声、近炮点强能量噪声、多次波、混叠噪声
   - 「面波」替换旧「面波噪声」，「50Hz工业噪声」替换旧「工业噪声」，删除「随机噪声」
   - **全部不设「不确定」选项**（用户：不确定标签不利于分类任务）；
     噪声类统一三项：存在/不存在/已压制但有残留
2. 新增「继承上一张的选项」按钮（Web 按钮；Qt 按钮 + 快捷键 I）：
   把上一张道集（已存记录优先，其次暂存）的选项填充到当前张，仅填充不保存；
   上一张无可继承时保持当前已选不变

**内容**：
- `label_config.yaml` 全量改写（10 特征、新句子模板、静校正用 phrase 拼接）
- `web_app.py` / `main_window.py` 新增 inherit_previous
- e2e 走查通过：10 特征保存（JSONL+句子）✔ 10 特征继承 ✔

**句子示例**：这是一条炮集，需要静校正，直达波存在，线性噪声不存在，面波已压制但有残留，异常振幅不存在，50Hz工业噪声存在，近炮点强能量噪声不存在，多次波不存在，混叠噪声不存在。

**注意**：旧版 labels.jsonl 记录的 labels 键（random_noise 等）与新配置不兼容，
旧记录在新界面中回显为空——旧的测试输出建议删除后重标。

---

## v1.2 — 导出图片改为纯数据图（宽512×高1024）

**需求**（用户）：生成的图片不要横纵坐标、标题等，只要数据；固定**宽 512 × 高 1024** 像素
（曾误做成 1024×512，用户指正后修正）。

**内容**：
- `imaging.py` 新增 `render_data_only()`：colormap 映射 → PIL 缩放 → 精确像素输出，
  无坐标轴/标题/边框；时间轴从上到下、道轴从左到右；默认 size=(512, 1024)（宽×高）
- `web_app.py` 与 `main_window.py` 的保存导出均改用 `render_data_only()`
- **界面显示（Web 预览/标注页、Qt 画布）也已改为无坐标轴/标题的纯数据图**
  （v1.2 初版曾保留坐标轴，用户确认后统一去除；Qt 侧用 `draw_gather(bare=True)`）
- 验证：直出 PNG、Web 端保存链路、Web 标注页显示图均为 512×1024（宽×高）RGB 纯数据图 ✔

**设计决策**：显示预览与训练用导出解耦——预览走 matplotlib（可缩放、有坐标），
导出走 PIL 精确像素，二者互不影响。

---

## v1.1 — Web 版上线（Gradio）

**背景**：用户在 Windows 本地通过 VNC/X11 使用 Qt 桌面版屡遭连接问题
（-R/-L 隧道方向混淆等），确认改用 Web 方案。

**内容**：
- 新增 `web_app.py`：Gradio 三步式界面（文件与道集 → 预处理 → 标注），
  复用全部核心模块；功能与桌面版一致（断点续标、保存导 PNG、句子预览、
  返回修改、跳过）；**无数字快捷键**（Web 版限制，用鼠标点选 Radio）
- gradio 6.25.0 安装于项目 `_vendor/`（系统与 conda site-packages 均只读，
  连 `/data/liuqi` 都不可写，`pip --target` 是唯一途径）
- 服务：`python web_app.py --port 7860`（0.0.0.0，校园网可直连）
- 端到端 API 测试（gradio_client）通过：加载→扫描36键值→生成道集→
  开始标注→保存（JSONL+PNG）✔

**踩坑记录**：
- `_vendor` 必须放 `sys.path` **最前**：conda 环境里的旧版 starlette 会让
  gradio 6 报 `HTTP_422_UNPROCESSABLE_CONTENT` 属性错误；
  _vendor 中 numpy 与环境同为 2.2.6，shadow 无害；matplotlib/segyio 不在其中
- 无头浏览器不可用（无 chromium/playwright），界面截图未做，由用户浏览器直接查看

**遗留问题**：同 v1.0（标签清单/句子模板待用户确认，改 label_config.yaml 即可）。

---

## v1.0.1 — 补充远程无显示环境运行说明

**背景**：用户在 SSH 服务器上直接 `python main_window.py` 报
`qt.qpa.xcb: could not connect to display`（服务器无 X 显示，`DISPLAY` 为空）。

**内容**：
- 实测 Qt **VNC 插件可用**：`python main_window.py -platform vnc:size=1500x950`
  输出 `QVncServer created on port 5900` ✔
- README 新增「远程服务器运行」一节：
  - 方案 A（推荐）：VNC 插件 + SSH 隧道 `ssh -L 5900:localhost:5900` + 本地 VNC 客户端
  - 方案 B：X11 转发（服务器 xauth 已装；注意 tmux 内 `DISPLAY` 失效需重新 export）

---

## v1.0 — 步骤 4/5 完成：GUI 整合与全流程走查（首个可用版本）

**内容**：
- 新增 `main_window.py`：PyQt5 向导式三步界面（文件与道集设置 → 预处理 → 标注）
  - 标注页：matplotlib Qt 画布（工具栏缩放/拖动）+ 右侧特征单选面板
  - 数字键选当前高亮特征选项并自动跳下一特征；←/→ 切换；S 跳过；Enter 保存并下一张
  - 保存时导出 PNG 到 images/；进度条；断点续标；已标可返回修改；未完成选择会话内暂存
- `imaging.py` 重构：抽出 `draw_gather(ax,...)` 供 Qt 画布复用；
  **移除强制 Agg 后端**（否则 GUI 画布无法工作；无显示环境 matplotlib 会自动回退 Agg）
- 新增 `README.md`（运行方法、使用流程、输出格式、配置说明）
- 离屏端到端测试（QT_QPA_PLATFORM=offscreen）全部通过：
  加载 → 扫描 36 键值 → 生成道集 → 数字键标注 → 保存（JSONL+PNG）→
  自动跳下一未标 → 新窗口续标定位 → 回读已标选项 → 修改保存
- 界面截图 `test_output/screenshot_label_page.png` 已给用户确认

**踩坑记录**：
- 离屏测试必须 monkeypatch QMessageBox 静态方法，否则模态框阻塞
- imaging.py 若 `matplotlib.use("Agg")`，Qt 画布会失败——后端选择留给入口

**遗留问题**：
- 标签清单（随机/工业噪声、「不确定」选项）与句子模板措辞待用户最终确认，改 `label_config.yaml` 即可
- 预处理目前仅 clip_percentile/normalize，用户提到"功能可能拓展"——用 `@register` 添加

---

## v0.4 — 步骤 3 完成：标签体系 + 存储

**内容**：
- 新增 `label_config.yaml`：5 个特征（集合类型/静校正/面波噪声/随机噪声/工业噪声），
  每特征含「不确定」选项；**每个选项支持 `phrase` 字段**（拼句子措辞与界面显示分离，
  保证"不确定"等选项也能拼出通顺句子）；句子模板可配置
- 新增 `labels.py`：配置加载、选择校验（缺项/非法项）、句子渲染、JSONL 字段转换
- 新增 `storage.py`：JSONL 存储，upsert 后全量原子重写（防崩溃丢数据），
  支持修改已标记录、断点续标（next_unlabeled）
- 测试通过：句子示例「这是一条炮集，需要静校正，面波噪声已压制但有残留，随机噪声存在，工业噪声不存在。」

**设计决策**：句子模板用 `{特征显示名}` 占位；选项缺省 phrase = label。

**遗留问题**：标签清单与句子模板措辞仍待用户最终确认（配置文件可直接改）。

---

## v0.3 — 步骤 2 完成：排序/抽道集 + 预处理 + 出图

**内容**：
- 新增 `gather.py`：多道头稳定字典序排序（np.lexsort）、按键道头抽道集
  （自动全部唯一值 / 手动指定值），键字符串解析（"95-96"）
- 新增 `preprocess.py`：注册表模式管线，初版含 clip_percentile、normalize；
  之后扩展新方法只需 `@register` 装饰器
- 新增 `imaging.py`：seismic 红-白-蓝配色出图，自动纵横比，中文标题/坐标轴
- 链路验证：示例文件 → 36 个炮集 → 炮集 1373（528 道）→ 99% clip →
  `test_output/shot_1373.png`，初至双曲线清晰 ✔

**踩坑记录**：
- matplotlib 后端**不做逐字符字体回退**，font.sans-serif 列表只用第一个字体
- 系统 `DroidSansFallbackFull.ttf` 缺拉丁字母 'l'（cmap 确认），不可用
- **解决**：下载 `fonts/NotoSansCJKsc-Regular.otf`（16MB，中英覆盖完整）到项目目录，
  imaging.py 优先注册它并置于 sans-serif 首位
- matplotlib 配置目录不可写问题：imaging.py 导入前设置 `MPLCONFIGDIR` 到临时目录

**遗留问题**：标签清单与句子模板仍待用户最终确认。

---

## v0.2 — 步骤 1 完成：SEG-Y 读取层

**内容**：
- 新增 `segy_reader.py`：numpy memmap 读取，支持大端/小端自动检测，
  格式码 1(IBM float)/2(int32)/3(int16)/5(IEEE float32)/8(int8)，
  1-based 任意字节区间道头读取，IBM→IEEE 浮点转换
- 用示例文件验证通过：
  - 端序自动检测 = little ✔
  - 19008 道 × 4000 采样点、dt=2ms、格式码 5 ✔
  - 字节 95-96 → 36 个炮号（1373~1793），炮集 1373 含 528 道 ✔
  - 道数据 float32 振幅范围约 ±110，数值正常 ✔
  - EBCDIC 文本头解码正常 ✔

**遗留问题**：无新增；标签清单与句子模板仍待用户最终确认（见 v0.1）。

---

## v0.1 — 项目启动（计划建立）

**日期**：项目初始化

**内容**：
- 与用户完成需求确认（8 项关键决策，详见 PLAN.md 第 2 节）
- 诊断示例文件 `01SB21_DenoiseB4_FFID19037.sgy`：
  - **关键发现：该文件为小端序 SEG-Y**（标准应为大端），segyio 无法读取
    → 决策：自实现 numpy memmap 读取层，支持端序自动检测
  - 19008 道 × 4000 采样点，float32，dt=2ms
  - 字节 95-96：36 个唯一值（炮号 1373~1793）；字节 13-16：道号 1~19008
- 建立计划文档 PLAN.md 与本记录文档
- 确定架构：8 个模块 + label_config.yaml 配置驱动标签体系

**环境基线**：
- conda env `seismic`：Python 3.10.16、PyQt5 ✔、numpy 2.2.6、matplotlib 3.10.7、segyio ✔（但本项目不依赖其读取）
- matplotlib 配置目录不可写 → 运行时需设置 `MPLCONFIGDIR` 到可写目录（代码内处理）

**遗留问题**：
- 初版标签清单（随机噪声/工业噪声/「不确定」选项）待用户最终确认
- 句子模板措辞待用户最终确认

---
