# 多人协作标注 + 腾讯云 COS 回传 —— 设计规格 v1.0

> 日期：2026-09-07　｜　状态：待用户审阅
> 现状代码：`web_app.py`（Gradio 6.26 单用户版），核心模块 `segy_reader / gather / preprocess / imaging / labels / storage`
> 运行环境：本机 Windows，Python 3.14.3（`C:\Python314`），Gradio 6.26.0 已装

## 1. 目标

把单用户 Gradio 标注网页改造成**同一局域网内 ≤5 人并发标注**的中央服务器，原始 sgy 只留在这台机器上；标注产物（图片 + labels.jsonl）自动汇聚到腾讯云 COS，供国内的人通过域名/默认域名看图。

## 2. 非目标（明确不做）

- 不暴露公网 / 不做公网 HTTPS 反代 —— 标注服务仅限内网。
- 不把原始数据（sgy / npy）上云。
- 不做双人复核、不做审核流（每张只标一次）。
- 不重写 Gradio UI（保留标注区交互）。
- 不改 SEG-Y/抽道集/预处理/出图/标签体系模块。

## 3. 现状与改动边界

| 现状 | 改动 |
|---|---|
| `web_app.py:73` 模块级全局 `SES = Session()`，全用户共享 | 删除全局 SES，改每用户名工作态 |
| 单人「加载→抽→标」三步串行 | 拆成「admin 建作业」+「标注者从池领取」 |
| `LabelStore` 无锁，单用户安全 | 每作业一把锁，多用户串行化写 |
| 保存即做 clip 增强（aug_count） | 标注保存不再增强；增强挪到离线后处理 |
| 保存必写 npy | 保留（本地磁盘），npy 不上云 |
| 无账号/权限 | 账号密码登录，分 admin / annotator 两角色 |

## 4. 角色与账号

- 账号表 `users.yaml`（不入库，放项目根目录）：
  ```yaml
  users:
    - username: boss
      password: xxx
      role: admin
    - username: ann1
      password: xxx
      role: annotator
  ```
- 登录用 Gradio 内置 `auth=`：传一个 **callable**，每次登录请求时重新读 `users.yaml`（校验 `username/password`）。→ 效果：admin 增删账号**改文件即生效，无需重启**（已登录会话不受影响，不做强制踢出，MVP）。
- 每个请求经 `gr.Request.username` 拿到当前用户名（Gradio 6 在 auth 启用时提供）；角色也按用户名实时查 `users.yaml`（用小缓存 + mtime 失效，避免每次请求读盘）。
- 权限矩阵：

| 操作 | admin | annotator |
|---|---|---|
| 建作业 / 管作业 / 全部重传 / 增删账号 | ✔ | ✘ |
| 从池领取、标注、保存、归还 | ✔ | ✔ |
| 修改「自己标过」的记录 | ✔ | ✔ |
| 修改他人记录 / 删除 | ✔（admin 复核用） | ✘ |

> **权限以 handler 内服务端校验为准，不只靠 UI 隐藏**：标注流程各入口先查 `role(gr.Request.username)`，越权请求直接拒绝。

## 5. 作业（Job）与任务池 —— 核心

### 5.1 作业建模
- admin 建作业 = 沿用现在第 1/2 步的输入（sgy 路径、排序键、抽道集键、勾选键值、clip%）→ 服务器**抽一次道集**生成一个作业：
  ```
  Job {
    job_id:      "<来源文件名>__<时间戳短串>"   # 例 01SB21_DenoiseB4_FFID19037__20260907a
    title:       作业显示名
    source_file, sort_keys, extract_keys, values, clip_percentile
    output_dir:  output/<job_id>/               # 本作业所有产物隔离存放
    gather_ids:  [有序道集 id 列表]（沿用现有含文件名前缀的 gather_id）
    created_by:  admin 用户名
    state:       open | closed                 # closed 后不再领新任务
  }
  ```
- 作业元数据落盘 `output/<job_id>/job.json`。服务启动扫描 `output/` 下的 `job.json` **自动恢复**所有作业。
- **多作业并存**：每个作业独立 output_dir + 独立 store，互不覆盖（也天然规避跨作业 gather_id 相同问题）。

### 5.2 任务池与领取（claim）
- 池 = 作业的 `gather_ids` 有序序列。
- 状态来源（都从磁盘可重建）：已标 = store 记录；在标 = 内存租约表。
- 租约表 `leases: {gather_id: {user, deadline}}`，**默认 TTL 30 分钟**，标注者任一交互续期；超时自动作废回池。
- `claim(user)`：在作业锁内，按池顺序返回第一个「未标 且 （无有效他人租约 或 租约即 user 本人）」的 gather_id，并写租约。→ 每张只被一人持有；关浏览器/挂机 → 超时回收，不卡死任务。
- `release(user, gather_id)`：标注者主动「归还」当前持有道集 → 撤销租约、不写任何记录，该张回池（给无法标注的道集一条出路，不必等租约过期）。
- 标注者可「恢复我的当前任务」（租约还没过期）或从「我标注的列表」重开自己的历史记录。

### 5.3 并发安全
- 每作业一把 `threading.RLock`。`claim / save / 重开 / store 读写 / 进度统计` 全部在该锁内执行。单进程内即可，无需分布式锁。
- 保存时校验租约归属：若用户对该 gather 既无有效租约、又不是该记录原作者 → 拒绝并提示「该道集已被他人领取/标注」，避免两人同时标完同一张。

### 5.4 保存流程（annotator 视角单条）
1. （锁内）校验归属 → 通过。
2. 校验完整性：所有特征必选 + bbox 特征「存在/已压制但有残留」必有框（复用现有 `LabelConfig.validate_selection` 与框校验逻辑）。
3. 写记录：现字段不变，**新增 `annotated_by: username`**；重开修改时保留原 `annotated_by`，`timestamp` 刷新。
4. 渲染/复用缓存图 → `output/<job_id>/images/<gather_id>.png`；npy 按开关存 `output/<job_id>/npy/<gather_id>.npy`。
5. （锁内释放租约）→ 返回「已保存，进度 x/y」。
6. 触发云上传（见 §7，异步不阻塞标注）。

### 5.5 渲染缓存
- 10k 张规模下避免重复出图：任一时刻需要展示/保存某道集时，若 `images/<gather_id>.png` 已存在则直接复用；否则渲染一次后写盘（显示图 = 导出图同一文件）。bbox 叠加仍为**显示层临时文件**，不入导出图（沿用 `overlay_boxes` 现行为）。

## 6. 会话状态重构

- 删除全局 `SES`；引入 `user_state: {username: UserWork}`，`UserWork` 含：
  ```
  job_id, current_gather_id,
  partial_sel,     # 未保存的 Radio 选择
  box_partial,     # 未保存的框 {feature_key: box}
  box_mode, pending_corner   # 框选进行态
  ```
- 所有原 handler 增 `request: gr.Request` 入参解析 username，取代对 `SES` 全局的读写。
- 断点续标语义：标注者只能回看/修改「自己持有或自己标过」的道集；他人的任务不可见、不可写。

## 7. 腾讯云 COS 同步层（新模块 `cloudsync.py`）

- 依赖：`pip install cos-python-sdk-v5`（装进 `C:\Python314`）。
- 配置 `cos_config.yaml`（不入库，放项目根目录）：
  ```yaml
  secret_id / secret_key / bucket(含 appid 后缀) / region   # 必填
  enabled: true            # false = 纯本地模式
  upload_npy: false        # 默认只传 图片 + labels.jsonl
  public_read: true        # 桶/对象允许直链预览（或绑自定义域名后给读权限）
  ```
  密钥只存在于本机，永不进代码/注释/日志。
- **对象键布局**（镜像本地目录，按作业隔离）：
  ```
  seismic/<job_id>/labels.jsonl
  seismic/<job_id>/images/<gather_id>.png
  ```
- **触发与重试**：
  - 单线程后台 worker + 队列（每作业或全局一把即可）。保存成功后入队：该条 `images/*.png` + 该作业整份 `labels.jsonl`。
  - 失败写 `output/<job_id>/upload.log` 并留在待重试队列；后台指数退避（1s/5s/30s/60s，到上限后每 60s 重试）。
  - admin「全部重传」= 遍历本地 images + labels.jsonl 全量重新入队。
- **域名**：`bucket 中国大陆地域默认域名` 即可供国内快速访问；若要绑自己的域名，在腾讯云 COS 控制台配置自定义域名/CDN，本代码不涉及。图片直链以对象键拼接（§上面布局）。

## 8. 增强（augmentation）的位置

- **标注阶段不做增强**：理由——多人对同一作业无法各自设置增强参数（会冲突重存）；且增强与基础记录标签完全相同，不必在标注循环里做。
- 标注完成后，admin 在**离线**用扩展现有 `backfill_augment.py` 思路：读本地 `npy/` + 基础 labels.jsonl → 为每条基础记录按 clip 范围采样渲染 N 张 `images/<gid>__clip<c>.png`、各写一条 `augmented_from` 记录 → 需要时一并上传 COS。
- 标注核心循环完全不感知增强；`web_app.py` 现有 `aug_lo/aug_hi/aug_count` UI 与保存分支从标注流程移除。

## 9. UI 改动（仍是一个 Gradio Blocks 应用）

- 顶层按角色分流（登录后 Tabs）：
  - **admin「作业管理」**：建作业表单（沿用原第 1/2 步组件）、作业列表（进度 labeled/total、每标注者计数、打开/关闭、全部重传）、账号增删（提示改 users.yaml）。
  - **「标注」**（admin 与 annotator 共用，annotator 只看到此页）：作业下拉（仅列出 state=open 或该用户参与过的作业）→「领下一张」→ 现有标注区（左图右栏、逐特征 Radio、bbox 框选、句子预览、保存）原样复用；底部操作按钮替换为「领取 / 归还此张（释放回池，不记录）/ 我标注的列表」。去掉增强输入框。
- 标注区交互（点图框选、Radio 联动句子、必选校验、纯数据导出图、叠加仅供显示）逐条保留现有实现。

## 10. 断点续标 / 重启恢复

- 一切可从磁盘重建：`output/*/job.json`（作业定义）+ 各作业 `labels.jsonl`（已标状态）。
- 启动流程：扫 `output/` → 载入作业 + store → **清空内存租约** → 未标者自动回池。已标数据零丢失。
- 内存 `user_state` 属会话级，重启后标注者重新领取即可（对应「我标注的列表」仍可回到历史记录）。

## 11. 部署（本机 Windows，纯内网）

- 运行：`python web_app.py`（默认绑 `0.0.0.0:7860`，保持）。标注者访问 `http://<本机内网IP>:7860`，用 `users.yaml` 账号登录。
- 后台常驻：提供 `启动.bat`；需要开机自启再用「任务计划程序」/ nssm 包一层（不阻塞本期）。
- 防火墙：放行 7860（若开启 Windows 防火墙需加内网入站规则）。
- 本机需联网**上行到腾讯云 COS**（仅需出站 HTTPS，无入站公网要求）。

## 12. 新增/改动文件清单

| 文件 | 动作 | 职责 |
|---|---|---|
| `users.yaml` | 新增（不入库） | 账号/角色 |
| `cos_config.yaml` | 新增（不入库） | COS 凭证与开关 |
| `cloudsync.py` | 新增 | COS 上传队列 / 重试 / dry-run / 全部重传 |
| `jobmanager.py` | 新增 | Job 定义加载、任务池、租约、作业锁、保存协调、进度统计（核心新逻辑，独立可测） |
| `web_app.py` | 重构 | 去掉全局 SES；auth；角色分流 UI；标注区接线到 jobmanager/cloudsync |
| `storage.py` | 微调 | 不改核心；确认记录字典透传 `annotated_by`（现已支持任意字段） |
| `README.md` / `PLAN.md` | 更新 | 部署与用法、进度记录 |
| `segy_reader / gather / preprocess / imaging / labels / label_config.yaml / main_window.py` | 不动 | — |

## 13. 配置默认值

| 项 | 默认 |
|---|---|
| 租约 TTL / 续期 | 30 分钟；每次交互续期 |
| 上传内容 | 图片 + labels.jsonl；`upload_npy=false` |
| 本地存 npy | 开（`save_npy=true`，10k 张约 80GB 本地磁盘，供离线增强） |
| 账号热加载 | users.yaml 每次登录重读，改文件即生效 |
| 多作业 | 支持，标注者下拉选择 open 作业 |
| COS region | 中国大陆地域（默认域名国内快速访问） |

## 14. 测试计划

1. **并发互斥单测**（对 jobmanager，无需渲染）：模拟 N 线程 `claim/save` → 断言每张 gather 恰好被标一次、无一张被两人同时持有、租约过期后回池。
2. **store 并发**：多线程 upsert 同一作业 store → 文件不损坏、原子写、无丢记录。
3. **cloudsync dry-run**：`enabled=false` 走日志假上传；再对假 COS 端点验证对象键、重试、全部重传幂等。
4. **规模冒烟**：合成 10k gather_id（mock 渲染）跑领取/保存/进度，验证时延与内存。
5. **e2e（Gradio API）**：两个不同用户名各领各标，验证 `annotated_by`、租约冲突拒绝、annotator 看不到建作业页、admin 全部重传。
6. **手工冒烟**：真实小 sgy 走「admin 建作业 → 两个浏览器账号并发标 → 本地产物齐全 → COS 出现对象」。

## 15. 默认决策表（用户已拍板）

| 决策 | 结论 |
|---|---|
| 整体方案 | A：中央服务器式（单 Gradio 进程多人） |
| 云存储 | 腾讯云 COS；传 图片+labels.jsonl；npy 留本地 |
| 并发协作 | 任务池领取，每张只标一次；租约 30min |
| 登录 | 账号密码（Gradio auth），admin/annotator 两角色 |
| 服务器 | 这台 Windows 机器；**纯内网**，所有标注者同局域网，无公网通路 |
| 规模 | ≤5 人并发；≤10000 张；首访渲染缓存 |
| 看图人 | 国内 → COS 中国大陆地域（默认域名或自绑域名） |
| 增强 | 标注循环不做；标注完成后离线脚本回补（基于本地 npy） |
