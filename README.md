# 地震道集标注器（Seismic Understanding Biaozhu）

地震道集（炮集 / CMP 道集 / 共检波点道集 / 残差等）特征标注工具，
产出「道集 ↔ 标签 ↔ 自然语言句子 ↔ 图片 ↔ 区域框」数据集，
供分类、多模态与目标检测模型训练。

**特性**

- 自实现 SEG-Y 读取层：大端/**小端**自动检测、1-based 任意字节道头、numpy memmap（大文件友好）
- Web 标注界面（Gradio）：浏览器即用，支持断点续标、多文件合并标注
- 8 项特征标签体系，配置驱动（改 `label_config.yaml` 即可增删特征，无需改代码）
- 面波 / 近炮点强能量噪声支持**矩形框选**（检测训练用，像素 + 原始数据双坐标）
- clip 随机采样数据增强：一个道集自动产出 N 张不同 clip 的训练图
- 每条记录自动渲染自然语言句子（句子模板可配置）

## 环境与依赖

- Python 3.10（开发环境为 conda env `seismic`）
- 必需：`numpy`、`matplotlib`、`PyYAML`
- Web 版：`gradio`（本仓库开发机的 gradio 6 装在项目 `_vendor/` 目录，**未入库**；
  新环境直接 `pip install gradio` 即可）
- Qt 桌面版（备用）：`PyQt5`

## 快速开始（Web 版，推荐）

```bash
python web_app.py            # 默认 0.0.0.0:7860，可加 --port 8000
```

浏览器访问 `http://<服务器IP>:7860`。本地访问远程服务器时用 SSH 隧道：

```bash
ssh -L 7860:localhost:7860 <用户名>@<服务器地址>
# 然后访问 http://localhost:7860
```

## Qt 桌面版（备用）

```bash
python main_window.py
```

在无显示器的远程服务器上运行（二选一）：

**方案 A：Qt VNC 插件（推荐，本地无需 X 服务器）**

```bash
# 服务器上（保持运行）
python main_window.py -platform vnc:size=1500x950   # 监听 5900 端口

# 本地电脑上：建立 SSH 隧道
ssh -L 5900:localhost:5900 <用户名>@<服务器地址>

# 再用 VNC 客户端连接 localhost:5900
# Windows: RealVNC / TightVNC Viewer；macOS: 自带「屏幕共享」；Linux: Remmina
```

**方案 B：X11 转发**

- 本地准备 X 服务器（Windows 装 VcXsrv/Xming，macOS 装 XQuartz，Linux 桌面自带）
- 重新登录：`ssh -X <用户名>@<服务器地址>`
- ⚠️ 若再接入已有 tmux 会话，`DISPLAY` 会失效：进 tmux **前**先 `echo $DISPLAY` 记下，
  进 tmux 后重新 `export DISPLAY=<记下的值>`
- 然后直接 `python main_window.py`

## 使用流程

### 第 1 步：文件与道集设置

- 填 sgy/segy 文件路径（支持大端/小端，默认自动检测）→「读取文件信息」
- 填排序键：如 `95-96, 13-16`（1-based 字节号，先按前者再按后者，升序）
- 填抽道集键：如 `95-96` → 点「扫描键值」→ 勾选要抽的值（默认全选）
- 选输出目录 →「生成道集」

### 第 2 步：预处理设置

- clip 分位数（默认 99%），可预览第一个道集效果
- **可选数据增强**：填增强 clip 下限/上限和张数 N（N=0 不增强）→「开始标注」

> **增强说明**（N>0 时）：保存时在 [下限,上限] 内随机采 N 个互不相同的 clip 值，
> 各渲染一张增强图（`images/<gather_id>__clip<值>.png`），每张在 labels.jsonl 中是
> 一条独立记录（labels/sentence/regions 与基础记录相同，另含 `augmented_from`、
> `clip_percentile` 字段）；基础记录保留但 image_path 为 null；npy 原始数据只存一份；
> 重复保存同一道集会自动清理其旧增强记录与图片再重新采样。

### 第 3 步：标注

左侧当前道集图，右侧逐特征单选，底部按钮：上一张 / 跳过 / 下一张 / 继承最近已标注 / 保存并下一张。

- **区域框选**（面波、近炮点强能量噪声）：点特征下方的「▣ 框选」，
  在左侧图上依次点击矩形两个对角即完成画框（显示图叠加框仅供查看，
  **导出的训练图保持干净**）；「✕ 清除」可重画。
  标签为「存在/已压制但有残留」时**必须画框才能保存**；「不存在」时框自动置空
- 「继承最近已标注」：回找最近一张**已保存**的道集复制其类别选项
  （自动跳过中间未标注的道集；框不继承，需另行画框）
- 全部特征选完才能保存；保存时自动导出 PNG 与 npy
- 重开程序自动从未标注的道集继续（断点续标）；已标注的可返回修改

## 输出与数据格式

```
输出目录/
├── labels.jsonl          # 每行一条道集记录
├── images/<文件名前缀>__<键>_<值>.png  # 纯数据图：无坐标轴/标题，固定 512×1024（宽×高）RGB
└── npy/<文件名前缀>__<键>_<值>.npy     # 原始数据：float32 (道数,采样点数)，无预处理，供数据增强
```

JSONL 单条示例：

```json
{
  "gather_id": "01SB21_DenoiseB4_FFID19037__95-96_1373",
  "gather_key": "95-96",
  "gather_value": 1373,
  "n_traces": 528,
  "labels": {"gather_type": "炮集", "statics": "需要", "direct_wave": "存在", "surface_wave": "已压制但有残留", "abnormal_amplitude": "不存在", "industrial_50hz": "存在", "near_shot_noise": "不存在", "aliasing_noise": "不存在"},
  "sentence": "这是一条炮集，需要静校正，直达波存在，面波已压制但有残留，……。",
  "regions": {"surface_wave": {"xyxy": [100, 200, 300, 600], "traces": [103, 311], "samples": [781, 2348]},
              "near_shot_noise": null},
  "image_path": "images/01SB21_DenoiseB4_FFID19037__95-96_1373.png",
  "npy_path": "npy/01SB21_DenoiseB4_FFID19037__95-96_1373.npy",
  "sort_keys": "95-96, 13-16",
  "extract_key": "95-96",
  "source_file": "/path/to/xxx.sgy",
  "timestamp": "2026-08-21T00:40:00"
}
```

**`regions` 字段**（仅 `label_config.yaml` 中 `bbox: true` 的特征有）：同一块区域的两套坐标——

| 字段 | 含义 | 用法 |
|---|---|---|
| `xyxy` | 512×1024 导出图上的像素坐标 `[左上x, 左上y, 右下x, 右下y]` | 检测模型训练直接用 |
| `traces` / `samples` | 原始数据的道/采样点下标范围，半开区间 `[起, 止)` | npy 切片：`data[t0:t1, s0:s1]` |

> **多文件合并**：gather_id 与图片名均带来源文件名前缀（如
> `01SB21_DenoiseB4_FFID19037__95-96_1373`），因此可以把多个 sgy 文件
> 标注到同一个输出目录，记录自动合并、互不覆盖；同一文件跨会话标注自动续标。

## 自定义标签体系

编辑 `label_config.yaml` 即可，**无需改代码**：

- 增删特征、选项、数字快捷键、选项拼句子措辞（`phrase`）、句子模板
- 某特征需要框选：加 `bbox: true`（可选 `bbox_color: "#rrggbb"`）

## 模块结构

| 文件 | 职责 |
|---|---|
| `segy_reader.py` | SEG-Y 读取（端序自动检测、1-based 任意字节道头、memmap） |
| `gather.py` | 多道头排序、按键抽道集 |
| `preprocess.py` | 预处理管线（注册表模式，可扩展）、clip 无放回采样 |
| `imaging.py` | 道集 → seismic 配色图；纯数据图渲染；框叠加显示 |
| `labels.py` / `label_config.yaml` | 标签体系与句子渲染 |
| `storage.py` | JSONL 存储与断点续标 |
| `web_app.py` | Gradio Web 界面（主用） |
| `main_window.py` | PyQt5 向导式界面（备用） |
| `backfill_npy.py` / `backfill_augment.py` / `migrate_remove_features.py` | 历史数据回补/迁移脚本 |

## 注意事项

- 道头字节号为 **1-based**（SEG-Y 标准文档惯例），区间长度须为 1/2/4/8 字节
- 小端 SEG-Y 非标准但常见，本工具自实现读取层支持
- `fonts/NotoSansCJKsc-Regular.otf` 为出图中文字体，勿删
- 计划与版本迭代记录见 `PLAN.md`、`MEMORY.md`
