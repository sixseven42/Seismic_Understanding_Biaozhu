# 地震道集标注器

地震道集（炮集 / CMP 道集 / 共检波点道集 / 残差等）特征标注工具，
产出「道集 ↔ 标签 ↔ 自然语言句子 ↔ 图片」数据集，供分类/多模态模型训练。

## 运行（Web 版，推荐）

```bash
conda activate seismic
cd /data/liuqi/code/seismic_biaozhu
python web_app.py            # 默认 0.0.0.0:7860，可加 --port 8000
```

浏览器打开（二选一）：

- 校园网直连：`http://162.105.91.239:7860`
- 或 SSH 隧道（注意是 `-L`）：`ssh -L 7860:localhost:7860 liuqi@162.105.91.239`，
  然后访问 `http://localhost:7860`

> gradio 依赖装在项目 `_vendor/` 目录（系统环境只读），勿删。

## 运行（Qt 桌面版，备用）

```bash
python main_window.py
```

### 在无显示器的远程服务器上运行 Qt 版（二选一）

**方案 A：Qt VNC 插件（推荐，本地无需 X 服务器）**

```bash
# 服务器上（保持运行）
python main_window.py -platform vnc:size=1500x950   # 监听 5900 端口

# 本地电脑上：建立 SSH 隧道
ssh -L 5900:localhost:5900 liuqi@<服务器地址>

# 再用 VNC 客户端连接 localhost:5900
# Windows: RealVNC / TightVNC Viewer；macOS: 自带「屏幕共享」；Linux: Remmina
```

**方案 B：X11 转发**

- 本地准备 X 服务器（Windows 装 VcXsrv/Xming，macOS 装 XQuartz，Linux 桌面自带）
- 重新登录：`ssh -X liuqi@<服务器地址>`（服务器端 xauth 已就绪）
- ⚠️ 若再接入已有 tmux 会话，`DISPLAY` 会失效：进 tmux **前**先 `echo $DISPLAY` 记下，
  进 tmux 后重新 `export DISPLAY=<记下的值>`
- 然后直接 `python main_window.py`


## 使用流程

1. **文件与道集设置**
   - 打开 sgy/segy 文件（支持大端/小端，默认自动检测）
   - 填排序键：如 `95-96, 13-16`（1-based 字节号，先按前者再按后者，升序）
   - 填抽道集键：如 `95-96` → 点「扫描键值」→ 勾选要抽的值（默认全选）
   - 选输出目录 → 「下一步：生成道集」
2. **预处理设置**：clip 分位数（默认 99%），可预览第一个道集效果；
   **可选数据增强**：填增强 clip 下限/上限和张数 N（N=0 不增强），
   保存时会在 [下限,上限] 内随机采 N 个 clip 值各存一张图 → 「开始标注」

> **增强说明**（N>0 时）：只存 N 张增强图（`images/<gather_id>__clip<值>.png`），
> 每张在 labels.jsonl 中是一条独立记录（labels/sentence 与基础记录相同，
> 另含 `augmented_from`、`clip_percentile` 字段）；基础记录保留但 image_path 为 null；
> 重复保存同一道集会自动清理其旧增强记录与图片再重新采样。
3. **标注**：左侧图像（工具栏可缩放/拖动），右侧逐特征单选
   - 数字键选当前高亮特征的选项，选完自动跳到下一特征
   - `←/→` 上一张/下一张，`S` 跳过，`Enter` 保存并下一张
   - **区域框选**（面波、近炮点强能量噪声）：点特征下方的「▣ 框选」，
     在左侧图上依次点击矩形两个对角即完成画框（显示图会叠加框，仅供查看，
     导出训练图保持干净）；「✕ 清除」可重画。标签为「存在/已压制但有残留」时
     必须画框才能保存；「不存在」时框自动置空
   - 全部特征选完才能保存；保存时自动导出 PNG 到 `输出目录/images/`
   - 「继承最近已标注」：回找最近一张**已保存**的道集复制其类别选项（跳过中间未标注的；
     框不继承，需另行画框）
   - 重开程序自动从未标注的道集继续；已标注的可返回修改

## 输出

```
输出目录/
├── labels.jsonl          # 每行一条道集记录
├── images/<文件名前缀>__<键>_<值>.png  # 纯数据图：无坐标轴/标题，固定 512×1024（宽×高）RGB
└── npy/<文件名前缀>__<键>_<值>.npy     # 原始数据：float32 (道数,采样点数)，无预处理，供数据增强
```

> **多文件合并**：gather_id 与图片名均带来源文件名前缀（如
> `01SB21_DenoiseB4_FFID19037__95-96_1373`），因此可以把多个 sgy 文件
> 标注到同一个输出目录，记录自动合并、互不覆盖；同一文件跨会话标注自动续标。

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

## 自定义标签体系

编辑 `label_config.yaml`：增删特征、选项、快捷键、句子措辞（`phrase`）、句子模板，**无需改代码**。

## 模块结构

| 文件 | 职责 |
|---|---|
| `segy_reader.py` | SEG-Y 读取（端序自动检测、1-based 任意字节道头、memmap） |
| `gather.py` | 多道头排序、按键抽道集 |
| `preprocess.py` | 预处理管线（注册表模式，可扩展） |
| `imaging.py` | 道集 → seismic 配色图 |
| `labels.py` / `label_config.yaml` | 标签体系与句子渲染 |
| `storage.py` | JSONL 存储与断点续标 |
| `main_window.py` | PyQt5 向导式界面 |

## 注意事项

- 道头字节号为 **1-based**（SEG-Y 标准文档惯例），区间长度须为 1/2/4/8 字节
- 小端 SEG-Y 非标准但常见（如示例文件），本工具自实现读取层支持
- `fonts/NotoSansCJKsc-Regular.otf` 为出图中文字体，勿删
- 计划与版本记录见 `PLAN.md`、`MEMORY.md`
