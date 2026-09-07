# -*- coding: utf-8 -*-
"""
main_window.py — 地震道集标注器主界面（PyQt5）

向导式三步：
  1) 文件与道集设置：打开 sgy → 端序 → 排序键 → 抽道集键 → 勾选键值 → 输出目录
  2) 预处理设置：clip 分位数（可预览第一个道集效果）
  3) 标注界面：左侧图像（可缩放），右侧逐特征单选；数字键选选项，
     ←/→ 上一张/下一张，S 跳过，Enter 保存并下一张；断点续标；可返回修改。

运行：python main_window.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QButtonGroup, QScrollArea, QShortcut, QSpinBox, QSplitter,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

from segy_reader import SegyReader, SegyReadError
from gather import extract_gathers, gather_values, parse_key, key_str
from preprocess import apply_pipeline, available as preprocessors_available, sample_clip_values
from labels import LabelConfig, LabelConfigError
from storage import LabelStore
from imaging import draw_gather, render_gather, render_data_only

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_config.yaml")


# ----------------------------------------------------------------------
class Session:
    """跨页面共享的会话状态。"""

    def __init__(self):
        self.reader: SegyReader | None = None
        self.file_path = ""
        self.endian = "auto"
        self.sort_keys: list[tuple[int, int]] = []
        self.gather_key: tuple[int, int] | None = None
        self.gathers: list = []
        self.clip_percentile = 99.0
        self.output_dir = ""
        self.store: LabelStore | None = None
        self.label_cfg: LabelConfig | None = None
        self.current = 0                     # 当前道集下标
        self.partial: dict[int, dict] = {}   # 道集下标 -> 未完成的选项 {特征名: label}
        # 数据增强：clip 范围 [aug_lo, aug_hi] 内随机采样 aug_count 张；0 = 不增强
        self.aug_lo = 80.0
        self.aug_hi = 100.0
        self.aug_count = 0

    def pipeline_steps(self) -> list[dict]:
        return [{"name": "clip_percentile",
                 "params": {"percentile": self.clip_percentile}}]

    def gather_data(self, index: int):
        """取某道集预处理后的数据 (n_traces, ns)。"""
        g = self.gathers[index]
        data = self.reader.get_traces(g.trace_indices)
        return apply_pipeline(data, self.pipeline_steps())


# ----------------------------------------------------------------------
class FileSetupPage(QWidget):
    """步骤 1：文件与道集设置。"""

    def __init__(self, session: Session, go_next):
        super().__init__()
        self.session = session
        self.go_next = go_next
        lay = QVBoxLayout(self)

        # --- 文件行 ---
        file_row = QHBoxLayout()
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        btn_browse = QPushButton("打开 sgy/segy 文件…")
        btn_browse.clicked.connect(self._browse)
        self.endian_combo = QComboBox()
        self.endian_combo.addItems(["自动检测", "大端", "小端"])
        btn_load = QPushButton("读取文件信息")
        btn_load.clicked.connect(self._load_file)
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(btn_browse)
        file_row.addWidget(QLabel("字节序:"))
        file_row.addWidget(self.endian_combo)
        file_row.addWidget(btn_load)
        lay.addLayout(file_row)

        self.info_label = QLabel("尚未加载文件")
        self.info_label.setWordWrap(True)
        lay.addWidget(self.info_label)

        # --- 排序与抽道集 ---
        form = QFormLayout()
        self.sort_edit = QLineEdit("95-96, 13-16")
        self.sort_edit.setPlaceholderText("例：95-96, 13-16（先按95-96，再按13-16，1-based 升序）")
        self.gkey_edit = QLineEdit("95-96")
        self.gkey_edit.setPlaceholderText("例：95-96（按此道头抽道集）")
        form.addRow("排序键（逗号分隔）:", self.sort_edit)
        form.addRow("抽道集键:", self.gkey_edit)
        lay.addLayout(form)

        scan_row = QHBoxLayout()
        btn_scan = QPushButton("扫描键值")
        btn_scan.clicked.connect(self._scan_values)
        btn_all = QPushButton("全选"); btn_all.clicked.connect(lambda: self._check_all(True))
        btn_none = QPushButton("全不选"); btn_none.clicked.connect(lambda: self._check_all(False))
        scan_row.addWidget(btn_scan); scan_row.addWidget(btn_all); scan_row.addWidget(btn_none)
        self.values_count = QLabel("键值列表：（未扫描）")
        scan_row.addWidget(self.values_count, 1)
        lay.addLayout(scan_row)

        self.values_list = QListWidget()
        self.values_list.setMaximumHeight(180)
        lay.addWidget(self.values_list)

        # --- 输出目录 ---
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(os.path.join(os.getcwd(), "output"))
        btn_out = QPushButton("选择输出目录…")
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(QLabel("输出目录:"))
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(btn_out)
        lay.addLayout(out_row)

        btn_next = QPushButton("下一步：生成道集 →")
        btn_next.setStyleSheet("font-weight: bold")
        btn_next.clicked.connect(self._build_gathers)
        lay.addWidget(btn_next)
        lay.addStretch(1)

    # ------------------------------------------------------------------
    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 SEG-Y 文件", "", "SEG-Y (*.sgy *.segy);;所有文件 (*)")
        if p:
            self.path_edit.setText(p)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.out_edit.setText(d)

    def _endian_arg(self) -> str:
        return {"自动检测": "auto", "大端": "big", "小端": "little"}[self.endian_combo.currentText()]

    def _load_file(self) -> bool:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择文件")
            return False
        try:
            if self.session.reader is not None:
                self.session.reader.close()
            self.session.reader = SegyReader(path, endian=self._endian_arg())
        except SegyReadError as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return False
        self.session.file_path = path
        self.session.endian = self._endian_arg()
        i = self.session.reader.info()
        self.info_label.setText(
            f"✔ 已加载 | 端序: {'小端' if i['endian']=='little' else '大端'}（{'自动' if self.session.endian=='auto' else '手动'}）"
            f" | 道数: {i['n_traces']} | 每道采样点: {i['ns']} | 采样间隔: {i['dt_ms']} ms"
            f" | 格式码: {i['format_code']} | 文件大小: {i['file_size_mb']} MB"
        )
        return True

    def _parse_keys(self) -> bool:
        try:
            self.session.sort_keys = [parse_key(t) for t in self.session_sort_text().split(",") if t.strip()]
            self.session.gather_key = parse_key(self.gkey_edit.text())
        except ValueError:
            QMessageBox.warning(self, "格式错误", "道头键格式应为如 95-96 或 13-16, 多个用英文逗号分隔")
            return False
        return True

    def session_sort_text(self) -> str:
        return self.sort_edit.text()

    def _scan_values(self) -> bool:
        if self.session.reader is None and not self._load_file():
            return False
        if not self._parse_keys():
            return False
        try:
            vals = gather_values(self.session.reader, self.session.gather_key)
        except SegyReadError as e:
            QMessageBox.critical(self, "道头读取失败", str(e))
            return False
        self.values_list.clear()
        for v in vals:
            it = QListWidgetItem(str(int(v)))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.values_list.addItem(it)
        self.values_count.setText(f"键值列表：共 {len(vals)} 个（勾选要抽取的）")
        return True

    def _check_all(self, checked: bool):
        for i in range(self.values_list.count()):
            self.values_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _selected_values(self) -> list[int]:
        out = []
        for i in range(self.values_list.count()):
            it = self.values_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(int(it.text()))
        return out

    def _build_gathers(self):
        if self.session.reader is None and not self._load_file():
            return
        if self.values_list.count() == 0 and not self._scan_values():
            return
        if not self._parse_keys():
            return
        values = self._selected_values()
        if not values:
            QMessageBox.warning(self, "提示", "请至少勾选一个键值")
            return
        try:
            gathers = extract_gathers(
                self.session.reader, self.session.sort_keys,
                self.session.gather_key, values=values,
            )
        except SegyReadError as e:
            QMessageBox.critical(self, "抽道集失败", str(e))
            return
        if not gathers:
            QMessageBox.warning(self, "提示", "没有抽到任何道集")
            return
        self.session.gathers = gathers
        # gather_id 前缀 = 来源文件名（不含扩展名），防止多文件标到同一输出目录时互相覆盖
        stem = os.path.splitext(os.path.basename(self.session.file_path))[0]
        for g in gathers:
            g.prefix = stem + "__"
        self.session.output_dir = self.out_edit.text().strip() or os.path.join(os.getcwd(), "output")
        sizes = [g.n_traces for g in gathers]
        QMessageBox.information(
            self, "道集生成完成",
            f"共生成 {len(gathers)} 个道集（键 {key_str(self.session.gather_key)}），\n"
            f"每道集道数: 最少 {min(sizes)} / 最多 {max(sizes)}。"
        )
        self.go_next()


# ----------------------------------------------------------------------
class PreprocessPage(QWidget):
    """步骤 2：预处理设置。"""

    def __init__(self, session: Session, go_next, go_back):
        super().__init__()
        self.session = session
        self.go_next = go_next
        lay = QVBoxLayout(self)

        form = QFormLayout()
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(50.0, 100.0)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.setValue(session.clip_percentile)
        self.clip_spin.setSuffix(" %")
        form.addRow("clip 分位数:", self.clip_spin)
        self.aug_lo = QDoubleSpinBox(); self.aug_lo.setRange(1, 100); self.aug_lo.setValue(80.0)
        self.aug_hi = QDoubleSpinBox(); self.aug_hi.setRange(1, 100); self.aug_hi.setValue(100.0)
        self.aug_count = QSpinBox(); self.aug_count.setRange(0, 50); self.aug_count.setValue(0)
        form.addRow("增强 clip 下限 (%):", self.aug_lo)
        form.addRow("增强 clip 上限 (%):", self.aug_hi)
        form.addRow("增强张数（0=不增强）:", self.aug_count)
        lay.addLayout(form)

        avail = "、".join(f"{k}（{v['desc']}）" for k, v in preprocessors_available().items())
        lay.addWidget(QLabel(f"已注册预处理方法：{avail}\n（后续会扩展更多方法，当前使用 clip_percentile）"))

        btn_preview = QPushButton("预览第一个道集效果")
        btn_preview.clicked.connect(self._preview)
        lay.addWidget(btn_preview)

        self.canvas = FigureCanvas(Figure(figsize=(6, 6)))
        lay.addWidget(NavigationToolbar(self.canvas, self))
        lay.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        btn_back = QPushButton("← 返回上一步"); btn_back.clicked.connect(go_back)
        btn_next = QPushButton("开始标注 →"); btn_next.setStyleSheet("font-weight: bold")
        btn_next.clicked.connect(self._start)
        row.addWidget(btn_back); row.addStretch(1); row.addWidget(btn_next)
        lay.addLayout(row)

    def _preview(self):
        if not self.session.gathers:
            QMessageBox.warning(self, "提示", "请先在第 1 步生成道集")
            return
        self.session.clip_percentile = self.clip_spin.value()
        data = self.session.gather_data(0)
        g = self.session.gathers[0]
        fig = self.canvas.figure; fig.clear()
        ax = fig.add_subplot(111)
        draw_gather(ax, data, dt_ms=self.session.reader.info()["dt_ms"], bare=True)
        fig.tight_layout()
        self.canvas.draw_idle()

    def _start(self):
        if not self.session.gathers:
            QMessageBox.warning(self, "提示", "请先在第 1 步生成道集")
            return
        self.session.clip_percentile = self.clip_spin.value()
        self.session.aug_lo = self.aug_lo.value()
        self.session.aug_hi = self.aug_hi.value()
        self.session.aug_count = self.aug_count.value()
        if self.session.aug_count > 0 and not (0 < self.session.aug_lo <= self.session.aug_hi <= 100):
            QMessageBox.warning(self, "参数错误", "增强 clip 范围须满足 0 < 下限 ≤ 上限 ≤ 100")
            return
        os.makedirs(self.session.output_dir, exist_ok=True)
        try:
            self.session.store = LabelStore(os.path.join(self.session.output_dir, "labels.jsonl"))
            self.session.label_cfg = LabelConfig(CONFIG_PATH)
        except (LabelConfigError, ValueError) as e:
            QMessageBox.critical(self, "初始化失败", str(e))
            return
        self.go_next()


# ----------------------------------------------------------------------
class LabelPage(QWidget):
    """步骤 3：标注界面。"""

    def __init__(self, session: Session, go_back):
        super().__init__()
        self.session = session
        self.go_back = go_back
        self._feature_radios: list[list[QRadioButton]] = []
        self._groups: list[QGroupBox] = []
        self._current_feature = 0
        self._loading = False

        split = QSplitter(self)

        # 左侧图像
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvas(Figure(figsize=(6, 8)))
        ll.addWidget(NavigationToolbar(self.canvas, self))
        ll.addWidget(self.canvas, 1)
        split.addWidget(left)

        # 右侧面板
        right = QWidget(); rl = QVBoxLayout(right)
        self.info_label = QLabel("—"); self.info_label.setWordWrap(True)
        rl.addWidget(self.info_label)
        self.progress = QProgressBar()
        rl.addWidget(self.progress)

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.feat_container = QWidget()
        self.feat_layout = QVBoxLayout(self.feat_container)
        self.scroll.setWidget(self.feat_container)
        rl.addWidget(self.scroll, 1)

        rl.addWidget(QLabel("句子预览:"))
        self.sentence_view = QTextEdit(); self.sentence_view.setReadOnly(True)
        self.sentence_view.setMaximumHeight(90)
        rl.addWidget(self.sentence_view)

        btn_row1 = QHBoxLayout()
        self.btn_prev = QPushButton("← 上一张"); self.btn_prev.clicked.connect(self.prev_gather)
        self.btn_skip = QPushButton("跳过 (S)"); self.btn_skip.clicked.connect(self.skip_gather)
        btn_row1.addWidget(self.btn_prev); btn_row1.addWidget(self.btn_skip)
        rl.addLayout(btn_row1)
        btn_row2 = QHBoxLayout()
        self.btn_next = QPushButton("下一张 →"); self.btn_next.clicked.connect(self.next_gather)
        self.btn_save = QPushButton("保存并下一张 (Enter)")
        self.btn_save.setStyleSheet("font-weight: bold")
        self.btn_save.clicked.connect(self.save_and_next)
        btn_row2.addWidget(self.btn_next); btn_row2.addWidget(self.btn_save)
        rl.addLayout(btn_row2)
        btn_inherit = QPushButton("⧉ 继承上一张的选项 (I)")
        btn_inherit.clicked.connect(self.inherit_previous)
        rl.addWidget(btn_inherit)
        btn_back = QPushButton("← 返回预处理设置"); btn_back.clicked.connect(go_back)
        rl.addWidget(btn_back)
        rl.addWidget(QLabel("快捷键: 数字键选当前特征选项并自动跳到下一特征 | "
                            "←/→ 切换道集 | S 跳过 | Enter 保存并下一张"))
        split.addWidget(right)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)

        # 快捷键
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_gather)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_gather)
        QShortcut(QKeySequence(Qt.Key_Return), self, activated=self.save_and_next)
        QShortcut(QKeySequence(Qt.Key_Enter), self, activated=self.save_and_next)
        QShortcut(QKeySequence(Qt.Key_S), self, activated=self.skip_gather)
        QShortcut(QKeySequence(Qt.Key_I), self, activated=self.inherit_previous)
        for d in range(1, 10):
            QShortcut(QKeySequence(getattr(Qt, f"Key_{d}")), self,
                      activated=lambda d=d: self._digit(d))

    # ------------------------------------------------------------------
    def enter_page(self):
        """进入本页：构建特征面板并跳到第一个未标注道集（断点续标）。"""
        self._build_feature_panel()
        ids = [g.gather_id for g in self.session.gathers]
        pos = self.session.store.next_unlabeled(ids, 0)
        self.show_gather(pos if pos is not None else 0)

    def _build_feature_panel(self):
        # 清空旧面板
        while self.feat_layout.count():
            item = self.feat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._feature_radios.clear(); self._groups.clear()
        cfg = self.session.label_cfg
        for fi, feat in enumerate(cfg.features):
            gb = QGroupBox(f"{fi+1}. {feat.name}")
            vb = QVBoxLayout(gb)
            radios = []
            for opt in feat.options:
                rb = QRadioButton(f"{opt.hotkey}  {opt.label}")
                rb.toggled.connect(lambda checked, f=feat.name, l=opt.label:
                                   self._on_option(f, l, checked))
                vb.addWidget(rb)
                radios.append(rb)
            self.feat_layout.addWidget(gb)
            self._feature_radios.append(radios)
            self._groups.append(gb)
        self.feat_layout.addStretch(1)

    # ------------------------------------------------------------------
    def show_gather(self, index: int):
        if not (0 <= index < len(self.session.gathers)):
            return
        self._loading = True
        self.session.current = index
        g = self.session.gathers[index]
        data = self.session.gather_data(index)

        fig = self.canvas.figure; fig.clear()
        ax = fig.add_subplot(111)
        draw_gather(ax, data, dt_ms=self.session.reader.info()["dt_ms"], bare=True)
        fig.tight_layout()
        self.canvas.draw_idle()

        # 信息/进度
        store = self.session.store
        labeled = sum(1 for gg in self.session.gathers if store.is_labeled(gg.gather_id))
        self.progress.setMaximum(len(self.session.gathers))
        self.progress.setValue(labeled)
        state = "已标注 ✔" if store.is_labeled(g.gather_id) else "未标注"
        self.info_label.setText(
            f"道集 {index+1}/{len(self.session.gathers)} | 键 {g.key} = {g.value}"
            f" | {g.n_traces} 道 | {state} | 已标 {labeled}/{len(self.session.gathers)}"
        )

        # 恢复选项：优先已存记录，其次会话内暂存
        self._restore_selection(g)
        self._current_feature = self._first_unanswered()
        self._refresh_highlight()
        self._refresh_sentence()
        self._loading = False

    def _restore_selection(self, g):
        rec = self.session.store.get(g.gather_id)
        sel_by_key = rec["labels"] if rec else {}
        partial = self.session.partial.get(self.session.current, {})
        cfg = self.session.label_cfg
        for fi, feat in enumerate(cfg.features):
            want = sel_by_key.get(feat.key) or partial.get(feat.name)
            for rb in self._feature_radios[fi]:
                rb.blockSignals(True)
                rb.setChecked(rb.text().split("  ", 1)[1] == want if want else False)
                rb.blockSignals(False)
            if want is None:
                # QButtonGroup 互斥清除：临时关掉互斥
                for rb in self._feature_radios[fi]:
                    rb.setAutoExclusive(False)
                    rb.setChecked(False)
                    rb.setAutoExclusive(True)

    # ------------------------------------------------------------------
    def _on_option(self, feat_name: str, label: str, checked: bool):
        if self._loading or not checked:
            return
        cfg = self.session.label_cfg
        sel = self._current_selection()
        sel[feat_name] = label
        self.session.partial[self.session.current] = sel
        # 跳到下一个未回答特征
        fi = [f.name for f in cfg.features].index(feat_name)
        self._current_feature = self._first_unanswered(after=fi)
        self._refresh_highlight()
        self._refresh_sentence()

    def _current_selection(self) -> dict:
        sel = {}
        cfg = self.session.label_cfg
        for fi, feat in enumerate(cfg.features):
            for rb in self._feature_radios[fi]:
                if rb.isChecked():
                    sel[feat.name] = rb.text().split("  ", 1)[1]
        return sel

    def _first_unanswered(self, after: int = -1) -> int:
        cfg = self.session.label_cfg
        sel = self._current_selection()
        n = len(cfg.features)
        for k in range(1, n + 1):
            i = (after + k) % n
            if cfg.features[i].name not in sel:
                return i
        return (after + 1) % n

    def _refresh_highlight(self):
        for i, gb in enumerate(self._groups):
            gb.setStyleSheet(
                "QGroupBox { border: 2px solid #2d7ff9; border-radius: 4px; margin-top: 6px; }"
                if i == self._current_feature else ""
            )
        # 让当前特征滚进视野
        self.scroll.ensureWidgetVisible(self._groups[self._current_feature])

    def _refresh_sentence(self):
        sel = self._current_selection()
        try:
            self.sentence_view.setPlainText(self.session.label_cfg.render_sentence(sel))
        except LabelConfigError as e:
            self.sentence_view.setPlainText(f"模板错误: {e}")

    def _digit(self, d: int):
        radios = self._feature_radios[self._current_feature]
        for rb in radios:
            if rb.text().startswith(f"{d}  "):
                rb.setChecked(True)
                return

    # ------------------------------------------------------------------
    def prev_gather(self):
        if self.session.current > 0:
            self.show_gather(self.session.current - 1)

    def next_gather(self):
        if self.session.current < len(self.session.gathers) - 1:
            self.show_gather(self.session.current + 1)

    def skip_gather(self):
        self.next_gather()

    def inherit_previous(self):
        """把上一张道集的选项复制到当前道集（未保存，仅填充选项）。"""
        if self.session.current == 0:
            QMessageBox.information(self, "提示", "当前已是第一张，无上一张可继承")
            return
        prev = self.session.gathers[self.session.current - 1]
        rec = self.session.store.get(prev.gather_id)
        by_key = rec["labels"] if rec else {}
        prev_partial = self.session.partial.get(self.session.current - 1, {})
        cfg = self.session.label_cfg
        sel = {}
        self._loading = True
        for fi, feat in enumerate(cfg.features):
            want = by_key.get(feat.key) or prev_partial.get(feat.name)
            if want:
                sel[feat.name] = want
            for rb in self._feature_radios[fi]:
                rb.setAutoExclusive(False)
                rb.setChecked(rb.text().split("  ", 1)[1] == want if want else False)
                rb.setAutoExclusive(True)
        self._loading = False
        if not sel:
            QMessageBox.information(self, "提示", f"上一张（{prev.gather_id}）没有任何已选/暂存选项")
            return
        self.session.partial[self.session.current] = sel
        self._current_feature = self._first_unanswered()
        self._refresh_highlight()
        self._refresh_sentence()

    def save_and_next(self):
        g = self.session.gathers[self.session.current]
        sel = self._current_selection()
        errs = self.session.label_cfg.validate_selection(sel)
        if errs:
            QMessageBox.warning(self, "尚有未选特征", "\n".join(errs))
            return
        cfg = self.session.label_cfg
        sentence = cfg.render_sentence(sel)
        import numpy as np
        # 原始数据（无预处理）：存 npy + 作为增强数据源
        raw = self.session.reader.get_traces(g.trace_indices).astype(np.float32)
        rel_npy = os.path.join("npy", f"{g.gather_id}.npy")
        abs_npy = os.path.join(self.session.output_dir, rel_npy)
        os.makedirs(os.path.dirname(abs_npy), exist_ok=True)
        np.save(abs_npy, raw)
        base_rec = {
            "gather_id": g.gather_id,
            "gather_key": g.key,
            "gather_value": g.value,
            "n_traces": g.n_traces,
            "labels": cfg.to_record_labels(sel),
            "sentence": sentence,
            "npy_path": rel_npy,
            "sort_keys": ", ".join(key_str(k) for k in self.session.sort_keys),
            "extract_key": key_str(self.session.gather_key),
            "source_file": self.session.file_path,
        }
        n_aug = self.session.aug_count
        if n_aug > 0:
            import glob as _glob
            self.session.store.remove_augmented(g.gather_id)
            img_dir = os.path.join(self.session.output_dir, "images")
            for old in _glob.glob(os.path.join(img_dir, f"{g.gather_id}__clip*.png")):
                os.remove(old)
            clips = sample_clip_values(self.session.aug_lo, self.session.aug_hi, n_aug)
            for cp in clips:
                d = apply_pipeline(raw, [{"name": "clip_percentile",
                                          "params": {"percentile": float(cp)}}])
                aug_id = f"{g.gather_id}__clip{cp:g}"
                rel_img = os.path.join("images", f"{aug_id}.png")
                render_data_only(d, out_path=os.path.join(self.session.output_dir, rel_img))
                self.session.store.upsert({**base_rec,
                                           "gather_id": aug_id,
                                           "image_path": rel_img,
                                           "augmented_from": g.gather_id,
                                           "clip_percentile": float(cp)})
            base_rec["image_path"] = None   # 只存增强图
        else:
            rel_img = os.path.join("images", f"{g.gather_id}.png")
            abs_img = os.path.join(self.session.output_dir, rel_img)
            render_data_only(self.session.gather_data(self.session.current), out_path=abs_img)
            base_rec["image_path"] = rel_img
        self.session.store.upsert(base_rec)
        self.session.partial.pop(self.session.current, None)
        # 更新进度后移动到下一张（优先下一个未标注）
        ids = [gg.gather_id for gg in self.session.gathers]
        nxt = self.session.store.next_unlabeled(ids, self.session.current + 1)
        if nxt is None:
            nxt = self.session.store.next_unlabeled(ids, 0)
        if nxt is None:
            QMessageBox.information(self, "完成", "全部道集均已标注！")
            self.show_gather(self.session.current)
        else:
            self.show_gather(nxt)


# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("地震道集标注器")
        self.resize(1300, 850)
        self.session = Session()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_file = FileSetupPage(self.session, self._goto_preprocess)
        self.page_prep = PreprocessPage(self.session, self._goto_label, lambda: self.stack.setCurrentIndex(0))
        self.page_label = LabelPage(self.session, lambda: self.stack.setCurrentIndex(1))
        self.stack.addWidget(self.page_file)
        self.stack.addWidget(self.page_prep)
        self.stack.addWidget(self.page_label)

    def _goto_preprocess(self):
        self.stack.setCurrentIndex(1)

    def _goto_label(self):
        self.stack.setCurrentIndex(2)
        self.page_label.enter_page()

    def closeEvent(self, event):
        if self.session.reader is not None:
            self.session.reader.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
