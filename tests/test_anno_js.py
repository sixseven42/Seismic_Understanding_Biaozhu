# -*- coding: utf-8 -*-
"""tests/test_anno_js.py — 用 Node + 最小 DOM 桩验证 ANNO_JS 里的纯逻辑。

浏览器侧的交互（数字键选题、Ctrl 两点框选换算）没法在 pytest 里真跑，
但这两处最容易出错、且逻辑是纯的：把 web_app.ANNO_JS 抽出来，在 Node 里以
假 DOM 加载，再直接调 window.__annoTest 暴露的函数断言行为。

没有 node 时自动跳过（不阻塞其他测试）。
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from labels import LabelConfig
from web_core import ABSENT_LABEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # 供导入 labels
NODE = shutil.which("node")

# 从 web_app.py 抽出 ANNO_JS 的 <script> 体，占位符按真实 label_config.yaml 填充
_HARNESS = r"""
'use strict';
// ---- 最小 DOM 桩：只够 IIFE 加载 + 调被测函数 ----
var CLICKS = [];
function mkClassList(){
  var set = {};
  return {
    add: function(c){ set[c] = true; },
    remove: function(c){ delete set[c]; },
    has: function(c){ return !!set[c]; },
  };
}
// 每题一个 #q-<key> 容器；选项 input **故意不给 name**（Gradio 6 实际如此）——
// 曾经的实现按 name 分组 + 下标兜底，会把每个选项当成一题，方向键就成了「同题内换选项」。
function mkGroup(key, texts, checkedIdx){
  var inputs = texts.map(function(t, i){
    var inp = {
      type: 'radio', name: '', checked: i === checkedIdx,
      closest: function(s){ return s === 'label' ? { textContent: t } : null; },
      click: function(){
        inputs.forEach(function(o){ o.checked = false; });
        inp.checked = true; CLICKS.push(t);
      },
    };
    return inp;
  });
  var container = {
    classList: mkClassList(), id: 'q-' + key,
    scrollIntoView: function(){ SCROLLED.push(key); },
    querySelectorAll: function(){ return inputs; },
  };
  return {inputs: inputs, container: container};
}
var SCROLLED = [];
// 题序：5 道选择题 + 2 个拉框项（与界面编号 1..7 一致）
var MG = [mkGroup('gather_type', ['1 炮集', '2 CMP道集', '3 共检波点道集', '4 残差'], -1),
          mkGroup('abnormal_amplitude', ['1 存在', '2 不存在', '3 已压制但有残留'], -1),
          mkGroup('aliasing_noise', ['1 存在', '2 不存在', '3 已压制但有残留'], -1),
          mkGroup('surface_wave', ['1 存在', '2 不存在', '3 已压制但有残留'], -1),
          mkGroup('near_shot_noise', ['1 存在', '2 不存在', '3 已压制但有残留'], -1)];
var G1 = MG[0].inputs, G2 = MG[1].inputs;
var SECS = [{id:'bx-surface_wave', classList: mkClassList(), scrollIntoView: function(){}},
            {id:'bx-near_shot_noise', classList: mkClassList(), scrollIntoView: function(){}}];
// 保存按钮桩：记录被点了几次
var SAVED = 0;
var SAVE_BTN = { click: function(){ SAVED++; } };
var SAVE_WRAP = { querySelector: function(sel){ return sel === 'button' ? SAVE_BTN : null; } };
var BY_ID = {'anno_featcol': {querySelectorAll: function(){ return []; }},
             'btn-save': SAVE_WRAP};
MG.forEach(function(g){ BY_ID[g.container.id] = g.container; });
SECS.forEach(function(s){ BY_ID[s.id] = s; });
var IMG = { complete: true, naturalWidth: 512, naturalHeight: 1024,
            getBoundingClientRect: function(){ return {left:0, top:0, width:512, height:1024}; } };

global.window = { addEventListener: function(){}, __ANNO_TEST__: true };
var ACTIVE = { tagName: 'BODY' };       // 可控的 document.activeElement
var FALLBACKS = [];                     // document.querySelectorAll('button') 的内容
global.document = {
  getElementById: function(id){ return BY_ID[id] || null; },
  querySelector: function(){ return IMG; },
  addEventListener: function(){},
  createElement: function(){ return { style: {} }; },
  querySelectorAll: function(){ return FALLBACKS; },
  get activeElement(){ return ACTIVE; },
};
global.setInterval = function(){ return 0; };
global.fetch = function(){ return { then: function(){ return this; }, catch: function(){} }; };

// ---- 载入被测脚本 ----
__ANNO_SCRIPT__

// ---- 断言 ----
var T = window.__annoTest;
function ok(cond, msg){ if (!cond){ console.error('FAIL: ' + msg); process.exit(1); } }

ok(T && typeof T.handleDigit === 'function', '测试钩子未暴露');

// ---- 1. 题序 = 5 道选择题 + 2 个拉框项，共 7 ----
// 注意 radio 无 name：分组必须按 #q-<key>，否则会把每个选项当成一题
// （= 方向键在同一题内跳选项的那个 bug）。
ok(T.items().length === 7, '应有 7 个题项（按题分组），实际 ' + T.items().length);
ok(T.items()[0].kind === 'radio' && T.items()[0].group.length === 4,
   '第 1 题应是 4 个选项的一组，实际 ' + T.items()[0].group.length);
ok(T.items()[1].group.length === 3, '第 2 题应是 3 个选项的一组');
ok(T.items()[5].kind === 'box' && T.items()[5].key === 'surface_wave'
   && T.items()[6].key === 'near_shot_noise', '拉框项应排在最后两项');

// ---- 2. 初始光标在第 1 题并高亮 ----
ok(T.cursorIndex() === 0, '初始光标应在第 1 题，实际 ' + T.cursorIndex());
T.paintCursor();
ok(MG[0].container.classList.has('anno-current'), '第 1 题应被高亮');
ok(!MG[1].container.classList.has('anno-current'), '第 2 题不应被高亮');

// ---- 3. ↑/↓ 跨题移动（不能在同一题内换选项、更不能改选中态）----
T.moveCursor(1);
ok(T.cursorIndex() === 1, '↓ 应跳到下一题，实际 ' + T.cursorIndex());
ok(G1.concat(G2).every(function(o){ return !o.checked; }),
   '方向键不得改动任何选项的选中态');
ok(!MG[0].container.classList.has('anno-current')
   && MG[1].container.classList.has('anno-current'), '高亮应整题移动');
T.setCursor(0);
T.moveCursor(1); T.moveCursor(1); T.moveCursor(1);
ok(T.cursorIndex() === 3, '连按 3 次 ↓ 应前进 3 题，实际 ' + T.cursorIndex());
ok(SCROLLED.length > 0, '移动后应把当前题滚进视野');
T.setCursor(2); T.moveCursor(-1);
ok(T.cursorIndex() === 1, '↑ 应回到上一题，实际 ' + T.cursorIndex());
ok(MG[1].container.classList.has('anno-current'), '↑ 后高亮应在第 2 题');
T.moveCursor(-1); T.moveCursor(-1);
ok(T.cursorIndex() === 0, '↑ 到顶应夹在第 1 题，实际 ' + T.cursorIndex());
T.setCursor(6); T.moveCursor(1);
ok(T.cursorIndex() === 6, '↓ 到底应夹在最后一项，实际 ' + T.cursorIndex());
ok(SECS[1].classList.has('anno-current'), '↓ 到底时高亮应在最后一个拉框项');

// ---- 4. 数字键答当前题并自动跳到下一题 ----
T.setCursor(0); T.paintCursor();
ok(T.handleDigit(3) === true, '第 1 题应被按键选中');
ok(CLICKS[CLICKS.length-1] === '3 共检波点道集',
   '应按数字选对应选项，实际 ' + CLICKS[CLICKS.length-1]);
ok(T.cursorIndex() === 1, '答完应自动跳到第 2 题，实际 ' + T.cursorIndex());
ok(MG[1].container.classList.has('anno-current'), '高亮应移到第 2 题');
ok(!MG[0].container.classList.has('anno-current'), '第 1 题高亮应撤掉');
ok(G2.every(function(o){ return !o.checked; }), '第 2 题不该被同时改动');

// 该题没有这个数字 → 不动作、光标不动
var n = CLICKS.length;
ok(T.handleDigit(9) === false, '无对应选项的数字应返回 false');
ok(CLICKS.length === n, '无对应选项时不应产生任何点击');
ok(T.cursorIndex() === 1, '无对应选项时光标不应移动');

// ---- 5. 一路答到拉框项；拉框项不吃数字键 ----
ok(T.handleDigit(2) === true, '第 2 题应接着被选中');
ok(T.handleDigit(1) === true, '第 3 题');
ok(T.handleDigit(1) === true, '第 4 题');
ok(T.handleDigit(1) === true, '第 5 题');
ok(T.cursorIndex() === 5, '答完第 5 题应停在第 6 项（面波拉框），实际 ' + T.cursorIndex());
ok(!MG[4].container.classList.has('anno-current'), '离开后第 5 题不应还高亮');
var n2 = CLICKS.length;
ok(T.handleDigit(1) === false, '拉框项不吃数字键');
ok(CLICKS.length === n2, '拉框项上按数字键不应产生点击');

// ---- 6. 回到第 1 题重答：覆盖原选择 ----
T.setCursor(0);
ok(T.handleDigit(1) === true, '回到第 1 题应可重答');
ok(G1[0].checked && !G1[2].checked, '重答应改选到新选项上');

// ---- 7. 选「不存在」→ 对应的拉框项自动跳过 ----
// 题序：0 集合类型 1 异常振幅 2 混叠噪声 3 面波 4 近炮点 5 面波拉框 6 近炮点拉框
ok(T.items()[5].key === 'surface_wave' && T.items()[6].key === 'near_shot_noise',
   '拉框项顺序应是 面波→近炮点');
ok(T.skippable(T.items()[5]) === false, '面波未作答时不该跳过');

// 面波选「不存在」(选项 2) → 面波拉框可跳过
T.setCursor(3);
T.handleDigit(2);
ok(T.skippable(T.items()[5]) === true, '面波选「不存在」后其拉框项应可跳过');
ok(T.cursorIndex() === 4, '答完面波应落到第 5 题（近炮点），实际 ' + T.cursorIndex());

// 近炮点也选「不存在」(选项 2) → 两个拉框项都可跳过；前方全是可跳过的 → 光标停在原地
T.handleDigit(2);
ok(T.skippable(T.items()[6]) === true, '近炮点选「不存在」后其拉框项应可跳过');
ok(T.cursorIndex() === 4, '两个拉框都可跳过时光标应停在最后一道题，实际 ' + T.cursorIndex());
T.moveCursor(1);
ok(T.cursorIndex() === 4, '↓ 前方无可停留题时应原地不动，实际 ' + T.cursorIndex());
T.paintCursor();
ok(SECS[0].classList.has('anno-skip') && SECS[1].classList.has('anno-skip'),
   '被跳过的拉框项应置灰标记');

// 面波改回「存在」→ 不再跳过；先把近炮点答完，再进入面波拉框
T.setCursor(3);
T.handleDigit(1);
ok(T.skippable(T.items()[5]) === false, '面波改选「存在」后拉框项应恢复');
ok(T.cursorIndex() === 4, '改回「存在」后仍先到第 5 题（近炮点），实际 ' + T.cursorIndex());
T.handleDigit(1);
ok(T.cursorIndex() === 5, '近炮点答完应进入面波拉框，实际 ' + T.cursorIndex());
ok(SECS[0].classList.has('anno-skip') === false, '恢复后不该再置灰');

// 拉框项不被跳过时，从最后一道题往后仍能停在它上面（已在上一步验证 index 5）

// ---- 8. Enter = 保存并下一张 ----
// 真实 keydown 的 target 就是获得焦点的元素 —— 桩里也用 ACTIVE，才能验到「焦点在文本框」
var fire = function(key){
  var e = {key: key, target: ACTIVE,
           preventDefault: function(){ e._prevented = true; },
           stopPropagation: function(){ e._stopped = true; }};
  T.onKeydown(e);
  return e;
};
var before = SAVED;
var e1 = fire('Enter');
ok(SAVED === before + 1, 'Enter 应触发保存按钮点击，实际点击次数 ' + (SAVED - before));
ok(e1._prevented === true, 'Enter 命中时应 preventDefault');
ok(e1._stopped === true, 'Enter 命中时应 stopPropagation');

// 焦点停在上一个点过的按钮上（例如「领取下一张」）时，Enter 仍必须保存：
// 不 preventDefault 的话浏览器会原生再点那个按钮，表现成「Enter 没反应」。
ACTIVE = {tagName: 'BUTTON'};
var n3 = SAVED;
var e2 = fire('Enter');
ok(SAVED === n3 + 1, '焦点在按钮上时 Enter 也必须保存（并吃掉原生点击）');
ok(e2._prevented === true, '焦点在按钮上也必须 preventDefault 以压掉原生点击');
ACTIVE = {tagName: 'INPUT', type: 'submit'};
var n4 = SAVED;
fire('Enter');
ok(SAVED === n4 + 1, '焦点在提交控件上时 Enter 也应保存');
// 可写文本框里 Enter 不抢：建作业表单照常输入
ACTIVE = {tagName: 'TEXTAREA'};                  // 无 readOnly → 视为正在输入
var n5 = SAVED;
fire('Enter');
ok(SAVED === n5, '可写文本框里 Enter 不应触发保存');
ACTIVE = {tagName: 'INPUT', type: 'text', readOnly: false};
fire('Enter');
ok(SAVED === n5, '可写数字/文本输入框里 Enter 不应触发保存');
// 只读字段（句子预览）不算输入，Enter 应保存
ACTIVE = {tagName: 'TEXTAREA', readOnly: true};
fire('Enter');
ok(SAVED === n5 + 1, '只读字段（句子预览）里 Enter 应保存');
ACTIVE = {tagName: 'BODY'};
// Enter 之外的其他键不触发保存
var n6 = SAVED;
fire('a'); fire('Tab');
ok(SAVED === n6, '非 Enter 键不应触发保存');

// 主路径：#btn-save 是个包一层的 div，里面才是 <button>
ok(T.clickSave() === true, 'clickSave 应能找到 #btn-save 里的按钮');
ok(SAVED === n6 + 1, 'clickSave 应点到 #btn-save 里的按钮');
// 兜底路径：elem_id 找不到时（Gradio 若把 id 放到别处 / 改名）按文案找
BY_ID['btn-save'] = null;
FALLBACKS.push({tagName: 'BUTTON', textContent: '某个无关按钮', click: function(){ SAVED += 1000; }});
FALLBACKS.push({tagName: 'BUTTON', textContent: '保存并释放（按 Enter 下一张）',
                click: function(){ SAVED += 1; }});
ok(T.saveBtn && T.saveBtn() !== null, 'saveBtn 应能兜底找到保存按钮');
var n7 = SAVED;
ok(T.clickSave() === true, 'elem_id 找不到时应按文案兜底');
ok(SAVED === n7 + 1, '兜底应点到「保存并释放」那个按钮而不是别的，实际 +' + (SAVED - n7));
// id 存在时不该走兜底
BY_ID['btn-save'] = SAVE_WRAP;

// 输入框判定：文本框/下拉不拦截数字键，单选/复选/按钮要放行
ok(T.isTyping({tagName:'INPUT', type:'text'}) === true, '文本框应被判定为打字中');
ok(T.isTyping({tagName:'TEXTAREA'}) === true, '文本域应被判定为打字中');
ok(T.isTyping({tagName:'INPUT', type:'number'}) === true, '数字输入框应被判定为打字中');
ok(T.isTyping({isContentEditable:true}) === true, 'contenteditable 应被判定为打字中');
ok(T.isTyping({tagName:'INPUT', type:'radio'}) === false, '单选不应被判定为打字中');
ok(T.isTyping({tagName:'BODY'}) === false, 'body 不应被判定为打字中');

// Ctrl 两点换算：两点顺序颠倒要归一化，且夹在自然像素范围内
var r = T.rectFrom([300, 700], [100, 200]);
ok(r.x0 === 100 && r.y0 === 200 && r.x1 === 300 && r.y1 === 700,
   '两点应归一化为左上→右下，实际 ' + JSON.stringify(r));
var r2 = T.rectFrom([-50, -80], [999, 5000]);
ok(r2.x0 === 0 && r2.y0 === 0 && r2.x1 === 511 && r2.y1 === 1023,
   '应夹到 512×1024 内（x1=511, y1=1023），实际 ' + JSON.stringify(r2));
var r3 = T.rectFrom([10, 20], [10, 20]);
ok(r3.x1 - r3.x0 === 0 && r3.y1 - r3.y0 === 0, '同一点应得到零面积矩形（后续会被 MIN 挡掉）');

console.log('ANNO_JS 逻辑检查通过');
"""


def _anno_script() -> str:
    """取出真正送进浏览器的 <script> 体（占位符按真实 label_config.yaml 填充）。

    注意：源码里的 ANNO_JS 是**非 raw** 三引号字符串，文件里写的是 `\\d`，浏览器收到的
    才是 `\\d`（单反斜杠）。所以必须用 ast.literal_eval 解一层转义，直接拿文件原文当 JS
    会把 `/^(\\d+)/` 变成「匹配反斜杠+d」，数字键永远失效 —— 这个坑本测试踩过。
    """
    src = open(os.path.join(ROOT, "web_app.py"), encoding="utf-8").read()
    m = re.search(r'ANNO_JS = """(.*?)"""\.replace\(', src, re.S)
    if not m:
        raise AssertionError("未能在 web_app.py 中定位 ANNO_JS")
    js = ast.literal_eval('"""' + m.group(1) + '"""')   # 解出真正的字符串
    cfg = LabelConfig(os.path.join(ROOT, "label_config.yaml"))

    def absent(f):
        """与 web_app._absent_label 同一规则：该特征「不存在」选项的 label，无则 ""。"""
        for o in f.options:
            if o.label == ABSENT_LABEL:
                return o.label
        return ""

    # 占位符内容与 web_app 一致（顺序 = 题号顺序），让桩与真实配置同步
    js = js.replace("__DRAG_BBOX__", repr([[f.key, f.name, f.bbox_color]
                                          for f in cfg.features if f.bbox]))
    js = js.replace("__ANNO_FEATS__",
                    repr([[f.key, f.name, absent(f)] for f in cfg.features]))
    body = re.search(r"<script>(.*)</script>", js, re.S)
    if not body:
        raise AssertionError("ANNO_JS 里没有 <script> 块")
    return body.group(1)


@unittest.skipUnless(NODE, "未安装 node，跳过前端逻辑检查")
class TestAnnoJsLogic(unittest.TestCase):
    def test_script_parses_and_logic_holds(self):
        harness = _HARNESS.replace("__ANNO_SCRIPT__", _anno_script())
        d = tempfile.mkdtemp()
        p = os.path.join(d, "harness.js")
        with open(p, "w", encoding="utf-8") as f:
            f.write(harness)
        r = subprocess.run([NODE, p], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0,
                         f"ANNO_JS 逻辑检查失败:\n{r.stdout}\n{r.stderr}")
        self.assertIn("逻辑检查通过", r.stdout)

    def test_script_has_no_stale_box_mode_hooks(self):
        """v3.8 起框选改 Ctrl 两点，旧的 ▣框选 模式钩子必须清干净。"""
        js = _anno_script()
        for stale in ("drawKey", "setDraw", "'bb-", "g.type === 'draw'"):
            self.assertNotIn(stale, js, f"ANNO_JS 残留旧框选模式代码: {stale}")


if __name__ == "__main__":
    unittest.main()
