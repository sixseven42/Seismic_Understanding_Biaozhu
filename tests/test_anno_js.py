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
var BY_ID = {'anno_featcol': {id: 'anno_featcol', querySelectorAll: function(){ return []; }},
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
  createElement: function(){ return {style: {}, getContext: function(){ return mkCtx(); }}; },
  querySelectorAll: function(){ return FALLBACKS; },
  get activeElement(){ return ACTIVE; },
};
global.setInterval = function(){ return 0; };
// setTimeout/clearTimeout 可控：scheduleTargetRefresh 的去抖靠它们，测试里手动 flush。
// clearTimeout 必须真的有（生产代码用它取消还没到点的回读），否则 postClear 会直接抛。
var TIMERS = {}, _tid = 0;
global.setTimeout = function(fn){ TIMERS[++_tid] = fn; return _tid; };
global.clearTimeout = function(id){ delete TIMERS[id]; };
function flushTimers(){
  var t = TIMERS; TIMERS = {};
  Object.keys(t).forEach(function(k){ t[k](); });
}
// rAF 立即执行：观察者回调里经 raf() 合并的那次重绘要能跑完
global.requestAnimationFrame = function(fn){ fn(); return 0; };

// fetch 桩：记录 URL，并可让 then 链**同步**跑完，好断言回读后的状态
var FETCHED = [], FETCH_PAYLOAD = null;
function SyncThenable(v){
  return {then: function(fn){ return SyncThenable(fn(v)); }, catch: function(){ return this; }};
}
global.fetch = function(url){
  FETCHED.push(url);
  if (FETCH_PAYLOAD === null){          // 默认：永不 resolve（沿用原桩行为）
    return {then: function(){ return this; }, catch: function(){ return this; }};
  }
  return SyncThenable({ok: true, json: function(){ return FETCH_PAYLOAD; }});
};

// MutationObserver 桩：记下每个观察者与其观察目标，测试里手动触发 featcol 那个
var OBSERVERS = [];
function MO(cb){
  this.cb = cb;
  this.observe = function(target, opts){ OBSERVERS.push({cb: cb, target: target, opts: opts}); };
}
global.MutationObserver = MO;
window.MutationObserver = MO;
function observerOn(id){
  for (var i = 0; i < OBSERVERS.length; i++){
    if (OBSERVERS[i].target && OBSERVERS[i].target.id === id) return OBSERVERS[i];
  }
  return null;
}

// <img> / canvas 桩：setup() + place() + redraw() 要能真跑一遍（观察者是 setup 里挂的）
var IMG_LISTENERS = {};
IMG.addEventListener = function(t, fn){ IMG_LISTENERS[t] = fn; };
function mkCtx(){
  var noop = function(){};
  return {clearRect:noop, strokeRect:noop, fillRect:noop, beginPath:noop, moveTo:noop,
          lineTo:noop, stroke:noop, fillText:noop, setLineDash:noop,
          measureText: function(t){ return {width: (t || '').length * 7}; }};
}
IMG.appendChild = function(cv){ BY_ID['anno_drag_canvas'] = cv; };
var DOC_LISTENERS = {};
document.addEventListener = function(t, fn){ DOC_LISTENERS[t] = fn; };

// 账号栏桩：currentUser() 靠它取当前用户，fetchBoxes/postBox 没用户就直接 return
BY_ID['who_md'] = {querySelector: function(sel){
  return sel === 'code' ? {textContent: 'ann1'} : null;
}};

// 「✕ 清除此框」按钮桩：右键清框会程序化点它
var CB_CLICKS = [];
BY_ID['cb-surface_wave'] = {tagName: 'BUTTON', id: 'cb-surface_wave',
                            click: function(){ CB_CLICKS.push('surface_wave'); }};
BY_ID['cb-near_shot_noise'] = {tagName: 'BUTTON', id: 'cb-near_shot_noise',
                               click: function(){ CB_CLICKS.push('near_shot_noise'); }};
BY_ID['anno_imgcol'] = {id: 'anno_imgcol'};

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

// ======================================================================
// v3.9 新增：松开 Ctrl 成框 / 右键清框 / Ctrl 目标在服务端 outputs 之后回读
// 这三处都是纯状态机，用存取器把状态摆到位再直接调处理函数。
// ======================================================================
T.setup();          // 跑一次真实 setup：观察者、事件绑定都在里面挂上

function noop(){}
function mkEvent(x, y, btn){
  var e = {clientX: x, clientY: y, button: btn === undefined ? 0 : btn,
           preventDefault: function(){ e._prevented = true; },
           stopPropagation: function(){ e._stopped = true; }};
  return e;
}

// ---- 9. 松开 Ctrl：以松开瞬间的鼠标位置作第二角 ----
T.setBoxes({});
T.setTarget('surface_wave');
T.setPending({key: 'surface_wave', p: [100, 200]});
T.setHover([300, 500]);
T.onKeyUp({key: 'Control'});
var kb = T.boxes()['surface_wave'];
ok(kb && kb.x0 === 100 && kb.y0 === 200 && kb.x1 === 300 && kb.y1 === 500,
   '松开 Ctrl 应以鼠标位置作第二角成框，实际 ' + JSON.stringify(kb));
ok(T.pending() === null, '成框后待定第一角应被消费掉');

// Cmd（Mac）/ 老 Firefox 的 OS 键同样算
T.setBoxes({});
T.setPending({key: 'surface_wave', p: [10, 10]});
T.setHover([60, 70]);
T.onKeyUp({key: 'Meta'});
ok(T.boxes()['surface_wave'] && T.boxes()['surface_wave'].x1 === 60, 'Meta（Cmd）也应成框');

// 其它键不触发
T.setBoxes({});
T.setPending({key: 'surface_wave', p: [10, 10]});
T.setHover([60, 70]);
T.onKeyUp({key: 'Shift'});
ok(!T.boxes()['surface_wave'], '松开 Shift 不该成框');
ok(T.pending() !== null, '未成框时待定第一角必须保留');

// 松手时框太小（点完没动）→ 保留第一角，仍可用「Ctrl 再点第二角」
T.setBoxes({});
T.setPending({key: 'surface_wave', p: [100, 200]});
T.setHover([101, 201]);
T.onKeyUp({key: 'Control'});
ok(!T.boxes()['surface_wave'], '退化框（太小的）不该成框');
ok(T.pending() && T.pending().p[0] === 100,
   '退化框时待定第一角不能丢（否则一次误松手就把第一角弄没了）');

// 指针已离开图片（hoverNat 被 onLeave 清空）→ 不知道第二角在哪，别瞎猜
T.setBoxes({});
T.setPending({key: 'surface_wave', p: [100, 200]});
T.setHover(null);
T.onKeyUp({key: 'Control'});
ok(!T.boxes()['surface_wave'], '指针不在图上时不该按旧位置成框');
ok(T.pending() !== null, '指针不在图上时待定第一角应保留');

// 目标已顺延（pending 与 target 不一致）→ 不落框，避免把框画给别的特征
T.setBoxes({});
T.setPending({key: 'surface_wave', p: [100, 200]});
T.setHover([300, 500]);
T.setTarget('near_shot_noise');
T.onKeyUp({key: 'Control'});
ok(!T.boxes()['surface_wave'], 'pending 与当前目标不符时不该成框');
T.setTarget('surface_wave');
T.setPending(null);

// 绑定确实挂上了（keydown 之外还要有 keyup）
ok(typeof DOC_LISTENERS['keyup'] === 'function', 'setup 应把 keyup 绑到 document 上');
ok(IMG_LISTENERS['contextmenu'] === T.onCtxMenu, 'img 上应绑定 contextmenu 处理');

// ---- 10. 右键：先清半成品，其次清鼠标下的框 ----
T.setBoxes({surface_wave: {x0: 0, y0: 0, x1: 100, y1: 100}});
T.setCanvas({_img: IMG, width: 512, height: 1024, getContext: function(){ return mkCtx(); }});
CB_CLICKS.length = 0;

// ① 有待定框 → 只丢弃它，不动已画好的框
T.setPending({key: 'surface_wave', p: [10, 10]});
var ce1 = mkEvent(50, 50, 2);
T.onCtxMenu(ce1);
ok(T.pending() === null, '右键应丢弃待定框');
ok(CB_CLICKS.length === 0, '有半成品时右键不该删掉已画好的框');
ok(T.boxes()['surface_wave'], '有半成品时右键不该动已画好的框');
ok(ce1._prevented === true, '右键必须 preventDefault 掉浏览器菜单');

// ② 无待定框 + 鼠标压在框上 → 本地立刻消失 **且**直连清框接口
// （不再程序化点 Gradio 的「✕ 清除此框」：右键不该依赖它的 DOM 层级）
T.setBoxes({surface_wave: {x0: 0, y0: 0, x1: 100, y1: 100}});
FETCHED.length = 0;
var ce2 = mkEvent(50, 50, 2);
T.onCtxMenu(ce2);
ok(CB_CLICKS.length === 0, '右键不该再程序化点 Gradio 清除按钮');
ok(!T.boxes()['surface_wave'], '右键框内应立刻在本地清掉该框（点了就有反馈）');
ok(FETCHED.filter(function(u){ return u.indexOf('/api/anno_box_clear') >= 0; }).length === 1,
   '右键框内应直连清框接口，实际 ' + JSON.stringify(FETCHED));

// 服务器确认后要回读一次目标（此时才 resolve，才能观察到回读）
T.setBoxes({surface_wave: {x0: 0, y0: 0, x1: 100, y1: 100}});
FETCH_PAYLOAD = {boxes: {}, target: 'near_shot_noise', note: ''};
FETCHED.length = 0;
T.onCtxMenu(mkEvent(50, 50, 2));
ok(FETCHED.some(function(u){ return u.indexOf('/api/boxes') >= 0; }),
   '清完服务端还应回读一次目标，实际 ' + JSON.stringify(FETCHED));
ok(!T.boxes()['surface_wave'], '回读后应以服务端为准（框确实清了）');
FETCH_PAYLOAD = null;

// ③ 无待定框 + 空白处 → 什么都不做（别误删）
T.setBoxes({surface_wave: {x0: 0, y0: 0, x1: 100, y1: 100}});
var ce3 = mkEvent(400, 900, 2);
FETCHED.length = 0;
T.onCtxMenu(ce3);
ok(T.boxes()['surface_wave'], '空白处右键不该清任何框');
ok(FETCHED.length === 0, '空白处右键不该发任何请求，实际 ' + JSON.stringify(FETCHED));
ok(ce3._prevented === true, '空白处右键也要吃掉浏览器菜单');

// ④ 右键落在另一个特征的框上 → 清的是那一个
T.setBoxes({surface_wave: {x0: 0, y0: 0, x1: 100, y1: 100},
            near_shot_noise: {x0: 200, y0: 200, x1: 400, y1: 400}});
FETCHED.length = 0;
T.onCtxMenu(mkEvent(300, 300, 2));
ok(T.boxes()['surface_wave'] && !T.boxes()['near_shot_noise'],
   '右键应只清鼠标下那个框，实际 ' + JSON.stringify(T.boxes()));

// ⑤ 清框前必须先取消还没到点的去抖回读 —— 否则它可能在服务端清掉**之前**发出，
// 把刚清掉的框又拉回来（clearKey 的注释里记过同一类时序坑）。
// FETCH_PAYLOAD 保持 null：postClear 的 then 链不 resolve，才不会掩盖"定时器有没有被取消"。
T.setBoxes({surface_wave: {x0: 0, y0: 0, x1: 100, y1: 100}});
T.scheduleTargetRefresh();                 // 先排一次去抖回读
T.onCtxMenu(mkEvent(50, 50, 2));           // 紧接着右键清框 → 应把上面那次取消掉
FETCHED.length = 0;
flushTimers();                             // 定时器若还在，就会在这里问一次 /api/boxes
ok(!FETCHED.some(function(u){ return u.indexOf('/api/boxes') >= 0; }),
   '清框时必须取消更早排下的去抖回读，实际 ' + JSON.stringify(FETCHED));
ok(!T.boxes()['surface_wave'], '框应保持清掉的状态');

// ---- 11. Ctrl 目标：服务端 outputs 落到 DOM 之后才回读 ----
// 回归锁：原来只在 document 的 change 监听里 fetchBoxes()，那比 Gradio 的
// on_radio_change 先跑，读到的 st['partial'] 还是改之前的 —— 于是
// 「两个都选不存在却仍能拉框」「改回存在反倒提示不存在无需画框」。
var fcol = observerOn('anno_featcol');
ok(fcol, 'setup 应在 #anno_featcol 上挂 MutationObserver（服务端 outputs 就写在这列）');

T.setBoxes({});
T.setTarget(null);
T.setGrab(null);
FETCHED.length = 0;
FETCH_PAYLOAD = {boxes: {}, target: 'surface_wave', note: ''};
fcol.cb();                       // 模拟 Gradio 写完 outputs → 触发观察者
flushTimers();                   // 去抖窗口结束
ok(FETCHED.length === 1 && FETCHED[0].indexOf('/api/boxes') >= 0,
   'featcol 变更后应回读一次 /api/boxes，实际 ' + JSON.stringify(FETCHED));
ok(T.targetKey() === 'surface_wave', '回读后 Ctrl 目标应更新为服务端算出的值');

// 同一批变更只问一次（去抖）
FETCHED.length = 0;
fcol.cb(); fcol.cb(); fcol.cb();
flushTimers();
ok(FETCHED.length === 1, '同一批 DOM 变更只该回读一次，实际 ' + FETCHED.length);

// 服务端说「两项均选不存在」→ 目标为空、提示语照原样拿到（并据此禁掉 Ctrl 画框）
FETCHED.length = 0;
FETCH_PAYLOAD = {boxes: {}, target: null, note: '两项均选「不存在」，本张无需画框'};
fcol.cb(); flushTimers();
ok(T.targetKey() === null, '服务端目标为空时应落到 null（Ctrl 随后不画框）');

// 拖动/缩放中收到回读 → 不能替换 boxes，否则手上的框会被打回服务端旧值
T.setGrab({type: 'move', key: 'surface_wave'});
T.setBoxes({surface_wave: {x0: 1, y0: 1, x1: 9, y1: 9}});
FETCHED.length = 0;
FETCH_PAYLOAD = {boxes: {surface_wave: {x0: 0, y0: 0, x1: 5, y1: 5}},
                 target: 'near_shot_noise', note: ''};
fcol.cb(); flushTimers();
ok(T.boxes()['surface_wave'].x0 === 1, '拖动中不该被回读覆盖本地框');
ok(T.targetKey() === 'near_shot_noise', '拖动中目标提示仍应更新（不影响几何）');
T.setGrab(null);

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

    def test_target_refresh_waits_for_server_outputs(self):
        """回归锁：Ctrl 目标的回读必须发生在**服务端 outputs 落到 DOM 之后**。

        原来的写法是在 document 的 change 监听里 fetchBoxes()。那个监听比 Gradio 的
        on_radio_change（写 st['partial'] 的那个）先跑，读到的还是改之前的选项；等服务端
        跑完，该事件的 outputs 里没有图片、place() 不触发，**再没人回读** —— 于是
        「两个都选「不存在」却仍能拉出框」「改回「存在」反倒提示不存在无需画框」。
        行为上的检验见 _HARNESS 第 11 节；这里再锁一次触发点，免得有人图省事加回去。
        """
        js = _anno_script()
        m = re.search(r"document\.addEventListener\('change'.*?\}, true\);", js, re.S)
        self.assertIsNotNone(m, "未能在 ANNO_JS 中定位 change 监听")
        self.assertNotIn("fetchBoxes()", m.group(0),
                         "change 监听里又直接回读 Ctrl 目标了：那里拿到的是改之前的选项")
        # 回读改挂在 featcol 观察者上（选项类 outputs 就写进这一列）
        self.assertIn("function scheduleTargetRefresh()", js)
        self.assertIn("scheduleTargetRefresh();", js)

    def test_scroll_hot_path_stays_cheap(self):
        """滚动卡顿的回归锁：滚动路径上不许做重排/重绘/抢滚动。

        v3.10.1 的卡顿三成因：①每次 scroll 同步 place()（重设 canvas.width 会重建画布
        缓冲并清空 + 全量重绘）②place() 里顺带 paintCursor()，其中的 scrollIntoView 会
        跟用户滚动抢方向盘 ③MutationObserver 连 class 一起听，Gradio 频繁切 class 就重排。
        """
        js = _anno_script()
        # ① 几何签名守卫 + 滚动走 rAF，且监听里不直接调 place()
        self.assertIn("sig === lastSig", js, "place() 缺少几何未变就跳过的守卫")
        self.assertNotIn("addEventListener('scroll', function(){ if (!grab) place(); }", js,
                         "滚动监听又变回同步 place() 了")
        self.assertIn("addEventListener('scroll', function(){ if (!grab) raf(", js,
                      "滚动监听应经 raf() 合并到一帧一次")
        # ② 所有 paintCursor 调用都必须显式给「是否滚动」参数，避免重绘时抢滚动
        self.assertNotIn("paintCursor();", js,
                         "存在无参 paintCursor() 调用：会在重绘时抢用户滚动")
        self.assertIn("function paintCursor(scrollIntoView)", js)
        # ③ 观察者不再监听 class
        self.assertIn("attributeFilter:['src']", js)
        self.assertNotIn("attributeFilter:['src','class']", js,
                         "观察者又在听 class 了（Gradio 切换 class 会频繁触发重排）")
        # canvas 尺寸赋值必须判等（赋同值也会清空画布）
        self.assertIn("if (canvas.width !== w2)", js)


if __name__ == "__main__":
    unittest.main()
