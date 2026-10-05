"use strict";

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const {
  autoFontSize,
  latexToSvgDataUri,
  svgToDataUri,
  safeOuterShadow,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./assets/pptxgenjs_helpers");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "李顺";
pptx.subject = "机器人仿真与运动控制技术面试";
pptx.title = "机器人仿真与运动控制";
pptx.company = "个人技术面试材料";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";
pptx.margin = 0;

const OUT = path.resolve(__dirname, "李顺_机器人仿真与运动控制_技术面试.pptx");
const ROOT = process.cwd();
const CSV_ON = path.join(
  ROOT,
  "mujoco_control_extract",
  "output",
  "spreadsheet",
  "stand_b_pitch0_3ms.csv"
);

const C = {
  navy: "10283F",
  navy2: "173A56",
  cyan: "12B8C8",
  cyan2: "55D6DF",
  orange: "FF9D4D",
  red: "E95F5F",
  ink: "173042",
  muted: "607587",
  paper: "F4F7F9",
  white: "FFFFFF",
  line: "D8E1E8",
  blueSoft: "DDF4F6",
  orangeSoft: "FFF0E3",
  green: "2CA58D",
  greenSoft: "E1F4EF",
  graySoft: "EAF0F4",
};

const FONT = "Microsoft YaHei";
const shadow = safeOuterShadow("000000", 0.14, 45, 2, 1);

function parseCsv(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = Number(values[i]);
    });
    return row;
  });
}

function addText(slide, text, x, y, w, h, options = {}) {
  const base = {
    x,
    y,
    w,
    h,
    fontFace: FONT,
    fontSize: 16,
    color: C.ink,
    margin: 0,
    valign: "mid",
    breakLine: false,
    ...options,
  };
  slide.addText(text, base);
}

function addAutoText(slide, text, x, y, w, h, options = {}) {
  const fitted = autoFontSize(text, FONT, {
    x,
    y,
    w,
    h,
    mode: "shrink",
    fontSize: options.fontSize || 24,
    minFontSize: options.minFontSize || 12,
    maxFontSize: options.fontSize || 24,
    margin: options.margin !== undefined ? options.margin : 0,
    bold: options.bold || false,
    valign: options.valign || "mid",
    breakLine: false,
  });
  slide.addText(text, { ...fitted, fontFace: FONT, color: C.ink, ...options });
}

function addRoundedRect(slide, x, y, w, h, fill, line = fill, radius = 0.08) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: radius,
    fill: { color: fill },
    line: { color: line, transparency: line === fill ? 100 : 0, width: 1 },
  });
}

function addPill(slide, text, x, y, w, fill, color, line = fill) {
  addRoundedRect(slide, x, y, w, 0.36, fill, line);
  addText(slide, text, x + 0.08, y + 0.01, w - 0.16, 0.32, {
    fontSize: 11.5,
    color,
    bold: true,
    align: "center",
  });
}

function addArrow(slide, x1, y1, x2, y2, color = C.cyan, width = 2.2, dash = "solid") {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: { color, width, beginArrowType: "none", endArrowType: "triangle", dash },
  });
}

function addDot(slide, x, y, r, fill, line = C.white, lineWidth = 1) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: x - r,
    y: y - r,
    w: 2 * r,
    h: 2 * r,
    fill: { color: fill },
    line: { color: line, width: lineWidth },
  });
}

function addSlideHeader(slide, pageNo, title, kicker) {
  slide.background = { color: C.paper };
  addText(slide, kicker.toUpperCase(), 0.62, 0.35, 2.5, 0.22, {
    fontSize: 9.5,
    color: C.cyan,
    bold: true,
    charSpacing: 1.2,
  });
  addAutoText(slide, title, 0.62, 0.62, 11.45, 0.6, {
    fontSize: 25,
    minFontSize: 20,
    bold: true,
    color: C.navy,
  });
  addText(slide, String(pageNo).padStart(2, "0"), 12.17, 0.44, 0.5, 0.28, {
    fontSize: 11,
    color: C.muted,
    bold: true,
    align: "right",
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.62,
    y: 1.3,
    w: 12.05,
    h: 0,
    line: { color: C.line, width: 1 },
  });
}

function addBottomPrompt(slide, text, pageNo) {
  // Intentional overlap: text sits on top of the prompt band.
  addRoundedRect(slide, 0.62, 6.82, 12.05, 0.42, C.navy, C.navy);
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.62,
    y: 6.82,
    w: 0.08,
    h: 0.42,
    fill: { color: C.cyan },
    line: { color: C.cyan, transparency: 100 },
  });
  addAutoText(slide, text, 0.86, 6.87, 10.95, 0.3, {
    fontSize: 12.5,
    minFontSize: 10.5,
    color: C.white,
    bold: true,
  });
  addText(slide, `P${pageNo}`, 11.94, 6.87, 0.45, 0.28, {
    fontSize: 9.5,
    color: C.cyan2,
    bold: true,
    align: "right",
  });
}

function addChip(slide, value, label, x, y, w, accent = C.cyan) {
  addRoundedRect(slide, x, y, w, 0.82, C.white, C.line);
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 0.06,
    h: 0.82,
    fill: { color: accent },
    line: { color: accent, transparency: 100 },
  });
  addAutoText(slide, value, x + 0.2, y + 0.08, w - 0.3, 0.34, {
    fontSize: 17,
    minFontSize: 11.5,
    color: C.navy,
    bold: true,
  });
  addAutoText(slide, label, x + 0.2, y + 0.43, w - 0.3, 0.24, {
    fontSize: 9.5,
    minFontSize: 8.2,
    color: C.muted,
  });
}

function addFormula(slide, tex, x, y, w, h) {
  slide.addImage({ data: latexToSvgDataUri(tex), x, y, w, h });
}

function drawRobotSchematic(slide, x, y, s = 1) {
  // Single SVG avoids PowerPoint/LibreOffice disagreements on negative line extents.
  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" width="800" height="560" viewBox="0 0 800 560">
    <rect x="105" y="35" width="590" height="150" rx="18" fill="#173A56" stroke="#55D6DF" stroke-width="4"/>
    <rect x="170" y="75" width="460" height="18" fill="#12B8C8"/>
    <text x="400" y="137" text-anchor="middle" font-family="Arial" font-size="25" font-weight="700" fill="#FFFFFF" letter-spacing="2">WHEEL-LEG / MUJOCO</text>
    <g fill="none" stroke="#55D6DF" stroke-width="10" stroke-linecap="round" stroke-linejoin="round">
      <path d="M185 185 L105 315 L175 455"/>
      <path d="M255 185 L315 300 L175 455"/>
      <path d="M545 185 L485 300 L625 455"/>
      <path d="M615 185 L695 315 L625 455"/>
    </g>
    <g fill="#FF9D4D" stroke="#FFFFFF" stroke-width="4">
      <circle cx="185" cy="185" r="16"/><circle cx="255" cy="185" r="16"/>
      <circle cx="545" cy="185" r="16"/><circle cx="615" cy="185" r="16"/>
    </g>
    <g fill="#FFFFFF" stroke="#12B8C8" stroke-width="4">
      <circle cx="105" cy="315" r="14"/><circle cx="315" cy="300" r="14"/>
      <circle cx="485" cy="300" r="14"/><circle cx="695" cy="315" r="14"/>
    </g>
    <g fill="#10283F" stroke="#55D6DF" stroke-width="7">
      <circle cx="175" cy="455" r="76"/><circle cx="625" cy="455" r="76"/>
    </g>
    <g fill="#FF9D4D" stroke="#FFFFFF" stroke-width="4">
      <circle cx="175" cy="455" r="19"/><circle cx="625" cy="455" r="19"/>
    </g>
  </svg>`;
  slide.addImage({ data: svgToDataUri(svg), x, y, w: 4.25 * s, h: 3.0 * s });
}

function drawLegGeometry(slide, x, y, s = 1) {
  // The complete diagram is one SVG object so all joints and links remain registered in PowerPoint.
  addRoundedRect(slide, x, y, 4.25 * s, 3.8 * s, C.white, C.line);
  addText(slide, "偏置闭链结构示意（非比例）", x + 0.22 * s, y + 0.18 * s, 2.4 * s, 0.25 * s, {
    fontSize: 10.5 * s,
    color: C.muted,
    bold: true,
  });

  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" width="620" height="540" viewBox="0 0 620 540">
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#FF9D4D"/></marker>
    </defs>
    <g fill="none" stroke-linecap="round" stroke-linejoin="round">
      <path d="M165 105 L105 270" stroke="#173A56" stroke-width="11"/>
      <path d="M455 105 L505 285" stroke="#173A56" stroke-width="11"/>
      <path d="M105 270 L300 405" stroke="#12B8C8" stroke-width="9"/>
      <path d="M505 285 L300 405" stroke="#12B8C8" stroke-width="9"/>
      <path d="M300 405 L330 500" stroke="#12B8C8" stroke-width="9"/>
      <path d="M310 80 L330 500" stroke="#FF9D4D" stroke-width="5" stroke-dasharray="13 10"/>
      <path d="M375 450 L375 325" stroke="#FF9D4D" stroke-width="6" marker-end="url(#arrow)"/>
      <path d="M335 105 C395 120 420 170 398 220" stroke="#FF9D4D" stroke-width="6" marker-end="url(#arrow)"/>
    </g>
    <g fill="#FFFFFF" stroke="#173A56" stroke-width="4">
      <circle cx="105" cy="270" r="15"/><circle cx="505" cy="285" r="15"/>
      <circle cx="300" cy="405" r="16"/><circle cx="330" cy="500" r="16"/><circle cx="310" cy="80" r="15"/>
    </g>
    <g fill="#FF9D4D" stroke="#FFFFFF" stroke-width="4">
      <circle cx="165" cy="105" r="17"/><circle cx="455" cy="105" r="17"/>
    </g>
    <g font-family="Arial" font-weight="700" font-size="24" fill="#173042">
      <text x="178" y="98">qf</text><text x="468" y="98">qr</text>
      <text x="122" y="262">J</text><text x="522" y="278">L</text>
      <text x="319" y="400">M</text><text x="350" y="505">W</text><text x="328" y="76">I</text>
    </g>
    <g font-family="Arial" font-weight="700" font-size="25" fill="#FF9D4D">
      <text x="385" y="372">F0</text><text x="345" y="120">Tp</text>
      <text transform="translate(280 340) rotate(-87)">L0 / phi0</text>
    </g>
  </svg>`;
  slide.addImage({ data: svgToDataUri(svg), x: x + 0.23 * s, y: y + 0.55 * s, w: 3.78 * s, h: 3.06 * s });
}

function runWarnings(slide) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

// ----------------------------- Slide 1 -----------------------------
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.12,
    fill: { color: C.cyan },
    line: { color: C.cyan, transparency: 100 },
  });
  addText(slide, "TECHNICAL INTERVIEW · 2026", 0.72, 0.62, 4.2, 0.28, {
    fontSize: 10.5,
    color: C.cyan2,
    bold: true,
    charSpacing: 1.6,
  });
  addAutoText(slide, "机器人仿真与运动控制", 0.72, 1.18, 7.0, 0.78, {
    fontSize: 31,
    minFontSize: 27,
    color: C.white,
    bold: true,
  });
  addAutoText(slide, "偏置五连杆串腿机器人的运动学、VMC 与 MuJoCo 仿真分析", 0.74, 2.06, 6.55, 0.68, {
    fontSize: 18,
    minFontSize: 14,
    color: "DCEAF2",
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.74,
    y: 2.92,
    w: 5.7,
    h: 0,
    line: { color: "31536C", width: 1.2 },
  });
  addText(slide, "李顺", 0.74, 3.15, 1.0, 0.34, {
    fontSize: 18,
    color: C.white,
    bold: true,
  });
  // Intentional tight alignment: name and school form one compact identity row.
  addText(slide, "香港中文大学（深圳） · 电子与计算机工程", 1.65, 3.16, 4.6, 0.3, {
    fontSize: 12.5,
    color: "B9CAD6",
  });
  addPill(slide, "MuJoCo", 0.74, 3.78, 1.05, C.navy2, C.cyan2, "31536C");
  addPill(slide, "运动学", 1.92, 3.78, 1.05, C.navy2, C.cyan2, "31536C");
  addPill(slide, "VMC", 3.1, 3.78, 0.85, C.navy2, C.cyan2, "31536C");
  addPill(slide, "仿真验证", 4.08, 3.78, 1.15, C.navy2, C.cyan2, "31536C");
  addPill(slide, "RoboMaster", 5.36, 3.78, 1.28, C.navy2, C.cyan2, "31536C");

  addRoundedRect(slide, 0.74, 4.55, 6.15, 1.05, C.navy2, "31536C");
  addText(slide, "近期学习", 0.98, 4.77, 0.85, 0.25, {
    fontSize: 10,
    color: C.orange,
    bold: true,
  });
  addText(slide, "机器人强化学习运动控制项目（起步阶段）", 1.85, 4.71, 4.55, 0.35, {
    fontSize: 13.5,
    color: C.white,
    bold: true,
  });
  addText(slide, "当前重点：跑通训练流程，理解观测、动作、奖励与低层控制接口", 0.98, 5.12, 5.45, 0.25, {
    fontSize: 10.5,
    color: "B9CAD6",
  });

  drawRobotSchematic(slide, 8.0, 1.38, 1.12);
  addText(slide, "MODEL  →  CONTROL  →  VERIFY", 8.43, 5.34, 3.65, 0.26, {
    fontSize: 10,
    color: C.cyan2,
    bold: true,
    align: "center",
    charSpacing: 1.3,
  });
  addRoundedRect(slide, 0.72, 6.63, 11.9, 0.44, C.white, C.white);
  addText(slide, "本页提示｜我的主要项目基础是 MuJoCo 仿真、机构运动学与机器人运动控制。", 0.96, 6.7, 11.2, 0.28, {
    fontSize: 12,
    color: C.navy,
    bold: true,
  });
  slide.addNotes(`讲述建议（约 50 秒）：\n1. 我是香港中文大学（深圳）电子与计算机工程专业大二学生。\n2. 目前主要关注机器人仿真和运动控制，最深入的项目是偏置五连杆串腿机器人的 MuJoCo 仿真分析。\n3. 我的重点不是强调会多少编程语言，而是机构运动学、VMC 物理含义和仿真问题分析。\n4. 最近也开始运行一个机器人强化学习运动控制项目，目前仍在起步阶段，因此本次主要讲掌握更深入的串腿项目。`);
  runWarnings(slide);
}

// ----------------------------- Slide 2 -----------------------------
{
  const slide = pptx.addSlide();
  addSlideHeader(slide, 2, "基于现有轮腿控制器，在 MuJoCo 中形成状态—控制—执行闭环", "SYSTEM LOOP");

  const y = 2.0;
  const blocks = [
    { x: 0.72, w: 2.1, title: "MuJoCo 物理模型", sub: "机构 / 接触 / 传感器状态", fill: C.blueSoft, accent: C.cyan },
    { x: 3.25, w: 2.0, title: "仿真适配层", sub: "关节映射 / 周期 / 接口", fill: C.graySoft, accent: C.navy2 },
    { x: 5.7, w: 2.15, title: "现有控制器", sub: "运动学 / VMC / LQR / MIT", fill: C.orangeSoft, accent: C.orange },
    { x: 8.3, w: 1.92, title: "执行器输出", sub: "关节力矩 / 轮端力矩", fill: C.greenSoft, accent: C.green },
    { x: 10.67, w: 1.92, title: "分层观测", sub: "状态 / 力矩 / 饱和 / A/B", fill: C.blueSoft, accent: C.cyan },
  ];
  blocks.forEach((b, i) => {
    addRoundedRect(slide, b.x, y, b.w, 1.2, b.fill, C.line);
    slide.addShape(pptx.ShapeType.rect, {
      x: b.x,
      y,
      w: b.w,
      h: 0.09,
      fill: { color: b.accent },
      line: { color: b.accent, transparency: 100 },
    });
    addText(slide, String(i + 1).padStart(2, "0"), b.x + 0.16, y + 0.2, 0.36, 0.22, {
      fontSize: 9.5,
      color: b.accent,
      bold: true,
    });
    addAutoText(slide, b.title, b.x + 0.16, y + 0.42, b.w - 0.32, 0.3, {
      fontSize: 14,
      minFontSize: 11,
      color: C.navy,
      bold: true,
    });
    addAutoText(slide, b.sub, b.x + 0.16, y + 0.8, b.w - 0.32, 0.24, {
      fontSize: 10.2,
      minFontSize: 8.5,
      color: C.muted,
    });
    if (i < blocks.length - 1) {
      addArrow(slide, b.x + b.w + 0.08, y + 0.6, blocks[i + 1].x - 0.08, y + 0.6, C.cyan, 1.8);
    }
  });

  addText(slide, "状态输入", 1.05, 3.47, 1.1, 0.25, { fontSize: 10.5, color: C.cyan, bold: true });
  addText(slide, "q / q̇ / 姿态 / 角速度", 1.05, 3.78, 2.55, 0.3, { fontSize: 13, color: C.navy, bold: true });
  addText(slide, "控制输出", 5.92, 3.47, 1.1, 0.25, { fontSize: 10.5, color: C.orange, bold: true });
  addText(slide, "F₀ / Tₚ / 关节与轮端力矩", 5.92, 3.78, 2.85, 0.3, { fontSize: 13, color: C.navy, bold: true });
  addText(slide, "验证输出", 10.03, 3.47, 1.1, 0.25, { fontSize: 10.5, color: C.green, bold: true });
  addText(slide, "时序曲线 / 对照实验 / 统计量", 10.03, 3.78, 2.4, 0.3, { fontSize: 13, color: C.navy, bold: true });

  addRoundedRect(slide, 0.72, 4.45, 11.87, 1.62, C.white, C.line);
  // Intentional tight alignment: the section label shares a row with the three ownership columns.
  addText(slide, "工作边界", 0.96, 4.68, 1.2, 0.28, { fontSize: 12, color: C.navy, bold: true });
  const cols = [
    { x: 2.1, label: "项目基础", title: "已有轮腿控制代码与 MuJoCo 模型", color: C.cyan },
    { x: 5.67, label: "个人参与", title: "偏置运动学调整、问题理解与方案讨论", color: C.orange },
    { x: 9.28, label: "AI 辅助", title: "仿真接口、分层遥测与实验脚本实现", color: C.green },
  ];
  cols.forEach((c, i) => {
    if (i > 0) {
      slide.addShape(pptx.ShapeType.line, {
        x: c.x - 0.25,
        y: 4.74,
        w: 0,
        h: 0.94,
        line: { color: C.line, width: 1 },
      });
    }
    addText(slide, c.label, c.x, 4.69, 1.0, 0.24, { fontSize: 9.5, color: c.color, bold: true });
    addAutoText(slide, c.title, c.x, 5.06, 2.82, 0.55, {
      fontSize: 12.5,
      minFontSize: 10.5,
      color: C.ink,
      bold: true,
      valign: "top",
    });
  });

  addBottomPrompt(slide, "MuJoCo 提供状态，已有控制器计算控制量，关节与轮端力矩返回物理仿真。", 2);
  slide.addNotes(`讲述建议（约 1 分 30 秒）：\n1. 这个项目不是从零重新写控制器，而是在已有轮腿控制工程基础上形成 MuJoCo 闭环。\n2. MuJoCo 负责机构、接触和传感器状态；适配层负责关节顺序、符号、时间周期与数据结构；控制器计算 VMC、LQR、MIT 和轮端控制量；最终力矩写回执行器。\n3. 我的真实参与集中在偏置五连杆运动学调整、理解控制链和讨论验证方案。接口、遥测和实验工具主要由 AI 辅助实现。\n4. 这一页讲的是系统如何工作，不把系统实现全部归为个人独立开发。`);
  runWarnings(slide);
}

// ----------------------------- Slide 3 -----------------------------
{
  const slide = pptx.addSlide();
  addSlideHeader(slide, 3, "雅可比描述微分运动，雅可比转置完成虚拟力到关节力矩的映射", "KINEMATICS → VMC");
  drawLegGeometry(slide, 0.72, 1.62, 1.0);

  addPill(slide, "运动变量", 5.33, 1.62, 1.1, C.blueSoft, C.navy2, C.line);
  addFormula(slide, String.raw`\mathbf y=\begin{bmatrix}L_0\\\phi_0\end{bmatrix},\quad \delta\mathbf y=\mathbf J_H\,\delta\mathbf q`, 5.42, 2.02, 3.15, 0.7);
  addText(slide, "JH：关节微小转动 → 虚拟腿长与腿角变化", 5.52, 2.76, 3.3, 0.27, {
    fontSize: 10.5,
    color: C.muted,
  });

  addPill(slide, "广义力", 8.92, 1.62, 1.05, C.orangeSoft, C.orange, C.line);
  addFormula(slide, String.raw`\mathbf f_v=\begin{bmatrix}F_0\\T_p\end{bmatrix}`, 9.02, 2.02, 1.85, 0.7);
  addText(slide, "F₀：轴向力    Tₚ：虚拟腿角广义力矩", 8.99, 2.76, 3.0, 0.27, {
    fontSize: 10.5,
    color: C.muted,
  });

  addRoundedRect(slide, 5.33, 3.2, 7.25, 1.43, C.graySoft, C.line);
  // Intentional composition: label, equation, arrow and boxed result form one derivation strip.
  addText(slide, "虚功一致", 5.62, 3.45, 1.05, 0.27, {
    fontSize: 10.5,
    color: C.cyan,
    bold: true,
  });
  addFormula(slide, String.raw`\delta W=\mathbf f_v^T\delta\mathbf y=\boldsymbol\tau_q^T\delta\mathbf q`, 6.44, 3.38, 2.8, 0.44);
  addArrow(slide, 9.48, 3.92, 10.15, 3.92, C.orange, 1.8);
  addFormula(slide, String.raw`\boxed{\boldsymbol\tau_q=\mathbf J_H^T\begin{bmatrix}F_0\\T_p\end{bmatrix}}`, 9.98, 3.35, 2.2, 0.7);
  addText(slide, "结果是两台主动关节电机的 raw VMC 力矩", 7.03, 4.1, 2.55, 0.26, {
    fontSize: 9.7,
    color: C.muted,
    align: "center",
  });

  addChip(slide, "122", "工作空间采样点", 5.33, 4.96, 1.65, C.cyan);
  addChip(slide, "3.58×10⁻¹⁰", "解析 JH vs. 有限差分", 7.19, 4.96, 1.65, C.orange);
  addChip(slide, "1.78×10⁻¹⁵ W", "虚功残差", 9.05, 4.96, 1.65, C.green);
  addChip(slide, "0.270 mm", "CAD vs. 完整 XML", 10.91, 4.96, 1.67, C.cyan);

  addRoundedRect(slide, 5.33, 5.98, 7.25, 0.56, C.orangeSoft, C.orangeSoft);
  addText(slide, "边界：Tₚ 不是轮毂电机力矩；raw VMC 力矩后面仍会叠加 MIT、符号映射与限幅。", 5.58, 6.11, 6.75, 0.27, {
    fontSize: 10.5,
    color: "8A4A18",
    bold: true,
  });

  addBottomPrompt(slide, "JH 映射运动；JHᵀ 根据虚功一致性映射与运动共轭的广义力。", 3);
  slide.addNotes(`讲述建议（约 3 分钟）：\n1. 我从两个主动关节角出发，先根据偏置闭链几何求轮轴位置 W，再得到虚拟腿长 L0 和腿角 phi0。\n2. 微分运动学写成 delta y = JH delta q。JH 的每个元素都是当前姿态下的瞬时传动系数。\n3. F0 与 L0 共轭，是沿虚拟腿方向的力；Tp 与 phi0 共轭，是让虚拟腿转动的广义力矩。Tp 不是轮毂力矩，也不是已经存在的某个髋关节电机力矩。\n4. 根据虚功一致：fv^T delta y = tau^T delta q。代入 delta y = JH delta q，并利用 fv^T JH = (JH^T fv)^T，可得 tau = JH^T fv。\n5. 所以 JH 映射运动，JH 转置映射共轭力。得到的是两个主动关节的 raw VMC 力矩。\n6. 数值验证证明运动学和力映射自洽，但不能证明闭环稳定或实车极性正确。`);
  runWarnings(slide);
}

// ----------------------------- Slide 4 -----------------------------
{
  const rows = parseCsv(CSV_ON).filter((r) => r.t_s >= 1.0);
  const sampled = rows.filter((_, i) => i % 10 === 0);
  const labels = sampled.map((r, i) => (i % 10 === 0 ? r.t_s.toFixed(1) : ""));
  const series = [
    { name: "MIT", labels, values: sampled.map((r) => r.tau_mit_j0_nm) },
    { name: "VMC", labels, values: sampled.map((r) => r.tau_vmc_j0_nm) },
    { name: "最终输出", labels, values: sampled.map((r) => r.tau_final_j0_nm) },
  ];

  const slide = pptx.addSlide();
  addSlideHeader(slide, 4, "分层力矩表明：站立摇摆不能只靠调整 LQR 参数解决", "LAYERED DIAGNOSIS");

  addRoundedRect(slide, 0.72, 1.58, 7.9, 4.76, C.white, C.line);
  addText(slide, "同一关节的分层力矩时序（pitch=0，MIT on，3 ms）", 0.98, 1.82, 5.4, 0.28, {
    fontSize: 12.5,
    color: C.navy,
    bold: true,
  });
  addText(slide, "t = 1–10 s · 单位 N·m", 6.7, 1.83, 1.55, 0.25, {
    fontSize: 9.5,
    color: C.muted,
    align: "right",
  });
  slide.addChart(pptx.ChartType.line, series, {
    x: 1.0,
    y: 2.22,
    w: 7.3,
    h: 3.28,
    showLegend: true,
    legendPos: "b",
    legendFontFace: FONT,
    legendFontSize: 9,
    chartColors: [C.orange, C.cyan, C.navy2],
    lineSize: 2.2,
    lineDataSymbol: "none",
    showTitle: false,
    showValue: false,
    catAxisLabelFontFace: FONT,
    catAxisLabelFontSize: 8,
    valAxisLabelFontFace: FONT,
    valAxisLabelFontSize: 8,
    valAxisMinVal: -9,
    valAxisMaxVal: 9,
    valAxisMajorUnit: 3,
    showCatName: false,
    catAxisTitle: "时间 / s",
    valAxisTitle: "关节力矩 / N·m",
    catAxisTitleFontFace: FONT,
    valAxisTitleFontFace: FONT,
    catAxisTitleFontSize: 9,
    valAxisTitleFontSize: 9,
    valGridLine: { color: "DCE5EB", width: 1 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: C.white, transparency: 100 }, border: { color: C.white, transparency: 100 } },
    plotArea: { fill: { color: C.white, transparency: 100 }, border: { color: C.white, transparency: 100 } },
    showBorder: false,
  });
  addRoundedRect(slide, 1.08, 5.62, 7.15, 0.47, C.graySoft, C.graySoft);
  addText(slide, "读图：MIT 与 VMC 在主要工作区间方向相反，最终输出明显小于任一分量。", 1.28, 5.72, 6.75, 0.25, {
    fontSize: 10.2,
    color: C.ink,
    bold: true,
  });

  addRoundedRect(slide, 8.92, 1.58, 3.67, 2.26, C.navy, C.navy);
  addText(slide, "同一工况的统计摘要", 9.2, 1.83, 2.7, 0.28, {
    fontSize: 12.5,
    color: C.white,
    bold: true,
  });
  const stats = [
    ["MIT", "6.29", C.orange],
    ["VMC", "8.43", C.cyan2],
    ["最终", "2.15", C.white],
  ];
  stats.forEach((s, i) => {
    addText(slide, s[0], 9.23, 2.32 + 0.42 * i, 0.72, 0.25, { fontSize: 10, color: "B9CAD6", bold: true });
    addText(slide, s[1], 10.1, 2.27 + 0.42 * i, 0.92, 0.3, { fontSize: 17, color: s[2], bold: true, align: "right" });
    addText(slide, "N·m", 11.12, 2.32 + 0.42 * i, 0.48, 0.22, { fontSize: 8.5, color: "B9CAD6" });
  });
  addText(slide, "平均绝对值；抵消仍以时序符号为主要证据", 9.2, 3.5, 2.95, 0.22, {
    fontSize: 8.7,
    color: "B9CAD6",
  });

  addRoundedRect(slide, 8.92, 4.08, 3.67, 2.26, C.white, C.line);
  addText(slide, "MIT on / off 隔离实验", 9.2, 4.33, 2.7, 0.28, {
    fontSize: 12.5,
    color: C.navy,
    bold: true,
  });
  addText(slide, "最大关节摆幅", 9.2, 4.78, 1.35, 0.25, { fontSize: 9.5, color: C.muted });
  addText(slide, "0.54°", 10.7, 4.7, 0.9, 0.35, { fontSize: 19, color: C.cyan, bold: true, align: "right" });
  addText(slide, "MIT ON", 11.7, 4.79, 0.55, 0.22, { fontSize: 8.5, color: C.muted, bold: true });
  slide.addShape(pptx.ShapeType.line, {
    x: 9.2,
    y: 5.18,
    w: 3.02,
    h: 0,
    line: { color: C.line, width: 1 },
  });
  addText(slide, "最大关节摆幅", 9.2, 5.35, 1.35, 0.25, { fontSize: 9.5, color: C.muted });
  addText(slide, "7.19°", 10.7, 5.27, 0.9, 0.35, { fontSize: 19, color: C.orange, bold: true, align: "right" });
  addText(slide, "MIT OFF", 11.7, 5.36, 0.62, 0.22, { fontSize: 8.5, color: C.muted, bold: true });
  addText(slide, "关闭 MIT 暴露冲突，但同时放大关节运动。", 9.2, 5.86, 3.0, 0.26, {
    fontSize: 9.7,
    color: "8A4A18",
    bold: true,
  });

  addBottomPrompt(slide, "先区分 MIT、VMC 与最终输出，再决定调哪一层；关闭 MIT 只是诊断，不是最终方案。", 4);
  slide.addNotes(`讲述建议（约 3 分钟）：\n1. 看到站立摇摆时，我没有把问题直接归因于 LQR，因为最终关节力矩同时包含 MIT 位置项和 VMC 力矩。\n2. 左图记录同一关节的 MIT、VMC 和最终输出。时序上 MIT 与 VMC 在主要区间方向相反，最终输出明显变小。\n3. 右上角三个数是平均绝对值，只用于摘要；严格的抵消判断要依赖同一时刻的符号和时序关系，不能只凭 8.43 减 6.29。\n4. 在保持 pitch=0、3 ms 等条件相同的情况下关闭 MIT，最大关节摆幅由约 0.54 度增大到 7.19 度。说明 MIT 与 VMC 确实存在目标冲突，但直接关闭 MIT 会破坏姿态约束，不能作为最终方案。\n5. 下一步应该建立与目标腿长一致的 MIT 位置目标，或重新设计纯 VMC 阻尼，然后再调 LQR。\n6. 数据来自当前 MuJoCo 10 秒站立实验，不等于实车稳定性证明。`);
  runWarnings(slide);
}

// ----------------------------- Slide 5 -----------------------------
{
  const slide = pptx.addSlide();
  addSlideHeader(slide, 5, "近期学习：机器人强化学习运动控制（起步阶段）", "CURRENT LEARNING");

  addRoundedRect(slide, 0.72, 1.62, 12.0, 1.55, C.navy, C.navy);
  const flow = [
    ["观测", "姿态 / 关节状态 / 指令"],
    ["策略网络", "从状态生成动作"],
    ["动作", "关节目标或力矩"],
    ["低层控制", "PD / 执行器 / 延迟"],
    ["物理仿真", "接触 / 摩擦 / 机体响应"],
    ["奖励与评测", "跟踪 / 稳定 / 平滑"],
  ];
  flow.forEach((item, i) => {
    const x = 0.98 + i * 1.93;
    addRoundedRect(slide, x, 1.94, 1.54, 0.9, i % 2 === 0 ? C.navy2 : "1E4968", "31536C");
    addText(slide, item[0], x + 0.12, 2.08, 1.3, 0.25, { fontSize: 11, color: C.cyan2, bold: true, align: "center" });
    addAutoText(slide, item[1], x + 0.12, 2.39, 1.3, 0.3, { fontSize: 9.4, minFontSize: 8, color: C.white, align: "center" });
    if (i < flow.length - 1) addArrow(slide, x + 1.6, 2.39, x + 1.85, 2.39, C.orange, 1.5);
  });

  addRoundedRect(slide, 0.72, 3.55, 5.75, 2.64, C.white, C.line);
  addText(slide, "已有基础可以迁移", 1.0, 3.84, 2.15, 0.3, { fontSize: 15, color: C.navy, bold: true });
  const leftBullets = [
    "MuJoCo 中状态、控制量与执行器接口的闭环理解",
    "控制周期、符号映射、限幅和接触对运动结果的影响",
    "通过时序数据区分算法输出与最终执行器行为",
  ];
  leftBullets.forEach((t, i) => {
    addDot(slide, 1.08, 4.43 + 0.53 * i, 0.055, C.cyan, C.cyan, 0);
    addAutoText(slide, t, 1.25, 4.28 + 0.53 * i, 4.75, 0.42, {
      fontSize: 12.5,
      minFontSize: 10.5,
      color: C.ink,
    });
  });

  addRoundedRect(slide, 6.75, 3.55, 5.97, 2.64, C.orangeSoft, C.orangeSoft);
  addText(slide, "当前阶段与下一步", 7.04, 3.84, 2.15, 0.3, { fontSize: 15, color: C.navy, bold: true });
  const rightItems = [
    ["当前", "正在运行相关强化学习运动控制项目，仍处于起步阶段"],
    ["关注", "观测、动作、奖励、低层控制与训练日志之间的对应关系"],
    ["下一步", "跑通可复现基线，再分析稳定性、动作平滑与执行器建模"],
  ];
  rightItems.forEach((item, i) => {
    addPill(slide, item[0], 7.04, 4.27 + 0.57 * i, 0.72, i === 0 ? C.orange : C.white, i === 0 ? C.white : C.orange, i === 0 ? C.orange : C.line);
    addAutoText(slide, item[1], 7.95, 4.23 + 0.57 * i, 4.25, 0.45, {
      fontSize: 11.5,
      minFontSize: 9.5,
      color: C.ink,
    });
  });

  addBottomPrompt(slide, "当前重点是跑通训练闭环，并理解观测、动作、奖励与低层控制接口。", 5);
  slide.addNotes(`讲述建议（约 1 分 30 秒）：\n1. 我最近确实开始运行一个机器人强化学习运动控制项目，但目前仍处于起步阶段，所以不把它包装成已经完成的项目成果。\n2. 我目前重点理解训练闭环：观测进入策略网络，策略输出关节目标或力矩，再经过低层控制和执行器作用到物理仿真，最后由奖励和评测反馈。\n3. 串腿项目带来的可迁移基础，是我已经理解控制周期、执行器接口、符号与限幅、接触动力学和分层数据的重要性。\n4. 下一步目标是先跑通可复现基线，再分析动作跟踪、稳定性、平滑性和执行器建模。`);
  runWarnings(slide);
}

pptx.writeFile({ fileName: OUT, compression: true });
