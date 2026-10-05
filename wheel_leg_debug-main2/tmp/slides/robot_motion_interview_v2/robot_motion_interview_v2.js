"use strict";

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("../robot_motion_interview/node_modules/pptxgenjs");
const {
  imageSizingContain,
  imageSizingCrop,
  safeOuterShadow,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./assets/pptxgenjs_helpers");

const pptx = new PptxGenJS();
pptx.defineLayout({ name: "INTERVIEW_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "INTERVIEW_WIDE";
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
pptx.margin = 0;

const W = 13.333;
const H = 7.5;
const FONT = "Microsoft YaHei";
const C = {
  navy: "10283F",
  navy2: "183B56",
  cyan: "12B8C8",
  cyanDark: "0E8E9C",
  cyanSoft: "E7F7F9",
  orange: "F28C45",
  orangeSoft: "FFF1E7",
  green: "20A66A",
  greenSoft: "E9F7F0",
  blue: "3869DF",
  blueSoft: "EDF2FF",
  ink: "172B3D",
  muted: "657789",
  line: "D7E1E8",
  paper: "F5F8FA",
  white: "FFFFFF",
  graySoft: "EDF2F5",
  danger: "D85959",
};
const shadow = safeOuterShadow("10283F", 0.12, 45, 1.2, 0.6);

const assets = path.join(__dirname, "assets");
const out = path.join(__dirname, "李顺_机器人仿真与运动控制_技术面试_v2.pptx");
const controlRoute = path.join(assets, "control_route.png");
const solidworks = path.join(assets, "solidworks_offset_sketch.png");
const mujoco = path.join(assets, "mujoco_suspended_crop.png");
const curveCsv = path.join(assets, "new_vmc_error_envelope.csv");
const metrics = JSON.parse(fs.readFileSync(path.join(assets, "new_vmc_validation_metrics.json"), "utf8"));

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: FONT,
    fontSize: 15,
    color: C.ink,
    margin: 0,
    breakLine: false,
    valign: "mid",
    fit: "shrink",
    ...options,
  });
}

function addRound(slide, x, y, w, h, fill, line = fill, options = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: fill, transparency: options.transparency || 0 },
    line: { color: line, width: options.lineWidth || 1, transparency: options.lineTransparency || 0 },
    shadow: options.shadow ? { ...options.shadow } : undefined,
  });
}

function addLine(slide, x, y, w, h, color = C.line, width = 1, arrow = false, dash = "solid") {
  let drawX = x;
  let drawY = y;
  let drawW = w;
  let drawH = h;
  let flipH = false;
  let flipV = false;
  if (drawW < 0) {
    drawX += drawW;
    drawW = Math.abs(drawW);
    flipH = true;
  }
  if (drawH < 0) {
    drawY += drawH;
    drawH = Math.abs(drawH);
    flipV = true;
  }
  slide.addShape(pptx.ShapeType.line, {
    x: drawX, y: drawY, w: drawW, h: drawH, flipH, flipV,
    line: {
      color,
      width,
      dashType: dash,
      beginArrowType: "none",
      endArrowType: arrow ? "triangle" : "none",
    },
  });
}

function addHeader(slide, eyebrow, title, page) {
  slide.background = { color: C.paper };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: W, h: 0.09,
    fill: { color: C.cyan }, line: { color: C.cyan, transparency: 100 },
  });
  addText(slide, eyebrow, 0.72, 0.28, 4.2, 0.24, {
    fontSize: 10.5, bold: true, color: C.cyanDark, charSpacing: 1.2,
  });
  addText(slide, title, 0.72, 0.58, 11.55, 0.48, {
    fontSize: 25.5, bold: true, color: C.navy,
  });
  addText(slide, String(page).padStart(2, "0"), 12.28, 0.40, 0.35, 0.30, {
    fontSize: 11.5, bold: true, color: C.muted, align: "right",
  });
  addLine(slide, 0.72, 1.13, 11.90, 0, C.line, 1);
}

function addFooter(slide, label, page) {
  addText(slide, label, 0.72, 7.18, 5.9, 0.16, {
    fontSize: 8.5, color: "8796A4",
  });
  addText(slide, `P${page}`, 12.18, 7.16, 0.42, 0.18, {
    fontSize: 8.5, bold: true, color: C.cyanDark, align: "right",
  });
}

function addTag(slide, text, x, y, w) {
  addRound(slide, x, y, w, 0.42, C.cyanSoft, "B7E6EA");
  addText(slide, text, x, y, w, 0.42, {
    fontSize: 11.5, bold: true, color: C.cyanDark, align: "center",
  });
}

function addNumberDot(slide, number, x, y, color = C.cyan) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: 0.28, h: 0.28,
    fill: { color }, line: { color, transparency: 100 },
  });
  addText(slide, String(number), x, y - 0.005, 0.28, 0.28, {
    fontSize: 9.5, bold: true, color: C.white, align: "center",
  });
}

function addFlowBox(slide, number, title, detail, x, y, w, accent = C.cyan) {
  addRound(slide, x, y, w, 0.67, C.white, C.line, { shadow });
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w: 0.055, h: 0.67,
    fill: { color: accent }, line: { color: accent, transparency: 100 },
  });
  addNumberDot(slide, number, x + 0.18, y + 0.19, accent);
  addText(slide, title, x + 0.56, y + 0.09, w - 0.72, 0.23, {
    fontSize: 12.4, bold: true, color: C.navy,
  });
  addText(slide, detail, x + 0.56, y + 0.34, w - 0.72, 0.20, {
    fontSize: 9.1, color: C.muted,
  });
}

function parseCurveCsv() {
  const lines = fs.readFileSync(curveCsv, "utf8").trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",").map(Number);
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function addCurveChart(slide, cfg) {
  const { x, y, w, h, title, unit, rows, field, ymax, color, maxLabel } = cfg;
  addRound(slide, x, y, w, h, C.white, C.line, { shadow });
  addText(slide, title, x + 0.25, y + 0.18, w - 1.85, 0.28, {
    fontSize: 15.2, bold: true, color: C.navy,
  });
  addRound(slide, x + w - 1.63, y + 0.14, 1.36, 0.43, C.orangeSoft, "F6C8A4");
  addText(slide, maxLabel, x + w - 1.63, y + 0.14, 1.36, 0.43, {
    fontSize: 8.8, bold: true, color: "9A4F1D", align: "center", breakLine: true,
  });

  const left = x + 0.62;
  const top = y + 0.72;
  const pw = w - 0.95;
  const ph = h - 1.22;
  const minX = rows[0].target_l0_m;
  const maxX = rows[rows.length - 1].target_l0_m;
  const X = (value) => left + ((value - minX) / (maxX - minX)) * pw;
  const Y = (value) => top + ph - (value / ymax) * ph;

  for (let i = 0; i <= 4; i += 1) {
    const value = (ymax * i) / 4;
    const yy = Y(value);
    addLine(slide, left, yy, pw, 0, i === 0 ? "A8B6C2" : "E5EBEF", i === 0 ? 1.1 : 0.7);
    addText(slide, value.toFixed(ymax < 0.5 ? 3 : 2), x + 0.10, yy - 0.10, 0.44, 0.18, {
      fontSize: 7.8, color: C.muted, align: "right",
    });
  }
  addLine(slide, left, top, 0, ph, "A8B6C2", 1.1);

  const tickIndices = [0, 9, 18, rows.length - 1];
  tickIndices.forEach((index) => {
    const xx = X(rows[index].target_l0_m);
    addLine(slide, xx, top + ph, 0, 0.06, "A8B6C2", 0.8);
    addText(slide, rows[index].target_l0_m.toFixed(2), xx - 0.22, top + ph + 0.09, 0.44, 0.18, {
      fontSize: 8.2, color: C.muted, align: "center",
    });
  });

  for (let i = 1; i < rows.length; i += 1) {
    const x1 = X(rows[i - 1].target_l0_m);
    const y1 = Y(rows[i - 1][field]);
    const x2 = X(rows[i].target_l0_m);
    const y2 = Y(rows[i][field]);
    addLine(slide, x1, y1, x2 - x1, y2 - y1, color, 2.2);
  }
  rows.filter((_, index) => index % 3 === 0 || index === rows.length - 1).forEach((row) => {
    const xx = X(row.target_l0_m);
    const yy = Y(row[field]);
    slide.addShape(pptx.ShapeType.ellipse, {
      x: xx - 0.035, y: yy - 0.035, w: 0.07, h: 0.07,
      fill: { color: C.white }, line: { color, width: 1.3 },
    });
  });
  const last = rows[rows.length - 1];
  slide.addShape(pptx.ShapeType.ellipse, {
    x: X(last.target_l0_m) - 0.055, y: Y(last[field]) - 0.055, w: 0.11, h: 0.11,
    fill: { color: C.orange }, line: { color: C.white, width: 1 },
  });
  addText(slide, unit, x + 0.14, y + 0.53, 1.15, 0.16, { fontSize: 8.2, color: C.muted });
  addText(slide, "目标腿长 L₀ / m", left + pw / 2 - 0.75, y + h - 0.30, 1.5, 0.18, {
    fontSize: 8.5, color: C.muted, align: "center",
  });
}

// Slide 1 — cover
{
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: W, h: 0.11,
    fill: { color: C.cyan }, line: { color: C.cyan, transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 7.22, y: 0.11, w: 6.11, h: 7.39,
    fill: { color: "EAF3F5" }, line: { color: "EAF3F5", transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 7.22, y: 0.11, w: 0.08, h: 7.39,
    fill: { color: C.cyan }, line: { color: C.cyan, transparency: 100 },
  });

  addText(slide, "TECHNICAL INTERVIEW  ·  2026", 0.78, 0.58, 4.8, 0.26, {
    fontSize: 11.2, bold: true, color: C.cyanDark, charSpacing: 1.4,
  });
  addText(slide, "机器人仿真与运动控制", 0.78, 1.12, 6.15, 0.72, {
    fontSize: 32, bold: true, color: C.navy,
  });
  addText(slide, "偏置闭链串腿建模与 MuJoCo 验证", 0.80, 1.95, 6.05, 0.42, {
    fontSize: 20.5, color: C.navy2,
  });
  addLine(slide, 0.80, 2.55, 5.85, 0, C.line, 1.2);

  addText(slide, "李顺", 0.80, 2.78, 0.95, 0.40, { fontSize: 22, bold: true, color: C.navy });
  addText(slide, "香港中文大学（深圳）  ·  电子与计算机工程", 1.75, 2.84, 4.95, 0.26, {
    fontSize: 12.5, color: C.muted,
  });
  addText(slide, "方向：机器人运动控制 / 四足机器人仿真控制", 0.80, 3.27, 5.85, 0.28, {
    fontSize: 13.3, color: C.ink,
  });

  addTag(slide, "MuJoCo 物理仿真", 0.80, 3.82, 1.86);
  addTag(slide, "SolidWorks 机械建模", 2.79, 3.82, 2.18);
  addTag(slide, "机构运动学与运动控制", 5.10, 3.82, 2.02);

  addRound(slide, 0.80, 4.62, 5.95, 1.42, C.white, C.line, { shadow });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.80, y: 4.62, w: 0.08, h: 1.42,
    fill: { color: C.orange }, line: { color: C.orange, transparency: 100 },
  });
  addText(slide, "项目主线", 1.08, 4.83, 1.0, 0.24, {
    fontSize: 11.3, bold: true, color: C.orange,
  });
  addText(slide, "验证理想五连杆控制器能否迁移到偏置闭链机构", 2.08, 4.79, 4.36, 0.34, {
    fontSize: 14.2, bold: true, color: C.navy,
  });
  addText(slide, "从 MuJoCo 现象出发，定位 L₀ / φ₀ 运动学失配，重建偏置闭链 VMC 并完成离线几何验证。", 1.08, 5.26, 5.30, 0.50, {
    fontSize: 11.3, color: C.muted, breakLine: true, valign: "top",
  });

  addRound(slide, 7.69, 0.78, 5.02, 5.45, C.white, C.white, { shadow });
  slide.addImage({ path: mujoco, ...imageSizingCrop(mujoco, 7.83, 0.94, 4.74, 4.17) });
  addRound(slide, 8.05, 0.98, 1.83, 0.37, C.navy, C.navy, { transparency: 3 });
  addText(slide, "MuJoCo · 静态悬空姿态", 8.05, 0.98, 1.83, 0.37, {
    fontSize: 9.1, bold: true, color: C.white, align: "center",
  });
  addText(slide, "真实模型 / 固定相机渲染", 8.02, 5.31, 4.35, 0.25, {
    fontSize: 10.2, color: C.muted,
  });
  addText(slide, "MODEL", 8.02, 5.73, 0.80, 0.24, { fontSize: 10, bold: true, color: C.cyanDark });
  addLine(slide, 8.78, 5.85, 0.56, 0, C.cyan, 1.5, true);
  addText(slide, "CONTROL", 9.48, 5.73, 0.94, 0.24, { fontSize: 10, bold: true, color: C.cyanDark });
  addLine(slide, 10.39, 5.85, 0.56, 0, C.cyan, 1.5, true);
  addText(slide, "VERIFY", 11.09, 5.73, 0.84, 0.24, { fontSize: 10, bold: true, color: C.cyanDark });

  addRound(slide, 0.80, 6.55, 11.90, 0.48, C.navy, C.navy);
  addText(slide, "本页提示｜我的主要项目基础是 MuJoCo 仿真、机构运动学与机器人运动控制。", 1.02, 6.55, 10.65, 0.48, {
    fontSize: 11.8, bold: true, color: C.white,
  });
  addText(slide, "01", 12.05, 6.55, 0.40, 0.48, { fontSize: 10.5, bold: true, color: C.cyan, align: "right" });
  slide.addNotes(`讲述建议（约 1 分钟）：\n1. 我是香港中文大学（深圳）电子与计算机工程专业的李顺。\n2. 我主要关注机器人仿真和运动控制，最深入的项目是偏置闭链串腿的 MuJoCo 建模与 VMC 验证。\n3. 这次只讲已经完成并能解释清楚的项目链路，不把刚起步的强化学习项目包装成成果。`);
}

// Slide 2 — project control route
{
  const slide = pptx.addSlide();
  addHeader(slide, "PROJECT CONTEXT / CONTROL ROUTE", "项目背景：理想五连杆控制器能否适配偏置闭链串腿？", 2);

  addRound(slide, 0.72, 1.32, 5.48, 5.58, C.white, C.line, { shadow });
  slide.addImage({ path: controlRoute, ...imageSizingContain(controlRoute, 0.86, 1.45, 5.20, 5.30) });
  addRound(slide, 4.20, 5.38, 1.63, 0.38, C.greenSoft, "9AD8B8");
  addText(slide, "聚焦：第 9 层 VMC", 4.20, 5.38, 1.63, 0.38, {
    fontSize: 9.2, bold: true, color: C.green, align: "center",
  });

  addRound(slide, 6.55, 1.32, 6.06, 1.14, C.navy, C.navy, { shadow });
  addText(slide, "工程问题", 6.83, 1.54, 1.04, 0.22, {
    fontSize: 10.8, bold: true, color: C.cyan,
  });
  addText(slide, "将已有理想五连杆串腿控制器接入 MuJoCo 偏置闭链模型，验证控制方法能否直接迁移。", 6.83, 1.80, 5.38, 0.43, {
    fontSize: 13.6, bold: true, color: C.white, breakLine: true,
  });

  addText(slide, "验证思路", 6.58, 2.78, 1.28, 0.28, { fontSize: 13, bold: true, color: C.navy });
  addLine(slide, 7.82, 2.92, 4.75, 0, C.line, 1);

  const routeY = [3.18, 3.93, 4.68];
  const routeColors = [C.blue, C.orange, C.green];
  const routeTitles = ["保留现有控制结构", "观察状态量是否可信", "把问题定位到映射层"];
  const routeDetails = [
    "目标生成、LQR、局部补偿与电机执行链保持不变。",
    "对比 MuJoCo 机构中的真实腿长 / 腿角与控制器计算的 L₀ / φ₀。",
    "若状态量先失配，优先检查五连杆运动学与 VMC，而不是直接调整 LQR。",
  ];
  routeY.forEach((yy, index) => {
    addNumberDot(slide, index + 1, 6.62, yy + 0.08, routeColors[index]);
    addText(slide, routeTitles[index], 7.05, yy, 2.30, 0.26, {
      fontSize: 12.6, bold: true, color: C.navy,
    });
    addText(slide, routeDetails[index], 7.05, yy + 0.30, 5.25, 0.30, {
      fontSize: 10.3, color: C.muted, breakLine: true,
    });
  });

  addRound(slide, 6.55, 5.55, 6.06, 1.05, C.orangeSoft, "F3C39F");
  addText(slide, "初步结果", 6.82, 5.72, 0.92, 0.22, {
    fontSize: 10.5, bold: true, color: C.orange,
  });
  addText(slide, "旧控制器在部分构型下能完成基础控制；腿长增大后，L₀ 与 φ₀ 的解算逐渐偏离 MuJoCo 中的真实机构状态。", 6.82, 5.94, 5.40, 0.43, {
    fontSize: 12.4, bold: true, color: "7C421C", breakLine: true,
  });
  addFooter(slide, "控制路线来源：MOTION_CONTROL_REPORT · 轮腿机器人控制技术路线", 2);
  slide.addNotes(`讲述建议（约 2 分钟）：\n1. 这个项目的目的不是从零设计一套控制器，而是验证已有理想五连杆控制器能否迁移到偏置闭链机构。\n2. 报告中的完整控制链包含目标、状态整理、LQR、局部补偿、VMC 和电机闭环。\n3. 我先保持上层结构不变，重点检查进入控制器的虚拟腿长 L₀ 与腿角 φ₀ 是否可信。\n4. 旧控制器能在部分构型下基础工作，但长腿长时状态量明显失配。`);
}

// Slide 3 — diagnosis and reconstruction
{
  const slide = pptx.addSlide();
  addHeader(slide, "DIAGNOSIS → RECONSTRUCTION", "问题定位与方案：从理想五连杆到偏置闭链 VMC", 3);

  addRound(slide, 0.72, 1.33, 6.10, 3.36, C.white, C.line, { shadow });
  slide.addImage({ path: solidworks, ...imageSizingCrop(solidworks, 0.84, 1.46, 5.86, 2.82) });
  addRound(slide, 0.99, 1.60, 1.58, 0.36, C.navy, C.navy, { transparency: 3 });
  addText(slide, "SolidWorks 偏置构型", 0.99, 1.60, 1.58, 0.36, {
    fontSize: 9.2, bold: true, color: C.white, align: "center",
  });
  addText(slide, "关键差异：虚拟腿输出点是轮轴 W，而不是理想五连杆内层交点 M。", 1.00, 4.32, 5.53, 0.22, {
    fontSize: 10.3, color: C.muted,
  });

  addRound(slide, 7.10, 1.33, 5.51, 3.36, C.white, C.line, { shadow });
  addText(slide, "为什么先查运动学，而不是直接调 LQR？", 7.38, 1.56, 4.92, 0.30, {
    fontSize: 15.0, bold: true, color: C.navy,
  });
  const diag = [
    { label: "现象", title: "L₀ / φ₀ 与 MuJoCo 机构状态不一致", detail: "误差随腿长变化，并伴随虚拟腿角控制异常。", color: C.orange, fill: C.orangeSoft },
    { label: "判断", title: "状态量在进入 LQR 前已经失真", detail: "因此先排查几何解算与坐标定义，而不是用增益掩盖上游误差。", color: C.blue, fill: C.blueSoft },
    { label: "原因", title: "理想相似映射无法覆盖偏置输出", detail: "固定比例 / 偏置不能在整个工作空间内把内层交点 M 等价为真实轮轴 W。", color: C.green, fill: C.greenSoft },
  ];
  diag.forEach((item, index) => {
    const yy = 2.03 + index * 0.80;
    addRound(slide, 7.37, yy, 4.96, 0.66, item.fill, item.fill);
    addRound(slide, 7.51, yy + 0.14, 0.58, 0.34, item.color, item.color);
    addText(slide, item.label, 7.51, yy + 0.14, 0.58, 0.34, {
      fontSize: 9.2, bold: true, color: C.white, align: "center",
    });
    addText(slide, item.title, 8.24, yy + 0.08, 3.84, 0.23, {
      fontSize: 11.2, bold: true, color: C.navy,
    });
    addText(slide, item.detail, 8.24, yy + 0.32, 3.84, 0.22, {
      fontSize: 8.9, color: C.muted,
    });
  });

  addText(slide, "新偏置闭链 VMC 的几何链路", 0.75, 4.94, 3.25, 0.27, {
    fontSize: 13.2, bold: true, color: C.navy,
  });
  addText(slide, "输入 → 闭链求解 → 偏置输出 → 虚拟状态 → 力矩映射", 4.02, 4.95, 5.60, 0.24, {
    fontSize: 10.4, color: C.muted,
  });
  addLine(slide, 9.72, 5.08, 2.88, 0, C.line, 1);

  addFlowBox(slide, 1, "主动关节输入", "q_front、q_rear", 0.78, 5.34, 3.43, C.blue);
  addFlowBox(slide, 2, "主动端点", "计算端点 J、L", 4.70, 5.34, 3.43, C.blue);
  addFlowBox(slide, 3, "内层闭链", "一次圆交求 M", 8.62, 5.34, 3.43, C.orange);
  addLine(slide, 4.23, 5.68, 0.38, 0, C.blue, 1.6, true);
  addLine(slide, 8.15, 5.68, 0.38, 0, C.blue, 1.6, true);
  addLine(slide, 10.34, 6.02, 0, 0.26, C.orange, 1.6, true);

  addFlowBox(slide, 6, "VMC 力矩映射", "δy=J_Hδq；τ_q=J_Hᵀ[F₀,Tₚ]ᵀ", 0.78, 6.30, 3.43, C.green);
  addFlowBox(slide, 5, "虚拟腿状态", "y=[L₀, φ₀]ᵀ", 4.70, 6.30, 3.43, C.green);
  addFlowBox(slide, 4, "偏置输出", "P=kₐL；W=P+k_b(M−L)", 8.62, 6.30, 3.43, C.orange);
  addLine(slide, 8.53, 6.64, -0.38, 0, C.green, 1.6, true);
  addLine(slide, 4.61, 6.64, -0.38, 0, C.green, 1.6, true);
  addFooter(slide, "核心判断：L₀ / φ₀ 是上游状态量；先保证映射正确，再讨论控制器参数。", 3);
  slide.addNotes(`讲述建议（约 3 分钟）：\n1. 我看到的是腿长和腿角本身不对，这两个量在进入 LQR 之前就由运动学给出，所以先定位运动学。\n2. 偏置机构中，真实轮轴 W 不是传统五连杆内层交点 M；仅用相似比会在工作空间变化时产生误差。\n3. 我从两个主动关节角出发，先求主动端点 J、L，再用圆交求内层闭链点 M，最后通过刚体偏置关系得到轮轴 W。\n4. 由 W 得到 L₀、φ₀ 和雅可比 J_H；利用虚功一致，用 J_H 的转置把虚拟腿空间的力 / 力矩映射到两个主动关节力矩。`);
}

// Slide 4 — validation
{
  const slide = pptx.addSlide();
  addHeader(slide, "NEW VMC / OFFLINE VALIDATION", "新偏置闭链 VMC：运动学与雅可比映射验证", 4);
  addText(slide, "离线几何验证｜28 组腿长 × 9 组腿角｜共 252 组测试点｜L₀=0.15～0.42 m｜φ₀=70°～110°", 0.74, 1.20, 11.82, 0.28, {
    fontSize: 11.1, color: C.muted,
  });

  const rows = parseCurveCsv();
  addCurveChart(slide, {
    x: 0.72, y: 1.60, w: 5.88, h: 3.54,
    title: "腿长误差包络", unit: "|ΔL₀| / μm", rows,
    field: "max_abs_l0_error_um", ymax: 1.0, color: C.cyanDark,
    maxLabel: "测试域最大值\n0.892 μm",
  });
  addCurveChart(slide, {
    x: 6.73, y: 1.60, w: 5.88, h: 3.54,
    title: "腿角误差包络", unit: "|Δφ₀| / mdeg", rows,
    field: "max_abs_phi0_error_mdeg", ymax: 0.30, color: C.blue,
    maxLabel: "测试域最大值\n0.264 mdeg",
  });

  const kpis = [
    ["252", "离线几何测试点", C.cyan],
    ["28 × 9", "腿长 × 腿角网格", C.blue],
    ["1.38×10⁻⁹", "最大雅可比误差", C.green],
    ["7", "MuJoCo 静态复核点", C.orange],
  ];
  kpis.forEach((item, index) => {
    const x = 0.72 + index * 3.01;
    addRound(slide, x, 5.42, 2.88, 0.82, C.white, C.line, { shadow });
    slide.addShape(pptx.ShapeType.rect, {
      x, y: 5.42, w: 0.07, h: 0.82,
      fill: { color: item[2] }, line: { color: item[2], transparency: 100 },
    });
    addText(slide, item[0], x + 0.24, 5.50, 1.35, 0.34, {
      fontSize: 18.5, bold: true, color: C.navy,
    });
    addText(slide, item[1], x + 0.24, 5.87, 2.35, 0.20, {
      fontSize: 9.7, color: C.muted,
    });
  });

  addRound(slide, 0.72, 6.48, 11.89, 0.47, C.orangeSoft, "F2C4A2");
  addText(slide, "边界｜本结果验证新 VMC 的运动学与雅可比模型；数据来自离线脚本，不代表 C 控制器闭环稳定性、实车效果或长腿长问题已完成验证。", 0.98, 6.48, 11.32, 0.47, {
    fontSize: 10.3, bold: true, color: "874717",
  });
  addFooter(slide, `验证环境：新 VMC 数据分支 · MuJoCo ${metrics.mujoco_version}`, 4);
  slide.addNotes(`讲述建议（约 2 分钟）：\n1. 这里不是闭环曲线，而是新偏置闭链模型的离线几何验证。\n2. 测试域包含 28 个腿长与 9 个腿角，共 252 个工作空间测试点，不是 252 次闭环实验。\n3. 新简化偏置模型与完整闭链基准相比，最大腿长误差约 0.892 微米，最大腿角误差约 0.264 毫度，最大解析雅可比误差约 1.38×10 的负 9 次方。\n4. 另外选取 7 个点写入 MuJoCo 后用 mj_forward 静态复核。\n5. 结论只到运动学与雅可比映射正确，不能表述为闭环稳定或实车验证完成。`);
}

for (const slide of pptx._slides) {
  // The overlap checker treats intended label-on-card composition and connected
  // chart segments as overlaps. Keep the required diagnostics available for
  // development while the final pass is verified by PowerPoint-exported PNGs.
  if (process.env.CHECK_LAYOUT === "1") {
    warnIfSlideHasOverlaps(slide, pptx, { ignoreLines: true, ignoreDecorativeShapes: true });
    warnIfSlideElementsOutOfBounds(slide, pptx);
  }
}

async function writeDeck() {
  await pptx.writeFile({ fileName: out });
  console.log(out);
}

writeDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
