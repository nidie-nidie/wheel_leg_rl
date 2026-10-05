"""Write the inline HTML fragment for A1 fixed-mean cross-validation plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


TEMPLATE = r'''<div id="a1-fixed-mean-crossval" class="a1cv">
  <div class="viz-controls" aria-label="trajectory controls">
    <label class="form-label">Dataset
      <select class="form-select" id="a1cv-dataset"></select>
    </label>
    <label class="form-label">Joint
      <select class="form-select" id="a1cv-joint"></select>
    </label>
    <label class="form-check form-switch">
      <input class="form-check-input" type="checkbox" id="a1cv-command">
      <span class="form-check-label">Command</span>
    </label>
    <div class="viz-row" aria-label="legend">
      <span class="a1cv-legend"><span class="a1cv-swatch a1cv-real"></span>real dof_pos</span>
      <span class="a1cv-legend"><span class="a1cv-swatch a1cv-sim"></span>fixed-mean replay</span>
      <span class="a1cv-legend"><span class="a1cv-swatch a1cv-cmd"></span>des_dof_pos</span>
    </div>
    <span class="viz-badge" id="a1cv-selected-metrics"></span>
  </div>
  <svg id="a1cv-summary" role="img" aria-label="fixed actuator replay RMSE by data set"></svg>
  <svg id="a1cv-focus" role="img" aria-label="selected A1 joint trajectory comparison"></svg>
  <svg id="a1cv-overview" role="img" aria-label="all A1 joint trajectory overlays for selected data set"></svg>
</div>
<style>
#a1-fixed-mean-crossval { color: var(--foreground); display: grid; gap: 12px; }
#a1-fixed-mean-crossval .a1cv-legend { align-items: center; display: inline-flex; gap: 6px; white-space: nowrap; }
#a1-fixed-mean-crossval .a1cv-swatch { display: inline-block; height: 3px; width: 22px; }
#a1-fixed-mean-crossval .a1cv-real { background: var(--viz-series-1); }
#a1-fixed-mean-crossval .a1cv-sim { background: var(--viz-series-2); }
#a1-fixed-mean-crossval .a1cv-cmd { background: var(--muted-foreground); }
#a1-fixed-mean-crossval svg { display: block; overflow: visible; width: 100%; }
#a1cv-summary { min-height: 150px; }
#a1cv-focus { min-height: 300px; }
#a1cv-overview { min-height: 650px; }
#a1-fixed-mean-crossval .axis, #a1-fixed-mean-crossval .grid { color: var(--border); stroke: currentColor; stroke-width: 1; }
#a1-fixed-mean-crossval .axis-label, #a1-fixed-mean-crossval .joint-label, #a1-fixed-mean-crossval .metric-label { fill: var(--muted-foreground); }
#a1-fixed-mean-crossval .real-line { fill: none; stroke: var(--viz-series-1); stroke-linejoin: round; stroke-linecap: round; }
#a1-fixed-mean-crossval .sim-line { fill: none; stroke: var(--viz-series-2); stroke-linejoin: round; stroke-linecap: round; }
#a1-fixed-mean-crossval .cmd-line { fill: none; stroke: var(--muted-foreground); stroke-dasharray: 5 4; stroke-linejoin: round; stroke-linecap: round; }
#a1-fixed-mean-crossval .zero-line { stroke: var(--border); stroke-width: 1; }
#a1-fixed-mean-crossval .bar-main { fill: var(--viz-series-2); opacity: 0.76; }
#a1-fixed-mean-crossval .bar-extra { fill: var(--viz-series-3); opacity: 0.68; }
#a1-fixed-mean-crossval .lane-divider { stroke: var(--border); stroke-width: 1; }
#a1-fixed-mean-crossval .selected-marker { fill: none; stroke: var(--ring); stroke-width: 1.5; }
@media (max-width: 520px) {
  #a1cv-overview { min-height: 820px; }
  #a1cv-focus { min-height: 330px; }
}
</style>
<script>
(() => {
  const root = document.getElementById('a1-fixed-mean-crossval');
  const data = __DATA__;
  const datasetSelect = root.querySelector('#a1cv-dataset');
  const jointSelect = root.querySelector('#a1cv-joint');
  const commandToggle = root.querySelector('#a1cv-command');
  const summary = root.querySelector('#a1cv-summary');
  const focus = root.querySelector('#a1cv-focus');
  const overview = root.querySelector('#a1cv-overview');
  const metricBadge = root.querySelector('#a1cv-selected-metrics');
  const fmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 });
  const fmt5 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 5 });
  const datasetLabel = ds => `${ds.id} · ${ds.label}`;
  data.datasets.forEach((dataset, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = datasetLabel(dataset);
    datasetSelect.appendChild(option);
  });
  data.datasets[0].joints.forEach((joint, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = joint.name;
    jointSelect.appendChild(option);
  });
  function makeEl(name, attrs = {}, parent) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (parent) parent.appendChild(el);
    return el;
  }
  function pathFor(xs, ys, xScale, yScale) {
    let d = '';
    for (let i = 0; i < xs.length; i += 1) {
      const x = xScale(xs[i]);
      const y = yScale(ys[i]);
      d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
    }
    return d;
  }
  function selectedDataset() { return data.datasets[Number(datasetSelect.value || 0)]; }
  function selectedJoint(dataset) { return dataset.joints[Number(jointSelect.value || 0)]; }
  function valueRange(joint, includeCommand) {
    const values = joint.real.concat(joint.sim, includeCommand ? joint.cmd : []);
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (max - min < 0.02) { min -= 0.01; max += 0.01; }
    const pad = Math.max(0.02, (max - min) * 0.08);
    return [min - pad, max + pad];
  }
  function renderSummary() {
    const width = Math.max(320, Math.floor(root.getBoundingClientRect().width));
    const height = width < 520 ? 190 : 160;
    summary.setAttribute('viewBox', `0 0 ${width} ${height}`);
    summary.replaceChildren();
    makeEl('title', {}, summary).textContent = 'Fixed actuator replay aggregate RMSE by data set';
    makeEl('desc', {}, summary).textContent = 'Encoder-frame replay error for three changba01 data sets and two additional chirp data sets using the same fitted actuator parameters.';
    const margin = { left: width < 520 ? 42 : 58, right: 14, top: 12, bottom: width < 520 ? 58 : 42 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const maxRmse = Math.max(...data.datasets.map(ds => ds.metrics.rmse), 0.01);
    const yMax = Math.ceil(maxRmse * 1200) / 1000;
    const yScale = v => margin.top + (1 - v / yMax) * plotH;
    [0, 0.5, 1].forEach(frac => {
      const y = margin.top + frac * plotH;
      makeEl('line', { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: 'grid' }, summary);
    });
    const gap = width < 520 ? 5 : 10;
    const barW = Math.max(24, (plotW - gap * (data.datasets.length - 1)) / data.datasets.length);
    data.datasets.forEach((dataset, index) => {
      const x = margin.left + index * (barW + gap);
      const y = yScale(dataset.metrics.rmse);
      const h = margin.top + plotH - y;
      const cls = dataset.group === 'changba01-three' ? 'bar-main' : 'bar-extra';
      makeEl('rect', { x, y, width: barW, height: h, class: cls }, summary);
      if (index === Number(datasetSelect.value || 0)) makeEl('rect', { x: x - 2, y: margin.top - 2, width: barW + 4, height: plotH + 4, class: 'selected-marker' }, summary);
      makeEl('text', { x: x + barW / 2, y: y - 5, 'text-anchor': 'middle', class: 'metric-label text-small' }, summary).textContent = fmt5.format(dataset.metrics.rmse);
      makeEl('text', { x: x + barW / 2, y: height - margin.bottom + 18, 'text-anchor': 'middle', class: 'axis-label text-small' }, summary).textContent = dataset.id;
      makeEl('text', { x: x + barW / 2, y: height - margin.bottom + 34, 'text-anchor': 'middle', class: 'axis-label text-small' }, summary).textContent = dataset.group === 'changba01-three' ? 'remote' : 'extra';
    });
    makeEl('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + plotH, class: 'axis' }, summary);
    [0, yMax / 2, yMax].forEach(v => {
      makeEl('text', { x: margin.left - 8, y: yScale(v) + 4, 'text-anchor': 'end', class: 'axis-label text-small' }, summary).textContent = fmt.format(v);
    });
    makeEl('text', { x: margin.left, y: 10, class: 'metric-label text-small' }, summary).textContent = 'encoder RMSE rad';
  }
  function renderFocus() {
    const dataset = selectedDataset();
    const joint = selectedJoint(dataset);
    const includeCommand = commandToggle.checked;
    metricBadge.textContent = `${dataset.id} · RMSE ${fmt5.format(dataset.metrics.rmse)} rad · joint ${fmt5.format(joint.rmse)} rad · P95 ${fmt5.format(joint.p95)}`;
    const width = Math.max(320, Math.floor(root.getBoundingClientRect().width));
    const height = width < 520 ? 330 : 300;
    focus.setAttribute('viewBox', `0 0 ${width} ${height}`);
    focus.replaceChildren();
    makeEl('title', {}, focus).textContent = `${dataset.id} ${joint.name} trajectory comparison`;
    makeEl('desc', {}, focus).textContent = 'Real encoder position, fixed-mean replay encoder position, and optional command trajectory on a shared time axis.';
    const margin = { left: width < 520 ? 44 : 60, right: 16, top: 22, bottom: 36 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const tMin = dataset.time[0];
    const tMax = dataset.time[dataset.time.length - 1];
    const [yMin, yMax] = valueRange(joint, includeCommand);
    const xScale = t => margin.left + ((t - tMin) / (tMax - tMin)) * plotW;
    const yScale = v => margin.top + (1 - ((v - yMin) / (yMax - yMin))) * plotH;
    [0, 0.25, 0.5, 0.75, 1].forEach(frac => {
      const y = margin.top + frac * plotH;
      makeEl('line', { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: 'grid' }, focus);
    });
    const zeroY = yScale(0);
    if (zeroY >= margin.top && zeroY <= margin.top + plotH) makeEl('line', { x1: margin.left, x2: width - margin.right, y1: zeroY, y2: zeroY, class: 'zero-line' }, focus);
    makeEl('path', { d: pathFor(dataset.time, joint.real, xScale, yScale), class: 'real-line', 'stroke-width': 2.2 }, focus);
    makeEl('path', { d: pathFor(dataset.time, joint.sim, xScale, yScale), class: 'sim-line', 'stroke-width': 2 }, focus);
    if (includeCommand) makeEl('path', { d: pathFor(dataset.time, joint.cmd, xScale, yScale), class: 'cmd-line', 'stroke-width': 1.5 }, focus);
    makeEl('line', { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: 'axis' }, focus);
    makeEl('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: 'axis' }, focus);
    [tMin, (tMin + tMax) / 2, tMax].forEach(t => {
      makeEl('text', { x: xScale(t), y: height - 12, 'text-anchor': 'middle', class: 'axis-label text-small' }, focus).textContent = `${fmt.format(t)}s`;
    });
    [yMin, (yMin + yMax) / 2, yMax].forEach(v => {
      makeEl('text', { x: margin.left - 8, y: yScale(v) + 4, 'text-anchor': 'end', class: 'axis-label text-small' }, focus).textContent = fmt.format(v);
    });
    makeEl('text', { x: margin.left, y: 15, class: 'joint-label' }, focus).textContent = `${dataset.id} · ${joint.name}`;
    makeEl('text', { x: width - margin.right, y: 15, 'text-anchor': 'end', class: 'metric-label text-small' }, focus).textContent = `joint RMSE ${fmt5.format(joint.rmse)} rad`;
  }
  function renderOverview() {
    const dataset = selectedDataset();
    const includeCommand = commandToggle.checked;
    const width = Math.max(320, Math.floor(root.getBoundingClientRect().width));
    const rowH = width < 520 ? 62 : 50;
    const margin = { left: width < 520 ? 88 : 132, right: width < 520 ? 58 : 88, top: 12, bottom: 28 };
    const height = margin.top + margin.bottom + rowH * dataset.joints.length;
    overview.setAttribute('viewBox', `0 0 ${width} ${height}`);
    overview.replaceChildren();
    makeEl('title', {}, overview).textContent = `${dataset.id} all joint trajectory overlays`;
    makeEl('desc', {}, overview).textContent = 'Twelve joint lanes compare real encoder position with the fixed-mean replay trajectory.';
    const plotW = width - margin.left - margin.right;
    const tMin = dataset.time[0];
    const tMax = dataset.time[dataset.time.length - 1];
    const xScale = t => margin.left + ((t - tMin) / (tMax - tMin)) * plotW;
    const maxJointRmse = Math.max(...dataset.joints.map(j => j.rmse), 0.01);
    dataset.joints.forEach((joint, index) => {
      const y0 = margin.top + index * rowH;
      const laneTop = y0 + 6;
      const laneH = rowH - 14;
      const [yMin, yMax] = valueRange(joint, includeCommand);
      const yScale = v => laneTop + (1 - ((v - yMin) / (yMax - yMin))) * laneH;
      makeEl('line', { x1: margin.left, x2: width - margin.right, y1: y0 + rowH - 4, y2: y0 + rowH - 4, class: 'lane-divider' }, overview);
      const zeroY = yScale(0);
      if (zeroY >= laneTop && zeroY <= laneTop + laneH) makeEl('line', { x1: margin.left, x2: width - margin.right, y1: zeroY, y2: zeroY, class: 'zero-line' }, overview);
      makeEl('path', { d: pathFor(dataset.time, joint.real, xScale, yScale), class: 'real-line', 'stroke-width': 1.35 }, overview);
      makeEl('path', { d: pathFor(dataset.time, joint.sim, xScale, yScale), class: 'sim-line', 'stroke-width': 1.25 }, overview);
      if (includeCommand) makeEl('path', { d: pathFor(dataset.time, joint.cmd, xScale, yScale), class: 'cmd-line', 'stroke-width': 1 }, overview);
      makeEl('text', { x: 2, y: y0 + rowH / 2 + 4, class: 'joint-label text-small' }, overview).textContent = joint.name.replace('_joint', '');
      const barW = Math.max(1, Math.min(margin.right - 26, (joint.rmse / maxJointRmse) * (margin.right - 26)));
      makeEl('rect', { x: width - margin.right + 10, y: y0 + rowH / 2 - 7, width: barW, height: 6, class: 'bar-main' }, overview);
      makeEl('text', { x: width - 2, y: y0 + rowH / 2 + 4, 'text-anchor': 'end', class: 'metric-label text-small' }, overview).textContent = fmt.format(joint.rmse);
    });
    makeEl('line', { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: 'axis' }, overview);
    [tMin, (tMin + tMax) / 2, tMax].forEach(t => {
      makeEl('text', { x: xScale(t), y: height - 8, 'text-anchor': 'middle', class: 'axis-label text-small' }, overview).textContent = `${fmt.format(t)}s`;
    });
    makeEl('text', { x: width - 2, y: 10, 'text-anchor': 'end', class: 'metric-label text-small' }, overview).textContent = 'joint RMSE rad';
  }
  function render() { renderSummary(); renderFocus(); renderOverview(); }
  datasetSelect.addEventListener('change', render);
  jointSelect.addEventListener('change', () => { renderFocus(); });
  commandToggle.addEventListener('change', render);
  if ('ResizeObserver' in window) { new ResizeObserver(render).observe(root); } else { window.addEventListener('resize', render); }
  render();
})();
</script>
'''


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = args.data.read_text(encoding="ascii")
    html = TEMPLATE.replace("__DATA__", data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    print(args.output)
    print(args.output.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
