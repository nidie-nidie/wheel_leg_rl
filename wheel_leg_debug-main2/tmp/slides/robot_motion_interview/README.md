# 机器人仿真与运动控制面试 PPT

交付文件：

- `李顺_机器人仿真与运动控制_技术面试.pptx`：可编辑演示文稿，含逐页演讲备注。
- `build_interview_ppt.js`：PptxGenJS 源文件。
- `preview_montage.png`：五页预览。

如需重新生成，请在仓库根目录运行：

```powershell
& 'C:\Program Files\nodejs\npm.cmd' install --prefix '.\output\ppt'
node '.\output\ppt\build_interview_ppt.js'
```

源文件读取仓库中的 `mujoco_control_extract/output/spreadsheet/stand_b_pitch0_3ms.csv` 生成第 4 页的原生 PowerPoint 曲线。
