# Glossary: English → 简体中文 (zh-Hans)

The terms every MKLight translation uses: the engine catalog
(`locales/zh-Hans/engine.json`) and the macOS, web and Windows UI catalogs.
They follow Chinese optical-engineering textbooks and design practice. Use
the left column's term wherever it occurs; if a term is missing, add it here
first. Scheme and file formats: `docs/I18N.md`.

## Style

* Full-width punctuation in Chinese text: ，。：；？！（）“”、. ASCII inside
  code, formulas, JSON and identifiers (`[x, y]`, `{"type": "all"}`).
* One half-width space between Chinese and Latin letters or digits
  (`第 3 面`, `RMS 半径`, `F 数`), none next to full-width punctuation.
* Short, plain UI Chinese: labels without 的 where possible (`主波长`, not
  `主要的波长`); sentences end with 。 only in descriptions, tooltips and
  messages, not in labels, buttons or menu items.
* Surfaces are counted as the engine counts them: `第 {n} 面`, or `面 {n}`
  in tables. The object is 物面, the image is 像面.
* `null` and `0` in setting descriptions are values a user types; keep them
  (`null 表示主波长`).
* Address the user as 你, not 您.
* Waves as a unit are shown as λ.

## Never translate

Identifiers (analysis names such as `spot_diagram`, operand mnemonics,
setting keys, JSON keys, `ParamPath`), units and symbols (mm, µm, nm, deg,
°, sr, W/mm², cycles/mm, λ, n_d, V_d, P_g,F, F/#), glass and catalogue
names (N-BK7, BUILTIN), file extensions (.zmx, .agf), numbers, and names of
methods or people used as names (Levenberg–Marquardt, CMA-ES, Nelder–Mead,
Philox, Brent, Coddington, Noll, Fringe, Gauss–Legendre). Keep the
abbreviations the industry keeps: MTF, PSF, LSF, OPD, RMS, PV, GEO, EFL,
BFL, NA, DOE, AGF, GPU, CPU, FFT, RGB, PNG, sRGB, CRA.

## Optics

| English | 简体中文 | Note |
|---|---|---|
| lens data, lens data editor | 镜头数据, 镜头数据编辑器 | |
| lens (system) | 镜头 | an imaging system |
| lens (element) | 透镜 | one element |
| element | 元件 | |
| cemented doublet | 双胶合透镜 | |
| singlet | 单透镜 | |
| mirror | 反射镜 | |
| surface | 面 | `第 3 面`; 表面 only in prose about the physical surface |
| object, object surface | 物, 物面 | |
| image, image surface | 像, 像面 | |
| object space, image space | 物方, 像方 | |
| radius (of curvature) | 曲率半径 | |
| curvature | 曲率 | |
| thickness | 厚度 | |
| centre thickness, edge thickness | 中心厚度, 边缘厚度 | |
| semi-diameter | 半口径 | |
| clear semi-diameter, clear aperture | 通光半口径, 通光孔径 | |
| mechanical semi-diameter | 机械半口径 | |
| aperture (system) | 孔径 | `孔径类型` |
| aperture (clipping, on a surface) | 孔径 | `固定孔径` |
| conic, conic constant | 圆锥系数 | k; 二次曲面系数 is the textbook synonym, not used |
| sphere, spherical surface | 球面 | |
| conic surface (surface type) | 二次曲面 | `球面 / 二次曲面`; not 圆锥曲面, which reads as a cone |
| surface type | 面型 | |
| kinoform | 连续闪耀 | |
| anamorphic squeeze | 压缩比 | |
| asphere, aspheric surface | 非球面 | |
| even asphere, odd asphere | 偶次非球面, 奇次非球面 | |
| Forbes Q-type (Qcon, Qbfs) | Forbes Q 型非球面 (Qcon, Qbfs) | |
| toroid, toroidal surface | 环曲面 | |
| biconic | 双锥面 | |
| cylinder, cylindrical | 柱面 | |
| freeform | 自由曲面 | |
| XY polynomial | XY 多项式 | |
| Zernike | 泽尼克 | `泽尼克系数` |
| diffractive surface | 衍射面 | |
| diffractive optical element (DOE) | 衍射光学元件 (DOE) | |
| diffraction order | 衍射级次 | |
| paraxial surface, ideal lens | 近轴面, 理想透镜 | |
| sag | 矢高 | |
| coordinate break | 坐标断点 | |
| decentre | 偏心 | |
| tilt | 倾斜 | |
| stop, aperture stop | 光阑, 孔径光阑 | |
| field | 视场 | |
| field point | 视场点 | |
| normalised field | 归一化视场 | |
| field angle | 视场角 | |
| object height | 物高 | |
| image height | 像高 | |
| wavelength | 波长 | |
| primary wavelength | 主波长 | |
| weight | 权重 | |
| pupil | 光瞳 | |
| entrance pupil, exit pupil | 入瞳, 出瞳 | |
| entrance pupil diameter | 入瞳直径 | |
| normalised pupil coordinate | 归一化光瞳坐标 | |
| pupil sampling | 光瞳采样 | |
| ray aiming | 光线瞄准 | |
| chief ray | 主光线 | |
| marginal ray | 边缘光线 | |
| real ray, paraxial ray | 实际光线, 近轴光线 | |
| ray trace | 光线追迹 | |
| reference tracer | 参考追迹器 | the CPU f64 tracer |
| vignetting, vignetting factor | 渐晕, 渐晕因子 | |
| first-order, paraxial | 一阶, 近轴 | `一阶特性` |
| effective focal length (EFL) | 有效焦距 (EFL) | |
| back focal length (BFL) | 后焦距 (BFL) | |
| front focal point | 前焦点 | |
| focal length | 焦距 | |
| F-number, F/# | F 数 | `像方 F 数`, `工作 F 数` |
| working F/# | 工作 F 数 | |
| numerical aperture (NA) | 数值孔径 (NA) | `物方 NA` in tables |
| magnification | 放大率 | |
| Lagrange invariant | 拉格朗日不变量 | |
| total track | 总长 | |
| Petzval sum, Petzval radius | 佩茨瓦尔和, 佩茨瓦尔半径 | |
| Airy radius, Airy disk | 艾里斑半径, 艾里斑 | |
| diffraction limit | 衍射极限 | |
| aberration | 像差 | |
| Seidel aberrations | 赛德尔像差 | |
| spherical aberration | 球差 | |
| coma | 彗差 | |
| astigmatism | 像散 | |
| field curvature | 场曲 | |
| distortion | 畸变 | |
| grid distortion | 网格畸变 | |
| longitudinal aberration | 纵向像差 | |
| lateral colour | 垂轴色差 | |
| axial colour, chromatic focal shift | 轴向色差, 色焦移 | |
| tangential, sagittal | 子午, 弧矢 | |
| spot diagram | 点列图 | |
| RMS spot radius | RMS 弥散斑半径 | |
| centroid | 质心 | |
| ray fan, ray aberration fan (the plot) | 光线像差曲线 | |
| fan (the set of rays) | 光线扇 | `子午光线扇` |
| OPD, optical path difference | 光程差 (OPD) | `OPD 曲线` for OPD fans |
| wavefront, wavefront map | 波前, 波前图 | |
| reference sphere | 参考球面 | |
| MTF, modulation transfer function | 调制传递函数 (MTF) | `MTF` in titles and labels |
| geometric MTF | 几何 MTF | |
| through-focus | 离焦 | `离焦 MTF`, `离焦扫描` |
| defocus plane | 离焦面 | |
| defocus | 离焦量 | |
| best focus | 最佳焦面 | |
| depth of focus | 焦深 | |
| PSF, point spread function | 点扩散函数 (PSF) | |
| line spread function (LSF) | 线扩散函数 (LSF) | |
| Strehl ratio | 斯特列尔比 | |
| encircled energy, ensquared energy | 包围能量, 方框包围能量 | |
| spatial frequency | 空间频率 | |
| footprint | 光束足迹 | |
| layout | 布局图 | |
| image simulation | 图像仿真 | |
| relative illumination | 相对照度 | |
| irradiance, intensity, radiance | 辐照度, 辐射强度, 辐亮度 | |
| transmission, transmittance | 透过率 | |
| reflectance | 反射率 | |
| ghost, ghost image | 鬼像 | |
| ghost path | 鬼像路径 | |
| stray light | 杂散光 | |
| scatter | 散射 | |
| sneak path | 潜行路径 | |
| polarisation, polarisation ray tracing | 偏振, 偏振光线追迹 | |
| Jones vector, Jones matrix | 琼斯矢量, 琼斯矩阵 | |
| Stokes vector | 斯托克斯矢量 | |
| diattenuation, retardance | 二向衰减, 相位延迟 | |
| unpolarised | 非偏振 | |
| coating | 膜层 | |
| sensor, detector | 传感器, 探测器 | |
| pixel pitch | 像元间距 | |
| chief ray angle (CRA) | 主光线角 (CRA) | |

## Materials

| English | 简体中文 | Note |
|---|---|---|
| material | 材料 | |
| glass | 玻璃 | |
| glass catalogue | 玻璃库 | one AGF file or the set; `玻璃库文件` |
| glass map | 玻璃图 | |
| refractive index | 折射率 | |
| Abbe number | 阿贝数 | |
| partial dispersion | 部分色散 | |
| dispersion | 色散 | |
| model glass | 模型玻璃 | |
| glass status: standard, preferred, obsolete, special, melt | 标准, 推荐, 停产, 特殊, 按需熔炼 | AGF status |
| climate resistance, stain resistance | 耐气候性, 耐污染性 | AGF CR, FR |
| relative cost | 相对成本 | |
| internal transmission | 内透过率 | |
| glass substitution | 玻璃替换 | |
| environment (temperature, pressure) | 环境（温度、气压） | |

## Optimisation and tolerancing

| English | 简体中文 | Note |
|---|---|---|
| optimise, optimisation | 优化 | |
| merit function | 评价函数 | |
| default merit function | 默认评价函数 | |
| operand | 操作数 | |
| variable | 变量 | |
| constraint | 约束 | |
| boundary (thickness) | 边界 | `厚度边界` |
| target, weight | 目标值, 权重 | |
| solve | 求解 | `曲率求解` |
| pickup | 跟随 | `厚度跟随` |
| configuration, multi-configuration | 结构, 多重结构 | |
| feasible | 可行 | |
| local minimum | 局部极小 | |
| finite differences | 有限差分 | |
| damped least squares, Levenberg–Marquardt | 阻尼最小二乘, Levenberg–Marquardt | |
| dogleg | dogleg | kept as a name; 狗腿法 in prose if needed |
| differential evolution | 差分进化 | |
| basin hopping (hammer) | 盆地跳跃 | |
| tabu search | 禁忌搜索 | |
| tolerance, tolerancing | 公差, 公差分析 | |
| sensitivity analysis | 灵敏度分析 | |
| compensator | 补偿器 | |
| Monte Carlo | 蒙特卡罗 | |
| trial | 试验 | `蒙特卡罗试验` |
| yield | 良率 | |
| percentile | 百分位 | |
| criterion | 判据 | |
| irregularity | 不规则度 | |
| test fringes, power (fringes) | 检验光圈, 光圈数 | |
| seed | 随机种子 | |
| candidate | 候选 | |

## User interface

| English | 简体中文 |
|---|---|
| analysis | 分析 |
| settings | 设置 |
| panel | 面板 |
| inspector | 检查器 |
| sidebar | 边栏 |
| toolbar | 工具栏 |
| preview | 预览 |
| apply | 应用 |
| run, stop, cancel | 运行, 停止, 取消 |
| job | 任务 |
| result | 结果 |
| backend | 计算后端 |
| CPU reference | CPU 参考 |
| automatic, auto | 自动 |
| none | 无 |
| default | 默认 |
| warning, error, issue | 警告, 错误, 问题 |
| validate, validation | 校验 |
| project | 项目 |
| sample (built-in system) | 示例 |
| import, export | 导入, 导出 |
| open, save, save as | 打开, 保存, 另存为 |
| undo, redo | 撤销, 重做 |
| grid | 网格 |
| map | 分布图 |
| table | 表格 |
| sampling, samples | 采样, 采样数 |
| index (position in a list) | 序号 |
| buffer | 缓冲区 |
| library | 资料库; the glass library is 玻璃库 |
| load | 载入 |
| placement (decentre and tilt section) | 位姿 |
| margin | 裕量 |
| pivot | 枢轴 |
| melt offset | 熔炼偏差 |
| build (a Monte Carlo trial lens) | 成品 |
