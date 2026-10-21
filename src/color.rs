//! Цветовая математика: температура/оттенок, матрицы камеры, Oklab, тональная кривая.

use crate::raw::CameraColor;

pub type M3 = [[f64; 3]; 3];

pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub fn mulv(a: &M3, v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

pub fn inv(m: &M3) -> M3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = if det.abs() < 1e-12 { 1e-12 } else { det };
    let mut r = [[0.0; 3]; 3];
    r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
    r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
    r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
    r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
    r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
    r
}

fn diag(v: [f64; 3]) -> M3 {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

fn from_flat(f: &[f32; 9]) -> M3 {
    [
        [f[0] as f64, f[1] as f64, f[2] as f64],
        [f[3] as f64, f[4] as f64, f[5] as f64],
        [f[6] as f64, f[7] as f64, f[8] as f64],
    ]
}

pub const XYZ_TO_SRGB: M3 = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

pub const SRGB_TO_XYZ: M3 = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

pub const D65_XY: (f64, f64) = (0.31270, 0.32900);

const BRADFORD: M3 = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];

pub fn xy_to_xyz(xy: (f64, f64)) -> [f64; 3] {
    let (x, y) = xy;
    [x / y, 1.0, (1.0 - x - y) / y]
}

pub fn xyz_to_xy(v: [f64; 3]) -> (f64, f64) {
    let s = (v[0] + v[1] + v[2]).max(1e-12);
    (v[0] / s, v[1] / s)
}

fn bradford(src_white: [f64; 3], dst_white: [f64; 3]) -> M3 {
    let s = mulv(&BRADFORD, src_white);
    let d = mulv(&BRADFORD, dst_white);
    let scale = diag([d[0] / s[0], d[1] / s[1], d[2] / s[2]]);
    mul(&inv(&BRADFORD), &mul(&scale, &BRADFORD))
}

/// Линия Планка в координатах CIE 1960 uv (аппроксимация Крыстека, 1000–15000 K).
pub fn planck_uv(t: f64) -> (f64, f64) {
    let t2 = t * t;
    let u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t2) / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t2);
    let v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t2) / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t2);
    (u, v)
}

/// Единичная нормаль к линии Планка, направленная в сторону пурпурного (вниз по v).
fn magenta_normal(t: f64) -> (f64, f64) {
    let (u1, v1) = planck_uv(t * 0.999);
    let (u2, v2) = planck_uv(t * 1.001);
    let (du, dv) = (u2 - u1, v2 - v1);
    let len = (du * du + dv * dv).sqrt().max(1e-12);
    let (mut nu, mut nv) = (-dv / len, du / len);
    if nv > 0.0 {
        nu = -nu;
        nv = -nv;
    }
    (nu, nv)
}

/// Шкала оттенка как в DNG/Lightroom: 1 единица = 1/3000 Duv.
pub const TINT_SCALE: f64 = 3000.0;
pub const TEMP_MIN: f64 = 1800.0;
pub const TEMP_MAX: f64 = 15000.0;

fn uv_to_xy(u: f64, v: f64) -> (f64, f64) {
    let d = 2.0 * u - 8.0 * v + 4.0;
    (3.0 * u / d, 2.0 * v / d)
}

fn xy_to_uv(x: f64, y: f64) -> (f64, f64) {
    let d = -2.0 * x + 12.0 * y + 3.0;
    (4.0 * x / d, 6.0 * y / d)
}

/// Температура (K) и оттенок (+ пурпурный, − зелёный) → хроматичность xy.
pub fn temp_tint_to_xy(t: f64, tint: f64) -> (f64, f64) {
    let t = t.clamp(TEMP_MIN, TEMP_MAX);
    let (u0, v0) = planck_uv(t);
    let (nu, nv) = magenta_normal(t);
    let off = tint / TINT_SCALE;
    uv_to_xy(u0 + nu * off, v0 + nv * off)
}

/// Хроматичность xy → (температура, оттенок).
pub fn xy_to_temp_tint(xy: (f64, f64)) -> (f64, f64) {
    let (u, v) = xy_to_uv(xy.0, xy.1);
    let dist = |mired: f64| {
        let (pu, pv) = planck_uv(1e6 / mired);
        (pu - u).powi(2) + (pv - v).powi(2)
    };
    let (lo, hi) = (1e6 / TEMP_MAX, 1e6 / TEMP_MIN);
    let mut best = lo;
    let mut best_d = f64::MAX;
    let mut m = lo;
    while m <= hi {
        let d = dist(m);
        if d < best_d {
            best_d = d;
            best = m;
        }
        m += 2.0;
    }
    let (mut a, mut b) = ((best - 2.0).max(lo), (best + 2.0).min(hi));
    for _ in 0..40 {
        let m1 = a + (b - a) * 0.382;
        let m2 = a + (b - a) * 0.618;
        if dist(m1) < dist(m2) {
            b = m2;
        } else {
            a = m1;
        }
    }
    let t = 1e6 / ((a + b) * 0.5);
    let (pu, pv) = planck_uv(t);
    let (nu, nv) = magenta_normal(t);
    let tint = ((u - pu) * nu + (v - pv) * nv) * TINT_SCALE;
    (t, tint)
}

impl CameraColor {
    /// Матрица XYZ → камера для источника с температурой t (интерполяция A ↔ D65 по 1/T, как в DNG).
    pub fn cm(&self, t: f64) -> M3 {
        let d65 = from_flat(&self.cm_d65);
        match &self.cm_a {
            None => d65,
            Some(a) => {
                let a = from_flat(a);
                let (t1, t2) = (2856.0, 6504.0);
                let g = ((1.0 / t - 1.0 / t2) / (1.0 / t1 - 1.0 / t2)).clamp(0.0, 1.0);
                let mut r = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        r[i][j] = a[i][j] * g + d65[i][j] * (1.0 - g);
                    }
                }
                r
            }
        }
    }

    /// Отклик камеры на белый при данном источнике.
    pub fn neutral(&self, t: f64, tint: f64) -> [f64; 3] {
        let xyz = xy_to_xyz(temp_tint_to_xy(t, tint));
        let n = mulv(&self.cm(t), xyz);
        [n[0].max(1e-6), n[1].max(1e-6), n[2].max(1e-6)]
    }

    /// Множители баланса белого (минимальный = 1).
    pub fn multipliers(&self, t: f64, tint: f64) -> [f32; 3] {
        let n = self.neutral(t, tint);
        let m = [1.0 / n[0], 1.0 / n[1], 1.0 / n[2]];
        let mn = m[0].min(m[1]).min(m[2]);
        [(m[0] / mn) as f32, (m[1] / mn) as f32, (m[2] / mn) as f32]
    }

    /// Обратная задача: отклик камеры на белый → (температура, оттенок).
    pub fn temp_tint_from_neutral(&self, n: [f64; 3]) -> (f64, f64) {
        let mut t = 5000.0;
        let mut tint = 0.0;
        for _ in 0..30 {
            let xyz = mulv(&inv(&self.cm(t)), n);
            let (t2, tint2) = xy_to_temp_tint(xyz_to_xy(xyz));
            let done = (t2 - t).abs() < 0.5;
            t = t2;
            tint = tint2;
            if done {
                break;
            }
        }
        (t, tint)
    }

    pub fn temp_tint_from_multipliers(&self, m: [f32; 3]) -> (f64, f64) {
        self.temp_tint_from_neutral([1.0 / m[0] as f64, 1.0 / m[1] as f64, 1.0 / m[2] as f64])
    }

    /// Матрица «камера после баланса белого» → линейный sRGB (D65). Белый (1,1,1) → (1,1,1).
    pub fn cam_to_srgb(&self, t: f64, tint: f64) -> M3 {
        let cm = self.cm(t);
        let white = xy_to_xyz(temp_tint_to_xy(t, tint));
        let n = mulv(&cm, white);
        let adapt = bradford(white, xy_to_xyz(D65_XY));
        let m = mul(&XYZ_TO_SRGB, &mul(&adapt, &mul(&inv(&cm), &diag(n))));
        let mut r = m;
        for row in r.iter_mut() {
            let s = row[0] + row[1] + row[2];
            if s.abs() > 1e-9 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }
        r
    }
}

// ---------- sRGB и Oklab ----------

pub fn srgb_encode(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

pub fn srgb_decode(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

pub fn oklab(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = [c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)];
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

pub fn oklab_to_rgb(lab: [f32; 3]) -> [f32; 3] {
    let [ll, a, b] = lab;
    let l = ll + 0.3963377774 * a + 0.2158037573 * b;
    let m = ll - 0.1055613458 * a - 0.0638541728 * b;
    let s = ll - 0.0894841775 * a - 1.2914855480 * b;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
}

/// Направления в плоскости (a, b) Oklab: «теплее» вдоль линии Планка и «пурпурнее» поперёк неё.
pub struct OklabAxes {
    pub warm: [f32; 2],
    pub magenta: [f32; 2],
}

fn ab_of_white(xy: (f64, f64)) -> [f32; 2] {
    let rgb = mulv(&XYZ_TO_SRGB, xy_to_xyz(xy));
    let lab = oklab([rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]);
    [lab[1], lab[2]]
}

fn unit2(v: [f32; 2]) -> [f32; 2] {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-9);
    [v[0] / n, v[1] / n]
}

pub fn oklab_axes() -> OklabAxes {
    let d65 = ab_of_white(D65_XY);
    let w = ab_of_white(temp_tint_to_xy(3500.0, 0.0));
    let warm = unit2([w[0] - d65[0], w[1] - d65[1]]);
    let (m1, m0) = (ab_of_white(temp_tint_to_xy(6504.0, 20.0)), ab_of_white(temp_tint_to_xy(6504.0, 0.0)));
    let mut m = [m1[0] - m0[0], m1[1] - m0[1]];
    let k = m[0] * warm[0] + m[1] * warm[1];
    m = [m[0] - k * warm[0], m[1] - k * warm[1]];
    OklabAxes { warm, magenta: unit2(m) }
}

pub fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

// ---------- Тональная кривая ----------

pub const SHOULDER_LUT: usize = 1024;
pub const CONTRAST_LUT: usize = 1024;
/// Начало мягкого «плеча» для светов (в линейных единицах, белый = 1).
const SHOULDER_START: f64 = 0.55;

/// Сжатие светов: значения до `max_in` плавно сводятся к 1, ниже 0.55 — без изменений.
pub fn shoulder(x: f64, max_in: f64) -> f64 {
    if max_in <= 1.0 + 1e-6 || x <= SHOULDER_START {
        return x.min(1.0);
    }
    let s = SHOULDER_START;
    let r = (max_in - s) / (1.0 - s);
    let t = ((x - s) / (max_in - s)).clamp(0.0, 1.0);
    s + (1.0 - s) * (1.0 - (1.0 - t).powf(r))
}

/// Базовая кривая: линейная яркость (1.0 = насыщение сенсора) → кодированный sRGB.
/// Подобрана по медианам 680 тыс. блоков JPEG камеры (Canon «Стандарт») на 78 кадрах с M50 Mark II:
/// глубокие тени, яркая середина, мягкое сжатие светов — привычный «камерный» вид без стилизации.
const TONE_POINTS: &[(f64, f64)] = &[
    (0.0, 0.0),
    (0.0023, 0.012),
    (0.0046, 0.026),
    (0.0093, 0.052),
    (0.0186, 0.104),
    (0.0372, 0.203),
    (0.0743, 0.356),
    (0.105, 0.453),
    (0.149, 0.560),
    (0.21, 0.665),
    (0.297, 0.755),
    (0.42, 0.825),
    (0.59, 0.893),
    (0.84, 0.958),
    (1.0, 1.0),
];

/// Монотонная кубическая интерполяция (Фритч — Карлсон).
pub fn monotone_spline(pts: &[(f64, f64)], x: f64) -> f64 {
    let n = pts.len();
    if x <= pts[0].0 {
        return pts[0].1;
    }
    if x >= pts[n - 1].0 {
        return pts[n - 1].1;
    }
    let mut d = vec![0.0; n - 1];
    for i in 0..n - 1 {
        d[i] = (pts[i + 1].1 - pts[i].1) / (pts[i + 1].0 - pts[i].0);
    }
    let mut m = vec![0.0; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for i in 1..n - 1 {
        m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) * 0.5 };
    }
    for i in 0..n - 1 {
        if d[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let a = m[i] / d[i];
            let b = m[i + 1] / d[i];
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * d[i];
                m[i + 1] = t * b * d[i];
            }
        }
    }
    let k = (0..n - 1).find(|&i| x < pts[i + 1].0).unwrap_or(n - 2);
    let h = pts[k + 1].0 - pts[k].0;
    let t = (x - pts[k].0) / h;
    let (t2, t3) = (t * t, t * t * t);
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * pts[k].1 + h10 * h * m[k] + h01 * pts[k + 1].1 + h11 * h * m[k + 1]
}

/// Базовая кривая для одного канала: линейное значение 0..1 → кодированное.
pub fn tone(x: f64) -> f64 {
    monotone_spline(TONE_POINTS, x.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Ручные поправки тона поверх базовой кривой, каждая −1..1: контраст (S-кривая вокруг
/// средне-серого), света (верх кривой) и тени (низ кривой).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ToneAdjust {
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
}

/// Средне-серый на выходе (кодированный, как у JPEG камеры) — опорная точка контраста.
const TONE_PIVOT: f64 = 0.46;
/// Сила «светов» и «теней» при ±1 и крутизна контраста.
const TONE_SH_GAIN: f64 = 0.9;
const TONE_CONTRAST_GAIN: f64 = 0.9;

/// Кодированная яркость (0..1) → с ручными поправками. Монотонно; чёрное, средне-серое (кроме
/// контраста) и белое остаются на месте, наклон кривой без изломов.
pub fn adjust_tone(y: f64, a: &ToneAdjust) -> f64 {
    let p = TONE_PIVOT;
    let mut y = y.clamp(0.0, 1.0);
    if a.shadows != 0.0 && y < p {
        // Подъём/опускание низа: сильнее всего около трети средне-серого, ноль у чёрного и у серого.
        y += a.shadows as f64 * TONE_SH_GAIN * y * (1.0 - y / p).powi(2);
    }
    if a.highlights != 0.0 && y > p {
        let u = (y - p) / (1.0 - p);
        y += a.highlights as f64 * TONE_SH_GAIN * (1.0 - y) * u * u;
    }
    if a.contrast != 0.0 {
        let k = 2f64.powf(TONE_CONTRAST_GAIN * a.contrast as f64);
        y = if y < p { p * (y / p).powf(k) } else { 1.0 - (1.0 - p) * ((1.0 - y) / (1.0 - p)).powf(k) };
    }
    y.clamp(0.0, 1.0)
}

/// Таблицы для GPU: [плечо (SHOULDER_LUT) | базовая кривая (CONTRAST_LUT)].
/// Индексация обеих — по sqrt(x / max) для точности в тенях.
pub fn build_tone_luts(max_in: f64) -> Vec<f32> {
    build_tone_luts_with(max_in, &ToneAdjust::default())
}

/// То же с ручными поправками тона (контраст, света, тени).
pub fn build_tone_luts_with(max_in: f64, adj: &ToneAdjust) -> Vec<f32> {
    let mut lut = Vec::with_capacity(SHOULDER_LUT + CONTRAST_LUT);
    let max_in = max_in.max(1.0);
    for i in 0..SHOULDER_LUT {
        let q = i as f64 / (SHOULDER_LUT - 1) as f64;
        let x = q * q * max_in;
        lut.push(shoulder(x, max_in) as f32);
    }
    for i in 0..CONTRAST_LUT {
        let q = i as f64 / (CONTRAST_LUT - 1) as f64;
        lut.push(adjust_tone(tone(q * q), adj) as f32);
    }
    lut
}

/// Яркость результата (кодированная, 0..1) для линейного серого значения — для анализа экспозиции.
pub fn render_gray(x: f64, max_in: f64) -> f64 {
    tone(shoulder(x, max_in.max(1.0)))
}

/// Быстрая табличная версия тона (как в шейдере): для массовой обработки на CPU.
pub struct ToneLut {
    lut: Vec<f32>,
    max: f32,
}

impl ToneLut {
    pub fn new(gain: f32) -> Self {
        let max = gain.max(1.0);
        ToneLut { lut: build_tone_luts(max as f64), max }
    }

    fn sample(&self, off: usize, n: usize, t: f32) -> f32 {
        let f = t.clamp(0.0, 1.0) * (n - 1) as f32;
        let i0 = f as usize;
        let i1 = (i0 + 1).min(n - 1);
        let a = self.lut[off + i0];
        a + (self.lut[off + i1] - a) * (f - i0 as f32)
    }

    /// Линейный sRGB × gain → кодированный sRGB (плечо по максимуму, кривая по каналам).
    pub fn apply(&self, lin: [f32; 3], gain: f32) -> [f32; 3] {
        let mut c = [lin[0].max(0.0) * gain, lin[1].max(0.0) * gain, lin[2].max(0.0) * gain];
        let n = c[0].max(c[1]).max(c[2]);
        if n > 1e-6 {
            let k = self.sample(0, SHOULDER_LUT, (n / self.max).sqrt()) / n;
            c = [c[0] * k, c[1] * k, c[2] * k];
        }
        [0, 1, 2].map(|i| self.sample(SHOULDER_LUT, CONTRAST_LUT, c[i].clamp(0.0, 1.0).sqrt()))
    }
}

/// Эмуляция шейдера цвета на CPU (без коррекции кожи): линейный sRGB → кодированный sRGB.
pub fn render_rgb(lin: [f32; 3], gain: f32, sat: f32) -> [f32; 3] {
    let mut rgb = [lin[0].max(0.0) * gain, lin[1].max(0.0) * gain, lin[2].max(0.0) * gain];
    if (sat - 1.0).abs() > 1e-3 {
        let lab = oklab(rgb);
        let back = oklab_to_rgb([lab[0], lab[1] * sat, lab[2] * sat]);
        rgb = [back[0].max(0.0), back[1].max(0.0), back[2].max(0.0)];
    }
    let max_in = gain.max(1.0) as f64;
    let n = rgb[0].max(rgb[1]).max(rgb[2]);
    if n > 1e-6 {
        let k = shoulder(n as f64, max_in) as f32 / n;
        rgb = [rgb[0] * k, rgb[1] * k, rgb[2] * k];
    }
    [tone(rgb[0] as f64) as f32, tone(rgb[1] as f64) as f32, tone(rgb[2] as f64) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m50() -> CameraColor {
        CameraColor {
            cm_a: Some([1.2356, -0.4649, -0.0678, -0.391, 1.1644, 0.2577, -0.0455, 0.1934, 0.7348]),
            cm_d65: [1.0463, -0.2173, -0.1437, -0.4856, 1.2635, 0.2482, -0.1216, 0.2915, 0.7237],
        }
    }

    #[test]
    fn temp_tint_roundtrip() {
        for &t in &[2500.0, 3200.0, 4300.0, 5500.0, 6500.0, 9000.0] {
            for &tint in &[-30.0, 0.0, 25.0] {
                let (t2, tint2) = xy_to_temp_tint(temp_tint_to_xy(t, tint));
                assert!((t2 - t).abs() / t < 0.002, "t {t} -> {t2}");
                assert!((tint2 - tint).abs() < 0.5, "tint {tint} -> {tint2}");
            }
        }
    }

    #[test]
    fn camera_wb_roundtrip() {
        let cam = m50();
        for &t in &[2800.0, 3500.0, 5000.0, 6500.0, 8000.0] {
            for &tint in &[-20.0, 0.0, 20.0] {
                let m = cam.multipliers(t, tint);
                let (t2, tint2) = cam.temp_tint_from_multipliers(m);
                assert!((t2 - t).abs() / t < 0.01, "t {t} -> {t2}");
                assert!((tint2 - tint).abs() < 1.0, "tint {tint} -> {tint2}");
            }
        }
    }

    #[test]
    fn white_maps_to_white() {
        let cam = m50();
        let m = cam.cam_to_srgb(3200.0, 5.0);
        let w = mulv(&m, [1.0, 1.0, 1.0]);
        for c in w {
            assert!((c - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn daylight_multipliers_plausible() {
        // Днём красный и синий каналы усиливаются, зелёный — опорный.
        let m = m50().multipliers(5500.0, 0.0);
        assert!(m[1] <= 1.0001 && m[0] > 1.3 && m[0] < 2.8 && m[2] > 1.0 && m[2] < 2.2, "{m:?}");
        // При лампе накаливания красному нужно меньше усиления, синему — больше.
        let t = m50().multipliers(2900.0, 0.0);
        assert!(t[0] < m[0] && t[2] > m[2], "{t:?}");
    }

    #[test]
    fn tone_curve_monotonic_and_bounded() {
        let lut = build_tone_luts(2.0);
        for w in lut[..SHOULDER_LUT].windows(2) {
            assert!(w[1] >= w[0]);
        }
        for w in lut[SHOULDER_LUT..].windows(2) {
            assert!(w[1] >= w[0]);
        }
        assert!((lut[SHOULDER_LUT - 1] - 1.0).abs() < 1e-6);
        // Средний серый камеры (≈0.11 от насыщения) → ≈0.46, как у JPEG Canon.
        assert!((render_gray(0.11, 1.0) - 0.47).abs() < 0.03);
    }

    #[test]
    fn tone_adjustments_are_monotonic_and_keep_ends() {
        for c in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            for h in [-1.0f32, 0.0, 1.0] {
                for s in [-1.0f32, 0.0, 1.0] {
                    let a = ToneAdjust { contrast: c, highlights: h, shadows: s };
                    let mut prev = -1.0;
                    for i in 0..=1000 {
                        let y = adjust_tone(i as f64 / 1000.0, &a);
                        assert!(y >= prev - 1e-12, "немонотонно: {a:?} в {i}");
                        prev = y;
                    }
                    assert!(adjust_tone(0.0, &a).abs() < 1e-9 && (adjust_tone(1.0, &a) - 1.0).abs() < 1e-9, "{a:?}");
                }
            }
        }
        let more = ToneAdjust { contrast: 0.5, ..Default::default() };
        assert!(adjust_tone(0.2, &more) < 0.2 && adjust_tone(0.8, &more) > 0.8, "контраст: тени темнее, света светлее");
        let lift = ToneAdjust { shadows: 1.0, ..Default::default() };
        assert!(adjust_tone(0.15, &lift) > 0.2, "тени поднимаются");
        let tame = ToneAdjust { highlights: -1.0, ..Default::default() };
        assert!(adjust_tone(0.85, &tame) < 0.8, "света приглушаются");
    }

    #[test]
    fn oklab_roundtrip() {
        let c = [0.6f32, 0.4, 0.3];
        let back = oklab_to_rgb(oklab(c));
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-4);
        }
    }
}
