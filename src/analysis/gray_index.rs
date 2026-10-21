//! Серые пиксели по Grayness Index (Qian et al., CVPR 2019, «On Finding Gray Pixels»).
//!
//! Серая поверхность отражает все цвета одинаково, поэтому её локальный контраст в логарифмах
//! одинаков во всех каналах — каким бы ни был свет (цвет источника при логарифме вычитается и
//! исчезает из контраста). GI — насколько лог-контраст красного и синего относительно общей
//! яркости отличается от нуля: чем меньше, тем серее поверхность. Цвет самых серых пикселей —
//! это цвет света, который на них падает.
//!
//! В отличие от «серого мира» и «ярких нейтралей», GI не смотрит на то, что ярче или чего больше:
//! экран проектора и окна не перетягивают оценку на себя. Если источников несколько, серые
//! пиксели собираются в несколько «мод» — по одной на источник.

use super::faces::Face;
use crate::raw::{CameraColor, LinearImage};

/// Какие пиксели вообще годятся: не в шуме, не у предела, не на плоском месте.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Минимум самого тёмного канала (линейный, 1 — предел).
    pub lo: f32,
    /// Максимум самого яркого канала.
    pub hi: f32,
    /// Минимальный лог-контраст (отклик LoG) в каждом канале.
    pub contrast: f32,
}

/// Для линейных данных сенсора (уменьшенный RAW).
pub const RAW_LIMITS: Limits = Limits { lo: 0.002, hi: 0.9, contrast: 0.25 };
/// Для готовой 8-битной картинки (после декодирования sRGB в линейный).
pub const SRGB_LIMITS: Limits = Limits { lo: 0.01, hi: 0.92, contrast: 0.25 };

/// LoG 5×5 (дискретное приближение).
const LOG5: [(i32, i32, f32); 13] = [
    (0, -2, -1.0),
    (-1, -1, -1.0),
    (0, -1, -2.0),
    (1, -1, -1.0),
    (-2, 0, -1.0),
    (-1, 0, -2.0),
    (0, 0, 16.0),
    (1, 0, -2.0),
    (2, 0, -1.0),
    (-1, 1, -1.0),
    (0, 1, -2.0),
    (1, 1, -1.0),
    (0, 2, -1.0),
];

#[inline]
fn log5(p: &[f32], w: usize, x: usize, y: usize) -> f32 {
    let mut s = 0.0;
    for &(dx, dy, k) in &LOG5 {
        s += k * p[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize];
    }
    s
}

/// Сглаживание [1 2 1]/4 по обеим осям (гасит шум перед логарифмом).
fn blur3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; src.len()];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let l = row[x.saturating_sub(1)];
            let r = row[(x + 1).min(w - 1)];
            tmp[y * w + x] = 0.25 * l + 0.5 * row[x] + 0.25 * r;
        }
    }
    let mut out = vec![0f32; src.len()];
    for y in 0..h {
        let (u, d) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            out[y * w + x] = 0.25 * tmp[u * w + x] + 0.5 * tmp[y * w + x] + 0.25 * tmp[d * w + x];
        }
    }
    out
}

/// Кандидаты в серые пиксели: (номер пикселя, GI), от самых серых. `px` — линейный RGB,
/// `skip` — пиксели, которые не рассматривать (например, кожа: её ровный тон в тени тоже даёт
/// одинаковый контраст каналов).
pub fn candidates(px: &[[f32; 3]], w: usize, h: usize, lim: &Limits, skip: Option<&[bool]>) -> Vec<(u32, f32)> {
    if w < 16 || h < 16 || px.len() != w * h {
        return Vec::new();
    }
    let n = w * h;
    const EPS: f32 = 1e-4;
    let ch: Vec<Vec<f32>> = (0..3).map(|k| blur3(&px.iter().map(|c| c[k].max(0.0)).collect::<Vec<_>>(), w, h)).collect();
    let lg: Vec<Vec<f32>> = ch.iter().map(|p| p.iter().map(|v| (v + EPS).ln()).collect()).collect();
    let ls: Vec<f32> = (0..n).map(|i| (ch[0][i] + ch[1][i] + ch[2][i] + 3.0 * EPS).ln()).collect();
    let rr: Vec<f32> = (0..n).map(|i| lg[0][i] - ls[i]).collect();
    let bb: Vec<f32> = (0..n).map(|i| lg[2][i] - ls[i]).collect();

    // GI в каждой годной точке.
    let mut gi = vec![f32::NAN; n];
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let i = y * w + x;
            if skip.is_some_and(|s| s[i]) {
                continue;
            }
            let (c0, c1, c2) = (ch[0][i], ch[1][i], ch[2][i]);
            if c0.min(c1).min(c2) < lim.lo || c0.max(c1).max(c2) > lim.hi {
                continue;
            }
            if (0..3).any(|k| log5(&lg[k], w, x, y).abs() < lim.contrast) {
                continue;
            }
            let a = log5(&rr, w, x, y);
            let b = log5(&bb, w, x, y);
            gi[i] = (a * a + b * b).sqrt();
        }
    }

    // Среднее по окну 7×7 (только годные точки): одиночные «серые» от шума не проходят.
    let (iw, ih) = (w + 1, h + 1);
    let mut sum = vec![0f64; iw * ih];
    let mut cnt = vec![0u32; iw * ih];
    for y in 0..h {
        let (mut rs, mut rc) = (0f64, 0u32);
        for x in 0..w {
            let v = gi[y * w + x];
            if v.is_finite() {
                rs += v as f64;
                rc += 1;
            }
            sum[(y + 1) * iw + x + 1] = sum[y * iw + x + 1] + rs;
            cnt[(y + 1) * iw + x + 1] = cnt[y * iw + x + 1] + rc;
        }
    }
    let mut out: Vec<(u32, f32)> = Vec::new();
    for y in 2..h - 2 {
        let (y0, y1) = (y.saturating_sub(3), (y + 4).min(h));
        for x in 2..w - 2 {
            let i = y * w + x;
            if !gi[i].is_finite() {
                continue;
            }
            let (x0, x1) = (x.saturating_sub(3), (x + 4).min(w));
            let s = sum[y1 * iw + x1] - sum[y0 * iw + x1] - sum[y1 * iw + x0] + sum[y0 * iw + x0];
            let c = cnt[y1 * iw + x1] - cnt[y0 * iw + x1] - cnt[y1 * iw + x0] + cnt[y0 * iw + x0];
            if c >= 4 {
                out.push((i as u32, (s / c as f64) as f32));
            }
        }
    }
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

/// Средний цвет самых серых пикселей (доля `frac` лучших, не меньше `min_n`), нормированный
/// на сумму каналов. None — серых почти нет.
pub fn gray_color(px: &[[f32; 3]], w: usize, h: usize, lim: &Limits, skip: Option<&[bool]>, frac: f32, min_n: usize) -> Option<[f32; 3]> {
    let c = candidates(px, w, h, lim, skip);
    if c.len() < min_n {
        return None;
    }
    let k = ((c.len() as f32 * frac) as usize).max(min_n).min(c.len());
    let mut acc = [0f64; 3];
    for &(i, _) in &c[..k] {
        let p = px[i as usize];
        let s = (p[0] + p[1] + p[2]).max(1e-6) as f64;
        for j in 0..3 {
            acc[j] += p[j] as f64 / s;
        }
    }
    Some(acc.map(|v| (v / k as f64) as f32))
}

/// Маска кожи: эллипсы «лицо + шея» (как у коррекции кожи), в координатах картинки `w×h`,
/// если лица найдены на картинке шириной `src_w`.
pub fn face_skip(faces: &[Face], src_w: usize, w: usize, h: usize) -> Option<Vec<bool>> {
    if faces.is_empty() || src_w == 0 {
        return None;
    }
    let s = w as f32 / src_w as f32;
    let mut m = vec![false; w * h];
    for f in faces {
        let (cx, cy) = ((f.x + 0.5 * f.w) * s, (f.y + 0.75 * f.h) * s);
        let (rx, ry) = ((0.75 * f.w * s).max(2.0), (0.95 * f.h * s).max(2.0));
        let (x0, x1) = ((cx - rx).floor().max(0.0) as usize, ((cx + rx).ceil() as usize).min(w));
        let (y0, y1) = ((cy - ry).floor().max(0.0) as usize, ((cy + ry).ceil() as usize).min(h));
        for y in y0..y1 {
            for x in x0..x1 {
                let (dx, dy) = ((x as f32 + 0.5 - cx) / rx, (y as f32 + 0.5 - cy) / ry);
                if dx * dx + dy * dy <= 1.0 {
                    m[y * w + x] = true;
                }
            }
        }
    }
    Some(m)
}

/// Уменьшение вдвое (среднее 2×2).
pub fn half(img: &LinearImage) -> (Vec<[f32; 3]>, usize, usize) {
    let (w, h) = (img.width / 2, img.height / 2);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut c = [0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = img.px(2 * x + dx, 2 * y + dy);
                for k in 0..3 {
                    c[k] += 0.25 * p[k];
                }
            }
            out.push(c);
        }
    }
    (out, w, h)
}

/// Источник света, найденный по серым пикселям: (температура, оттенок) и доля серых пикселей
/// (с весами «свет на людях»).
#[derive(Clone, Copy, Debug)]
pub struct LightMode {
    pub temp: f32,
    pub tint: f32,
    pub share: f32,
}

/// Результат по кадру: моды света и надёжность (0..1, по числу серых пикселей).
#[derive(Clone, Debug, Default)]
pub struct FrameLight {
    pub modes: Vec<LightMode>,
    pub reliability: f32,
}

/// Доля лучших кандидатов, из которых строятся моды.
const TOP_FRAC: f32 = 0.03;
const TOP_MIN: usize = 150;
const TOP_MAX: usize = 12000;
/// Шаг гистограммы лог-хромы и её охват.
const BIN: f32 = 0.02;
const U_LO: f32 = -1.5;
const U_HI: f32 = 2.5;
/// Моды ближе этого (в лог-хроме) — один источник.
const MODE_SEP: f32 = 0.07;
/// Мода слабее этой доли от главной не считается.
const MODE_MIN: f32 = 0.12;

/// Моды света кадра по серым пикселям. `img` — уменьшенный кадр в цветах камеры, `faces` —
/// лица на нём: серые пиксели рядом с людьми весят больше (нас интересует свет на людях),
/// самые яркие (экраны, окна, лампы) — меньше.
pub fn frame_light(img: &LinearImage, cam: &CameraColor, faces: &[Face]) -> FrameLight {
    let (px, w, h) = if img.width > 1000 { half(img) } else { (img.data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(), img.width, img.height) };
    let skip = face_skip(faces, img.width, w, h);
    let cand = candidates(&px, w, h, &RAW_LIMITS, skip.as_deref());
    if cand.len() < TOP_MIN {
        return FrameLight::default();
    }
    let k = ((cand.len() as f32 * TOP_FRAC) as usize).clamp(TOP_MIN, TOP_MAX).min(cand.len());
    let top = &cand[..k];

    // Порог «самых ярких» по годным кандидатам (экраны, окна, лампы).
    let mut ys: Vec<f32> = cand.iter().step_by(4).map(|&(i, _)| {
        let p = px[i as usize];
        p[0] + p[1] + p[2]
    }).collect();
    let bright = if ys.is_empty() {
        f32::MAX
    } else {
        let q = ((ys.len() - 1) as f32 * 0.98) as usize;
        ys.select_nth_unstable_by(q, |a, b| a.total_cmp(b));
        ys[q]
    };
    let s = w as f32 / img.width.max(1) as f32;
    let fc: Vec<(f32, f32, f32)> = faces.iter().map(|f| ((f.x + 0.5 * f.w) * s, (f.y + 0.5 * f.h) * s, (f.h * s).max(2.0))).collect();

    let nb = ((U_HI - U_LO) / BIN) as usize;
    let mut hist = vec![0f32; nb * nb];
    let mut pts: Vec<(f32, f32, f32)> = Vec::with_capacity(k);
    for &(i, _) in top {
        let p = px[i as usize];
        let (x, y) = ((i as usize % w) as f32, (i as usize / w) as f32);
        let u = (p[1] / p[0].max(1e-6)).ln();
        let v = (p[1] / p[2].max(1e-6)).ln();
        let mut wt = 1.0f32;
        if p[0] + p[1] + p[2] > bright {
            wt *= 0.3;
        }
        if y < h as f32 / 3.0 {
            wt *= 0.7;
        }
        if !fc.is_empty() {
            let near = fc
                .iter()
                .map(|&(cx, cy, fh)| {
                    let d2 = (x - cx).powi(2) + (y - cy).powi(2);
                    (-d2 / (2.0 * (2.0 * fh).powi(2))).exp()
                })
                .fold(0.0f32, f32::max);
            wt *= 1.0 + 2.0 * near;
        }
        let (bu, bv) = (((u - U_LO) / BIN).floor(), ((v - U_LO) / BIN).floor());
        if bu < 0.0 || bv < 0.0 || bu >= nb as f32 || bv >= nb as f32 {
            continue;
        }
        hist[bv as usize * nb + bu as usize] += wt;
        pts.push((u, v, wt));
    }
    let total: f32 = pts.iter().map(|p| p.2).sum();
    if pts.len() < TOP_MIN / 2 || total <= 0.0 {
        return FrameLight::default();
    }

    // Сглаживание гистограммы (σ ≈ 1.5 бина) и поиск пиков.
    let kern: Vec<f32> = (-4i32..=4).map(|d| (-(d * d) as f32 / (2.0 * 1.5 * 1.5)).exp()).collect();
    let mut tmp = vec![0f32; nb * nb];
    for y in 0..nb {
        for x in 0..nb {
            let mut s = 0.0;
            for (j, kv) in kern.iter().enumerate() {
                let xx = x as i32 + j as i32 - 4;
                if xx >= 0 && (xx as usize) < nb {
                    s += kv * hist[y * nb + xx as usize];
                }
            }
            tmp[y * nb + x] = s;
        }
    }
    let mut sm = vec![0f32; nb * nb];
    for y in 0..nb {
        for x in 0..nb {
            let mut s = 0.0;
            for (j, kv) in kern.iter().enumerate() {
                let yy = y as i32 + j as i32 - 4;
                if yy >= 0 && (yy as usize) < nb {
                    s += kv * tmp[yy as usize * nb + x];
                }
            }
            sm[y * nb + x] = s;
        }
    }
    let peak_max = sm.iter().copied().fold(0.0f32, f32::max);
    let mut peaks: Vec<(f32, f32, f32)> = Vec::new();
    for y in 1..nb - 1 {
        for x in 1..nb - 1 {
            let v = sm[y * nb + x];
            if v < MODE_MIN * peak_max {
                continue;
            }
            let is_max = (-1i32..=1).all(|dy| (-1i32..=1).all(|dx| (dx == 0 && dy == 0) || sm[(y as i32 + dy) as usize * nb + (x as i32 + dx) as usize] <= v));
            if is_max {
                peaks.push((U_LO + (x as f32 + 0.5) * BIN, U_LO + (y as f32 + 0.5) * BIN, v));
            }
        }
    }
    peaks.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut kept: Vec<(f32, f32)> = Vec::new();
    for (u, v, _) in peaks {
        if kept.iter().all(|&(ku, kv)| ((u - ku).powi(2) + (v - kv).powi(2)).sqrt() >= MODE_SEP) {
            kept.push((u, v));
        }
        if kept.len() == 4 {
            break;
        }
    }

    // Каждая точка — к ближайшему пику; центр моды — взвешенное среднее её точек рядом с пиком.
    let mut acc = vec![(0f64, 0f64, 0f64); kept.len()];
    for &(u, v, wt) in &pts {
        let (best, d2) = kept
            .iter()
            .enumerate()
            .map(|(j, &(ku, kv))| (j, (u - ku).powi(2) + (v - kv).powi(2)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        if d2 > 0.1 * 0.1 {
            continue;
        }
        let a = &mut acc[best];
        let wt = wt as f64;
        a.0 += wt * u as f64;
        a.1 += wt * v as f64;
        a.2 += wt;
    }
    let mut modes = Vec::new();
    for a in acc {
        if a.2 <= 0.0 {
            continue;
        }
        let (u, v) = (a.0 / a.2, a.1 / a.2);
        let neutral = [(-u).exp(), 1.0, (-v).exp()];
        let (t, tint) = cam.temp_tint_from_neutral(neutral);
        if !t.is_finite() || !tint.is_finite() {
            continue;
        }
        modes.push(LightMode {
            temp: t.clamp(crate::color::TEMP_MIN, crate::color::TEMP_MAX) as f32,
            tint: tint.clamp(-60.0, 60.0) as f32,
            share: (a.2 / total as f64) as f32,
        });
    }
    modes.sort_by(|a, b| b.share.total_cmp(&a.share));
    FrameLight { modes, reliability: (pts.len() as f32 / 600.0).min(1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Текстура «серые и цветные пятна» под источником `light` (множители каналов).
    fn scene(w: usize, h: usize, light_at: impl Fn(usize, usize) -> [f32; 3]) -> Vec<[f32; 3]> {
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let band = (x / 9 + 2 * (y / 11)) % 5;
                let refl: [f32; 3] = match band {
                    0 => [0.6, 0.6, 0.6],
                    1 => [0.25, 0.25, 0.25],
                    2 => [0.55, 0.3, 0.15],
                    3 => [0.12, 0.12, 0.12],
                    _ => [0.15, 0.35, 0.5],
                };
                let shade = 0.55 + 0.45 * ((x as f32 * 0.31).sin() * (y as f32 * 0.23).cos()).abs();
                let l = light_at(x, y);
                v.push([0, 1, 2].map(|k| refl[k] * shade * l[k] * 0.8));
            }
        }
        v
    }

    #[test]
    fn grey_patches_give_light_colour() {
        let light = [1.0, 0.75, 0.4]; // тёплый свет
        let px = scene(160, 120, |_, _| light);
        let g = gray_color(&px, 160, 120, &RAW_LIMITS, None, 0.05, 40).expect("серые найдены");
        let s = light[0] + light[1] + light[2];
        for k in 0..3 {
            assert!((g[k] - light[k] / s).abs() < 0.03, "{g:?} vs {:?}", light.map(|v| v / s));
        }
    }

    #[test]
    fn bright_cool_screen_does_not_win_over_warm_light_on_people() {
        // Как на панели у экрана: сверху яркий холодный экран, внизу тёплый свет на людях.
        let cam = CameraColor {
            cm_a: Some([1.2356, -0.4649, -0.0678, -0.391, 1.1644, 0.2577, -0.0455, 0.1934, 0.7348]),
            cm_d65: [1.0463, -0.2173, -0.1437, -0.4856, 1.2635, 0.2482, -0.1216, 0.2915, 0.7237],
        };
        let (w, h) = (240, 180);
        let unit = |n: [f64; 3]| {
            let m = n[0].max(n[1]).max(n[2]);
            [(n[0] / m) as f32, (n[1] / m) as f32, (n[2] / m) as f32]
        };
        let warm = unit(cam.neutral(3000.0, 0.0));
        let cool = unit(cam.neutral(6500.0, 0.0));
        // Экран занимает половину кадра и вдвое ярче — «серый мир» и «яркие нейтрали» пошли бы за ним.
        let px = scene(w, h, |_, y| if y < h / 2 { cool.map(|c| c * 1.6) } else { warm.map(|c| c * 0.7) });
        let data: Vec<f32> = px.iter().flat_map(|c| *c).collect();
        let img = LinearImage { width: w, height: h, data };
        let face = Face { x: 100.0, y: 110.0, w: 20.0, h: 24.0, score: 0.9, landmarks: [[0.0; 2]; 5] };
        let fl = frame_light(&img, &cam, &[face]);
        assert!(!fl.modes.is_empty());
        let main = fl.modes[0];
        let d = |t: f32, r: f64| (1e6 / t as f64 - 1e6 / r).abs();
        assert!(d(main.temp, 3000.0) < d(main.temp, 6500.0), "главная мода {main:?}; все: {:?}", fl.modes);
        assert!(d(main.temp, 3000.0) < 15.0 && main.tint.abs() < 5.0, "{main:?}");
    }
}
