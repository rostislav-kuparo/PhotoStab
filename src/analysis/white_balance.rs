//! Автобаланс белого для репортажа со смешанным светом.
//!
//! Свет кадра измеряется по серым пикселям (Grayness Index, см. `gray_index`): у каждого
//! найденного источника — своя оценка с весом по доле серых пикселей рядом с людьми. Если
//! серого почти нет — запасная оценка gray-edge. Если в кадре есть лица, кожа даёт ещё одно
//! условие: её оттенок должен попасть в естественный диапазон. Итог — минимум робастной энергии
//! на сетке (майред × оттенок). ББ камеры в оценке не участвует: автобаланс Canon сам прыгает от
//! кадра к кадру и намеренно оставляет тепло ламп.
//!
//! Вся съёмка приводится к одной белой точке: нейтраль с одинаковым лёгким теплом
//! (`WARMTH_MIRED`) — так кадры под разным светом выглядят снятыми при одном.

use super::faces::Face;
use super::gray_index;
use crate::color::{self, TEMP_MAX, TEMP_MIN};
use crate::raw::{CameraColor, LinearImage};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WbEstimate {
    pub method: String,
    pub temp: f32,
    pub tint: f32,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WbResult {
    /// Оценка источника света.
    pub illuminant_temp: f32,
    pub illuminant_tint: f32,
    /// Итоговые настройки ББ (источник → общая белая точка съёмки).
    pub temp: f32,
    pub tint: f32,
    pub confidence: f32,
    pub estimates: Vec<WbEstimate>,
}

/// Средний цвет кожи одного лица в цветах камеры (до ББ).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SkinSample {
    pub cam: [f32; 3],
    pub weight: f32,
}

/// Естественный оттенок кожи в Oklab (≈ 49°) и допуск.
pub const SKIN_HUE: f32 = 0.855;
const SKIN_HUE_SIGMA: f32 = 0.14;
const SKIN_CHROMA_LO: f32 = 0.025;
const SKIN_CHROMA_HI: f32 = 0.13;

/// Общее для всей съёмки лёгкое тепло (майред): одинаковое у каждого кадра, при любом свете.
/// Сдвиг в майредах — это одинаковый сдвиг цвета на выходе (в логарифмах каналов свет Планка
/// почти линеен по 1/T), поэтому тепло выглядит одинаково и под лампами, и у окна.
pub const WARMTH_MIRED: f64 = 6.0;
/// Полный вес оценки по серым пикселям (делится между найденными источниками).
const GI_WEIGHT: f64 = 3.0;
/// Вес запасной оценки gray-edge, когда серых пикселей мало.
const EDGE_WEIGHT: f64 = 0.6;

fn valid(c: [f32; 3]) -> bool {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    mx < 0.95 && c[1] > 0.003 && mn > 0.0003
}

fn to_temp_tint(cam: &CameraColor, e: [f64; 3]) -> Option<(f64, f64)> {
    if e.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    let (t, tint) = cam.temp_tint_from_neutral(e);
    Some((t.clamp(TEMP_MIN, TEMP_MAX), tint.clamp(-60.0, 60.0)))
}

fn gray_edge(img: &LinearImage, p: f64) -> Option<[f64; 3]> {
    let (w, h) = (img.width, img.height);
    if w < 8 || h < 8 {
        return None;
    }
    let mut acc = [0f64; 3];
    let mut n = 0f64;
    for y in (1..h - 1).step_by(2) {
        for x in (1..w - 1).step_by(2) {
            let (l, r, u, d) = (img.px(x - 1, y), img.px(x + 1, y), img.px(x, y - 1), img.px(x, y + 1));
            if !(valid(l) && valid(r) && valid(u) && valid(d)) {
                continue;
            }
            for i in 0..3 {
                let gx = (r[i] - l[i]) as f64;
                let gy = (d[i] - u[i]) as f64;
                acc[i] += (gx * gx + gy * gy).sqrt().powf(p);
            }
            n += 1.0;
        }
    }
    if n < 100.0 {
        return None;
    }
    Some([(acc[0] / n).powf(1.0 / p), (acc[1] / n).powf(1.0 / p), (acc[2] / n).powf(1.0 / p)])
}

fn ang_diff(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > std::f32::consts::PI {
        d -= 2.0 * std::f32::consts::PI;
    }
    while d < -std::f32::consts::PI {
        d += 2.0 * std::f32::consts::PI;
    }
    d
}

/// Штраф за неестественный цвет кожи при данных множителях ББ и матрице.
pub(crate) fn skin_penalty_with(mul: &[f32; 3], m: &color::M3, skin: &[SkinSample]) -> f64 {
    if skin.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut wsum = 0.0;
    for s in skin {
        let c = [s.cam[0] * mul[0], s.cam[1] * mul[1], s.cam[2] * mul[2]];
        let c = color::mulv(m, [c[0] as f64, c[1] as f64, c[2] as f64]);
        let lab = color::oklab([c[0] as f32, c[1] as f32, c[2] as f32]);
        let ch = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
        let hue = lab[2].atan2(lab[1]);
        let mut p = (ang_diff(hue, SKIN_HUE) / SKIN_HUE_SIGMA).powi(2) as f64;
        if ch < SKIN_CHROMA_LO {
            p += ((SKIN_CHROMA_LO - ch) / 0.012).powi(2) as f64;
        } else if ch > SKIN_CHROMA_HI {
            p += ((ch - SKIN_CHROMA_HI) / 0.035).powi(2) as f64;
        }
        total += s.weight as f64 * (1.0 + p).ln();
        wsum += s.weight as f64;
    }
    total / wsum.max(1e-9)
}

/// Правдоподобный тон кожи: ±26° от естественного (≈ 23–75° в Oklab). Дальше — уже не кожа,
/// а ошибка света: синие, зелёные, фиолетовые или морковные лица.
const SKIN_VETO_HUE: f32 = 0.45;
const SKIN_VETO_SCALE: f32 = 0.15;

/// Насколько кожа ушла за пределы правдоподобного тона при данных множителях ББ и матрице:
/// 0 внутри коридора, дальше — квадратично. Медиана по лицам: одно «лицо» с плаката не
/// перевешивает настоящие. Почти серые участки (тон не определён) не считаются.
pub(crate) fn skin_veto_with(mul: &[f32; 3], m: &color::M3, skin: &[SkinSample]) -> f64 {
    let mut v: Vec<f64> = skin
        .iter()
        .filter_map(|s| {
            let c = [s.cam[0] * mul[0], s.cam[1] * mul[1], s.cam[2] * mul[2]];
            let c = color::mulv(m, [c[0] as f64, c[1] as f64, c[2] as f64]);
            let lab = color::oklab([c[0] as f32, c[1] as f32, c[2] as f32]);
            let ch = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
            if ch < 0.015 {
                return None;
            }
            let d = (ang_diff(lab[2].atan2(lab[1]), SKIN_HUE).abs() - SKIN_VETO_HUE).max(0.0) / SKIN_VETO_SCALE;
            Some((d * d) as f64)
        })
        .collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Данные одного кадра для совместной оценки ББ по группе.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrameWb {
    pub estimates: Vec<WbEstimate>,
    pub skin: Vec<SkinSample>,
    /// Оттенок ББ камеры. В оценке кадра не участвует; медиана по съёмке — общий якорь оси
    /// «зелёный ↔ пурпурный» (по ней камера права: она знает свою матрицу).
    #[serde(default)]
    pub prior_tint: Option<f32>,
}

/// Оценка ББ одного кадра (как будто кадр единственный).
#[allow(dead_code)]
pub fn estimate(img: &LinearImage, cam: &CameraColor, asshot: Option<[f32; 3]>, skin: &[SkinSample]) -> WbResult {
    let prior_tint = asshot.map(|m| cam.temp_tint_from_multipliers(m).1 as f32);
    let f = FrameWb { estimates: frame_estimates(img, cam, asshot, &[]), skin: skin.to_vec(), prior_tint };
    solve(cam, std::slice::from_ref(&f))
}

/// Оценки источника света по одному кадру: по одной на каждый источник, найденный по серым
/// пикселям (`faces` — лица кадра: свет на людях важнее), и запасная gray-edge, если серых мало.
/// ББ камеры — только если больше ничего нет.
pub fn frame_estimates(img: &LinearImage, cam: &CameraColor, asshot: Option<[f32; 3]>, faces: &[Face]) -> Vec<WbEstimate> {
    let mut est: Vec<WbEstimate> = Vec::new();
    let fl = gray_index::frame_light(img, cam, faces);
    for m in &fl.modes {
        let weight = GI_WEIGHT * fl.reliability as f64 * m.share as f64;
        if weight > 0.02 {
            est.push(WbEstimate { method: "серые пиксели".into(), temp: m.temp, tint: m.tint, weight: weight as f32 });
        }
    }
    let edge_w = EDGE_WEIGHT * (1.0 - fl.reliability as f64);
    if edge_w > 0.02 {
        if let Some((t, tint)) = gray_edge(img, 2.0).and_then(|e| to_temp_tint(cam, e)) {
            est.push(WbEstimate { method: "gray-edge".into(), temp: t as f32, tint: tint as f32, weight: edge_w as f32 });
        }
    }
    if est.is_empty() {
        if let Some((t, tint)) = asshot.and_then(|m| to_temp_tint(cam, [1.0 / m[0] as f64, 1.0 / m[1] as f64, 1.0 / m[2] as f64])) {
            est.push(WbEstimate { method: "камера".into(), temp: t as f32, tint: tint as f32, weight: 0.3 });
        }
    }
    est
}

/// Совместная оценка ББ по одному или нескольким кадрам, снятым при одном свете.
/// Каждый кадр вносит одинаковый вклад; кожа всех лиц — общее условие.
pub fn solve(cam: &CameraColor, frames: &[FrameWb]) -> WbResult {
    let est: Vec<WbEstimate> = frames.iter().flat_map(|f| f.estimates.iter().cloned()).collect();
    let skin_frames: Vec<(f64, &[SkinSample])> = frames
        .iter()
        .filter(|f| !f.skin.is_empty())
        .map(|f| (1.2 + (f.skin.len() as f64 * 0.3).min(1.0), f.skin.as_slice()))
        .collect();
    if est.is_empty() && skin_frames.is_empty() {
        return WbResult {
            illuminant_temp: 5200.0,
            illuminant_tint: 0.0,
            temp: 5200.0,
            tint: 0.0,
            confidence: 0.0,
            estimates: est,
        };
    }

    let energy = |mired: f64, tint: f64| -> f64 {
        let mut e = 0.0;
        for k in &est {
            let dm = (mired - 1e6 / k.temp as f64) / 25.0;
            let dt = (tint - k.tint as f64) / 10.0;
            e += k.weight as f64 * (1.0 + dm * dm + dt * dt).ln();
        }
        if !skin_frames.is_empty() {
            let mul = cam.multipliers(1e6 / mired, tint);
            let m = cam.cam_to_srgb(1e6 / mired, tint);
            for (w, s) in &skin_frames {
                e += w * skin_penalty_with(&mul, &m, s);
            }
        }
        e
    };

    let (m_lo, m_hi) = (1e6 / 12000.0, 1e6 / 2000.0);
    let mut best = (200.0, 0.0, f64::MAX);
    let mut m = m_lo;
    while m <= m_hi {
        let mut t = -50.0;
        while t <= 50.0 {
            let e = energy(m, t);
            if e < best.2 {
                best = (m, t, e);
            }
            t += 2.5;
        }
        m += 4.0;
    }
    for (dm_step, dt_step) in [(1.0, 0.5), (0.25, 0.125)] {
        let (cm, ct) = (best.0, best.1);
        for i in -8..=8 {
            for j in -8..=8 {
                let mm = cm + i as f64 * dm_step;
                let tt = ct + j as f64 * dt_step;
                let e = energy(mm, tt);
                if e < best.2 {
                    best = (mm, tt, e);
                }
            }
        }
    }
    let (mired, tint) = (best.0, best.1);

    // Уверенность: насколько согласны между собой независимые оценки.
    let wsum: f64 = est.iter().map(|k| k.weight as f64).sum();
    let spread = if wsum > 0.0 {
        (est.iter().map(|k| k.weight as f64 * (1e6 / k.temp as f64 - mired).powi(2)).sum::<f64>() / wsum).sqrt()
    } else {
        60.0
    };
    let mut confidence = (-(spread / 40.0).powi(2)).exp();
    if !skin_frames.is_empty() {
        confidence = (confidence + 0.2).min(1.0);
    }

    from_illuminant(mired, tint, confidence as f32, est)
}

/// Итоговые настройки ББ для источника (майред, оттенок): источник приводится к общей белой
/// точке съёмки — нейтраль с одинаковым лёгким теплом, по оттенку полностью.
pub fn from_illuminant(mired: f64, tint: f64, confidence: f32, estimates: Vec<WbEstimate>) -> WbResult {
    // ББ «чуть холоднее источника» (меньше майред) недокомпенсирует его — кадр чуть теплее.
    let mired_out = mired - WARMTH_MIRED;
    WbResult {
        illuminant_temp: (1e6 / mired) as f32,
        illuminant_tint: tint as f32,
        temp: (1e6 / mired_out).clamp(TEMP_MIN, TEMP_MAX) as f32,
        tint: tint as f32,
        confidence,
        estimates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> CameraColor {
        CameraColor {
            cm_a: Some([1.2356, -0.4649, -0.0678, -0.391, 1.1644, 0.2577, -0.0455, 0.1934, 0.7348]),
            cm_d65: [1.0463, -0.2173, -0.1437, -0.4856, 1.2635, 0.2482, -0.1216, 0.2915, 0.7237],
        }
    }

    /// Сцена из нейтральных и цветных поверхностей, снятая при источнике (t, tint).
    fn scene(cam: &CameraColor, t: f64, tint: f64) -> LinearImage {
        let n = cam.neutral(t, tint);
        let mx = n[0].max(n[1]).max(n[2]);
        let (w, h) = (200, 150);
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                // Отражательная способность в «цветах камеры»: в основном серые оттенки + немного цвета.
                let band = (x / 25 + y / 30) % 6;
                let refl: [f64; 3] = match band {
                    0 => [0.7, 0.7, 0.7],
                    1 => [0.35, 0.35, 0.35],
                    2 => [0.5, 0.4, 0.3],
                    3 => [0.2, 0.2, 0.2],
                    4 => [0.3, 0.38, 0.45],
                    _ => [0.12, 0.12, 0.12],
                };
                let shade = 0.6 + 0.4 * ((x as f64 * 0.07).sin() * (y as f64 * 0.05).cos()).abs();
                for i in 0..3 {
                    data.push((refl[i] * shade * n[i] / mx * 0.9) as f32);
                }
            }
        }
        LinearImage { width: w, height: h, data }
    }

    #[test]
    fn one_white_point_for_any_light() {
        // Серый под лампой 2900 K и под дневным светом 6000 K после ББ выглядит одинаково.
        let cam = cam();
        let out = |t: f64, tint: f64| {
            let r = from_illuminant(1e6 / t, tint, 1.0, vec![]);
            let n = cam.neutral(t, tint);
            let mul = cam.multipliers(r.temp as f64, r.tint as f64);
            let m = cam.cam_to_srgb(r.temp as f64, r.tint as f64);
            let c = color::mulv(&m, [n[0] * mul[0] as f64, n[1] * mul[1] as f64, n[2] * mul[2] as f64]);
            let lab = color::oklab([c[0] as f32, c[1] as f32, c[2] as f32]);
            [lab[1] / lab[0], lab[2] / lab[0]]
        };
        let (a, b) = (out(2900.0, 0.0), out(6000.0, -8.0));
        assert!((a[0] - b[0]).abs() < 0.004 && (a[1] - b[1]).abs() < 0.004, "{a:?} vs {b:?}");
        // И это лёгкое тепло, а не нейтраль и не оранжевый.
        assert!(a[1] > 0.002 && a[1] < 0.02, "{a:?}");
    }

    #[test]
    fn recovers_tungsten_and_daylight() {
        let cam = cam();
        for &(t, tint) in &[(2900.0, 0.0), (3400.0, 8.0), (5500.0, 0.0), (6800.0, -6.0)] {
            let img = scene(&cam, t, tint);
            // ББ камеры ошибся на 600 K в тёплую сторону.
            let wrong = cam.multipliers(t + 600.0, 0.0);
            let r = estimate(&img, &cam, Some(wrong), &[]);
            let err_mired = (1e6 / r.illuminant_temp as f64 - 1e6 / t).abs();
            assert!(err_mired < 22.0, "t={t}: оценка {} K ({err_mired:.1} майред)", r.illuminant_temp);
            assert!((r.illuminant_tint as f64 - tint).abs() < 8.0, "tint {tint}: {}", r.illuminant_tint);
        }
    }
}
