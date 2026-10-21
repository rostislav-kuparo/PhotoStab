//! Кожа: замер по щекам и лбу (по ключевым точкам YuNet) и мягкая коррекция оттенка.

use super::faces::Face;
use super::white_balance::{SKIN_HUE, SkinSample};
use crate::color;
use crate::develop::SkinShift;
use crate::raw::LinearImage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SkinStats {
    /// Оттенок в Oklab, радианы.
    pub hue: f32,
    pub chroma: f32,
    /// Линейная яркость Y (медиана).
    pub luma: f32,
}

const HUE_LO: f32 = 0.66; // ≈ 38°
const HUE_HI: f32 = 1.08; // ≈ 62°
const MAX_SHIFT: f32 = 0.24; // ≈ 14°
const CHROMA_MAX: f32 = 0.115;

/// Центры и радиус участков кожи: две щеки и лоб.
fn patches(f: &Face) -> ([[f32; 2]; 3], f32) {
    let [re, le, nose, rm, lm] = f.landmarks;
    let eye_mid = [(re[0] + le[0]) * 0.5, (re[1] + le[1]) * 0.5];
    let mouth_mid = [(rm[0] + lm[0]) * 0.5, (rm[1] + lm[1]) * 0.5];
    let down = mouth_mid[1] - eye_mid[1];
    let cheek = |eye: [f32; 2], mouth: [f32; 2]| -> [f32; 2] {
        let x = eye[0] * 0.6 + mouth[0] * 0.4 + (eye[0] - nose[0]) * 0.25;
        let y = eye[1] + (mouth[1] - eye[1]) * 0.55;
        [x, y]
    };
    let forehead = [eye_mid[0], eye_mid[1] - down.abs() * 0.5];
    let r = (f.w * 0.08).max(1.5);
    ([cheek(re, rm), cheek(le, lm), forehead], r)
}

fn collect<F: Fn(usize, usize) -> [f32; 3]>(f: &Face, w: usize, h: usize, get: F) -> Vec<[f32; 3]> {
    let (centers, r) = patches(f);
    let mut out = Vec::new();
    for c in centers {
        let (x0, x1) = ((c[0] - r).floor().max(0.0) as usize, ((c[0] + r).ceil() as usize).min(w.saturating_sub(1)));
        let (y0, y1) = ((c[1] - r).floor().max(0.0) as usize, ((c[1] + r).ceil() as usize).min(h.saturating_sub(1)));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (dx, dy) = (x as f32 + 0.5 - c[0], y as f32 + 0.5 - c[1]);
                if dx * dx + dy * dy <= r * r {
                    out.push(get(x, y));
                }
            }
        }
    }
    out
}

/// Отбрасывает тени и блики: оставляет пиксели между 20-м и 90-м процентилем яркости.
fn trim_by_luma(px: &mut Vec<[f32; 3]>) {
    if px.len() < 10 {
        return;
    }
    px.sort_by(|a, b| (a[0] + a[1] + a[2]).total_cmp(&(b[0] + b[1] + b[2])));
    let lo = px.len() / 5;
    let hi = px.len() * 9 / 10;
    *px = px[lo..hi].to_vec();
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let k = v.len() / 2;
    v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    v[k]
}

fn usable(f: &Face) -> bool {
    f.score >= 0.7 && f.w >= 14.0
}

/// Цвет кожи в цветах камеры — для оценки баланса белого.
pub fn camera_samples(img: &LinearImage, faces: &[Face]) -> Vec<SkinSample> {
    let mut out = Vec::new();
    for f in faces.iter().filter(|f| usable(f)) {
        let mut px = collect(f, img.width, img.height, |x, y| img.px(x, y));
        px.retain(|c| c[0].max(c[1]).max(c[2]) < 0.95 && c[1] > 0.003);
        trim_by_luma(&mut px);
        if px.len() < 6 {
            continue;
        }
        let cam = [
            median(px.iter().map(|c| c[0]).collect()),
            median(px.iter().map(|c| c[1]).collect()),
            median(px.iter().map(|c| c[2]).collect()),
        ];
        out.push(SkinSample { cam, weight: f.area().sqrt() * f.score });
    }
    out
}

/// Цвет кожи в линейном sRGB (после ББ, до экспокоррекции).
pub fn measure(lin: &[[f32; 3]], w: usize, h: usize, faces: &[Face]) -> Option<SkinStats> {
    let mut hues = Vec::new();
    let mut acc = (0f32, 0f32, 0f32, 0f32);
    for f in faces.iter().filter(|f| usable(f)) {
        let mut px = collect(f, w, h, |x, y| lin[y * w + x]);
        px.retain(|c| c[0].max(c[1]).max(c[2]) < 0.98);
        trim_by_luma(&mut px);
        if px.len() < 6 {
            continue;
        }
        let labs: Vec<[f32; 3]> = px.iter().map(|c| color::oklab(*c)).collect();
        let a = median(labs.iter().map(|l| l[1]).collect());
        let b = median(labs.iter().map(|l| l[2]).collect());
        let y = median(px.iter().map(|c| super::exposure::luma(*c)).collect());
        let wgt = f.area().sqrt() * f.score;
        acc.0 += a * wgt;
        acc.1 += b * wgt;
        acc.2 += y * wgt;
        acc.3 += wgt;
        hues.push(b.atan2(a));
    }
    if acc.3 <= 0.0 {
        return None;
    }
    let (a, b) = (acc.0 / acc.3, acc.1 / acc.3);
    Some(SkinStats { hue: b.atan2(a), chroma: (a * a + b * b).sqrt(), luma: acc.2 / acc.3 })
}

/// Общая коррекция кожи для группы: усредняем цвет кожи по всем кадрам группы.
pub fn group_correction(stats: &[SkinStats]) -> Option<SkinShift> {
    if stats.is_empty() {
        return None;
    }
    let n = stats.len() as f32;
    let a = stats.iter().map(|s| s.chroma * s.hue.cos()).sum::<f32>() / n;
    let b = stats.iter().map(|s| s.chroma * s.hue.sin()).sum::<f32>() / n;
    let mut chroma: Vec<f32> = stats.iter().map(|s| s.chroma).collect();
    chroma.sort_by(|x, y| x.total_cmp(y));
    let luma = stats.iter().map(|s| s.luma).sum::<f32>() / n;
    correction(&SkinStats { hue: b.atan2(a), chroma: chroma[chroma.len() / 2], luma })
}

/// Мягкая коррекция: возвращает тон кожи в естественный диапазон и гасит «варёную» насыщенность.
/// Яркость кожи не меняется.
pub fn correction(s: &SkinStats) -> Option<SkinShift> {
    let shift = if s.hue < HUE_LO {
        (HUE_LO - s.hue) * 0.8
    } else if s.hue > HUE_HI {
        (HUE_HI - s.hue) * 0.8
    } else {
        // Внутри диапазона — лёгкое подтягивание к центру.
        (SKIN_HUE - s.hue) * 0.15
    }
    .clamp(-MAX_SHIFT, MAX_SHIFT);
    let chroma_scale = if s.chroma > CHROMA_MAX { (CHROMA_MAX + (s.chroma - CHROMA_MAX) * 0.35) / s.chroma } else { 1.0 };
    if shift.abs() < 0.015 && chroma_scale > 0.98 {
        return None;
    }
    Some(SkinShift {
        hue_center: s.hue,
        half_width: 0.55,
        shift,
        chroma_scale,
        chroma_min: 0.02,
        chroma_max: 0.22,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reddish_skin_is_pulled_back() {
        let s = SkinStats { hue: 0.45, chroma: 0.16, luma: 0.3 };
        let c = correction(&s).unwrap();
        assert!(c.shift > 0.1 && c.chroma_scale < 0.9, "{c:?}");
    }

    #[test]
    fn natural_skin_untouched() {
        let s = SkinStats { hue: SKIN_HUE, chroma: 0.07, luma: 0.3 };
        assert!(correction(&s).is_none());
    }

    #[test]
    fn reference_skin_patch_hue_in_range() {
        // Macbeth «светлая кожа» sRGB(194,150,130) и «тёмная кожа» (115,82,68).
        for rgb in [[194u8, 150, 130], [115, 82, 68]] {
            let lin = [0, 1, 2].map(|i| color::srgb_decode(rgb[i] as f32 / 255.0));
            let lab = color::oklab(lin);
            let hue = lab[2].atan2(lab[1]);
            assert!(hue > HUE_LO && hue < HUE_HI, "{rgb:?}: {}", hue.to_degrees());
        }
    }
}
