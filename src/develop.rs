//! Параметры проявки (то, что видит пользователь) и их перевод в «рецепт» для GPU.

use crate::color;
use crate::raw::CameraColor;
use serde::{Deserialize, Serialize};

/// Коррекция тона кожи: работает только в окне оттенков, похожих на кожу.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct SkinShift {
    /// Центр окна оттенков (Oklab, радианы).
    pub hue_center: f32,
    pub half_width: f32,
    /// Сдвиг оттенка (радианы).
    pub shift: f32,
    /// Множитель насыщенности.
    pub chroma_scale: f32,
    pub chroma_min: f32,
    pub chroma_max: f32,
}

/// Где на кадре кожа: до 8 эллипсов «лицо + шея» в долях кадра (ориентация показа) —
/// центр x, центр y, полуось x, полуось y. Коррекция кожи действует только внутри них,
/// чтобы не трогать дерево, стены и нарисованных людей на картинах.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct SkinMask {
    pub n: u32,
    pub e: [[f32; 4]; 8],
}

/// Эллипс из долей кадра в ориентации показа — в доли кадра в ориентации сенсора.
pub fn ellipse_to_sensor(e: [f32; 4], orientation: u16) -> [f32; 4] {
    let [x, y, rx, ry] = e;
    match orientation {
        2 => [1.0 - x, y, rx, ry],
        3 => [1.0 - x, 1.0 - y, rx, ry],
        4 => [x, 1.0 - y, rx, ry],
        5 => [y, x, ry, rx],
        6 => [y, 1.0 - x, ry, rx],
        7 => [1.0 - y, 1.0 - x, ry, rx],
        8 => [1.0 - y, x, ry, rx],
        _ => e,
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;

    #[test]
    fn ellipse_follows_orientation() {
        // Ориентация 6 (повернуть по часовой): точка показа (X, Y) лежит на сенсоре в (Y, 1 − X),
        // как в `sharpness::orient_plane`; полуоси меняются местами.
        assert_eq!(ellipse_to_sensor([0.9, 0.2, 0.1, 0.3], 6), [0.2, 1.0 - 0.9, 0.3, 0.1]);
        assert_eq!(ellipse_to_sensor([0.9, 0.2, 0.1, 0.3], 8), [1.0 - 0.2, 0.9, 0.3, 0.1]);
        assert_eq!(ellipse_to_sensor([0.9, 0.2, 0.1, 0.3], 1), [0.9, 0.2, 0.1, 0.3]);
    }
}

/// Параметры проявки одного кадра.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct DevelopParams {
    pub temp: f32,
    pub tint: f32,
    pub exposure_ev: f32,
    /// Поворот изображения, градусы, положительный — против часовой.
    pub angle_deg: f32,
    pub skin: Option<SkinShift>,
    /// Где на кадре кожа (без маски коррекция кожи не применяется).
    #[serde(default)]
    pub skin_mask: Option<SkinMask>,
    pub saturation: f32,
    pub sharpen: f32,
    /// ИИ-шумодав (None — выключен).
    #[serde(default)]
    pub denoise: Option<crate::denoise::DenoiseParams>,
    /// Восстановление пересвета 0..1 (0 — обрезанные участки просто белые).
    #[serde(default)]
    pub recovery: f32,
    /// Белая точка содержимого кадра (линейная, до экспокоррекции). None — предел сенсора.
    #[serde(default)]
    pub white: Option<f32>,
    /// Шумодав в RAW до демозаика (профиль шума кадра и сила). None — выключен.
    #[serde(default)]
    pub raw_nr: Option<crate::raw_denoise::RawNr>,
    /// Ручные поправки тона −1..1: контраст, света, тени.
    #[serde(default)]
    pub contrast: f32,
    #[serde(default)]
    pub highlights: f32,
    #[serde(default)]
    pub shadows: f32,
}

impl Default for DevelopParams {
    fn default() -> Self {
        Self {
            temp: 5500.0,
            tint: 0.0,
            exposure_ev: 0.0,
            angle_deg: 0.0,
            skin: None,
            skin_mask: None,
            saturation: 1.0,
            sharpen: 0.35,
            denoise: None,
            recovery: 0.0,
            white: None,
            raw_nr: None,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
        }
    }
}

/// Рецепт для GPU: всё уже посчитано на CPU.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub mul: [f32; 3],
    pub clip: f32,
    pub matrix: [[f32; 3]; 3],
    pub gain: f32,
    pub saturation: f32,
    pub skin: Option<SkinShift>,
    /// Эллипсы кожи в долях кадра в ориентации сенсора и их число.
    pub skin_ell: [[f32; 4]; 8],
    pub skin_n: u32,
    pub orientation: u16,
    pub angle_deg: f32,
    /// Ограничение длинной стороны результата (None — полный размер).
    pub out_long: Option<u32>,
    pub sharpen: f32,
    pub denoise: Option<crate::denoise::DenoiseParams>,
    pub raw_nr: Option<crate::raw_denoise::RawNr>,
    pub recovery: f32,
    pub luts: Vec<f32>,
    pub lut_max: f32,
}

pub fn recipe(p: &DevelopParams, cam: &CameraColor, orientation: u16, out_long: Option<u32>) -> Recipe {
    let t = p.temp as f64;
    let tint = p.tint as f64;
    let mul = cam.multipliers(t, tint);
    let m = cam.cam_to_srgb(t, tint);
    let matrix = [
        [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32],
        [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
        [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32],
    ];
    let gain = 2f32.powf(p.exposure_ev);
    let recovery = p.recovery.clamp(0.0, 1.0);
    // «Плечо» сжимает света только до реальной белой точки кадра: выбитое сенсором уходит
    // в белое, как у камеры, а не сплющивает весь верх. С восстановлением светов плечо
    // продлевается, чтобы достроенные участки легли ниже белого.
    let white = if recovery > 0.0 { recovery_white(recovery) } else { p.white.unwrap_or(1.0).clamp(0.05, 1.0) };
    let lut_max = (gain * white).max(1.0);
    let mask = p.skin_mask.unwrap_or_default();
    let skin_n = mask.n.min(8);
    let mut skin_ell = [[0f32; 4]; 8];
    for i in 0..skin_n as usize {
        skin_ell[i] = ellipse_to_sensor(mask.e[i], orientation);
    }
    Recipe {
        mul,
        clip: if recovery > 0.0 { 1e6 } else { 1.0 },
        matrix,
        gain,
        saturation: p.saturation,
        skin: if skin_n > 0 { p.skin } else { None },
        skin_ell,
        skin_n,
        orientation,
        angle_deg: p.angle_deg,
        out_long,
        sharpen: p.sharpen,
        denoise: p.denoise,
        raw_nr: p.raw_nr,
        recovery,
        luts: color::build_tone_luts_with(lut_max as f64, &color::ToneAdjust { contrast: p.contrast, highlights: p.highlights, shadows: p.shadows }),
        lut_max,
    }
}

/// Уровень «белого» (в единицах обрезки после ББ) при данной силе восстановления светов.
pub fn recovery_white(recovery: f32) -> f32 {
    1.0 + 1.5 * recovery.clamp(0.0, 1.0)
}

/// Геометрия результата: размеры и аффинное отображение «пиксель результата → пиксель источника».
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub dw: u32,
    pub dh: u32,
    /// src_x = a[0]*ox + a[1]*oy + a[2], src_y = b[0]*ox + b[1]*oy + b[2] (координаты с центрами на +0.5).
    pub a: [f32; 3],
    pub b: [f32; 3],
    /// Во сколько раз источник крупнее результата (>= 1).
    pub scale: f32,
}

pub fn display_dims(sw: u32, sh: u32, orientation: u16) -> (u32, u32) {
    if matches!(orientation, 5..=8) { (sh, sw) } else { (sw, sh) }
}

/// Доля кадра, остающаяся после поворота на `angle` без чёрных углов (с сохранением пропорций).
pub fn crop_scale(w: f64, h: f64, angle_deg: f64) -> f64 {
    let a = angle_deg.to_radians().abs();
    let (s, c) = (a.sin(), a.cos());
    (w / (w * c + h * s)).min(h / (w * s + h * c))
}

pub fn geometry(sw: u32, sh: u32, orientation: u16, angle_deg: f32, out_long: Option<u32>) -> Geometry {
    let (dw, dh) = display_dims(sw, sh, orientation);
    let (dwf, dhf) = (dw as f64, dh as f64);
    let k = crop_scale(dwf, dhf, angle_deg as f64);
    let (cw, ch) = (dwf * k, dhf * k);
    let scale_out = match out_long {
        Some(l) => (l as f64 / cw.max(ch)).min(1.0),
        None => 1.0,
    };
    let ow = (cw * scale_out).round().max(1.0);
    let oh = (ch * scale_out).round().max(1.0);
    let (kx, ky) = (ow / cw, oh / ch);
    let th = (angle_deg as f64).to_radians();
    let (sn, cs) = (th.sin(), th.cos());
    let (swf, shf) = (sw as f64, sh as f64);

    let map = |ox: f64, oy: f64| -> (f64, f64) {
        let u = ox / kx - cw / 2.0;
        let v = oy / ky - ch / 2.0;
        // Изображение повёрнуто на th против часовой (ось y вниз): источник = R(-th) * результат.
        let xd = cs * u - sn * v + dwf / 2.0;
        let yd = sn * u + cs * v + dhf / 2.0;
        match orientation {
            2 => (swf - xd, yd),
            3 => (swf - xd, shf - yd),
            4 => (xd, shf - yd),
            5 => (yd, xd),
            6 => (yd, shf - xd),
            7 => (swf - yd, shf - xd),
            8 => (swf - yd, xd),
            _ => (xd, yd),
        }
    };
    let (x0, y0) = map(0.0, 0.0);
    let (x1, y1) = map(1.0, 0.0);
    let (x2, y2) = map(0.0, 1.0);
    Geometry {
        dw: ow as u32,
        dh: oh as u32,
        a: [(x1 - x0) as f32, (x2 - x0) as f32, x0 as f32],
        b: [(y1 - y0) as f32, (y2 - y0) as f32, y0 as f32],
        scale: (1.0 / kx.min(ky)).max(1.0) as f32,
    }
}

/// Для анализа: угол, найденный в координатах сенсора, переводится в координаты показа.
pub fn sensor_angle_to_display(angle: f32, orientation: u16) -> f32 {
    if matches!(orientation, 2 | 4 | 5 | 7) { -angle } else { angle }
}

/// Результат проявки: RGBA8.
pub struct Rgba {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(g: &Geometry, ox: f64, oy: f64) -> (f64, f64) {
        (
            g.a[0] as f64 * ox + g.a[1] as f64 * oy + g.a[2] as f64,
            g.b[0] as f64 * ox + g.b[1] as f64 * oy + g.b[2] as f64,
        )
    }

    #[test]
    fn identity() {
        let g = geometry(600, 400, 1, 0.0, None);
        assert_eq!((g.dw, g.dh), (600, 400));
        let (x, y) = apply(&g, 10.5, 20.5);
        assert!((x - 10.5).abs() < 1e-3 && (y - 20.5).abs() < 1e-3);
    }

    #[test]
    fn orientation_6_rotates_cw() {
        // Сенсор 600x400, EXIF 6: на экране 400x600. Верх-левый угол экрана — это низ-левый угол сенсора.
        let g = geometry(600, 400, 6, 0.0, None);
        assert_eq!((g.dw, g.dh), (400, 600));
        let (x, y) = apply(&g, 0.5, 0.5);
        assert!((x - 0.5).abs() < 1e-3 && (y - 399.5).abs() < 1e-3, "{x} {y}");
    }

    #[test]
    fn downscale() {
        let g = geometry(6000, 4000, 1, 0.0, Some(1500));
        assert_eq!((g.dw, g.dh), (1500, 1000));
        assert!((g.scale - 4.0).abs() < 1e-4);
    }

    #[test]
    fn rotation_stays_inside() {
        let g = geometry(6000, 4000, 1, 4.0, None);
        for &(ox, oy) in &[(0.0, 0.0), (g.dw as f64, 0.0), (0.0, g.dh as f64), (g.dw as f64, g.dh as f64)] {
            let (x, y) = apply(&g, ox, oy);
            assert!(x >= -0.01 && x <= 6000.01 && y >= -0.01 && y <= 4000.01, "{x} {y}");
        }
        assert!(g.dw < 6000 && g.dh < 4000);
    }

    #[test]
    fn positive_angle_is_counter_clockwise() {
        // Точка справа от центра результата берётся из источника ниже центра,
        // т.е. содержимое повёрнуто против часовой стрелки.
        let g = geometry(1000, 1000, 1, 10.0, None);
        let c = g.dw as f64 / 2.0;
        let (_, y) = apply(&g, c + 100.0, c);
        assert!(y > 500.0 + 5.0, "{y}");
    }
}
