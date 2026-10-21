//! Шумодав в RAW, до демозаика: PMRID (Wang et al., «Practical Deep Raw Image Denoising on
//! Mobile Devices», ECCV 2020; megvii-research/PMRID, Apache-2.0), ONNX на видеокарте через
//! DirectML.
//!
//! В RAW шум ещё простой: независимый в каждом пикселе и предсказуемый по уровню сигнала. После
//! демозаика, баланса белого и кривой он превращается в цветные пятна в десятки пикселей, с
//! которыми сети по готовой картинке справляются плохо. Сеть обучена на шуме одного сенсора при
//! ISO 1600; k-sigma-преобразование по профилю шума нашей камеры (`noise::NoiseProfile`, меряется
//! по самим RAW) переводит её шум при любом ISO к этому эталону и обратно.

use crate::noise::NoiseProfile;
use crate::raw::RawFrame;
use anyhow::{Result, anyhow};
use ort::session::Session;
use std::sync::{Mutex, OnceLock};

static MODEL: &[u8] = include_bytes!("../models/pmrid.onnx");
/// Тайл в упакованных пикселях (4 канала RGGB, половинное разрешение) и перекрытие.
const TILE: usize = 512;
const OVERLAP: usize = 32;
/// Шкала сенсора, на котором обучена сеть (10 бит, белое − чёрное), и множитель входа сети.
const V: f64 = 959.0;
const INP_SCALE: f32 = 256.0;
/// Шум эталонного сенсора при ISO 1600: k = K(1600), σ = Σ(1600) (полиномы из PMRID).
fn anchor() -> (f64, f64) {
    let iso = 1600.0f64;
    (0.0005995267 * iso + 0.00868861, 7.11772e-7 * iso * iso + 6.514934e-4 * iso + 0.11492713)
}
/// Почти выбитые пиксели не трогаем: восстановлению светов нужны их исходные значения.
const NEAR_CLIP: f32 = 0.95;

/// Доля шумодава в RAW и доля яркостного шумодава по картинке после него. RAW снимает цветные
/// пятна и часть зерна у источника, сеть по картинке дочищает яркость, не размывая детали
/// (подобрано на ISO 1600 при +1.4 EV: в полную силу RAW-сеть смягчает брови и оправы).
pub const RAW_SHARE: f32 = 0.5;
pub const LUMA_AFTER_RAW: f32 = 0.7;

/// Шумодав в RAW для кадра, если шумодав нужен (`p.denoise`) и профиль шума известен; яркостный
/// шумодав по картинке тогда слабее. Иначе — только шумодав по картинке, как раньше.
pub fn apply(p: &mut crate::develop::DevelopParams, noise: Option<NoiseProfile>) {
    p.raw_nr = None;
    let (Some(d), Some(n)) = (p.denoise, noise) else { return };
    let strength = RAW_SHARE * d.luma.max(d.chroma).clamp(0.0, 1.0);
    if strength < 0.01 {
        return;
    }
    p.raw_nr = Some(RawNr { noise: n, strength });
    p.denoise = Some(crate::denoise::DenoiseParams { luma: d.luma * LUMA_AFTER_RAW, chroma: d.chroma });
}

/// Шумодав в RAW: профиль шума кадра и сила (0..1, доля очищенного сигнала).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RawNr {
    pub noise: NoiseProfile,
    pub strength: f32,
}

fn session() -> Result<&'static Mutex<Session>> {
    static S: OnceLock<std::result::Result<Mutex<Session>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let model = crate::nn::with_input_shape(MODEL, &[1, 4, TILE as i64, TILE as i64]).map_err(|e| e.to_string())?;
        crate::nn::gpu_session(&model).map(Mutex::new).map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| anyhow!("шумодав RAW недоступен: {e}"))
}

/// Начала тайлов вдоль оси длиной `n` (последний прижат к краю).
fn starts(n: usize) -> Vec<usize> {
    if n <= TILE {
        return vec![0];
    }
    let step = TILE - OVERLAP;
    let mut v: Vec<usize> = (0..).map(|i| i * step).take_while(|&s| s + TILE < n).collect();
    v.push(n - TILE);
    v.dedup();
    v
}

/// Вес в зоне перекрытия тайлов: плавный спад, кроме краёв изображения.
fn ramp(i: usize, len: usize, at_start: bool, at_end: bool) -> f32 {
    let o = OVERLAP as f32;
    let mut w = 1.0f32;
    if !at_start {
        w = w.min((i as f32 + 0.5) / o);
    }
    if !at_end {
        w = w.min((len as f32 - i as f32 - 0.5) / o);
    }
    w.clamp(0.0, 1.0)
}

fn reflect(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = i % period;
    if m < n { m } else { period - m }
}

/// Позиции мозаики 2×2 в порядке каналов сети R, G (в ряду R), G (в другом ряду), B.
fn rggb_positions(cfa: [u32; 4]) -> Option<[usize; 4]> {
    let r = (0..4).find(|&p| cfa[p] == 0)?;
    let b = (0..4).find(|&p| cfa[p] == 2)?;
    let g1 = (0..4).find(|&p| cfa[p] == 1 && p / 2 == r / 2)?;
    let g2 = (0..4).find(|&p| cfa[p] == 1 && p != g1)?;
    Some([r, g1, g2, b])
}

/// Очищенные данные сенсора (тот же формат, что `frame.data`).
pub fn run(frame: &RawFrame, nr: &RawNr) -> Result<Vec<u16>> {
    let (w, h) = (frame.width, frame.height);
    let (pw, ph) = (w / 2, h / 2);
    let pos = rggb_positions(frame.cfa).ok_or_else(|| anyhow!("мозаика не байеровская"))?;
    let strength = nr.strength.clamp(0.0, 1.0);
    if strength < 0.01 || pw < 8 || ph < 8 {
        return Ok(frame.data.clone());
    }
    // k-sigma: шум нашей камеры → шум эталона при ISO 1600 (и обратно).
    let (ka, sa) = anchor();
    let kv = (V * nr.noise.k as f64).max(1e-6);
    let sv = V * V * nr.noise.s as f64;
    let cvt_k = ka / kv;
    let cvt_b = (sv / (kv * kv) - sa / (ka * ka)) * ka;
    let fwd = |x: f32| -> f32 { ((x.clamp(0.0, 1.0) as f64 * V * cvt_k + cvt_b) / V) as f32 * INP_SCALE };
    let inv = |y: f32| -> f32 { (((y / INP_SCALE) as f64 * V - cvt_b) / cvt_k / V) as f32 };

    // Упакованные плоскости в долях белого.
    let mut planes: Vec<Vec<f32>> = vec![vec![0f32; pw * ph]; 4];
    for (c, &p) in pos.iter().enumerate() {
        let (oy, ox) = (p / 2, p % 2);
        let black = frame.black[p];
        let range = (frame.white - black).max(1.0);
        let pl = &mut planes[c];
        for y in 0..ph {
            let row = &frame.data[(2 * y + oy) * w..];
            for x in 0..pw {
                pl[y * pw + x] = (row[2 * x + ox] as f32 - black) / range;
            }
        }
    }

    let session = session()?;
    let mut acc = vec![0f32; 4 * pw * ph];
    let mut wsum = vec![0f32; pw * ph];
    let xs = starts(pw);
    let ys = starts(ph);
    let mut input = vec![0f32; 4 * TILE * TILE];
    for (iy, &ty) in ys.iter().enumerate() {
        for (ix, &tx) in xs.iter().enumerate() {
            for c in 0..4 {
                for y in 0..TILE {
                    let sy = reflect(ty + y, ph);
                    for x in 0..TILE {
                        let sx = reflect(tx + x, pw);
                        input[c * TILE * TILE + y * TILE + x] = fwd(planes[c][sy * pw + sx]);
                    }
                }
            }
            let out = crate::nn::run(session, [1, 4, TILE, TILE], &input)?;
            let o = &out.first().ok_or_else(|| anyhow!("шумодав RAW: нет выхода"))?.1;
            let (tw, th) = (TILE.min(pw), TILE.min(ph));
            for y in 0..th {
                let wy = ramp(y, th, iy == 0, iy + 1 == ys.len());
                for x in 0..tw {
                    let wgt = wy * ramp(x, tw, ix == 0, ix + 1 == xs.len());
                    let v = [0, 1, 2, 3].map(|c| o[c * TILE * TILE + y * TILE + x]);
                    if v.iter().all(|c| c.is_finite()) {
                        let i = (ty + y) * pw + tx + x;
                        for c in 0..4 {
                            acc[c * pw * ph + i] += wgt * inv(v[c]);
                        }
                        wsum[i] += wgt;
                    }
                }
            }
        }
    }

    let mut data = frame.data.clone();
    for (c, &p) in pos.iter().enumerate() {
        let (oy, ox) = (p / 2, p % 2);
        let black = frame.black[p];
        let range = (frame.white - black).max(1.0);
        for y in 0..ph {
            for x in 0..pw {
                let i = y * pw + x;
                let orig = planes[c][i];
                if wsum[i] < 1e-3 || orig >= NEAR_CLIP {
                    continue;
                }
                let den = acc[c * pw * ph + i] / wsum[i];
                let v = orig + strength * (den - orig);
                data[(2 * y + oy) * w + 2 * x + ox] = (black + v * range).round().clamp(0.0, 65535.0) as u16;
            }
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rggb_order_for_common_mosaics() {
        assert_eq!(rggb_positions([0, 1, 1, 2]), Some([0, 1, 2, 3]));
        assert_eq!(rggb_positions([2, 1, 1, 0]), Some([3, 2, 1, 0]));
        assert_eq!(rggb_positions([1, 0, 2, 1]), Some([1, 0, 3, 2]));
    }

    #[test]
    fn ksigma_roundtrip() {
        let nr = RawNr { noise: NoiseProfile { k: 7e-4, s: 4e-6 }, strength: 1.0 };
        let (ka, sa) = anchor();
        let kv = V * nr.noise.k as f64;
        let sv = V * V * nr.noise.s as f64;
        let (ck, cb) = (ka / kv, (sv / (kv * kv) - sa / (ka * ka)) * ka);
        for x in [0.0f64, 0.01, 0.2, 0.9] {
            let y = (x * V * ck + cb) / V;
            let back = (y * V - cb) / ck / V;
            assert!((back - x).abs() < 1e-9);
        }
    }
}
