//! ИИ-шумодав по готовой картинке: NAFNet-SIDD-width32 (megvii-research/NAFNet, MIT), ONNX fp16,
//! на видеокарте через ONNX Runtime + DirectML. Работает тайлами 512×512 с плавным сшиванием;
//! сила задаётся отдельно для яркостного и цветового шума (как в DaVinci Resolve). При экспорте
//! и в лупе основную часть шума сенсора снимает шумодав в RAW (`raw_denoise`), этот дочищает.
//! (SCUNet по качеству чуть лучше, но через DirectML на порядок медленнее — для сотен кадров
//! не годится.)

use anyhow::{Result, anyhow};
use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::sync::{Mutex, OnceLock};

static MODEL: &[u8] = include_bytes!("../models/nafnet_sidd_w32_512_fp16.onnx");
const TILE: usize = 512;
const OVERLAP: usize = 64;

/// Сила шумоподавления 0..1: яркостный и цветовой шум.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct DenoiseParams {
    pub luma: f32,
    pub chroma: f32,
}

impl DenoiseParams {
    pub fn active(&self) -> bool {
        self.luma > 0.01 || self.chroma > 0.01
    }
}

/// Сила по умолчанию: от «эффективного ISO» — ISO, умноженного на осветление при проявке
/// (вытянутые тени шумят как более высокий ISO).
pub fn auto_strength(iso: Option<u32>, exposure_ev: f32) -> Option<DenoiseParams> {
    let iso = iso? as f32;
    let eff = iso * 2f32.powf(exposure_ev.max(0.0));
    // До ~ISO 1600 шум у APS-C почти незаметен — не тратим время.
    if eff < 1600.0 {
        return None;
    }
    let luma = ((eff / 1600.0).log2() / 2.0).clamp(0.0, 1.0) * 0.8;
    let chroma = ((eff / 800.0).log2() / 2.0).clamp(0.0, 1.0);
    let p = DenoiseParams { luma, chroma };
    p.active().then_some(p)
}

struct Engine {
    session: Mutex<Session>,
}

fn engine() -> Result<&'static Engine> {
    static E: OnceLock<std::result::Result<Engine, String>> = OnceLock::new();
    E.get_or_init(|| {
        let build = || -> ort::Result<Engine> {
            let env = crate::nn::env().map_err(|e| ort::Error::new(e.to_string()))?;
            let session = Session::builder(env)?
                .with_execution_providers([ort::ep::DirectML::default().build()])?
                .commit_from_memory(MODEL)?;
            Ok(Engine { session: Mutex::new(session) })
        };
        build().map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| anyhow!("шумодав недоступен: {e}"))
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

/// Вес пикселя тайла: плавный спад в зоне перекрытия, кроме краёв изображения.
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

/// Наибольшая правка шумодавом (доля диапазона кодированного sRGB): яркости и цветности.
/// Цветовой шум мал, а сбой сети — всегда в цвете, поэтому цветность ограничена строже.
const MAX_LUMA_CHANGE: f32 = 0.15;
const MAX_CHROMA_CHANGE: f32 = 0.05;

fn unpack10(v: u32) -> [f32; 3] {
    [(v & 1023) as f32 / 1023.0, ((v >> 10) & 1023) as f32 / 1023.0, ((v >> 20) & 1023) as f32 / 1023.0]
}

fn pack10(c: [f32; 3]) -> u32 {
    let q = |x: f32| (x.clamp(0.0, 1.0) * 1023.0).round() as u32;
    q(c[0]) | (q(c[1]) << 10) | (q(c[2]) << 20)
}

/// Прямоугольник в пикселях: x, y, ширина, высота.
pub type Region = (usize, usize, usize, usize);

/// Шумоподавление изображения в упаковке 10:10:10 (кодированный sRGB) на месте.
/// `region` — обработать только часть кадра (для лупы).
pub fn denoise_packed(px: &mut [u32], w: usize, h: usize, p: DenoiseParams, region: Option<Region>) -> Result<()> {
    if !p.active() {
        return Ok(());
    }
    let eng = engine()?;
    let (rx, ry, rw, rh) = region.unwrap_or((0, 0, w, h));
    let (rw, rh) = (rw.min(w - rx), rh.min(h - ry));
    // Рабочая область с запасом, чтобы у краёв региона был контекст.
    let pad = if region.is_some() { 48 } else { 0 };
    let (ax, ay) = (rx.saturating_sub(pad), ry.saturating_sub(pad));
    let (aw, ah) = ((rx + rw + pad).min(w) - ax, (ry + rh + pad).min(h) - ay);

    let mut acc = vec![0f32; aw * ah * 3];
    let mut wsum = vec![0f32; aw * ah];
    let xs = starts(aw);
    let ys = starts(ah);
    let mut input = vec![0f32; 3 * TILE * TILE];
    let mut session = eng.session.lock().map_err(|_| anyhow!("шумодав занят"))?;
    for (iy, &ty) in ys.iter().enumerate() {
        for (ix, &tx) in xs.iter().enumerate() {
            // Тайл с зеркальным дополнением, если область меньше тайла.
            for y in 0..TILE {
                let sy = reflect(ty + y, ah);
                for x in 0..TILE {
                    let sx = reflect(tx + x, aw);
                    let c = unpack10(px[(ay + sy) * w + ax + sx]);
                    for ch in 0..3 {
                        input[ch * TILE * TILE + y * TILE + x] = c[ch];
                    }
                }
            }
            let o: Vec<f32> = {
                let half: Vec<f16> = input.iter().map(|v| f16::from_f32(*v)).collect();
                let t = Tensor::from_array(([1usize, 3, TILE, TILE], half)).map_err(|e| anyhow!("{e}"))?;
                let out = session.run(ort::inputs![t]).map_err(|e| anyhow!("шумодав: {e}"))?;
                let (_, o) = out[0].try_extract_tensor::<f16>().map_err(|e| anyhow!("{e}"))?;
                o.iter().map(|v| v.to_f32()).collect()
            };
            let (tw, th) = (TILE.min(aw), TILE.min(ah));
            let bad = suspicious_blocks(&input, &o, tw, th);
            for y in 0..th {
                let wy = ramp(y, th, iy == 0, iy + 1 == ys.len());
                for x in 0..tw {
                    let wgt = wy * ramp(x, tw, ix == 0, ix + 1 == xs.len());
                    let i = (ty + y) * aw + tx + x;
                    let v = [0, 1, 2].map(|ch| o[ch * TILE * TILE + y * TILE + x]);
                    // На ярких насыщенных цветах (жёлтый у предела) вычисления в половинной точности
                    // иногда переполняются — такие ответы сети не берём, там останется исходник.
                    if !bad[(y / BLOCK) * TILE.div_ceil(BLOCK) + x / BLOCK] && v.iter().all(|c| c.is_finite() && *c > -0.5 && *c < 1.5) {
                        for ch in 0..3 {
                            acc[i * 3 + ch] += wgt * v[ch];
                        }
                        wsum[i] += wgt;
                    }
                }
            }
        }
    }
    drop(session);

    // Смешивание с исходником отдельно по яркости и цветности (YCbCr).
    for y in ry..ry + rh {
        for x in rx..rx + rw {
            let i = (y - ay) * aw + (x - ax);
            if wsum[i] < 1e-3 {
                continue;
            }
            let ws = wsum[i];
            let s = unpack10(px[y * w + x]);
            let d = [acc[i * 3] / ws, acc[i * 3 + 1] / ws, acc[i * 3 + 2] / ws];
            let ys_ = 0.299 * s[0] + 0.587 * s[1] + 0.114 * s[2];
            let yd = 0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2];
            // Шумодав убирает зерно, а не меняет картинку: правка больше этих пределов — сбой сети
            // (на насыщенном жёлтом у предела она рисует зелёную сетку), а не шум.
            let yo = ys_ + p.luma * (yd - ys_).clamp(-MAX_LUMA_CHANGE, MAX_LUMA_CHANGE);
            let dcb = ((d[2] - yd) - (s[2] - ys_)).clamp(-MAX_CHROMA_CHANGE, MAX_CHROMA_CHANGE);
            let dcr = ((d[0] - yd) - (s[0] - ys_)).clamp(-MAX_CHROMA_CHANGE, MAX_CHROMA_CHANGE);
            let cb = (s[2] - ys_) + p.chroma * dcb;
            let cr = (s[0] - ys_) + p.chroma * dcr;
            let r = yo + cr;
            let b = yo + cb;
            let g = (yo - 0.299 * r - 0.114 * b) / 0.587;
            px[y * w + x] = pack10([r, g, b]);
        }
    }
    Ok(())
}

/// Блоки проверки ответа сети.
const BLOCK: usize = 8;
/// Блок, изменённый сетью сильнее этого числа «типичных» блоков куска (и не меньше порога), —
/// сбой (сетка на насыщенном жёлтом), а не шум: настоящий шум по куску однороден.
const BLOCK_OUTLIER: f32 = 5.0;
const BLOCK_FLOOR: f32 = 0.035;

/// Какие блоки 8×8 куска сеть, похоже, испортила: средняя правка блока много больше типичной.
fn suspicious_blocks(input: &[f32], out: &[f32], tw: usize, th: usize) -> Vec<bool> {
    let nbx = TILE.div_ceil(BLOCK);
    let nby = TILE.div_ceil(BLOCK);
    let mut rms = vec![f32::NAN; nbx * nby];
    for by in 0..th.div_ceil(BLOCK) {
        for bx in 0..tw.div_ceil(BLOCK) {
            let (mut acc, mut n) = (0f32, 0f32);
            for y in by * BLOCK..((by + 1) * BLOCK).min(th) {
                for x in bx * BLOCK..((bx + 1) * BLOCK).min(tw) {
                    for ch in 0..3 {
                        let k = ch * TILE * TILE + y * TILE + x;
                        let d = out[k] - input[k];
                        acc += if d.is_finite() { d * d } else { 1.0 };
                        n += 1.0;
                    }
                }
            }
            rms[by * nbx + bx] = (acc / n.max(1.0)).sqrt();
        }
    }
    let mut v: Vec<f32> = rms.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return vec![false; nbx * nby];
    }
    v.sort_by(f32::total_cmp);
    let limit = (BLOCK_OUTLIER * v[v.len() / 2]).max(BLOCK_FLOOR);
    rms.iter().map(|r| r.is_finite() && *r > limit).collect()
}

fn reflect(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = i % period;
    if m < n { m } else { period - m }
}

/// Прогрев: загрузка модели в видеокарту заранее (≈2 с), чтобы первый кадр не ждал.
pub fn warm_up() {
    std::thread::spawn(|| {
        let _ = engine();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn print_face_model_inputs() {
        let env = ort::init().build().unwrap();
        for (name, bytes) in [
            ("yunet", &include_bytes!("../models/face_detection_yunet_2023mar.onnx")[..]),
            ("mesh", &include_bytes!("../models/face_landmarks.onnx")[..]),
        ] {
            let s = Session::builder(&env).unwrap().commit_from_memory(bytes).unwrap();
            for i in s.inputs() {
                println!("{name} вход {} {:?}", i.name(), i.dtype());
            }
            for o in s.outputs() {
                println!("{name} выход {} {:?}", o.name(), o.dtype());
            }
        }
    }

    #[test]
    fn tiles_cover_everything() {
        for n in [100usize, 512, 513, 1000, 4000, 6000] {
            let s = starts(n);
            assert_eq!(s[0], 0);
            assert!(s.last().unwrap() + TILE.min(n) >= n);
            for w in s.windows(2) {
                assert!(w[1] - w[0] <= TILE - OVERLAP);
            }
        }
    }

    #[test]
    fn auto_strength_by_iso() {
        assert!(auto_strength(Some(200), 0.0).is_none());
        assert!(auto_strength(Some(640), 0.8).is_none());
        let mid = auto_strength(Some(3200), 0.0).unwrap();
        let high = auto_strength(Some(12800), 0.0).unwrap();
        assert!(high.luma > mid.luma && high.chroma >= mid.chroma);
        // Осветление на 2 EV — как ISO ×4.
        assert!(auto_strength(Some(800), 2.0).unwrap().luma > mid.luma - 1e-3);
    }

    #[test]
    fn removes_synthetic_noise() {
        let (w, h) = (700usize, 600usize);
        let mut clean = vec![[0f32; 3]; w * h];
        let mut px = vec![0u32; w * h];
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        for y in 0..h {
            for x in 0..w {
                let c = [0.3 + 0.4 * x as f32 / w as f32, 0.5 + 0.2 * (y as f32 * 0.02).sin(), 0.45];
                clean[y * w + x] = c;
                px[y * w + x] = pack10([c[0] + 0.12 * rnd(), c[1] + 0.12 * rnd(), c[2] + 0.12 * rnd()]);
            }
        }
        let err = |px: &[u32]| -> f32 {
            px.iter().zip(&clean).map(|(p, c)| {
                let u = unpack10(*p);
                (0..3).map(|i| (u[i] - c[i]).powi(2)).sum::<f32>()
            }).sum::<f32>() / (w * h * 3) as f32
        };
        let before = err(&px);
        if denoise_packed(&mut px, w, h, DenoiseParams { luma: 1.0, chroma: 1.0 }, None).is_err() {
            eprintln!("DirectML недоступен — тест пропущен");
            return;
        }
        let after = err(&px);
        let gain_db = 10.0 * (before / after).log10();
        println!("шумодав: +{gain_db:.1} дБ");
        assert!(gain_db > 6.0, "{gain_db}");
    }
}
