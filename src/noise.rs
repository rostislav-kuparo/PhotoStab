//! Профиль шума сенсора по самим RAW: модель «Пуассон + Гаусс» — дисперсия = k·сигнал + s,
//! сигнал в долях белого после вычитания чёрного. Нужна шумодаву в RAW: он работает с шумом
//! «эталонного» сенсора, и профиль переводит к нему шум нашей камеры при любом ISO.
//!
//! Оценка: в блоках 8×8 каждой цветовой плоскости — средний уровень и дисперсия разностей
//! соседних пикселей того же цвета (плавная текстура почти не мешает). Для каждого уровня яркости
//! берётся нижний квантиль дисперсии — это участки без текстуры, — и через них проводится прямая.
//! Калибровочные снимки не нужны: на любом репортажном кадре хватает ровных стен и фона.

use crate::raw::RawFrame;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NoiseProfile {
    /// Дисперсия шума = k·x + s, где x — сигнал в долях белого (0..1).
    pub k: f32,
    pub s: f32,
}

const BLOCK: usize = 8;
/// Блоки берём через один по каждой оси — для оценки хватает, а быстрее вчетверо.
const BLOCK_STEP: usize = 2;
/// Нижний квантиль дисперсии на уровне и поправка его смещения (для ~56 степеней свободы).
const LOW_Q: f32 = 0.2;
const LOW_Q_BIAS: f32 = 0.84;
const BINS: usize = 24;
const X_LO: f32 = 0.002;
const X_HI: f32 = 0.7;

/// Пары (уровень, дисперсия) по блокам кадра — для оценки по одному кадру или по многим.
pub fn samples(frame: &RawFrame) -> Vec<(f32, f32)> {
    let (w, h) = (frame.width, frame.height);
    let (pw, ph) = (w / 2, h / 2);
    let mut out = Vec::new();
    for pos in 0..4 {
        let (oy, ox) = (pos / 2, pos % 2);
        let black = frame.black[pos];
        let range = (frame.white - black).max(1.0);
        let at = |px: usize, py: usize| -> f32 { (frame.data[(2 * py + oy) * w + 2 * px + ox] as f32 - black) / range };
        let mut by = 0;
        while (by + 1) * BLOCK <= ph {
            let mut bx = 0;
            while (bx + 1) * BLOCK + 1 <= pw {
                let (mut sum, mut n, mut d2, mut nd, mut top) = (0f64, 0f64, 0f64, 0f64, 0f32);
                for y in by * BLOCK..(by + 1) * BLOCK {
                    for x in bx * BLOCK..(bx + 1) * BLOCK {
                        let v = at(x, y);
                        let r = at(x + 1, y);
                        top = top.max(v).max(r);
                        sum += v as f64;
                        n += 1.0;
                        d2 += ((v - r) as f64).powi(2);
                        nd += 1.0;
                    }
                }
                // Возле пересвета шум срезан — не годится.
                if top < 0.9 {
                    out.push(((sum / n) as f32, (d2 / nd / 2.0) as f32));
                }
                bx += BLOCK_STEP;
            }
            by += BLOCK_STEP;
        }
    }
    out
}

/// Прямая «дисперсия от уровня» по нижнему квантилю каждого уровня. None — мало данных.
pub fn fit(samples: &[(f32, f32)]) -> Option<NoiseProfile> {
    let lx = |x: f32| ((x.max(X_LO).ln() - X_LO.ln()) / (X_HI.ln() - X_LO.ln()) * BINS as f32) as usize;
    let mut bins: Vec<Vec<f32>> = vec![Vec::new(); BINS];
    let mut xs: Vec<Vec<f32>> = vec![Vec::new(); BINS];
    for &(x, v) in samples {
        if !(X_LO..X_HI).contains(&x) || !v.is_finite() || v <= 0.0 {
            continue;
        }
        let b = lx(x).min(BINS - 1);
        bins[b].push(v);
        xs[b].push(x);
    }
    // (уровень, дисперсия без текстуры, вес)
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    for (b, v) in bins.iter_mut().enumerate() {
        if v.len() < 40 {
            continue;
        }
        v.sort_by(f32::total_cmp);
        let q = v[((v.len() - 1) as f32 * LOW_Q) as usize] / LOW_Q_BIAS;
        let mut x = xs[b].clone();
        x.sort_by(f32::total_cmp);
        let xm = x[x.len() / 2];
        // Вес: число блоков; относительная точность одинакова, поэтому делим на квадрат уровня.
        pts.push((xm as f64, q as f64, (v.len() as f64).sqrt() / (q as f64).powi(2)));
    }
    if pts.len() < 4 {
        return None;
    }
    let solve = |pts: &[(f64, f64, f64)]| -> (f64, f64) {
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(x, y, w) in pts {
            sw += w;
            sx += w * x;
            sy += w * y;
            sxx += w * x * x;
            sxy += w * x * y;
        }
        let det = sw * sxx - sx * sx;
        if det.abs() < 1e-30 {
            return (0.0, sy / sw.max(1e-30));
        }
        ((sw * sxy - sx * sy) / det, (sxx * sy - sx * sxy) / det)
    };
    let (mut k, mut s) = solve(&pts);
    // Один проход отсева: уровни, далеко ушедшие от прямой (текстура, полосы), не считаем.
    let kept: Vec<(f64, f64, f64)> = pts.iter().copied().filter(|&(x, y, _)| {
        let p = (k * x + s).max(1e-12);
        (y / p).ln().abs() < 0.5
    }).collect();
    if kept.len() >= 4 {
        (k, s) = solve(&kept);
    }
    if !(k > 0.0) {
        return None;
    }
    Some(NoiseProfile { k: k as f32, s: s.max(0.0) as f32 })
}

pub fn estimate(frame: &RawFrame) -> Option<NoiseProfile> {
    fit(&samples(frame))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::{CameraColor, PhotoMeta};

    /// Простой детерминированный шум: приблизительно нормальный (сумма равномерных).
    fn gauss(i: usize) -> f32 {
        let mut s = 0.0;
        for k in 0..4 {
            let mut x = ((i * 4 + k) as u32).wrapping_mul(2_654_435_761);
            x ^= x >> 15;
            x = x.wrapping_mul(2_246_822_519);
            x ^= x >> 13;
            s += (x % 10_000) as f32 / 10_000.0 - 0.5;
        }
        s * (12.0f32 / 4.0).sqrt()
    }

    #[test]
    fn recovers_poisson_gaussian_profile() {
        // Плавный градиент по кадру с шумом var = k·x + s.
        let (w, h) = (1200usize, 800usize);
        let (k, s) = (2.0e-4f32, 3.0e-7f32);
        let (black, white) = (512.0f32, 16383.0f32);
        let mut data = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let level = 0.003 + 0.6 * (x as f32 / w as f32) * (0.3 + 0.7 * y as f32 / h as f32);
                let sd = (k * level + s).sqrt();
                let v = level + sd * gauss(y * w + x);
                data[y * w + x] = (black + v * (white - black)).round().clamp(0.0, 65535.0) as u16;
            }
        }
        let frame = RawFrame {
            width: w,
            height: h,
            data,
            cfa: [0, 1, 1, 2],
            black: [black; 4],
            white,
            asshot_wb: None,
            color: CameraColor { cm_a: None, cm_d65: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] },
            meta: PhotoMeta::default(),
        };
        let p = estimate(&frame).expect("профиль");
        assert!((p.k / k - 1.0).abs() < 0.2, "k {} vs {k}", p.k);
        assert!(p.s > 0.0 && (p.s / s).ln().abs() < 1.0, "s {} vs {s}", p.s);
    }
}
