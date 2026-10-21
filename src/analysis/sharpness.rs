//! Резкость (для автоотбора) и миниатюры для сравнения кадров.

use super::faces::Face;
use crate::raw::RawFrame;
use rayon::prelude::*;

/// Зелёный канал в половинном разрешении (среднее двух G ячейки 2×2), в «перцептивной» шкале sqrt.
pub fn green_half(frame: &RawFrame) -> (Vec<f32>, usize, usize) {
    let (w, h) = (frame.width / 2, frame.height / 2);
    let gpos: Vec<usize> = (0..4).filter(|&i| frame.cfa[i] == 1).collect();
    let scale: Vec<f32> = (0..4).map(|i| 1.0 / (frame.white - frame.black[i]).max(1.0)).collect();
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut s = 0.0;
            for &i in &gpos {
                let (dy, dx) = (i / 2, i % 2);
                let v = frame.data[(2 * y + dy) * frame.width + 2 * x + dx] as f32;
                s += ((v - frame.black[i]) * scale[i]).clamp(0.0, 1.0);
            }
            row[x] = (s / gpos.len().max(1) as f32).sqrt();
        }
    });
    (out, w, h)
}

pub fn orient_plane(p: &[f32], w: usize, h: usize, o: u16) -> (Vec<f32>, usize, usize) {
    let swap = matches!(o, 5..=8);
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    let mut out = vec![0f32; ow * oh];
    out.par_chunks_mut(ow).enumerate().for_each(|(yd, row)| {
        for xd in 0..ow {
            let (xs, ys) = match o {
                2 => (w - 1 - xd, yd),
                3 => (w - 1 - xd, h - 1 - yd),
                4 => (xd, h - 1 - yd),
                5 => (yd, xd),
                6 => (yd, h - 1 - xd),
                7 => (w - 1 - yd, h - 1 - xd),
                8 => (w - 1 - yd, xd),
                _ => (xd, yd),
            };
            row[xd] = p[ys * w + xs];
        }
    });
    (out, ow, oh)
}

/// Энергия лапласиана в прямоугольнике (среднее L²). Сравнивается только внутри серии,
/// где свет и шум одинаковы, поэтому нормировка не нужна.
fn region(p: &[f32], w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
    let (x0, y0) = (x0.max(1), y0.max(1));
    let (x1, y1) = (x1.min(w - 1), y1.min(h - 1));
    if x1 <= x0 + 2 || y1 <= y0 + 2 {
        return 0.0;
    }
    let sum: f64 = (y0..y1)
        .into_par_iter()
        .map(|y| {
            let mut s = 0f64;
            for x in x0..x1 {
                let c = p[y * w + x];
                let l = 4.0 * c - p[y * w + x - 1] - p[y * w + x + 1] - p[(y - 1) * w + x] - p[(y + 1) * w + x];
                s += (l * l) as f64;
            }
            s
        })
        .sum();
    (sum / ((x1 - x0) * (y1 - y0)) as f64 * 1e4) as f32
}

/// Резкость лица: центральная часть рамки (глаза, нос, рот).
pub fn face(p: &[f32], w: usize, h: usize, f: &Face, scale: f32) -> f32 {
    let (cx, cy) = f.center();
    let (hw, hh) = (f.w * 0.35 * scale, f.h * 0.35 * scale);
    let (cx, cy) = (cx * scale, cy * scale);
    region(p, w, h, (cx - hw).max(0.0) as usize, (cy - hh).max(0.0) as usize, (cx + hw) as usize, (cy + hh) as usize)
}

/// Резкость кадра: среднее по четверти самых детальных участков сетки 6×4
/// (при малой ГРИП резкой бывает только часть кадра).
pub fn frame(p: &[f32], w: usize, h: usize) -> f32 {
    let (nx, ny) = (6, 4);
    let mut tiles: Vec<f32> = Vec::new();
    for ty in 0..ny {
        for tx in 0..nx {
            tiles.push(region(p, w, h, tx * w / nx, ty * h / ny, (tx + 1) * w / nx, (ty + 1) * h / ny));
        }
    }
    tiles.sort_by(|a, b| b.total_cmp(a));
    let k = (tiles.len() / 4).max(1);
    tiles[..k].iter().sum::<f32>() / k as f32
}

/// Миниатюра 32×N яркости (sqrt), нормированная к нулевому среднему и единичному разбросу.
pub fn thumb(lin: &[[f32; 3]], w: usize, h: usize) -> Vec<f32> {
    let tw = 32usize;
    let th = ((32 * h) as f32 / w as f32).round().max(1.0) as usize;
    let mut v = vec![0f32; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let (x0, x1) = (tx * w / tw, (tx + 1) * w / tw);
            let (y0, y1) = (ty * h / th, (ty + 1) * h / th);
            let mut s = 0f32;
            let mut n = 0f32;
            for y in (y0..y1).step_by(2) {
                for x in (x0..x1).step_by(2) {
                    s += super::exposure::luma(lin[y * w + x]).max(0.0).sqrt();
                    n += 1.0;
                }
            }
            v[ty * tw + tx] = s / n.max(1.0);
        }
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt().max(1e-4);
    v.iter().map(|x| (x - mean) / std).collect()
}

// ---------- чёткость: абсолютная мера смаза ----------
//
// На резком краю перепад яркости укладывается в пиксель, и градиент на каждом уровне пирамиды
// (уменьшение вдвое) вдвое меньше, чем на предыдущем — отношение ≈ 2. Размытый край шире
// пикселя на обоих уровнях, и отношение падает к 1. Отношение не зависит от контраста и
// яркости сцены. Считаем отдельно по x и по y (смаз от движения — в одну сторону) и берём
// худшее. Мелкий уровень пирамиды не используем: на высоком ISO шум там выглядит как резкость.

/// Уменьшение плоскости вдвое (среднее 2×2).
fn half(p: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (hw, hh) = (w / 2, h / 2);
    let mut out = vec![0f32; hw * hh];
    out.par_chunks_mut(hw.max(1)).enumerate().for_each(|(y, row)| {
        for x in 0..hw {
            let i = 2 * y * w + 2 * x;
            row[x] = 0.25 * (p[i] + p[i + 1] + p[i + w] + p[i + w + 1]);
        }
    });
    (out, hw, hh)
}

struct Level {
    p: Vec<f32>,
    w: usize,
    h: usize,
    /// Во сколько раз уровень мельче плоскости G (полуразрешение RAW).
    scale: f32,
}

/// Пирамида плоскости G: уровни ½, ¼ и ⅛ от полуразрешения RAW.
pub struct Pyramid {
    levels: Vec<Level>,
}

impl Pyramid {
    pub fn new(p: &[f32], w: usize, h: usize) -> Self {
        let mut levels = Vec::new();
        let (mut cur, mut cw, mut ch) = half(p, w, h);
        let mut scale = 2.0;
        for _ in 0..3 {
            let next = half(&cur, cw, ch);
            levels.push(Level { p: cur, w: cw, h: ch, scale });
            (cur, cw, ch) = next;
            scale *= 2.0;
        }
        Pyramid { levels }
    }
}

/// Модуль градиента (центральная разность) в единицах «на пиксель плоскости G».
fn grad(l: &Level, x: usize, y: usize, dir: usize) -> f32 {
    let d = if dir == 0 { l.p[y * l.w + x + 1] - l.p[y * l.w + x - 1] } else { l.p[(y + 1) * l.w + x] - l.p[(y - 1) * l.w + x] };
    d.abs() * 0.5 / l.scale
}

/// Перепад, который считается краем (в шкале sqrt-яркости на пиксель плоскости G).
const MIN_EDGE: f32 = 0.0008;

/// Медиана отношения градиентов уровней `fine` / `coarse` на самых сильных краях прямоугольника
/// (в координатах плоскости G). None — краёв мало.
fn edge_ratio(fine: &Level, coarse: &Level, rect: (usize, usize, usize, usize), dir: usize) -> Option<f32> {
    let k = coarse.scale as usize;
    let (x0, y0) = ((rect.0 / k).max(1), (rect.1 / k).max(1));
    let (x1, y1) = ((rect.2 / k).min(coarse.w.saturating_sub(1)), (rect.3 / k).min(coarse.h.saturating_sub(1)));
    if x1 <= x0 + 4 || y1 <= y0 + 4 {
        return None;
    }
    let mut g: Vec<(f32, usize, usize)> = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for y in y0..y1 {
        for x in x0..x1 {
            g.push((grad(coarse, x, y, dir), x, y));
        }
    }
    let top = (g.len() / 20).max(12);
    if g.len() < top * 2 {
        return None;
    }
    g.select_nth_unstable_by(top, |a, b| b.0.total_cmp(&a.0));
    let edges: Vec<&(f32, usize, usize)> = g[..top].iter().filter(|e| e.0 > MIN_EDGE).collect();
    if edges.len() < 12 {
        return None;
    }
    let mut r: Vec<f32> = edges
        .iter()
        .map(|&&(gc, x, y)| {
            // Пик градиента на мелком уровне рядом с той же точкой.
            let (fx, fy) = (2 * x, 2 * y);
            let mut best = 0f32;
            for dy in -1i32..=2 {
                for dx in -1i32..=2 {
                    let (xx, yy) = (fx as i32 + dx, fy as i32 + dy);
                    if xx >= 1 && yy >= 1 && (xx as usize) < fine.w - 1 && (yy as usize) < fine.h - 1 {
                        best = best.max(grad(fine, xx as usize, yy as usize, dir));
                    }
                }
            }
            best / gc
        })
        .collect();
    let m = r.len() / 2;
    r.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    Some(r[m])
}

/// Чёткость участка: ≈2 — резкий край, ≈1 — размыто сильнее ~8 пикселей полного кадра.
pub fn clarity(pyr: &Pyramid, rect: (usize, usize, usize, usize)) -> Option<f32> {
    let (l1, l2) = (&pyr.levels[0], &pyr.levels[1]);
    let r: Vec<f32> = (0..2).filter_map(|d| edge_ratio(l1, l2, rect, d)).collect();
    r.into_iter().reduce(f32::min)
}

/// Чёткость кадра: самый чёткий участок сетки 4×3 (при малой ГРИП резкой бывает только часть)
/// и центр этого участка в долях кадра.
pub fn frame_clarity(pyr: &Pyramid, w: usize, h: usize) -> (f32, [f32; 2]) {
    let (nx, ny) = (4, 3);
    let mut best = (0f32, [0.5f32, 0.5]);
    for ty in 0..ny {
        for tx in 0..nx {
            if let Some(c) = clarity(pyr, (tx * w / nx, ty * h / ny, (tx + 1) * w / nx, (ty + 1) * h / ny)) {
                if c > best.0 {
                    best = (c, [(tx as f32 + 0.5) / nx as f32, (ty as f32 + 0.5) / ny as f32]);
                }
            }
        }
    }
    best
}

/// Чёткость лица (рамка с небольшим запасом), `scale` — плоскость G / анализируемый кадр.
pub fn face_clarity(pyr: &Pyramid, w: usize, h: usize, f: &Face, scale: f32) -> Option<f32> {
    let (cx, cy) = f.center();
    let (hw, hh) = (f.w * 0.6 * scale, f.h * 0.6 * scale);
    let (cx, cy) = (cx * scale, cy * scale);
    let rect = ((cx - hw).max(0.0) as usize, (cy - hh).max(0.0) as usize, ((cx + hw) as usize).min(w), ((cy + hh) as usize).min(h));
    clarity(pyr, rect)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Шум для тестов: равномерный в [-1, 1].
    fn noise(i: usize) -> f32 {
        let mut x = (i as u32).wrapping_mul(2_654_435_761);
        x ^= x >> 15;
        x = x.wrapping_mul(2_246_822_519);
        x ^= x >> 13;
        (x % 20_001) as f32 / 10_000.0 - 1.0
    }

    /// Сцена из прямоугольников разной яркости (резкие края во все стороны).
    fn scene(w: usize, h: usize) -> Vec<f32> {
        let mut p = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let cell = (x / 37) * 7 + (y / 29) * 13;
                p[y * w + x] = 0.25 + 0.5 * ((cell * 2_654_435_761usize) % 97) as f32 / 97.0;
            }
        }
        p
    }

    /// Размытие коробкой радиуса r по x и/или по y.
    fn blur(p: &[f32], w: usize, h: usize, rx: usize, ry: usize) -> Vec<f32> {
        let mut out = p.to_vec();
        if rx > 0 {
            let src = out.clone();
            for y in 0..h {
                for x in 0..w {
                    let (a, b) = (x.saturating_sub(rx), (x + rx).min(w - 1));
                    out[y * w + x] = (a..=b).map(|xx| src[y * w + xx]).sum::<f32>() / (b - a + 1) as f32;
                }
            }
        }
        if ry > 0 {
            let src = out.clone();
            for y in 0..h {
                for x in 0..w {
                    let (a, b) = (y.saturating_sub(ry), (y + ry).min(h - 1));
                    out[y * w + x] = (a..=b).map(|yy| src[yy * w + x]).sum::<f32>() / (b - a + 1) as f32;
                }
            }
        }
        out
    }

    fn whole(p: &[f32], w: usize, h: usize) -> f32 {
        frame_clarity(&Pyramid::new(p, w, h), w, h).0
    }

    #[test]
    fn sharp_beats_blurred_and_noise_is_not_sharpness() {
        let (w, h) = (640, 480);
        let sharp = scene(w, h);
        let soft = blur(&sharp, w, h, 6, 6);
        let moved = blur(&sharp, w, h, 9, 0);
        // Высокий ISO: размытый кадр с сильным шумом.
        let noisy: Vec<f32> = soft.iter().enumerate().map(|(i, v)| v + 0.04 * noise(i)).collect();
        let (cs, cb, cm, cn) = (whole(&sharp, w, h), whole(&soft, w, h), whole(&moved, w, h), whole(&noisy, w, h));
        assert!(cs > 1.6, "резкий {cs}");
        assert!(cb < 1.3, "размытый {cb}");
        assert!(cm < 1.3, "смаз по горизонтали {cm}");
        assert!(cn < 1.4, "шум не резкость {cn}");
    }
}
