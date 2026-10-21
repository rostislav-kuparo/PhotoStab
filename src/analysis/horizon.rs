//! Автогоризонт: поиск длинных прямых около горизонтали и вертикали и оценка завала камеры.
//!
//! Вертикали в помещении (двери, колонны, углы стен) — самый надёжный признак. При наклоне
//! камеры вверх/вниз вертикали сходятся: их отклонение растёт с удалением от центра кадра.
//! Поэтому отклонения аппроксимируются как `крен + k * смещение`, а крен — свободный член.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HorizonResult {
    /// Поворот для исправления, градусы, + против часовой.
    pub angle: f32,
    pub confidence: f32,
    pub apply: bool,
    pub vertical_lines: usize,
    pub horizontal_lines: usize,
}

impl Default for HorizonResult {
    fn default() -> Self {
        Self { angle: 0.0, confidence: 0.0, apply: false, vertical_lines: 0, horizontal_lines: 0 }
    }
}

#[derive(Clone, Copy, Debug)]
struct Line {
    /// Отклонение от горизонтали/вертикали (град), со знаком: + значит содержимое повёрнуто по часовой.
    tilt: f32,
    length: f32,
    /// Смещение от центра кадра (доли размера), со знаком.
    offset: f32,
}

const RANGE: f32 = 12.0;
const STEP: f32 = 0.1;
const MAX_SIDE: usize = 1200;
pub const MIN_APPLY: f32 = 0.25;
pub const MAX_APPLY: f32 = 8.0;

struct Edges {
    w: usize,
    h: usize,
    /// Угол направления линии (град, (-90, 90]) или NaN, если не край.
    angle: Vec<f32>,
    weight: Vec<f32>,
}

fn downsample(lum: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let f = w.max(h).div_ceil(MAX_SIDE).max(1);
    if f == 1 {
        return (lum.iter().map(|v| v.max(0.0).sqrt()).collect(), w, h);
    }
    let (ow, oh) = (w / f, h / f);
    let mut out = vec![0f32; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let mut s = 0.0;
            for by in 0..f {
                for bx in 0..f {
                    s += lum[(y * f + by) * w + x * f + bx];
                }
            }
            out[y * ow + x] = (s / (f * f) as f32).max(0.0).sqrt();
        }
    }
    (out, ow, oh)
}

fn edges(g: &[f32], w: usize, h: usize) -> Edges {
    // Сглаживание 3×3.
    let mut s = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p = |dx: isize, dy: isize| g[((y as isize + dy) as usize) * w + (x as isize + dx) as usize];
            s[y * w + x] = (4.0 * p(0, 0) + 2.0 * (p(-1, 0) + p(1, 0) + p(0, -1) + p(0, 1))
                + p(-1, -1)
                + p(1, -1)
                + p(-1, 1)
                + p(1, 1))
                / 16.0;
        }
    }
    let mut mag = vec![0f32; w * h];
    let mut ang = vec![f32::NAN; w * h];
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let p = |dx: isize, dy: isize| s[((y as isize + dy) as usize) * w + (x as isize + dx) as usize];
            let gx = (p(1, -1) + 2.0 * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2.0 * p(-1, 0) + p(-1, 1));
            let gy = (p(-1, 1) + 2.0 * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2.0 * p(0, -1) + p(1, -1));
            let m = (gx * gx + gy * gy).sqrt();
            mag[y * w + x] = m;
            // Направление линии перпендикулярно градиенту: d = (-gy, gx).
            let mut a = gx.atan2(-gy).to_degrees();
            if a > 90.0 {
                a -= 180.0;
            }
            if a <= -90.0 {
                a += 180.0;
            }
            ang[y * w + x] = a;
        }
    }
    let mut sample: Vec<f32> = mag.iter().step_by(7).copied().collect();
    let k = sample.len() * 88 / 100;
    sample.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    let thr = sample[k].max(0.03);
    let mut weight = vec![0f32; w * h];
    for i in 0..w * h {
        if mag[i] >= thr {
            weight[i] = (mag[i] / (2.0 * thr)).min(1.0);
        } else {
            ang[i] = f32::NAN;
        }
    }
    Edges { w, h, angle: ang, weight }
}

/// Перевод угла линии в отклонение семейства (горизонтальное или вертикальное).
fn family_tilt(a: f32, vertical: bool) -> f32 {
    if vertical {
        if a > 0.0 { a - 90.0 } else { a + 90.0 }
    } else {
        a
    }
}

fn find_lines(e: &Edges, vertical: bool) -> Vec<Line> {
    let (w, h) = (e.w, e.h);
    let nbins = (2.0 * RANGE / STEP).round() as usize + 1;
    let diag = ((w * w + h * h) as f32).sqrt();
    let nrho = (2.0 * diag) as usize + 2;
    let mut acc = vec![0f32; nbins * nrho];
    let trig: Vec<(f32, f32)> = (0..nbins)
        .map(|i| {
            let t = (-RANGE + i as f32 * STEP).to_radians();
            (t.sin(), t.cos())
        })
        .collect();
    let rho_of = |x: f32, y: f32, (sn, cs): (f32, f32)| -> f32 {
        if vertical {
            // Направление (−sin δ, cos δ) → нормаль (cos δ, sin δ).
            x * cs + y * sn
        } else {
            -x * sn + y * cs
        }
    };
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a = e.angle[i];
            if a.is_nan() {
                continue;
            }
            let t = family_tilt(a, vertical);
            if t.abs() > RANGE + 2.0 {
                continue;
            }
            let center = ((t + RANGE) / STEP).round() as isize;
            for b in (center - 15).max(0)..=(center + 15).min(nbins as isize - 1) {
                let b = b as usize;
                let rho = rho_of(x as f32, y as f32, trig[b]) + diag;
                let r = rho.round() as usize;
                if r < nrho {
                    acc[b * nrho + r] += e.weight[i];
                }
            }
        }
    }

    let min_votes = w.min(h) as f32 * 0.08;
    let mut peaks: Vec<(f32, usize, usize)> = Vec::new();
    for b in 0..nbins {
        for r in 1..nrho - 1 {
            let v = acc[b * nrho + r];
            if v < min_votes {
                continue;
            }
            let mut is_max = true;
            'n: for db in -4isize..=4 {
                for dr in -3isize..=3 {
                    let (bb, rr) = (b as isize + db, r as isize + dr);
                    if bb < 0 || bb >= nbins as isize || rr < 0 || rr >= nrho as isize || (db == 0 && dr == 0) {
                        continue;
                    }
                    let o = acc[bb as usize * nrho + rr as usize];
                    if o > v || (o == v && (db, dr) < (0, 0)) {
                        is_max = false;
                        break 'n;
                    }
                }
            }
            if is_max {
                peaks.push((v, b, r));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut lines: Vec<Line> = Vec::new();
    // Жадный отбор: пиксели принятой линии помечаются занятыми, поэтому «призраки» Хафа —
    // соседние углы, пересекающие настоящую линию на коротком участке, — отсеиваются.
    let mut used = vec![false; w * h];
    for &(_, b, r) in peaks.iter().take(120) {
        let tilt = -RANGE + b as f32 * STEP;
        let rho = r as f32 - diag;
        let (len, s0, s1) = segment(e, &used, vertical, tilt, rho);
        if len < w.min(h) as f32 * 0.1 {
            continue;
        }
        let refined = refine(e, &used, vertical, tilt, rho, s0, s1);
        mark(e, &mut used, vertical, tilt, rho, s0, s1);
        let Some((tilt, (mx, my))) = refined else { continue };
        let t = tilt.to_radians();
        let offset = if vertical {
            // x на центральной строке для линии через (mx, my) с направлением (−sin t, cos t).
            (mx + (h as f32 * 0.5 - my) * (-t.tan()) - w as f32 * 0.5) / w as f32
        } else {
            (my + (w as f32 * 0.5 - mx) * t.tan() - h as f32 * 0.5) / h as f32
        };
        lines.push(Line { tilt, length: len, offset });
        if lines.len() >= 30 {
            break;
        }
    }
    lines
}

/// Точка на линии и её направление.
fn line_frame(vertical: bool, tilt: f32, rho: f32) -> ((f32, f32), (f32, f32), (f32, f32)) {
    let t = tilt.to_radians();
    let (sn, cs) = (t.sin(), t.cos());
    let (dir, p0) = if vertical {
        // x*cs + y*sn = rho, направление (−sn, cs).
        ((-sn, cs), (rho * cs, rho * sn))
    } else {
        ((cs, sn), (-rho * sn, rho * cs))
    };
    (dir, p0, (-dir.1, dir.0))
}

/// Самый длинный непрерывный участок линии на свободных краевых пикселях (разрывы до 4 px).
/// Возвращает (длина, начало, конец) в параметре вдоль линии.
fn segment(e: &Edges, used: &[bool], vertical: bool, tilt: f32, rho: f32) -> (f32, f32, f32) {
    let (w, h) = (e.w as f32, e.h as f32);
    let (dir, p0, normal) = line_frame(vertical, tilt, rho);
    let span = (w * w + h * h).sqrt();
    let (mut best, mut run, mut gap) = (0f32, 0f32, 0f32);
    let (mut start, mut best_range) = (0f32, (0f32, 0f32));
    let mut s = -span;
    while s <= span {
        let (x, y) = (p0.0 + dir.0 * s, p0.1 + dir.1 * s);
        if x >= 2.0 && y >= 2.0 && x < w - 2.0 && y < h - 2.0 {
            let mut hit = false;
            for k in -1..=1 {
                let xi = (x + normal.0 * k as f32).round() as usize;
                let yi = (y + normal.1 * k as f32).round() as usize;
                let i = yi * e.w + xi;
                let a = e.angle[i];
                if !used[i] && !a.is_nan() && (family_tilt(a, vertical) - tilt).abs() < 4.0 {
                    hit = true;
                    break;
                }
            }
            if hit {
                if run == 0.0 {
                    start = s;
                }
                run += 1.0 + gap;
                gap = 0.0;
                if run > best {
                    best = run;
                    best_range = (start, s);
                }
            } else {
                gap += 1.0;
                if gap > 4.0 {
                    run = 0.0;
                    gap = 0.0;
                }
            }
        } else {
            run = 0.0;
            gap = 0.0;
        }
        s += 1.0;
    }
    (best, best_range.0, best_range.1)
}

/// Уточнение угла: главная ось (PCA) краевых пикселей вдоль найденного участка.
/// Возвращает (отклонение, центр масс).
fn refine(e: &Edges, used: &[bool], vertical: bool, tilt: f32, rho: f32, s0: f32, s1: f32) -> Option<(f32, (f32, f32))> {
    let (w, h) = (e.w as f32, e.h as f32);
    let (dir, p0, normal) = line_frame(vertical, tilt, rho);
    let mut seen = std::collections::HashSet::new();
    let (mut sw, mut sx, mut sy) = (0f64, 0f64, 0f64);
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    let mut s = s0;
    while s <= s1 {
        for k in -2..=2 {
            let x = p0.0 + dir.0 * s + normal.0 * k as f32;
            let y = p0.1 + dir.1 * s + normal.1 * k as f32;
            if x < 0.0 || y < 0.0 || x >= w || y >= h {
                continue;
            }
            let i = y as usize * e.w + x as usize;
            let a = e.angle[i];
            if used[i] || a.is_nan() || (family_tilt(a, vertical) - tilt).abs() > 3.0 || !seen.insert(i) {
                continue;
            }
            let (px, py, wt) = ((i % e.w) as f64 + 0.5, (i / e.w) as f64 + 0.5, e.weight[i] as f64);
            pts.push((px, py, wt));
            sw += wt;
            sx += wt * px;
            sy += wt * py;
        }
        s += 1.0;
    }
    if pts.len() < 20 || sw <= 0.0 {
        return None;
    }
    let (mx, my) = (sx / sw, sy / sw);
    let (mut cxx, mut cyy, mut cxy) = (0f64, 0f64, 0f64);
    for &(x, y, wt) in &pts {
        cxx += wt * (x - mx) * (x - mx);
        cyy += wt * (y - my) * (y - my);
        cxy += wt * (x - mx) * (y - my);
    }
    // Угол главной оси ковариационной матрицы.
    let theta = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let mut a = theta.to_degrees() as f32;
    if a > 90.0 {
        a -= 180.0;
    }
    if a <= -90.0 {
        a += 180.0;
    }
    let t = family_tilt(a, vertical);
    if (t - tilt).abs() > 1.5 {
        return None;
    }
    Some((t, (mx as f32, my as f32)))
}

fn mark(e: &Edges, used: &mut [bool], vertical: bool, tilt: f32, rho: f32, s0: f32, s1: f32) {
    let (w, h) = (e.w as f32, e.h as f32);
    let (dir, p0, normal) = line_frame(vertical, tilt, rho);
    let mut s = s0;
    while s <= s1 {
        for k in -3..=3 {
            let x = p0.0 + dir.0 * s + normal.0 * k as f32;
            let y = p0.1 + dir.1 * s + normal.1 * k as f32;
            if x >= 0.0 && y >= 0.0 && x < w && y < h {
                used[y as usize * e.w + x as usize] = true;
            }
        }
        s += 0.5;
    }
}

struct Fit {
    roll: f32,
    rms: f32,
    /// Длина линий, согласных с найденным креном.
    support: f32,
    /// Вес согласных линий для найденного крена и для гипотезы «камера стояла ровно».
    score: f32,
    level_score: f32,
}

fn center_weight(offset: f32) -> f32 {
    (-(offset / 0.45).powi(2)).exp()
}

/// Робастная подгонка tilt = roll + k * offset (перебор моделей + МНК по согласным линиям).
fn fit(lines: &[Line], prior_w: f32) -> Option<Fit> {
    if lines.is_empty() {
        return None;
    }
    let base: Vec<f32> = lines.iter().map(|l| l.length * center_weight(l.offset) * prior_w).collect();
    // Сходимость вертикалей (перспектива) оцениваем, только если линии есть по обе стороны от центра,
    // иначе наклон модели путается с креном.
    let total: f32 = base.iter().sum();
    let left: f32 = lines.iter().zip(&base).filter(|(l, _)| l.offset < -0.1).map(|(_, b)| b).sum();
    let right: f32 = lines.iter().zip(&base).filter(|(l, _)| l.offset > 0.1).map(|(_, b)| b).sum();
    let use_slope = lines.len() >= 4 && left > 0.2 * total && right > 0.2 * total;

    // Перебор моделей (в духе RANSAC): каждая линия сама по себе (k = 0) и, если есть разброс
    // по смещению, каждая пара линий. Побеждает модель с наибольшим весом согласных линий.
    const INLIER: f32 = 0.35;
    let score = |roll: f32, k: f32| -> f32 {
        lines
            .iter()
            .zip(&base)
            .map(|(l, b)| {
                let r = (l.tilt - roll - k * l.offset) / INLIER;
                if r.abs() < 1.0 { b * (1.0 - r * r) } else { 0.0 }
            })
            .sum()
    };
    let mut best = (lines[0].tilt, 0.0f32, -1.0f32);
    for l in lines {
        let s = score(l.tilt, 0.0);
        if s > best.2 {
            best = (l.tilt, 0.0, s);
        }
    }
    if use_slope {
        let flat_best = best.2;
        for i in 0..lines.len() {
            for j in i + 1..lines.len() {
                let d = lines[i].offset - lines[j].offset;
                if d.abs() < 0.1 {
                    continue;
                }
                let k = (lines[i].tilt - lines[j].tilt) / d;
                if k.abs() > 30.0 {
                    continue;
                }
                let roll = lines[i].tilt - k * lines[i].offset;
                let s = score(roll, k);
                // Модель со сходимостью должна объяснять заметно больше, чем без неё.
                if s > best.2 && s > flat_best * 1.25 {
                    best = (roll, k, s);
                }
            }
        }
    }
    let (mut roll, mut k, _) = best;
    // Уточнение взвешенными МНК по согласным линиям (веса Тьюки в окне INLIER).
    let mut inl: Vec<(f32, f32, f32)> = Vec::new();
    for _ in 0..3 {
        inl = lines
            .iter()
            .zip(&base)
            .filter_map(|(l, b)| {
                let r = (l.tilt - roll - k * l.offset) / INLIER;
                (r.abs() < 1.0).then(|| (l.offset, l.tilt, b * (1.0 - r * r).powi(2)))
            })
            .collect();
        let sw: f32 = inl.iter().map(|p| p.2).sum();
        if sw <= 0.0 {
            return None;
        }
        if k != 0.0 {
            let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
            for &(x, y, w) in &inl {
                sx += w * x;
                sy += w * y;
                sxx += w * x * x;
                sxy += w * x * y;
            }
            let den = sw * sxx - sx * sx;
            if den.abs() > 1e-6 {
                k = (sw * sxy - sx * sy) / den;
                roll = (sy - k * sx) / sw;
            }
        } else {
            roll = inl.iter().map(|p| p.1 * p.2).sum::<f32>() / sw;
        }
    }
    let sw: f32 = inl.iter().map(|p| p.2).sum();
    let rms = (inl.iter().map(|&(x, y, w)| w * (y - roll - k * x).powi(2)).sum::<f32>() / sw).sqrt();
    let support: f32 = lines
        .iter()
        .filter(|l| (l.tilt - roll - k * l.offset).abs() < INLIER * 1.5)
        .map(|l| l.length * center_weight(l.offset))
        .sum();
    let level_score = score(0.0, 0.0).max(score(0.0, k));
    Some(Fit { roll, rms, support, score: score(roll, k), level_score })
}

/// `lum` — линейная яркость кадра в ориентации показа.
pub fn detect(lum: &[f32], w: usize, h: usize) -> HorizonResult {
    if w < 32 || h < 32 {
        return HorizonResult::default();
    }
    let (g, gw, gh) = downsample(lum, w, h);
    let e = edges(&g, gw, gh);
    let v = find_lines(&e, true);
    let hz = find_lines(&e, false);
    if std::env::var("HZ_DEBUG").is_ok() {
        for (tag, ls) in [("V", &v), ("H", &hz)] {
            for l in ls.iter() {
                eprintln!("{tag} tilt {:+.2} len {:.0} off {:+.2}", l.tilt, l.length, l.offset);
            }
        }
    }
    let fv = fit(&v, 1.3);
    let fh = fit(&hz, 1.0);
    let norm = (gw + gh) as f32 * 0.5;
    let score = |f: &Option<Fit>| -> Option<(f32, f32)> {
        f.as_ref().map(|f| {
            let agree = (-(f.rms / 0.6).powi(2)).exp();
            let support = (f.support / norm).min(1.0);
            (f.roll, agree * support)
        })
    };
    let (sv, sh) = (score(&fv), score(&fh));
    // Гипотеза «камера стояла ровно» — по умолчанию. Если она объясняет линии почти так же
    // хорошо, как найденный наклон, ничего не поворачиваем.
    let best: f32 = fv.iter().chain(fh.iter()).map(|f| f.score).sum();
    let level: f32 = fv.iter().chain(fh.iter()).map(|f| f.level_score).sum();
    let level_wins = best > 0.0 && level >= 0.6 * best;
    let (roll, mut conf) = match (sv, sh) {
        (Some((rv, cv)), Some((rh, ch))) => {
            let wv = cv * 1.3;
            let roll = (rv * wv + rh * ch) / (wv + ch).max(1e-6);
            let mut c = (cv.max(ch) + 0.3 * cv.min(ch)).min(1.0);
            if (rv - rh).abs() > 1.0 && cv > 0.2 && ch > 0.2 {
                c *= 0.5;
            }
            (roll, c)
        }
        (Some((r, c)), None) | (None, Some((r, c))) => (r, c),
        (None, None) => (0.0, 0.0),
    };
    let lines = v.len() + hz.len();
    if lines < 2 {
        conf *= 0.6;
    }
    // Сильный завал при съёмке с рук бывает редко — большим углам нужно больше доказательств.
    conf *= (-(roll / 7.0).powi(2)).exp();
    if level_wins {
        conf = conf.min(0.3);
    }
    // Содержимое повёрнуто по часовой на roll → исправляем поворотом против часовой на roll.
    let angle = roll;
    let apply = conf >= 0.45 && !level_wins && angle.abs() >= MIN_APPLY && angle.abs() <= MAX_APPLY;
    HorizonResult { angle, confidence: conf, apply, vertical_lines: v.len(), horizontal_lines: hz.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// «Комната»: дверные косяки и плинтус, повёрнутые на `deg` против часовой стрелки.
    fn room(deg: f32, w: usize, h: usize, keystone: f32) -> Vec<f32> {
        let mut img = vec![0.2f32; w * h];
        let t = deg.to_radians();
        let (sn, cs) = (t.sin(), t.cos());
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        for y in 0..h {
            for x in 0..w {
                // В «исходные» координаты сцены (поворот обратно по часовой).
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let sx = cs * dx - sn * dy;
                let sy = sn * dx + cs * dy;
                let u = sx / w as f32 + 0.5;
                let v = sy / h as f32 + 0.5;
                // Вертикали сходятся кверху (камера чуть смотрит вверх).
                let ku = 0.5 + (u - 0.5) * (1.0 + keystone * (v - 0.5));
                // Сглаженные края (как после оптики и демозаика).
                let cover = |d: f32, half: f32, px: f32| ((half - d) * px + 0.5).clamp(0.0, 1.0);
                let mut val = 0.2f32;
                for door in [0.18f32, 0.32, 0.62, 0.8] {
                    val = val.max(0.2 + 0.6 * cover((ku - door).abs(), 0.004, w as f32));
                }
                val = val.max(0.2 + 0.55 * cover((v - 0.78).abs(), 0.005, h as f32));
                val = val.max(0.2 + 0.55 * cover((v - 0.25).abs(), 0.004, h as f32));
                // Шум сенсора.
                let mut z = (x as u64) << 32 | y as u64;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                let n = ((z ^ (z >> 31)) % 1000) as f32 / 1000.0;
                img[y * w + x] = val + (n - 0.5) * 0.03;
            }
        }
        img
    }

    #[test]
    fn detects_known_tilt() {
        for &deg in &[2.5f32, -3.0, 0.8, -6.0] {
            let img = room(deg, 900, 600, 0.0);
            let r = detect(&img, 900, 600);
            // Содержимое повёрнуто против часовой на deg → исправление по часовой: angle ≈ −deg.
            assert!((r.angle + deg).abs() < 0.2, "tilt {deg}: найдено {:?}", r);
            assert!(r.apply, "{r:?}");
        }
    }

    #[test]
    fn keystone_does_not_fool_roll() {
        let img = room(1.5, 900, 600, 0.25);
        let r = detect(&img, 900, 600);
        assert!((r.angle + 1.5).abs() < 0.35, "{r:?}");
    }

    #[test]
    fn level_image_not_rotated() {
        let img = room(0.0, 900, 600, 0.0);
        let r = detect(&img, 900, 600);
        assert!(!r.apply, "{r:?}");
    }

    #[test]
    fn texture_without_lines_not_rotated() {
        let (w, h) = (600, 400);
        let img: Vec<f32> = (0..w * h).map(|i| (((i * 2654435761usize) >> 7) % 1000) as f32 / 1000.0 * 0.5).collect();
        let r = detect(&img, w, h);
        assert!(!r.apply, "{r:?}");
    }
}
