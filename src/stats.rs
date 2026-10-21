//! Сравнение нашей проявки с JPEG камеры: яркость, доля светлых участков, насыщенность и
//! остаточный оттенок нейтралей (тёплый и «зелёный ↔ пурпурный»). Нужен для настройки
//! автоматики на реальных съёмках; в папку съёмки ничего не пишет.

use crate::{analysis, batch, color, develop, gpu, raw};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

/// Длинная сторона сравниваемых картинок.
const SIDE: u32 = 480;

/// Характеристики одной картинки в Oklab.
#[derive(Clone, Copy, Default, Debug)]
pub struct ImgStats {
    /// Процентили светлоты: 2, 10, 50, 90, 98.
    pub l: [f32; 5],
    /// Доля светлых участков (L > 0.75).
    pub light: f32,
    /// Средняя насыщенность (хрома) в светлых (L > 0.75) и средних (0.35–0.75) тонах.
    pub chroma_light: f32,
    pub chroma_mid: f32,
    /// Остаточный оттенок почти нейтральных участков, ×1000: тёплый (+) / холодный (−)
    /// и пурпурный (+) / зелёный (−).
    pub warm: f32,
    pub gm: f32,
}

pub use color::OklabAxes as Axes;

pub fn axes() -> Axes {
    color::oklab_axes()
}

/// Характеристики картинки RGBA (sRGB, 8 бит).
pub fn measure(rgba: &[u8], ax: &Axes) -> ImgStats {
    let lut: Vec<f32> = (0..256).map(|i| color::srgb_decode(i as f32 / 255.0)).collect();
    let mut ls = Vec::with_capacity(rgba.len() / 4);
    let (mut c_hi, mut n_hi, mut c_mid, mut n_mid) = (0f64, 0f64, 0f64, 0f64);
    let (mut sa, mut sb, mut nn) = (0f64, 0f64, 0f64);
    for px in rgba.chunks_exact(4) {
        let lab = color::oklab([lut[px[0] as usize], lut[px[1] as usize], lut[px[2] as usize]]);
        let c = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
        ls.push(lab[0]);
        if lab[0] > 0.75 {
            c_hi += c as f64;
            n_hi += 1.0;
        } else if lab[0] > 0.35 {
            c_mid += c as f64;
            n_mid += 1.0;
        }
        if c < 0.05 && lab[0] > 0.25 && lab[0] < 0.95 {
            sa += lab[1] as f64;
            sb += lab[2] as f64;
            nn += 1.0;
        }
    }
    if ls.is_empty() {
        return ImgStats::default();
    }
    ls.sort_by(f32::total_cmp);
    let q = |p: f32| ls[((ls.len() - 1) as f32 * p) as usize];
    let n = ls.len() as f64;
    let (a, b) = if nn > 0.0 { ((sa / nn) as f32, (sb / nn) as f32) } else { (0.0, 0.0) };
    ImgStats {
        l: [q(0.02), q(0.10), q(0.50), q(0.90), q(0.98)],
        light: (n_hi / n) as f32,
        chroma_light: if n_hi > 0.0 { (c_hi / n_hi) as f32 } else { 0.0 },
        chroma_mid: if n_mid > 0.0 { (c_mid / n_mid) as f32 } else { 0.0 },
        warm: 1000.0 * (a * ax.warm[0] + b * ax.warm[1]),
        gm: 1000.0 * (a * ax.magenta[0] + b * ax.magenta[1]),
    }
}

pub fn camera_jpeg(path: &Path, orientation: u16, w: u32, h: u32) -> Result<image::RgbaImage> {
    let img = raw::camera_jpeg(path)?;
    let img = match orientation {
        3 => img.rotate180(),
        6 => img.rotate90(),
        8 => img.rotate270(),
        _ => img,
    };
    Ok(image::imageops::resize(&img.to_rgba8(), w, h, image::imageops::FilterType::Triangle))
}

/// Наша проявка (авто, без шумодава) и JPEG камеры того же размера.
pub fn pair(gpu: &gpu::Gpu, it: &batch::Item, ax: &Axes) -> Result<(ImgStats, ImgStats)> {
    let p = analysis::auto_params(&it.analysis);
    let r = develop::recipe(&p, &it.cam, 1, Some(SIDE));
    let ours = gpu.develop_linear(&it.small, &r)?;
    let cam = camera_jpeg(&it.path, it.meta.orientation, ours.width, ours.height)?;
    Ok((measure(&ours.data, ax), measure(cam.as_raw(), ax)))
}

#[derive(Default)]
struct Acc {
    n: usize,
    ours: [f64; 11],
    cam: [f64; 11],
}

fn flat(s: &ImgStats) -> [f64; 11] {
    let mut v = [0f64; 11];
    for i in 0..5 {
        v[i] = s.l[i] as f64;
    }
    v[5] = s.light as f64;
    v[6] = s.chroma_light as f64;
    v[7] = s.chroma_mid as f64;
    v[8] = s.warm as f64;
    v[9] = s.gm as f64;
    v
}

impl Acc {
    fn add(&mut self, o: &ImgStats, c: &ImgStats, ev: f32) {
        self.n += 1;
        let (fo, fc) = (flat(o), flat(c));
        for i in 0..10 {
            self.ours[i] += fo[i];
            self.cam[i] += fc[i];
        }
        self.ours[10] += ev as f64;
    }
    fn line(&self, title: &str) -> String {
        let n = self.n.max(1) as f64;
        let o = |i: usize| self.ours[i] / n;
        let c = |i: usize| self.cam[i] / n;
        format!(
            "{title} ({} кадр., экспо {:+.2} EV)\n  L p2/p10/p50/p90/p98  камера {:.2}/{:.2}/{:.2}/{:.2}/{:.2}   наша {:.2}/{:.2}/{:.2}/{:.2}/{:.2}\n  \
             светлое {:>3.0}% → {:>3.0}%   хрома светлых {:.3} → {:.3}, средних {:.3} → {:.3}\n  \
             нейтрали: тепло {:+5.1} → {:+5.1}   пурпур(+)/зелень(−) {:+5.1} → {:+5.1}",
            self.n,
            o(10),
            c(0), c(1), c(2), c(3), c(4),
            o(0), o(1), o(2), o(3), o(4),
            c(5) * 100.0,
            o(5) * 100.0,
            c(6), o(6), c(7), o(7),
            c(8), o(8), c(9), o(9),
        )
    }
}

/// `photostab stats <папка>`: сводка по группам (и по кадрам при STATS_FRAMES=1).
pub fn run(items: &[batch::Item]) -> Result<()> {
    let gpu = gpu::Gpu::new()?;
    let ax = axes();
    let per_frame = std::env::var_os("STATS_FRAMES").is_some();
    let res: Vec<Option<(ImgStats, ImgStats)>> = items.par_iter().map(|it| pair(&gpu, it, &ax).ok()).collect();
    let mut groups: BTreeMap<usize, Acc> = BTreeMap::new();
    let mut all = Acc::default();
    for (it, r) in items.iter().zip(&res) {
        let Some((o, c)) = r else {
            eprintln!("  нет сравнения для {}", it.path.display());
            continue;
        };
        let ev = it.analysis.exposure.ev;
        groups.entry(it.group).or_default().add(o, c, ev);
        all.add(o, c, ev);
        if per_frame {
            println!(
                "{:<14} гр.{:<2} экспо {:+.2}  L50 {:.2} → {:.2}  светлое {:>3.0}% → {:>3.0}%  тепло {:+5.1} → {:+5.1}  п/з {:+5.1} → {:+5.1}",
                it.path.file_name().unwrap().to_string_lossy(),
                it.group,
                ev,
                c.l[2], o.l[2],
                c.light * 100.0, o.light * 100.0,
                c.warm, o.warm,
                c.gm, o.gm,
            );
        }
    }
    for (g, acc) in &groups {
        let first = items.iter().find(|it| it.group == *g).map(|it| &it.analysis);
        let mut title = match first {
            Some(a) => format!("Группа {} · ББ {:.0}K/{:+.1}", g + 1, a.wb.temp, a.wb.tint),
            None => format!("Группа {}", g + 1),
        };
        if let Some((parent, plan)) = first.and_then(|a| a.subgroup) {
            title.push_str(&format!(" · {} из группы {}", analysis::grouping::plan_name(plan), parent + 1));
        }
        println!("{}", acc.line(&title));
    }
    println!("{}", all.line("ВСЕГО"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_point_where_expected() {
        let ax = axes();
        // Тёплое — в сторону жёлто-оранжевого (b > 0), пурпурное — в сторону a > 0.
        assert!(ax.warm[1] > 0.5, "{:?}", ax.warm);
        assert!(ax.magenta[0] > 0.5, "{:?}", ax.magenta);
        assert!((ax.warm[0] * ax.magenta[0] + ax.warm[1] * ax.magenta[1]).abs() < 1e-4);
    }

    #[test]
    fn grey_is_neutral_and_green_is_negative() {
        let ax = axes();
        let grey: Vec<u8> = [128u8, 128, 128, 255].repeat(100);
        let s = measure(&grey, &ax);
        assert!(s.warm.abs() < 1.0 && s.gm.abs() < 1.0, "{s:?}");
        let green: Vec<u8> = [120u8, 136, 120, 255].repeat(100);
        assert!(measure(&green, &ax).gm < -3.0);
    }
}
