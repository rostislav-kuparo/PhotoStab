//! Служебная калибровка базового вида по JPEG самой камеры (команда `photostab calibrate <папка>`).
//! Сравнивает нашу линейную проявку (ББ камеры) с встроенным JPEG по блокам и печатает
//! таблицу тональной кривой и отношения насыщенности.

use crate::color;
use crate::raw;
use anyhow::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const BW: usize = 120;
const BH: usize = 80;

struct Block {
    ours: [f32; 3],
    cam: [f32; 3],
}

fn blocks_for(path: &Path) -> Result<Vec<Block>> {
    let frame = raw::decode(path)?;
    let Some(mul) = frame.asshot_wb else { return Ok(vec![]) };
    let (t, tint) = frame.color.temp_tint_from_multipliers(mul);
    let m = frame.color.cam_to_srgb(t, tint);
    let mul = frame.color.multipliers(t, tint);
    let small = raw::superpixel(&frame, 1600);
    let (w, h) = (small.width, small.height);
    let cam = raw::embedded_preview(path)?.to_rgb8();
    let cam = image::imageops::resize(&cam, w as u32, h as u32, image::imageops::FilterType::Triangle);

    let mut out = Vec::new();
    for by in 0..BH {
        for bx in 0..BW {
            let (x0, x1) = (bx * w / BW, (bx + 1) * w / BW);
            let (y0, y1) = (by * h / BH, (by + 1) * h / BH);
            let mut so = [0f64; 3];
            let mut sc = [0f64; 3];
            let mut n = 0f64;
            let mut clipped = false;
            for y in y0..y1 {
                for x in x0..x1 {
                    let c = small.px(x, y);
                    let wb = [(c[0] * mul[0]).min(1.0), (c[1] * mul[1]).min(1.0), (c[2] * mul[2]).min(1.0)];
                    if wb.iter().any(|v| *v > 0.9) {
                        clipped = true;
                    }
                    let lin = color::mulv(&m, [wb[0] as f64, wb[1] as f64, wb[2] as f64]);
                    let p = cam.get_pixel(x as u32, y as u32).0;
                    if p.iter().any(|v| *v >= 250) {
                        clipped = true;
                    }
                    for i in 0..3 {
                        so[i] += lin[i].max(0.0);
                        sc[i] += color::srgb_decode(p[i] as f32 / 255.0) as f64;
                    }
                    n += 1.0;
                }
            }
            if clipped || n == 0.0 {
                continue;
            }
            out.push(Block {
                ours: [(so[0] / n) as f32, (so[1] / n) as f32, (so[2] / n) as f32],
                cam: [(sc[0] / n) as f32, (sc[1] / n) as f32, (sc[2] / n) as f32],
            });
        }
    }
    Ok(out)
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let k = v.len() / 2;
    v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    v[k]
}

pub fn run(files: Vec<PathBuf>) -> Result<()> {
    let all: Vec<Block> = files
        .par_iter()
        .filter_map(|p| blocks_for(p).map_err(|e| eprintln!("{}: {e}", p.display())).ok())
        .flatten()
        .collect();
    println!("блоков: {}", all.len());

    println!("\nТональная кривая: линейная яркость (наша) → кодированная яркость JPEG камеры");
    println!("{:>8} {:>8} {:>8} {:>8} {:>7}", "log2 Y", "Y", "камера", "сейчас", "блоков");
    let mut b = -9.0f32;
    while b < 0.5 {
        let (lo, hi) = (2f32.powf(b), 2f32.powf(b + 0.5));
        let ys: Vec<f32> = all
            .iter()
            .filter(|k| {
                let y = color::luminance(k.ours);
                y >= lo && y < hi
            })
            .map(|k| color::srgb_encode(color::luminance(k.cam)))
            .collect();
        if ys.len() >= 20 {
            let mid = (lo * hi).sqrt();
            println!(
                "{:>8.2} {:>8.4} {:>8.3} {:>8.3} {:>7}",
                b + 0.25,
                mid,
                median(ys.clone()),
                color::render_gray(mid as f64, 1.0),
                ys.len()
            );
        }
        b += 0.5;
    }

    println!("\nНасыщенность (Oklab, отношение C камеры к нашей при равной яркости) по секторам тона:");
    let mut sectors: Vec<Vec<f32>> = vec![Vec::new(); 12];
    let mut hue_d: Vec<Vec<f32>> = vec![Vec::new(); 12];
    for k in &all {
        let lo = color::oklab(k.ours);
        let lc = color::oklab(k.cam);
        let co = (lo[1] * lo[1] + lo[2] * lo[2]).sqrt();
        let cc = (lc[1] * lc[1] + lc[2] * lc[2]).sqrt();
        if co < 0.02 || lo[0] < 0.2 {
            continue;
        }
        let ho = lo[2].atan2(lo[1]);
        let hc = lc[2].atan2(lc[1]);
        let s = (((ho.to_degrees() + 360.0) % 360.0) / 30.0) as usize % 12;
        // Сравниваем насыщенность относительно светлоты, т.к. кривые у нас и у камеры разные.
        sectors[s].push((cc / lc[0].max(0.05)) / (co / lo[0].max(0.05)));
        let mut d = (hc - ho).to_degrees();
        if d > 180.0 {
            d -= 360.0;
        }
        if d < -180.0 {
            d += 360.0;
        }
        hue_d[s].push(d);
    }
    for s in 0..12 {
        if sectors[s].len() >= 30 {
            println!(
                "  тон {:>3}–{:>3}°: насыщ. ×{:.2}, сдвиг тона {:+.1}°, блоков {}",
                s * 30,
                s * 30 + 30,
                median(sectors[s].clone()),
                median(hue_d[s].clone()),
                sectors[s].len()
            );
        }
    }
    let all_ratio: Vec<f32> = sectors.iter().flatten().copied().collect();
    println!("  всего: ×{:.2}", median(all_ratio));

    println!("\nС текущей кривой: ΔE (Oklab ×100) и отношение насыщенности камера/наша");
    for sat in [1.0f32, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4] {
        let mut de = Vec::with_capacity(all.len() / 4);
        let mut cr = Vec::new();
        let mut dl = Vec::new();
        for k in all.iter().step_by(4) {
            let r = color::render_rgb(k.ours, 1.0, sat);
            let ol = color::oklab([0, 1, 2].map(|i| color::srgb_decode(r[i])));
            let cl = color::oklab(k.cam);
            de.push(((ol[0] - cl[0]).powi(2) + (ol[1] - cl[1]).powi(2) + (ol[2] - cl[2]).powi(2)).sqrt() * 100.0);
            dl.push((cl[0] - ol[0]) * 100.0);
            let (co, cc) = ((ol[1] * ol[1] + ol[2] * ol[2]).sqrt(), (cl[1] * cl[1] + cl[2] * cl[2]).sqrt());
            if co > 0.02 && ol[0] > 0.25 {
                cr.push(cc / co);
            }
        }
        println!(
            "  насыщенность ×{sat:.2}: ΔE медиана {:.2}, ΔL {:+.2}, C камера/наша ×{:.2}",
            median(de),
            median(dl),
            median(cr)
        );
    }
    Ok(())
}
