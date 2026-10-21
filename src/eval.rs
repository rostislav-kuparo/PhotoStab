//! `photostab eval <папка> <выход>`: насколько однородно выглядит съёмка — у JPEG камеры и у
//! нашей авто-проявки. Цель программы — чтобы вся пачка выглядела снятой при одном свете и одних
//! настройках, поэтому мерим не «правильность» отдельного кадра, а разброс между кадрами:
//! - остаток цвета на серых поверхностях (Grayness Index по готовой картинке);
//! - цвет и яркость кожи лиц (независимая проверка — не по серым пикселям);
//! - общая яркость (медиана L).
//!
//! Пишет в `<выход>` таблицу по кадрам (CSV) и контактные листы; в папку съёмки ничего не пишет.

use crate::analysis::{faces::Face, gray_index, skin};
use crate::{batch, color, develop, gpu, stats};
use anyhow::Result;
use rayon::prelude::*;
use std::fmt::Write as _;
use std::path::Path;

/// Длинная сторона проявки для замеров.
const SIDE: u32 = 640;
/// Размер плитки контактного листа.
const TW: u32 = 160;
const TH: u32 = 107;
/// Плитка листа худших кадров.
const WW: u32 = 240;
const WH: u32 = 160;

/// Замеры одной картинки.
#[derive(Clone, Copy, Default, Debug)]
struct M {
    /// Остаток на серых: тепло (+ жёлтый/оранжевый) и пурпур (+) / зелень (−), Oklab ×1000.
    gray: Option<(f32, f32)>,
    /// Кожа: тон (°), насыщенность ×1000, яркость (log2 линейной Y).
    skin: Option<(f32, f32, f32)>,
    l50: f32,
    /// Насыщенность средних тонов (Oklab ×1000) и контраст (светлота p90 − p10).
    chroma: f32,
    contrast: f32,
}

struct Row {
    name: String,
    group: usize,
    temp: f32,
    tint: f32,
    ev: f32,
    ours: M,
    cam: M,
    tile_ours: image::RgbaImage,
    tile_cam: image::RgbaImage,
}

fn scaled(faces: &[Face], s: f32) -> Vec<Face> {
    faces.iter().map(|f| f.scaled(s)).collect()
}

fn measure(rgba: &[u8], w: usize, h: usize, faces: &[Face], ax: &color::OklabAxes) -> M {
    let lut: Vec<f32> = (0..256).map(|i| color::srgb_decode(i as f32 / 255.0)).collect();
    let px: Vec<[f32; 3]> = rgba.chunks_exact(4).map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]]).collect();
    let skip = gray_index::face_skip(faces, w, w, h);
    let gray = gray_index::gray_color(&px, w, h, &gray_index::SRGB_LIMITS, skip.as_deref(), 0.01, 40).map(|g| {
        // Яркость приводим к одной (Y = 0.2), чтобы a/b не зависели от светлоты серых.
        let k = 0.2 / color::luminance(g).max(1e-6);
        let lab = color::oklab(g.map(|v| v * k));
        (1000.0 * (lab[1] * ax.warm[0] + lab[2] * ax.warm[1]), 1000.0 * (lab[1] * ax.magenta[0] + lab[2] * ax.magenta[1]))
    });
    let skin = skin::measure(&px, w, h, faces).map(|s| (s.hue.to_degrees(), s.chroma * 1000.0, s.luma.max(1e-5).log2()));
    let st = stats::measure(rgba, ax);
    M { gray, skin, l50: st.l[2], chroma: st.chroma_mid * 1000.0, contrast: st.l[3] - st.l[1] }
}

fn tile(img: &image::RgbaImage, tw: u32, th: u32) -> image::RgbaImage {
    let s = (tw as f64 / img.width() as f64).min(th as f64 / img.height() as f64);
    let (w, h) = (((img.width() as f64 * s) as u32).max(1), ((img.height() as f64 * s) as u32).max(1));
    let t = image::imageops::resize(img, w, h, image::imageops::FilterType::Triangle);
    let mut out = image::RgbaImage::from_pixel(tw, th, image::Rgba([20, 20, 20, 255]));
    image::imageops::overlay(&mut out, &t, ((tw - w) / 2) as i64, ((th - h) / 2) as i64);
    out
}

fn row(gpu: &gpu::Gpu, it: &batch::Item, ax: &color::OklabAxes, save_dir: Option<&Path>) -> Result<Row> {
    let a = &it.analysis;
    let p = crate::analysis::auto_params(a);
    let r = develop::recipe(&p, &it.cam, 1, Some(SIDE));
    let ours = gpu.develop_linear(&it.small, &r)?;
    let (w, h) = (ours.width, ours.height);
    let cam = stats::camera_jpeg(&it.path, it.meta.orientation, w, h)?;
    let faces = scaled(&a.faces, w as f32 / a.width.max(1) as f32);
    let ours_img = image::RgbaImage::from_raw(w, h, ours.data).ok_or_else(|| anyhow::anyhow!("размер"))?;
    // EVAL_SAVE=IMG_1,IMG_2 — сохранить эти кадры целиком (камера | наша) для просмотра.
    let name = it.path.file_stem().unwrap_or_default().to_string_lossy().to_string();
    if let (Ok(list), Some(dir)) = (std::env::var("EVAL_SAVE"), save_dir) {
        if list.split(',').any(|n| n.trim() == name) {
            let mut pair = image::RgbaImage::from_pixel(w * 2 + 8, h, image::Rgba([0, 0, 0, 255]));
            image::imageops::overlay(&mut pair, &cam, 0, 0);
            image::imageops::overlay(&mut pair, &ours_img, (w + 8) as i64, 0);
            let _ = image::DynamicImage::ImageRgba8(pair).to_rgb8().save(dir.join(format!("{name}.jpg")));
        }
    }
    Ok(Row {
        name,
        group: it.group,
        temp: a.wb.temp,
        tint: a.wb.tint,
        ev: a.exposure.ev,
        ours: measure(ours_img.as_raw(), w as usize, h as usize, &faces, ax),
        cam: measure(cam.as_raw(), w as usize, h as usize, &faces, ax),
        tile_ours: tile(&ours_img, WW, WH),
        tile_cam: tile(&cam, WW, WH),
    })
}

/// Среднее, разброс (sd), 5-й и 95-й процентили.
fn summary(v: &[f32]) -> Option<(f32, f32, f32, f32)> {
    if v.is_empty() {
        return None;
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    let q = |p: f32| s[((s.len() - 1) as f32 * p).round() as usize];
    Some((mean, sd, q(0.05), q(0.95)))
}

fn line(out: &mut String, title: &str, cam: &[f32], ours: &[f32]) {
    let f = |v: &[f32]| match summary(v) {
        Some((m, sd, lo, hi)) => format!("{m:+7.1} ± {sd:5.1}  [{lo:+6.1} … {hi:+6.1}]"),
        None => format!("{:^31}", "—"),
    };
    let _ = writeln!(out, "{title:<24} {}   {}   (n={})", f(cam), f(ours), ours.len());
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    s[s.len() / 2]
}

fn sheet(tiles: &[&image::RgbaImage], cols: u32, tw: u32, th: u32) -> image::RgbaImage {
    let rows = (tiles.len() as u32).div_ceil(cols).max(1);
    let mut s = image::RgbaImage::from_pixel(tw * cols, th * rows, image::Rgba([10, 10, 10, 255]));
    for (i, t) in tiles.iter().enumerate() {
        let (x, y) = ((i as u32 % cols) * tw, (i as u32 / cols) * th);
        image::imageops::overlay(&mut s, *t, x as i64, y as i64);
    }
    s
}

pub fn run(items: &[batch::Item], out: &Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let gpu = gpu::Gpu::new()?;
    let ax = color::oklab_axes();
    let all = std::env::var_os("EVAL_ALL").is_some();
    let mut sel: Vec<&batch::Item> = items.iter().filter(|it| all || it.analysis.cull.pick).collect();
    sel.sort_by(|a, b| {
        let (ta, tb) = (a.meta.capture_seconds().unwrap_or(0.0), b.meta.capture_seconds().unwrap_or(0.0));
        ta.total_cmp(&tb).then(a.path.cmp(&b.path))
    });
    let rows: Vec<Row> = sel
        .par_iter()
        .filter_map(|it| match row(&gpu, it, &ax, Some(out)) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("  {}: {e}", it.path.display());
                None
            }
        })
        .collect();

    let pick = |f: &dyn Fn(&M) -> Option<f32>, m: fn(&Row) -> &M| -> Vec<f32> { rows.iter().filter_map(|r| f(m(r))).collect() };
    let cam: fn(&Row) -> &M = |r| &r.cam;
    let ours: fn(&Row) -> &M = |r| &r.ours;
    let warm = |m: &M| m.gray.map(|g| g.0);
    let gm = |m: &M| m.gray.map(|g| g.1);
    let hue = |m: &M| m.skin.map(|s| s.0);
    let chroma = |m: &M| m.skin.map(|s| s.1);
    let face_ev = |m: &M| m.skin.map(|s| s.2);
    let l50 = |m: &M| Some(m.l50);
    let sat = |m: &M| Some(m.chroma);
    let contrast = |m: &M| Some(m.contrast);

    let mut rep = String::new();
    let _ = writeln!(rep, "Кадров: {} ({})", rows.len(), if all { "все" } else { "отобранные автоотбором; EVAL_ALL=1 — все" });
    let _ = writeln!(rep, "{:<24} {:^31}   {:^31}", "", "камера: среднее ± sd [p5 … p95]", "PhotoStab");
    line(&mut rep, "серые: тепло", &pick(&warm, cam), &pick(&warm, ours));
    line(&mut rep, "серые: пурпур/зелень", &pick(&gm, cam), &pick(&gm, ours));
    line(&mut rep, "кожа: тон, °", &pick(&hue, cam), &pick(&hue, ours));
    line(&mut rep, "кожа: насыщенность", &pick(&chroma, cam), &pick(&chroma, ours));
    line(&mut rep, "лица: яркость, EV", &pick(&face_ev, cam), &pick(&face_ev, ours));
    line(&mut rep, "L50 кадра", &pick(&l50, cam), &pick(&l50, ours));
    line(&mut rep, "насыщенность средних", &pick(&sat, cam), &pick(&sat, ours));
    line(&mut rep, "контраст (L p90−p10)", &pick(&contrast, cam), &pick(&contrast, ours));

    // Худшие по теплу: дальше всех от медианы нашей съёмки.
    let med = median(&pick(&warm, ours));
    let mut worst: Vec<&Row> = rows.iter().filter(|r| r.ours.gray.is_some()).collect();
    worst.sort_by(|a, b| (b.ours.gray.unwrap().0 - med).abs().total_cmp(&(a.ours.gray.unwrap().0 - med).abs()));
    worst.truncate(12);
    let _ = writeln!(rep, "\nДальше всех от медианы по теплу (медиана {med:+.1}):");
    for r in &worst {
        let g = r.ours.gray.unwrap();
        let c = r.cam.gray.map(|c| format!("{:+6.1} / {:+6.1}", c.0, c.1)).unwrap_or("—".into());
        let _ = writeln!(rep, "  {:<10} гр.{:<2} ББ {:>5.0}K/{:+5.1}  тепло/п-з {:+6.1} / {:+6.1}   камера {}", r.name, r.group + 1, r.temp, r.tint, g.0, g.1, c);
    }
    print!("{rep}");
    std::fs::write(out.join("eval.txt"), &rep)?;

    let mut csv = String::from("кадр;группа;темп;оттенок;экспо;наша_тепло;наша_пз;камера_тепло;камера_пз;наша_кожа_тон;камера_кожа_тон;наша_лица_ev;камера_лица_ev;наша_l50;камера_l50\n");
    let f = |v: Option<f32>| v.map(|x| format!("{x:.2}")).unwrap_or_default();
    for r in &rows {
        let _ = writeln!(
            csv,
            "{};{};{:.0};{:.1};{:.2};{};{};{};{};{};{};{};{};{:.3};{:.3}",
            r.name,
            r.group + 1,
            r.temp,
            r.tint,
            r.ev,
            f(warm(&r.ours)),
            f(gm(&r.ours)),
            f(warm(&r.cam)),
            f(gm(&r.cam)),
            f(hue(&r.ours)),
            f(hue(&r.cam)),
            f(face_ev(&r.ours)),
            f(face_ev(&r.cam)),
            r.ours.l50,
            r.cam.l50
        );
    }
    std::fs::write(out.join("frames.csv"), csv)?;

    // Данные света по каждому кадру всей съёмки: из чего сложилось решение.
    let mut lights = String::new();
    let mut by_time: Vec<&batch::Item> = items.iter().collect();
    by_time.sort_by(|a, b| a.meta.capture_seconds().unwrap_or(0.0).total_cmp(&b.meta.capture_seconds().unwrap_or(0.0)).then(a.path.cmp(&b.path)));
    for it in by_time {
        let a = &it.analysis;
        let est: Vec<String> = a.frame_wb.estimates.iter().map(|e| format!("{} {:.0}K/{:+.0}×{:.2}", e.method, e.temp, e.tint, e.weight)).collect();
        let _ = writeln!(
            lights,
            "{:<10} {} серия {:<3} гр.{:<2} свет {:>5.0}K/{:+5.1}  камера {}  лиц {}  [{}]",
            it.path.file_stem().unwrap_or_default().to_string_lossy(),
            if a.cull.pick { "★" } else { " " },
            a.scene,
            it.group + 1,
            a.wb.illuminant_temp,
            a.wb.illuminant_tint,
            a.asshot.map(|(t, ti)| format!("{t:.0}K/{ti:+.0}")).unwrap_or("-".into()),
            a.frame_wb.skin.len(),
            est.join("; ")
        );
        // Кожа «по модели»: при итоговом ББ и при усилении из анализа / итоговом.
        let gain0 = 2f32.powf(a.exposure.ev);
        if let Some((h, c)) = crate::analysis::light_graph::model_skin(&a.frame_wb, &it.cam, a.wb.temp, a.wb.tint, gain0) {
            let _ = writeln!(lights, "           кожа по модели: тон {h:.1}°, насыщенность {c:.3} (экспо {:+.2})", a.exposure.ev);
        }
    }
    std::fs::write(out.join("lights.txt"), lights)?;

    let small = |t: &image::RgbaImage| tile(t, TW, TH);
    let cam_tiles: Vec<image::RgbaImage> = rows.iter().map(|r| small(&r.tile_cam)).collect();
    let our_tiles: Vec<image::RgbaImage> = rows.iter().map(|r| small(&r.tile_ours)).collect();
    image::DynamicImage::ImageRgba8(sheet(&cam_tiles.iter().collect::<Vec<_>>(), 12, TW, TH)).to_rgb8().save(out.join("camera.jpg"))?;
    image::DynamicImage::ImageRgba8(sheet(&our_tiles.iter().collect::<Vec<_>>(), 12, TW, TH)).to_rgb8().save(out.join("ours.jpg"))?;
    let pairs: Vec<&image::RgbaImage> = worst.iter().flat_map(|r| [&r.tile_cam, &r.tile_ours]).collect();
    image::DynamicImage::ImageRgba8(sheet(&pairs, 6, WW, WH)).to_rgb8().save(out.join("worst.jpg"))?;
    println!("листы и таблица: {}", out.display());
    Ok(())
}
