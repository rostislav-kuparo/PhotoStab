#![recursion_limit = "256"]
// В релизе — оконное приложение без консоли; консольные команды подключаются к терминалу сами.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod batch;
mod calibrate;
mod color;
mod denoise;
mod desktop;
mod develop;
mod eval;
mod export;
mod gpu;
mod history;
mod nn;
mod noise;
mod prof;
mod raw;
mod raw_denoise;
mod server;
mod stats;

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn usage() {
    eprintln!(
        "PhotoStab — единая проявка RAW-съёмки для репортажа\n\n\
         photostab                         открыть окно программы\n\
         photostab <папка>                 открыть съёмку в окне программы\n\
         photostab --browser [папка]       интерфейс в браузере вместо окна\n\
         photostab process <вход> <выход>  обработать всё без интерфейса (JPEG + report.txt)\n\
         photostab cull <вход> <выход>     только автоотбор: таблица и контактные листы\n\
         photostab info <файл>             сведения о RAW\n"
    );
}

fn serve(open: Option<PathBuf>) -> Result<()> {
    let gpu = gpu::Gpu::new()?;
    server::run(gpu, open)
}

/// Окно программы; если WebView2 недоступен — интерфейс в браузере.
fn window(open: Option<PathBuf>) -> Result<()> {
    match desktop::run(open.clone()) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("окно не открылось ({e}), запускаю в браузере");
            serve(open)
        }
    }
}

/// В оконной сборке у процесса нет консоли — для команд подключаемся к консоли терминала.
fn attach_console() {
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Console::AttachConsole(windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS);
    }
}

fn show_error(e: &anyhow::Error) {
    eprintln!("Ошибка: {e:#}");
    let _ = rfd::MessageDialog::new()
        .set_title("PhotoStab")
        .set_description(format!("{e:#}"))
        .set_level(rfd::MessageLevel::Error)
        .show();
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let gui = args.is_empty() || (args.len() == 1 && Path::new(&args[0]).is_dir());
    if !gui {
        attach_console();
    }
    if let Err(e) = run(&mut args) {
        if gui {
            show_error(&e);
        } else {
            eprintln!("Ошибка: {e:#}");
        }
        std::process::exit(1);
    }
}

fn run(args: &mut Vec<String>) -> Result<()> {
    if let Some(i) = args.iter().position(|a| a == "--port") {
        args.remove(i);
        if i < args.len() {
            let port = args.remove(i);
            unsafe { std::env::set_var("PHOTOSTAB_PORT", port) };
        }
    }
    if let Some(i) = args.iter().position(|a| a == "--no-browser") {
        args.remove(i);
        unsafe { std::env::set_var("PHOTOSTAB_NO_BROWSER", "1") };
        return serve(args.first().map(PathBuf::from).filter(|p| p.is_dir()));
    }
    if let Some(i) = args.iter().position(|a| a == "--browser") {
        args.remove(i);
        return serve(args.first().map(PathBuf::from).filter(|p| p.is_dir()));
    }
    match args.first().map(|s| s.as_str()) {
        Some("info") if args.len() >= 2 => info(Path::new(&args[1])),
        Some("process") if args.len() >= 3 => process(Path::new(&args[1]), Path::new(&args[2])),
        Some("calibrate") if args.len() >= 2 => calibrate::run(collect_raws(Path::new(&args[1]))),
        Some("sheet") if args.len() >= 3 => contact_sheets(Path::new(&args[1]), Path::new(&args[2])),
        Some("makernotes") if args.len() >= 2 => makernotes(Path::new(&args[1])),
        Some("nrtest") if args.len() >= 3 => nrtest(Path::new(&args[1]), Path::new(&args[2])),
        Some("horizon") if args.len() >= 2 => horizon_debug(Path::new(&args[1])),
        Some("faces") if args.len() >= 3 => faces_debug(Path::new(&args[1]), Path::new(&args[2])),
        Some("cull") if args.len() >= 3 => cull_report(Path::new(&args[1]), Path::new(&args[2])),
        Some("meta") if args.len() >= 2 => meta_table(Path::new(&args[1])),
        Some("report") if args.len() >= 2 => report(Path::new(&args[1])),
        Some("blur") if args.len() >= 3 => blur_sheets(Path::new(&args[1]), Path::new(&args[2])),
        Some("stats") if args.len() >= 2 => stats::run(&analyze_with_progress(&collect_raws(Path::new(&args[1])))),
        Some("noise") if args.len() >= 2 => noise_report(Path::new(&args[1])),
        Some("nrcompare") if args.len() >= 3 => nr_compare(Path::new(&args[1]), Path::new(&args[2])),
        Some("eval") if args.len() >= 3 => eval::run(&analyze_with_progress(&collect_raws(Path::new(&args[1]))), Path::new(&args[2])),
        Some("compare") if args.len() >= 3 => compare(Path::new(&args[1]), Path::new(&args[2])),
        Some("-h") | Some("--help") => {
            usage();
            Ok(())
        }
        None => window(None),
        Some(p) if Path::new(p).is_dir() => window(Some(PathBuf::from(p))),
        _ => {
            usage();
            bail!("неизвестная команда")
        }
    }
}

fn collect_raws(input: &Path) -> Vec<PathBuf> {
    if input.is_file() {
        return vec![input.to_path_buf()];
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(input)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| raw::is_raw_file(p)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn info(path: &Path) -> Result<()> {
    let t = Instant::now();
    let f = raw::decode(path)?;
    let dt = t.elapsed();
    println!("{}", path.display());
    println!("  камера:      {} {}", f.meta.make, f.meta.model);
    println!("  объектив:    {:?}", f.meta.lens);
    println!("  размер:      {}x{}  cfa={:?}", f.width, f.height, f.cfa);
    println!("  чёрный:      {:?}  белый: {}", f.black, f.white);
    println!("  ББ камеры:   {:?}", f.asshot_wb);
    if let Some(m) = f.asshot_wb {
        let (t, tint) = f.color.temp_tint_from_multipliers(m);
        println!("               ≈ {:.0} K, оттенок {:+.1}", t, tint);
    }
    println!("  матрица A:   {}", f.color.cm_a.is_some());
    println!(
        "  съёмка:      {:?} ISO {:?} выдержка {:?} f/{:?} вспышка {:?}",
        f.meta.datetime_original,
        f.meta.iso,
        f.meta.exposure_time,
        f.meta.fnumber_f(),
        f.meta.flash
    );
    println!("  ориентация:  {}", f.meta.orientation);
    // Доля пересвеченных отсчётов по цветам фильтра и «почти пересвеченных».
    let mut clip = [0u64; 3];
    let mut near = [0u64; 3];
    let mut tot = [0u64; 3];
    for y in 0..f.height {
        for x in 0..f.width {
            let i = (y & 1) * 2 + (x & 1);
            let c = f.cfa[i] as usize;
            let v = (f.data[y * f.width + x] as f32 - f.black[i]) / (f.white - f.black[i]);
            tot[c] += 1;
            if v >= 0.985 {
                clip[c] += 1;
            } else if v >= 0.5 {
                near[c] += 1;
            }
        }
    }
    for (c, n) in ["R", "G", "B"].iter().enumerate() {
        println!(
            "  {n}: обрезано {:.1}%, ярко (0.5–0.98) {:.1}%",
            100.0 * clip[c] as f64 / tot[c] as f64,
            100.0 * near[c] as f64 / tot[c] as f64
        );
    }
    println!("  декодирование: {:.0} мс", dt.as_secs_f64() * 1000.0);
    Ok(())
}

/// Встроенный JPEG камеры в ориентации показа.
fn oriented_preview(path: &Path, orientation: u16) -> Result<image::RgbaImage> {
    let img = raw::embedded_preview(path)?;
    let img = match orientation {
        3 => img.rotate180(),
        6 => img.rotate90(),
        8 => img.rotate270(),
        _ => img,
    };
    Ok(img.to_rgba8())
}

fn analyze_with_progress(files: &[PathBuf]) -> Vec<batch::Item> {
    let t = Instant::now();
    let (items, failed) = batch::analyze_all(files, &|i, n| {
        if i % 10 == 0 || i == n {
            eprintln!("  анализ {i}/{n}");
        }
    });
    for (p, e) in failed {
        eprintln!("  ошибка {}: {e}", p.display());
    }
    eprintln!("анализ: {:.1} с", t.elapsed().as_secs_f64());
    prof::report();
    items
}

/// Автоотбор: таблица по сериям и контактные листы с рамками (зелёная — отобран, красная — нет).
fn cull_report(input: &Path, output: &Path) -> Result<()> {
    use rayon::prelude::*;
    std::fs::create_dir_all(output)?;
    let files = collect_raws(input);
    let mut items = analyze_with_progress(&files);
    items.sort_by(|a, b| a.analysis.cull.series.cmp(&b.analysis.cull.series).then(a.path.cmp(&b.path)));
    let picked = items.iter().filter(|i| i.analysis.cull.pick).count();
    let series = items.iter().map(|i| i.analysis.cull.series).max().map(|s| s + 1).unwrap_or(0);
    println!("кадров {}, серий {}, отобрано {}", items.len(), series, picked);
    for it in &items {
        let a = &it.analysis;
        let c = &a.cull;
        println!(
            "серия {:>3} ({}) {} {:<14} оценка {:.2} глаза {} резкость {:>7.1} лиц {}  {}",
            c.series,
            c.series_len,
            if c.pick { "★" } else { " " },
            it.path.file_name().unwrap().to_string_lossy(),
            c.score,
            c.eyes.map(|e| format!("{e:.2}")).unwrap_or("-".into()),
            a.sharpness,
            a.faces.len(),
            c.reasons.join(", ")
        );
    }
    const TW: u32 = 320;
    const TH: u32 = 214;
    let tiles: Vec<image::RgbaImage> = items
        .par_iter()
        .map(|it| {
            let img = oriented_preview(&it.path, it.meta.orientation).unwrap_or_else(|_| image::RgbaImage::new(TW, TH));
            let s = ((TW - 8) as f64 / img.width() as f64).min((TH - 8) as f64 / img.height() as f64);
            let (w, h) = ((img.width() as f64 * s) as u32, (img.height() as f64 * s) as u32);
            let t = image::imageops::resize(&img, w.max(1), h.max(1), image::imageops::FilterType::Triangle);
            let col = if it.analysis.cull.pick { image::Rgba([40, 200, 60, 255]) } else { image::Rgba([200, 40, 40, 255]) };
            let mut tile = image::RgbaImage::from_pixel(TW, TH, image::Rgba([40, 40, 40, 255]));
            let (ox, oy) = ((TW - w) / 2, (TH - h) / 2);
            for y in oy.saturating_sub(4)..(oy + h + 4).min(TH) {
                for x in ox.saturating_sub(4)..(ox + w + 4).min(TW) {
                    tile.put_pixel(x, y, col);
                }
            }
            image::imageops::overlay(&mut tile, &t, ox as i64, oy as i64);
            tile
        })
        .collect();
    for (k, chunk) in tiles.chunks(24).enumerate() {
        let mut sheet = image::RgbaImage::from_pixel(TW * 6, TH * 4, image::Rgba([20, 20, 20, 255]));
        for (i, t) in chunk.iter().enumerate() {
            image::imageops::overlay(&mut sheet, t, ((i % 6) as u32 * TW) as i64, ((i / 6) as u32 * TH) as i64);
        }
        let names: Vec<String> = items[k * 24..k * 24 + chunk.len()]
            .iter()
            .map(|i| i.path.file_stem().unwrap().to_string_lossy().to_string())
            .collect();
        println!("лист {k}: {}", names.join(" "));
        image::DynamicImage::ImageRgba8(sheet).to_rgb8().save(output.join(format!("cull_{k}.jpg")))?;
    }
    Ok(())
}

fn describe(it: &batch::Item) -> String {
    let a = &it.analysis;
    let asshot = a.asshot.map(|(t, ti)| format!("{t:.0}K/{ti:+.0}")).unwrap_or("-".into());
    format!(
        "{:<14} {} гр.{:<2} {} камера {:<11} кадр {:>5.0}K/{:+.0} → ББ {:>5.0}K/{:+.0} (ув. {:.2})  экспо {:+.2} EV [key {:.3} p995 {:.2} лицо {}; {}]  лица {}  кожа {}  горизонт {:+.2}° (ув. {:.2}{})",
        it.path.file_name().unwrap().to_string_lossy(),
        it.meta.capture_seconds().map(|t| format!("t{:>9.0}", t)).unwrap_or("t        -".into()),
        it.group,
        // Яркость сцены: экспопара камеры + насколько светлым вышел кадр.
        analysis::ev100(&it.meta)
            .map(|e| format!("EV {:>5.1} сцена {:>5.1}", e, e + (a.exposure.key.max(1e-4) / 0.18).log2()))
            .unwrap_or("EV     - сцена     -".into()),
        asshot,
        a.own_wb.illuminant_temp,
        a.own_wb.illuminant_tint,
        a.wb.temp,
        a.wb.tint,
        a.wb.confidence,
        a.exposure.ev,
        a.exposure.key,
        a.exposure.p995,
        a.exposure.face_luma.map(|v| format!("{v:.3}")).unwrap_or("-".into()),
        a.exposure.reason,
        a.faces.len(),
        a.skin_shift.map(|s| format!("{:+.1}°×{:.2}", s.shift.to_degrees(), s.chroma_scale)).unwrap_or("-".into()),
        a.horizon.angle,
        a.horizon.confidence,
        if a.horizon.apply { ", поворот" } else { "" }
    )
}

/// Склейка «JPEG камеры | наш результат (авто)» для визуальной проверки.
fn compare(input: &Path, output: &Path) -> Result<()> {
    let files = collect_raws(input);
    std::fs::create_dir_all(output)?;
    let gpu = gpu::Gpu::new()?;
    let items = analyze_with_progress(&files);
    const SIDE: u32 = 900;
    let only: Vec<String> = std::env::var("CMP_ONLY").map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    for it in &items {
        let nm = it.path.file_name().unwrap().to_string_lossy().to_string();
        if !only.is_empty() && !only.iter().any(|o| nm.contains(o.as_str())) {
            continue;
        }
        println!("{}", describe(it));
        let frame = raw::decode(&it.path)?;
        let p = analysis::auto_params(&it.analysis);
        let r = develop::recipe(&p, &frame.color, frame.meta.orientation, Some(SIDE));
        let ours = gpu.develop_raw(&frame, &r)?;
        let ours = image::RgbaImage::from_raw(ours.width, ours.height, ours.data).unwrap();
        let cam = oriented_preview(&it.path, frame.meta.orientation)?;
        let ch = ours.height();
        let cw = (cam.width() as f64 * ch as f64 / cam.height() as f64).round() as u32;
        let cam = image::imageops::resize(&cam, cw, ch, image::imageops::FilterType::Triangle);
        let mut canvas = image::RgbaImage::from_pixel(cw + ours.width() + 8, ch, image::Rgba([0, 0, 0, 255]));
        image::imageops::overlay(&mut canvas, &cam, 0, 0);
        image::imageops::overlay(&mut canvas, &ours, cw as i64 + 8, 0);
        let rgb = image::DynamicImage::ImageRgba8(canvas).to_rgb8();
        let name = it.path.file_stem().unwrap().to_string_lossy().to_string() + "_cmp.jpg";
        rgb.save(output.join(&name))?;
    }
    Ok(())
}

/// Служебное: листы 100%-кропов (лицо или центр кадра) от самых нечётких кадров к чётким —
/// для настройки порога «размыто». Порядок кадров на листах печатается.
fn blur_sheets(input: &Path, output: &Path) -> Result<()> {
    use rayon::prelude::*;
    let files = collect_raws(input);
    std::fs::create_dir_all(output)?;
    let items = analyze_with_progress(&files);
    let by_frame = std::env::var("BLUR_BY").map(|v| v == "frame").unwrap_or(false);
    let score = |a: &analysis::Analysis| {
        let fc = batch::best_face_clarity(a);
        if by_frame || fc <= 0.0 { a.clarity } else { fc }
    };
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&a, &b| score(&items[a].analysis).total_cmp(&score(&items[b].analysis)));
    let n = items.len();
    let mut pick: Vec<usize> = order.iter().copied().take(40).collect();
    pick.extend(order.iter().copied().skip(n.saturating_sub(8)));
    const TW: u32 = 480;
    const TH: u32 = 320;
    let tiles: Vec<image::RgbaImage> = pick
        .par_iter()
        .map(|&i| {
            let it = &items[i];
            let a = &it.analysis;
            let img = match oriented_preview(&it.path, it.meta.orientation) {
                Ok(v) => v,
                Err(_) => return image::RgbaImage::new(TW, TH),
            };
            let (iw, ih) = (img.width(), img.height());
            let face = if by_frame { None } else { a.faces.iter().max_by(|x, y| x.area().total_cmp(&y.area())) };
            let (cx, cy) = match face {
                Some(f) => {
                    let (fx, fy) = f.center();
                    (fx / a.width as f32, fy / a.height as f32)
                }
                None => (a.clarity_at[0], a.clarity_at[1]),
            };
            let x = ((cx * iw as f32) as i64 - TW as i64 / 2).clamp(0, iw.saturating_sub(TW) as i64) as u32;
            let y = ((cy * ih as f32) as i64 - TH as i64 / 2).clamp(0, ih.saturating_sub(TH) as i64) as u32;
            image::imageops::crop_imm(&img, x, y, TW.min(iw), TH.min(ih)).to_image()
        })
        .collect();
    let blurred: Vec<String> = items
        .iter()
        .filter(|it| it.analysis.cull.reasons.iter().any(|r| r == "размыто"))
        .map(|it| it.path.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    println!("размыто: {} из {}: {}", blurred.len(), items.len(), blurred.join(" "));
    for (k, chunk) in tiles.chunks(16).enumerate() {
        let mut sheet = image::RgbaImage::from_pixel(TW * 4, TH * 4, image::Rgba([40, 40, 40, 255]));
        for (j, t) in chunk.iter().enumerate() {
            image::imageops::overlay(&mut sheet, t, ((j % 4) as u32 * TW) as i64, ((j / 4) as u32 * TH) as i64);
        }
        let names: Vec<String> = pick[k * 16..(k * 16 + chunk.len())]
            .iter()
            .map(|&i| {
                let a = &items[i].analysis;
                let fc = batch::best_face_clarity(a);
                format!("{} {:.2}/{:.2}{}", items[i].path.file_stem().unwrap().to_string_lossy(), fc, a.clarity, if a.cull.reasons.iter().any(|r| r == "размыто") { "✗" } else { "" })
            })
            .collect();
        println!("лист {k}: {}", names.join("  "));
        image::DynamicImage::ImageRgba8(sheet).to_rgb8().save(output.join(format!("blur_{k}.jpg")))?;
    }
    Ok(())
}

/// Контактные листы встроенных превью (по 24 кадра, 6×4) — для просмотра всей съёмки.
fn contact_sheets(input: &Path, output: &Path) -> Result<()> {
    use rayon::prelude::*;
    let files = collect_raws(input);
    std::fs::create_dir_all(output)?;
    const TW: u32 = 320;
    const TH: u32 = 214;
    let tiles: Vec<image::RgbaImage> = files
        .par_iter()
        .map(|p| {
            let o = raw::read_meta(p).map(|m| m.orientation).unwrap_or(1);
            let img = oriented_preview(p, o).unwrap_or_else(|_| image::RgbaImage::new(TW, TH));
            let s = (TW as f64 / img.width() as f64).min(TH as f64 / img.height() as f64);
            let (w, h) = ((img.width() as f64 * s) as u32, (img.height() as f64 * s) as u32);
            image::imageops::resize(&img, w.max(1), h.max(1), image::imageops::FilterType::Triangle)
        })
        .collect();
    for (k, chunk) in tiles.chunks(24).enumerate() {
        let mut sheet = image::RgbaImage::from_pixel(TW * 6, TH * 4, image::Rgba([40, 40, 40, 255]));
        for (i, t) in chunk.iter().enumerate() {
            let (x, y) = ((i % 6) as u32 * TW, (i / 6) as u32 * TH);
            image::imageops::overlay(&mut sheet, t, (x + (TW - t.width()) / 2) as i64, (y + (TH - t.height()) / 2) as i64);
        }
        let first = k * 24;
        let names: Vec<String> = files[first..(first + chunk.len())]
            .iter()
            .map(|p| p.file_stem().unwrap().to_string_lossy().to_string())
            .collect();
        println!("лист {k}: {}", names.join(" "));
        image::DynamicImage::ImageRgba8(sheet).to_rgb8().save(output.join(format!("sheet_{k}.jpg")))?;
    }
    Ok(())
}

/// Служебное: сравнение шумодавов на одном RAW — 100% кропы вокруг самого крупного лица:
/// без шумодава | NAFNet (авто) | NAFNet в полную силу / RAW | правило программы | как в параметрах.
/// `NR_EV` — экспозиция (как в программе), иначе своя по кадру.
fn nr_compare(path: &Path, out: &Path) -> Result<()> {
    let gpu = gpu::Gpu::new()?;
    let frame = raw::decode(path)?;
    let cam = raw::camera_preview_fast(path);
    let (_, a) = analysis::analyze(&frame, cam.as_ref());
    let mut base = analysis::auto_params(&a);
    if let Some(ev) = std::env::var("NR_EV").ok().and_then(|v| v.parse::<f32>().ok()) {
        base.exposure_ev = ev;
    }
    let noise = noise::estimate(&frame).ok_or_else(|| anyhow::anyhow!("не удалось оценить шум"))?;
    println!("ISO {:?}, экспо {:+.2} EV, шум: k {:.3e}, s {:.3e}", frame.meta.iso, base.exposure_ev, noise.k, noise.s);
    let (cx, cy) = a
        .faces
        .iter()
        .max_by(|x, y| x.area().total_cmp(&y.area()))
        .map(|f| ((f.x + f.w / 2.0) / a.width as f32, (f.y + f.h / 2.0) / a.height as f32))
        .unwrap_or((0.5, 0.5));
    let (cw, ch) = (640u32, 640u32);
    // Кроп в пикселях результата; шумодав по картинке — только в нём (быстрее).
    let probe = gpu.develop_raw(&frame, &develop::recipe(&base, &frame.color, frame.meta.orientation, None))?;
    let x0 = ((cx * probe.width as f32) as i64 - cw as i64 / 2).clamp(0, (probe.width - cw) as i64) as u32;
    let y0 = ((cy * probe.height as f32) as i64 - ch as i64 / 2).clamp(0, (probe.height - ch) as i64) as u32;
    let region = (x0 as usize, y0 as usize, cw as usize, ch as usize);
    let render = |p: develop::DevelopParams, title: &str| -> Result<develop::Rgba> {
        let r = develop::recipe(&p, &frame.color, frame.meta.orientation, None);
        let _ = gpu.develop_raw_ext(&frame, &r, Some(region))?; // прогрев сетей
        let t = Instant::now();
        let img = gpu.develop_raw_ext(&frame, &r, Some(region))?;
        println!("{title}: {:.2} с", t.elapsed().as_secs_f64());
        Ok(img)
    };
    let auto = denoise::auto_strength(frame.meta.iso, base.exposure_ev).unwrap_or(denoise::DenoiseParams { luma: 0.8, chroma: 1.0 });
    let mut plain = base;
    plain.denoise = None;
    let mut naf = base;
    naf.denoise = Some(auto);
    let mut raw_only = base;
    raw_only.denoise = None;
    raw_only.raw_nr = Some(raw_denoise::RawNr { noise, strength: 1.0 });
    let mut chosen = base;
    chosen.denoise = Some(auto);
    raw_denoise::apply(&mut chosen, Some(noise));
    let full_naf = develop::DevelopParams { denoise: Some(denoise::DenoiseParams { luma: 1.0, chroma: 1.0 }), ..base };
    let imgs = [
        render(plain, "1 без шумодава")?,
        render(naf, "2 NAFNet (авто)")?,
        render(full_naf, "3 NAFNet в полную силу")?,
        render(raw_only, "4 RAW в полную силу")?,
        render(chosen, "5 программа: RAW ½ + NAFNet 0.7")?,
        render(base, "6 как есть в параметрах")?,
    ];
    let mut canvas = image::RgbaImage::from_pixel(cw * 3 + 16, ch * 2 + 8, image::Rgba([0, 0, 0, 255]));
    for (k, img) in imgs.iter().enumerate() {
        let full = image::RgbaImage::from_raw(img.width, img.height, img.data.clone()).unwrap();
        let crop = image::imageops::crop_imm(&full, x0, y0, cw, ch).to_image();
        image::imageops::overlay(&mut canvas, &crop, ((k % 3) as u32 * (cw + 8)) as i64, ((k / 3) as u32 * (ch + 8)) as i64);
    }
    image::DynamicImage::ImageRgba8(canvas).to_rgb8().save(out)?;
    Ok(())
}

/// Служебное: профиль шума сенсора по ISO (по самим RAW, до 12 кадров на ISO).
fn noise_report(input: &Path) -> Result<()> {
    use rayon::prelude::*;
    use std::collections::BTreeMap;
    let files = collect_raws(input);
    let mut by_iso: BTreeMap<u32, Vec<PathBuf>> = BTreeMap::new();
    for f in &files {
        if let Ok(m) = raw::read_meta(f) {
            by_iso.entry(m.iso.unwrap_or(0)).or_default().push(f.clone());
        }
    }
    for (iso, list) in by_iso {
        let step = (list.len() / 12).max(1);
        let pick: Vec<&PathBuf> = list.iter().step_by(step).take(12).collect();
        let per: Vec<(Vec<(f32, f32)>, Option<noise::NoiseProfile>)> = pick
            .par_iter()
            .filter_map(|p| raw::decode(p).ok())
            .map(|fr| {
                let s = noise::samples(&fr);
                let f = noise::fit(&s);
                (s, f)
            })
            .collect();
        let all: Vec<(f32, f32)> = per.iter().flat_map(|(s, _)| s.iter().copied()).collect();
        let pooled = noise::fit(&all);
        let ks: Vec<String> = per.iter().filter_map(|(_, f)| f.map(|f| format!("{:.2e}", f.k))).collect();
        match pooled {
            Some(p) => println!(
                "ISO {iso:>5} ({} кадр.): k {:.3e}  s {:.3e}  (шум на чёрном {:.1} DN из 16383 при белом)   по кадрам k: {}",
                list.len(),
                p.k,
                p.s,
                p.s.sqrt() * 15800.0,
                ks.join(" ")
            ),
            None => println!("ISO {iso:>5} ({} кадр.): мало ровных участков", list.len()),
        }
    }
    Ok(())
}

/// Служебное: шумодав на полном кадре — время и склейка 100% кропов «без | с шумодавом».
fn nrtest(path: &Path, out: &Path) -> Result<()> {
    let gpu = gpu::Gpu::new()?;
    let frame = raw::decode(path)?;
    let (_, a) = analysis::analyze(&frame, None);
    let mut p = analysis::auto_params(&a);
    let nr = denoise::auto_strength(frame.meta.iso, p.exposure_ev).unwrap_or(denoise::DenoiseParams { luma: 0.8, chroma: 1.0 });
    println!("ISO {:?}, экспо {:+.2} EV, шумодав: яркость {:.0}%, цвет {:.0}%", frame.meta.iso, p.exposure_ev, nr.luma * 100.0, nr.chroma * 100.0);
    let r0 = develop::recipe(&p, &frame.color, frame.meta.orientation, None);
    let t = Instant::now();
    let plain = gpu.develop_raw(&frame, &r0)?;
    println!("без шумодава: {:.2} с", t.elapsed().as_secs_f64());
    p.denoise = Some(nr);
    let r1 = develop::recipe(&p, &frame.color, frame.meta.orientation, None);
    let _ = gpu.develop_raw(&frame, &r1)?; // прогрев модели
    let t = Instant::now();
    let clean = gpu.develop_raw(&frame, &r1)?;
    println!("с шумодавом: {:.2} с", t.elapsed().as_secs_f64());
    // Центр кропа — самое крупное лицо или центр кадра.
    let (cx, cy) = a
        .faces
        .iter()
        .max_by(|x, y| x.area().total_cmp(&y.area()))
        .map(|f| ((f.x + f.w / 2.0) / a.width as f32, (f.y + f.h / 2.0) / a.height as f32))
        .unwrap_or((0.5, 0.5));
    let (cw, ch) = (800u32, 800u32);
    let crop = |img: &develop::Rgba| -> image::RgbaImage {
        let x0 = ((cx * img.width as f32) as i64 - cw as i64 / 2).clamp(0, (img.width - cw) as i64) as u32;
        let y0 = ((cy * img.height as f32) as i64 - ch as i64 / 2).clamp(0, (img.height - ch) as i64) as u32;
        let full = image::RgbaImage::from_raw(img.width, img.height, img.data.clone()).unwrap();
        image::imageops::crop_imm(&full, x0, y0, cw, ch).to_image()
    };
    let mut canvas = image::RgbaImage::from_pixel(cw * 2 + 8, ch, image::Rgba([0, 0, 0, 255]));
    image::imageops::overlay(&mut canvas, &crop(&plain), 0, 0);
    image::imageops::overlay(&mut canvas, &crop(&clean), cw as i64 + 8, 0);
    image::DynamicImage::ImageRgba8(canvas).to_rgb8().save(out)?;
    Ok(())
}

/// Служебное: подробности поиска горизонта по одному файлу (HZ_DEBUG).
fn horizon_debug(path: &Path) -> Result<()> {
    unsafe { std::env::set_var("HZ_DEBUG", "1") };
    let frame = raw::decode(path)?;
    let small = analysis::orient(&raw::superpixel(&frame, analysis::ANALYSIS_SIDE), frame.meta.orientation);
    let lin = analysis::linear_srgb(&small, &frame.color, 4200.0, -20.0);
    let lum: Vec<f32> = lin.iter().map(|c| analysis::exposure::luma(*c)).collect();
    let r = analysis::horizon::detect(&lum, small.width, small.height);
    println!("{r:?}");
    Ok(())
}

/// Служебное: лица и точки лица на одном кадре, с отрисовкой.
fn faces_debug(path: &Path, out: &Path) -> Result<()> {
    let frame = raw::decode(path)?;
    let (small, a) = analysis::analyze(&frame, None);
    let lin = analysis::linear_srgb(&small, &frame.color, a.wb.temp, a.wb.tint);
    let gain = 2f32.powf(a.exposure.ev);
    let rgb: Vec<u8> = lin
        .iter()
        .flat_map(|c| {
            let r = color::render_rgb(*c, gain, 1.0);
            [0, 1, 2].map(|i| (r[i] * 255.0).round() as u8)
        })
        .collect();
    let img8 = analysis::faces::Rgb8 { width: small.width, height: small.height, data: &rgb };
    let mut canvas = image::RgbImage::from_raw(small.width as u32, small.height as u32, rgb.clone()).unwrap();
    let all = analysis::faces::detect(&img8)?;
    for f in &all {
        let m = analysis::landmarks::run(&img8, f)?;
        let col = if m.presence >= 0.5 { image::Rgb([0, 255, 0]) } else { image::Rgb([255, 0, 0]) };
        let (x0, y0, x1, y1) = (f.x.max(0.0) as u32, f.y.max(0.0) as u32, (f.x + f.w) as u32, (f.y + f.h) as u32);
        for x in x0..x1.min(canvas.width() - 1) {
            for y in [y0, y1.min(canvas.height() - 1)] {
                canvas.put_pixel(x, y, col);
            }
        }
        for y in y0..y1.min(canvas.height() - 1) {
            for x in [x0, x1.min(canvas.width() - 1)] {
                canvas.put_pixel(x, y, col);
            }
        }
    }
    for f in &a.faces {
        let m = analysis::landmarks::run(&img8, f)?;
        let met = analysis::landmarks::metrics(&m);
        println!("лицо {:.0}x{:.0} score {:.2}: {:?}", f.w, f.h, f.score, met);
        for (i, p) in m.pts.iter().enumerate() {
            let col = if [33, 133, 159, 145, 160, 144, 158, 153, 362, 263, 386, 374, 385, 380, 387, 373].contains(&i) {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 255, 0])
            };
            let (x, y) = (p[0] as i64, p[1] as i64);
            if x >= 0 && y >= 0 && (x as u32) < canvas.width() && (y as u32) < canvas.height() {
                canvas.put_pixel(x as u32, y as u32, col);
            }
        }
    }
    canvas.save(out)?;
    Ok(())
}

/// Служебное: теги makernotes Canon (CMT3) в CR3.
fn makernotes(path: &Path) -> Result<()> {
    use rawler::formats::tiff::reader::TiffReader;
    let f = std::fs::File::open(path)?;
    let bmff = rawler::formats::bmff::Bmff::new(std::io::BufReader::new(f)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let desc = bmff.filebox.moov.cr3desc.as_ref().ok_or_else(|| anyhow::anyhow!("не CR3"))?;
    for (tag, e) in desc.cmt3.tiff.root_ifd().entries() {
        let s = format!("{:?}", e.value);
        println!("{:#06x} {}", tag, &s[..s.len().min(220)]);
    }
    Ok(())
}

/// Таблица метаданных: время, экспопараметры, уровень белого.
fn meta_table(input: &Path) -> Result<()> {
    use rayon::prelude::*;
    let files = collect_raws(input);
    let rows: Vec<String> = files
        .par_iter()
        .map(|p| match raw::decode(p) {
            Ok(f) => {
                let m = &f.meta;
                let small = raw::superpixel(&f, 400);
                let mean_g = small.data.chunks_exact(3).map(|c| c[1] as f64).sum::<f64>() / (small.data.len() / 3) as f64;
                format!(
                    "{:<14} {:<20} ISO {:>5} t {:>8} f/{:<4} EV100 {:>6.2} вспышка {:<3} белый {:>6.0} чёрный {:>6.0} G {:.4}",
                    p.file_name().unwrap().to_string_lossy(),
                    m.datetime_original.clone().unwrap_or_default(),
                    m.iso.unwrap_or(0),
                    m.exposure_time.map(|(n, d)| format!("{n}/{d}")).unwrap_or_default(),
                    m.fnumber_f().map(|v| format!("{v:.1}")).unwrap_or_default(),
                    analysis::ev100(m).unwrap_or(f32::NAN),
                    if m.flash_fired() { "да" } else { "нет" },
                    f.white,
                    f.black[0],
                    mean_g
                )
            }
            Err(e) => format!("{}: {e}", p.display()),
        })
        .collect();
    for r in rows {
        println!("{r}");
    }
    Ok(())
}

fn report(input: &Path) -> Result<()> {
    let files = collect_raws(input);
    for it in analyze_with_progress(&files) {
        println!("{}", describe(&it));
    }
    Ok(())
}

fn process(input: &Path, output: &Path) -> Result<()> {
    let files = collect_raws(input);
    if files.is_empty() {
        bail!("RAW-файлы не найдены в {}", input.display());
    }
    std::fs::create_dir_all(output)?;
    let gpu = gpu::Gpu::new()?;
    eprintln!("GPU: {}", gpu.name);
    let t0 = Instant::now();
    let items = analyze_with_progress(&files);
    let jobs: Vec<(PathBuf, develop::DevelopParams)> = items
        .iter()
        .map(|it| {
            let mut p = analysis::auto_params(&it.analysis);
            p.denoise = denoise::auto_strength(it.meta.iso, p.exposure_ev);
            raw_denoise::apply(&mut p, it.analysis.noise);
            (it.path.clone(), p)
        })
        .collect();
    let mut report = String::new();
    for it in &items {
        report.push_str(&describe(it));
        report.push('\n');
    }
    std::fs::write(output.join("report.txt"), report)?;
    let t1 = Instant::now();
    let opts = batch::ExportOptions { quality: 92, long_side: None, sequence: None };
    let results = batch::export_all(&gpu, &jobs, output, &opts, &|i, n, _| {
        if i % 10 == 0 || i == n {
            eprintln!("  экспорт {i}/{n}");
        }
    });
    for (p, r) in &results {
        if let Err(e) = r {
            eprintln!("  ошибка {}: {e}", p.display());
        }
    }
    eprintln!(
        "готово: {} кадров, анализ {:.1} с, экспорт {:.1} с",
        results.len(),
        (t1 - t0).as_secs_f64(),
        t1.elapsed().as_secs_f64()
    );
    Ok(())
}
