//! Анализ кадра (лица, оценки света, горизонт) и согласование группы кадров:
//! общий баланс белого, общая экспозиция и общая коррекция кожи для каждой группы освещения.

pub mod culling;
pub mod exposure;
pub mod faces;
pub mod gray_index;
pub mod grouping;
pub mod horizon;
pub mod landmarks;
pub mod light_graph;
pub mod sharpness;
pub mod skin;
pub mod white_balance;

use crate::color;
use crate::develop::{DevelopParams, SkinShift};
use crate::raw::{CameraColor, LinearImage, PhotoMeta, RawFrame, superpixel};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Длинная сторона изображения для анализа и превью.
pub const ANALYSIS_SIDE: usize = 1440;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analysis {
    pub width: usize,
    pub height: usize,
    pub asshot: Option<(f32, f32)>,
    /// Оценки света по самому кадру (для совместной оценки по группе).
    pub frame_wb: white_balance::FrameWb,
    /// ББ, каким он был бы для кадра в отрыве от остальных.
    pub own_wb: white_balance::WbResult,
    pub faces: Vec<faces::Face>,
    /// Метрики лиц (глаза, рот, поворот) — в том же порядке, что и `faces`.
    pub face_metrics: Vec<landmarks::FaceMetrics>,
    /// Резкость каждого лица (по RAW в половинном разрешении).
    pub face_sharp: Vec<f32>,
    /// Резкость кадра.
    pub sharpness: f32,
    /// Абсолютная чёткость кадра (самый чёткий участок): ≈2 — резко, ≈1 — размыто.
    #[serde(default)]
    pub clarity: f32,
    /// Чёткость каждого лица (0 — не удалось оценить).
    #[serde(default)]
    pub face_clarity: Vec<f32>,
    /// Где самый чёткий участок кадра (центр, доли кадра).
    #[serde(default)]
    pub clarity_at: [f32; 2],
    /// Нормированная миниатюра яркости для поиска серий.
    pub thumb: Vec<f32>,
    pub horizon: horizon::HorizonResult,
    /// Автоотбор (заполняется для всей съёмки).
    #[serde(default)]
    pub cull: culling::CullResult,
    /// Группа освещения и решения, принятые для всей группы.
    pub group: usize,
    pub wb: white_balance::WbResult,
    pub exposure: exposure::ExposureResult,
    pub skin: Option<skin::SkinStats>,
    pub skin_shift: Option<SkinShift>,
    /// Доля кадра, где обрезан не каждый канал (там пересвет можно восстановить).
    #[serde(default)]
    pub clip_partial: f32,
    /// Сила восстановления светов (авто).
    #[serde(default)]
    pub recovery: f32,
    /// Белая точка содержимого (линейная, до экспокоррекции; 0 — неизвестна).
    #[serde(default)]
    pub white: f32,
    /// Медиана яркости JPEG камеры (эталон экспозиции).
    #[serde(default)]
    pub cam_y50: Option<f32>,
    /// Цвет кожи лиц на JPEG камеры (Oklab a, b): эталон общего тона съёмки.
    #[serde(default)]
    pub cam_skin: Option<[f32; 2]>,
    /// Профиль шума сенсора (по кадру, затем общий для его ISO в съёмке) — для шумодава в RAW.
    #[serde(default)]
    pub noise: Option<crate::noise::NoiseProfile>,
    /// Если группа — выделенный план: (группа освещения, план 0/1/2). Сейчас не выделяются:
    /// свет каждого кадра решается сам, поле осталось для старых сохранений.
    #[serde(default)]
    pub subgroup: Option<(usize, u8)>,
    /// Серия (кадры подряд в одном месте): номер и число кадров.
    #[serde(default)]
    pub scene: usize,
    #[serde(default)]
    pub scene_len: usize,
    /// Типичный ББ группы (медиана кадров): для списка групп и пересчёта старых правок.
    #[serde(default)]
    pub group_wb: Option<(f32, f32)>,
    /// Полный размер кадра в ориентации показа.
    #[serde(default)]
    pub full_w: usize,
    #[serde(default)]
    pub full_h: usize,
}

/// Поворот/отражение уменьшенного кадра в ориентацию показа (EXIF).
pub fn orient(img: &LinearImage, o: u16) -> LinearImage {
    let (w, h) = (img.width, img.height);
    let swap = matches!(o, 5..=8);
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    let mut data = vec![0f32; ow * oh * 3];
    for yd in 0..oh {
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
            let s = (ys * w + xs) * 3;
            let d = (yd * ow + xd) * 3;
            data[d..d + 3].copy_from_slice(&img.data[s..s + 3]);
        }
    }
    LinearImage { width: ow, height: oh, data }
}

/// Линейный sRGB после баланса белого (без экспокоррекции).
pub fn linear_srgb(img: &LinearImage, cam: &CameraColor, temp: f32, tint: f32) -> Vec<[f32; 3]> {
    let mul = cam.multipliers(temp as f64, tint as f64);
    let m = cam.cam_to_srgb(temp as f64, tint as f64);
    let m: [[f32; 3]; 3] = [0, 1, 2].map(|i| [m[i][0] as f32, m[i][1] as f32, m[i][2] as f32]);
    img.data
        .par_chunks_exact(3)
        .map(|c| {
            let wb = [(c[0] * mul[0]).min(1.0), (c[1] * mul[1]).min(1.0), (c[2] * mul[2]).min(1.0)];
            [0, 1, 2].map(|i| (m[i][0] * wb[0] + m[i][1] * wb[1] + m[i][2] * wb[2]).max(0.0))
        })
        .collect()
}

/// sRGB 8 бит для детектора лиц: с автоматическим осветлением тёмных кадров.
fn detector_image(lin: &[[f32; 3]]) -> Vec<u8> {
    let mut ys: Vec<f32> = lin.iter().step_by(17).map(|c| exposure::luma(*c)).collect();
    let k = ys.len() / 2;
    ys.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    let gain = (0.06 / ys[k].max(1e-4)).clamp(0.7, 8.0);
    let lut = color::ToneLut::new(gain);
    lin.par_iter()
        .flat_map_iter(|c| {
            let r = lut.apply(*c, gain);
            [0, 1, 2].map(|i| (r[i] * 255.0).round() as u8)
        })
        .collect()
}

/// Яркость кадра и кожа при данном ББ.
struct Measure {
    key: f32,
    p995: f32,
    hist: Vec<f32>,
    skin: Option<skin::SkinStats>,
    white: f32,
    /// Поправка, при которой наша проявка в среднем так же светла, как JPEG камеры.
    cam_ev: Option<f32>,
}

fn measure(small: &LinearImage, cam: &CameraColor, faces: &[faces::Face], temp: f32, tint: f32, cam_y50: Option<f32>) -> Measure {
    let lin = linear_srgb(small, cam, temp, tint);
    let skin = skin::measure(&lin, small.width, small.height, faces);
    let (key, p995, hist) = exposure::frame_stats(&lin, small.width, small.height);
    let white = content_white(small, &lin);
    let cam_ev = cam_y50.map(|y| camera_match_ev(&lin, white, y));
    Measure { key, p995, hist, skin, white, cam_ev }
}

/// Медиана кодированной яркости картинки (0..1) — для JPEG камеры.
pub fn camera_luma50(img: &image::DynamicImage) -> Option<f32> {
    let rgb = img.to_rgb8();
    let mut ys: Vec<f32> = rgb
        .as_raw()
        .chunks_exact(3)
        .step_by(5)
        .map(|p| (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0)
        .collect();
    if ys.is_empty() {
        return None;
    }
    let k = ys.len() / 2;
    ys.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    Some(ys[k])
}

/// Поправка экспозиции, при которой медиана кодированной яркости нашей проявки (с нашей кривой
/// и белой точкой) совпадает с медианой JPEG камеры.
fn camera_match_ev(lin: &[[f32; 3]], white: f32, cam_y50: f32) -> f32 {
    let px: Vec<[f32; 3]> = lin.iter().step_by(7).copied().collect();
    if px.is_empty() {
        return 0.0;
    }
    let median_at = |ev: f32| {
        let g = 2f32.powf(ev);
        let lut = color::ToneLut::new((g * white).max(1.0));
        let mut ys: Vec<f32> = px
            .iter()
            .map(|c| {
                let r = lut.apply(*c, g);
                0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]
            })
            .collect();
        let k = ys.len() / 2;
        ys.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
        ys[k]
    };
    let (mut a, mut b) = (-3.0f32, 3.0f32);
    for _ in 0..14 {
        let m = 0.5 * (a + b);
        if median_at(m) < cam_y50 {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// «Белая точка» содержимого кадра (линейный sRGB при данном ББ, до экспокоррекции):
/// 99.9-й процентиль самого яркого канала по участкам, где ни один канал не упёрся в предел.
/// Уже выбитые окна и лампы всё равно уйдут в белое — под них не нужно сжимать весь верх.
fn content_white(small: &LinearImage, lin: &[[f32; 3]]) -> f32 {
    let mut v: Vec<f32> = small
        .data
        .par_chunks_exact(3)
        .zip(lin.par_iter())
        .filter(|(c, l)| c.iter().all(|x| *x < 0.98) && l.iter().all(|x| *x < 0.999))
        .map(|(_, l)| l[0].max(l[1]).max(l[2]))
        .collect();
    if v.len() < 100 {
        return 1.0;
    }
    let k = ((v.len() - 1) as f32 * 0.999) as usize;
    v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    (v[k] * 1.05).clamp(0.05, 1.0)
}

/// Цвет кожи найденных лиц на JPEG камеры (Oklab a, b линейного sRGB). `w` — ширина кадра
/// анализа (в ориентации показа), в его координатах заданы лица.
fn camera_skin(img: &image::DynamicImage, orientation: u16, faces: &[faces::Face], w: usize) -> Option<[f32; 2]> {
    if faces.is_empty() || w == 0 {
        return None;
    }
    let img = match orientation {
        3 => img.rotate180(),
        6 => img.rotate90(),
        8 => img.rotate270(),
        _ => img.clone(),
    };
    let rgb = img.to_rgb8();
    let (cw, ch) = (rgb.width() as usize, rgb.height() as usize);
    let lut: Vec<f32> = (0..256).map(|i| color::srgb_decode(i as f32 / 255.0)).collect();
    let lin: Vec<[f32; 3]> = rgb.pixels().map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]]).collect();
    let s = cw as f32 / w as f32;
    let faces: Vec<faces::Face> = faces.iter().map(|f| f.scaled(s)).collect();
    let st = skin::measure(&lin, cw, ch, &faces)?;
    Some([st.chroma * st.hue.cos(), st.chroma * st.hue.sin()])
}

/// Анализ одного кадра. Решения (ББ, экспозиция, кожа) — как для группы из одного кадра;
/// `harmonize` затем пересчитывает их по всей съёмке. `cam_jpeg` — JPEG камеры (эталон яркости и
/// общего тона кожи), если есть.
pub fn analyze(frame: &RawFrame, cam_jpeg: Option<&image::DynamicImage>) -> (LinearImage, Analysis) {
    let cam_y50 = cam_jpeg.and_then(camera_luma50);
    let small = crate::prof::time("1 уменьшение кадра", || orient(&superpixel(frame, ANALYSIS_SIDE), frame.meta.orientation));
    let cam = &frame.color;
    let (w, h) = (small.width, small.height);
    let asshot = frame.asshot_wb.map(|m| {
        let (t, tint) = cam.temp_tint_from_multipliers(m);
        (t as f32, tint as f32)
    });
    let (t0, tint0) = asshot.unwrap_or((5200.0, 0.0));
    let lin0 = crate::prof::time("2 линейный sRGB", || linear_srgb(&small, cam, t0, tint0));
    let det = crate::prof::time("3 картинка для детектора", || detector_image(&lin0));
    let det_img = faces::Rgb8 { width: w, height: h, data: &det };
    // Лица: YuNet находит, Face Mesh подтверждает (отсекает ложные — принты, узоры) и даёт метрики.
    let mut faces = Vec::new();
    let mut face_metrics = Vec::new();
    for f in crate::prof::time("4 YuNet лица", || faces::detect(&det_img)).unwrap_or_default() {
        if let Ok(m) = crate::prof::time("5 Face Mesh", || landmarks::run(&det_img, &f)) {
            if std::env::var("FM_DEBUG").is_ok() {
                eprintln!("лицо {:.0}x{:.0} @({:.0},{:.0}) score {:.2} presence {:.3} {:?}", f.w, f.h, f.x, f.y, f.score, m.presence, landmarks::metrics(&m));
            }
            if m.presence >= 0.5 {
                faces.push(f);
                face_metrics.push(landmarks::metrics(&m));
            }
        }
    }
    // Резкость — по зелёному каналу RAW в половинном разрешении (детальнее, чем превью).
    let (g, gw, gh) = crate::prof::time("6 резкость: плоскость G", || { let (g, gw, gh) = sharpness::green_half(frame); sharpness::orient_plane(&g, gw, gh, frame.meta.orientation) });
    let scale = gw as f32 / w as f32;
    let face_sharp: Vec<f32> = faces.iter().map(|f| sharpness::face(&g, gw, gh, f, scale)).collect();
    let sharp = crate::prof::time("7 резкость: лапласиан", || sharpness::frame(&g, gw, gh));
    let ((clarity, clarity_at), face_clarity) = crate::prof::time("7б чёткость", || {
        let pyr = sharpness::Pyramid::new(&g, gw, gh);
        let fc: Vec<f32> = faces.iter().map(|f| sharpness::face_clarity(&pyr, gw, gh, f, scale).unwrap_or(0.0)).collect();
        (sharpness::frame_clarity(&pyr, gw, gh), fc)
    });
    drop(g);
    let thumb = sharpness::thumb(&lin0, w, h);
    let cam_skin = cam_jpeg.and_then(|img| camera_skin(img, frame.meta.orientation, &faces, w));
    let noise = crate::prof::time("6б профиль шума", || crate::noise::estimate(frame));
    let clip_partial = partial_clip_fraction(&small);

    let frame_wb = white_balance::FrameWb {
        estimates: crate::prof::time("8 ББ: оценки", || white_balance::frame_estimates(&small, cam, frame.asshot_wb, &faces)),
        skin: skin::camera_samples(&small, &faces),
        prior_tint: asshot.map(|a| a.1),
    };
    let own_wb = crate::prof::time("9 ББ: решение", || white_balance::solve(cam, std::slice::from_ref(&frame_wb)));
    let lum: Vec<f32> = lin0.iter().map(|c| exposure::luma(*c)).collect();
    let horizon = crate::prof::time("10 горизонт", || horizon::detect(&lum, w, h));

    let m = crate::prof::time("11 яркость и кожа", || measure(&small, cam, &faces, own_wb.temp, own_wb.tint, cam_y50));
    let skin_stats = m.skin;
    let fe = exposure::FrameExpo {
        key: m.key,
        p995: m.p995,
        face_luma: skin_stats.map(|s| s.luma),
        ev100: ev100(&frame.meta),
        flash: frame.meta.flash_fired(),
        hist: m.hist,
        cam_ev: m.cam_ev,
    };
    let exposure = exposure::group(&[fe]).remove(0);
    let skin_shift = skin_stats.as_ref().and_then(skin::correction);
    let a = Analysis {
        width: w,
        height: h,
        asshot,
        frame_wb,
        wb: own_wb.clone(),
        own_wb,
        faces,
        face_metrics,
        face_sharp,
        sharpness: sharp,
        clarity,
        face_clarity,
        clarity_at,
        thumb,
        cull: Default::default(),
        horizon,
        group: 0,
        exposure,
        skin: skin_stats,
        skin_shift,
        clip_partial,
        recovery: auto_recovery(clip_partial),
        white: m.white,
        cam_y50,
        cam_skin,
        noise,
        subgroup: None,
        scene: 0,
        scene_len: 1,
        group_wb: None,
        full_w: if matches!(frame.meta.orientation, 5..=8) { frame.height } else { frame.width },
        full_h: if matches!(frame.meta.orientation, 5..=8) { frame.width } else { frame.height },
    };
    (small, a)
}

/// Маска кожи кадра: эллипсы «лицо + шея» вокруг найденных лиц (самые крупные первыми).
pub fn skin_mask(faces: &[faces::Face], w: usize, h: usize) -> Option<crate::develop::SkinMask> {
    let mut fs: Vec<&faces::Face> = faces.iter().collect();
    fs.sort_by(|a, b| b.area().total_cmp(&a.area()));
    fs.truncate(8);
    if fs.is_empty() || w == 0 || h == 0 {
        return None;
    }
    let mut m = crate::develop::SkinMask { n: fs.len() as u32, e: [[0.0; 4]; 8] };
    for (i, f) in fs.iter().enumerate() {
        // От лба (без волос) до шеи, по ширине — с ушами.
        m.e[i] = [(f.x + 0.5 * f.w) / w as f32, (f.y + 0.75 * f.h) / h as f32, 0.75 * f.w / w as f32, 0.85 * f.h / h as f32];
    }
    Some(m)
}

pub fn auto_params(a: &Analysis) -> DevelopParams {
    DevelopParams {
        temp: a.wb.temp,
        tint: a.wb.tint,
        exposure_ev: a.exposure.ev,
        angle_deg: if a.horizon.apply { a.horizon.angle } else { 0.0 },
        skin: a.skin_shift,
        skin_mask: skin_mask(&a.faces, a.width, a.height),
        recovery: a.recovery,
        white: (a.white > 0.0).then_some(a.white),
        ..DevelopParams::default()
    }
}

/// Доля пикселей, где часть каналов упёрлась в предел сенсора, а часть — нет.
pub fn partial_clip_fraction(small: &LinearImage) -> f32 {
    let n = small.data.len() / 3;
    let partial = small
        .data
        .chunks_exact(3)
        .filter(|c| {
            let s = c.iter().filter(|v| **v >= 0.98).count();
            s > 0 && s < 3
        })
        .count();
    partial as f32 / n.max(1) as f32
}

/// Восстанавливаем пересвет, только если есть что восстанавливать (частично обрезанные участки).
pub fn auto_recovery(partial: f32) -> f32 {
    let t = ((partial - 0.002) / (0.02 - 0.002)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn ev100(m: &PhotoMeta) -> Option<f32> {
    let n = m.fnumber_f()?;
    let t = m.exposure_time_secs()?;
    let iso = m.iso? as f64;
    Some(((n * n / t).log2() - (iso / 100.0).log2()) as f32)
}

/// Разброс ББ камеры по съёмке (майред, между 10-м и 90-м процентилем), больше которого камера
/// считается стоявшей на автобалансе.
const CAMERA_AWB_SPREAD: f32 = 4.0;

fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Согласование всей съёмки.
/// 1. Свет каждого кадра — одна задача на всю съёмку (`light_graph`): кадры при одном свете
///    согласуются между собой, общий сдвиг задают камера (оттенок) и кожа всех лиц (температура).
/// 2. Каждый кадр приводится к одной белой точке (`white_balance::from_illuminant`).
/// 3. Группы освещения (по найденному свету и яркости сцены) — для правок, экспозиции и шумодава.
/// 4. Экспозиция: по группе «как у камеры» с переводом по EV100, затем яркость людей каждой
///    серии подтягивается к общей для съёмки (`exposure::balance_shoot`).
/// `items` — (метаданные, уменьшенный кадр, цвет камеры, анализ). Возвращает номер группы для каждого.
pub fn harmonize(items: &mut [(PhotoMeta, LinearImage, CameraColor, Analysis)]) -> Vec<usize> {
    let n = items.len();
    // Профиль шума — общий для каждого ISO съёмки (медиана по кадрам): по одному кадру с малым
    // числом ровных участков он неточен, а сенсор и ISO у кадров одни и те же.
    {
        let mut by_iso: std::collections::HashMap<u32, Vec<crate::noise::NoiseProfile>> = std::collections::HashMap::new();
        for (m, _, _, a) in items.iter() {
            if let (Some(iso), Some(p)) = (m.iso, a.noise) {
                by_iso.entry(iso).or_default().push(p);
            }
        }
        for (m, _, _, a) in items.iter_mut() {
            if let Some(v) = m.iso.and_then(|iso| by_iso.get(&iso)).filter(|v| v.len() >= 3) {
                a.noise = Some(crate::noise::NoiseProfile {
                    k: median_f32(v.iter().map(|p| p.k).collect()),
                    s: median_f32(v.iter().map(|p| p.s).collect()),
                });
            }
        }
    }
    // Серии (кадры подряд в одном месте) — связи графа и «сцена» в интерфейсе.
    let series = {
        let si: Vec<culling::SeriesItem> = items
            .iter()
            .map(|(m, _, _, a)| culling::SeriesItem { time: m.capture_seconds(), landscape: a.width >= a.height, thumb: &a.thumb })
            .collect();
        culling::series(&si)
    };
    let mut series_of = vec![0usize; n];
    let mut series_len = vec![1usize; n];
    for (k, s) in series.iter().enumerate() {
        for &i in s {
            series_of[i] = k;
            series_len[i] = s.len();
        }
    }
    let scene_ev: Vec<Option<f32>> = items.iter().map(|(m, _, _, a)| ev100(m).map(|e| e + (a.exposure.key.max(1e-4) / 0.18).log2())).collect();

    // Общий тон по коже на JPEG камеры — только если камера была на автобалансе: на
    // фиксированном ББ её JPEG показывает цвет пресета (под лампами — оранжевый), а не
    // привычный приятный вид.
    let camera_awb = {
        let mut m: Vec<f32> = items.iter().filter(|(meta, _, _, _)| !meta.flash_fired()).filter_map(|(_, _, _, a)| a.asshot.map(|(t, _)| 1e6 / t.max(1.0))).collect();
        m.sort_by(f32::total_cmp);
        m.len() >= 10 && m[m.len() * 9 / 10] - m[m.len() / 10] > CAMERA_AWB_SPREAD
    };
    let solved = {
        let nodes: Vec<light_graph::Node> = items
            .iter()
            .enumerate()
            .map(|(i, (m, _, cam, a))| light_graph::Node {
                frame: &a.frame_wb,
                cam,
                time: m.capture_seconds(),
                flash: m.flash_fired(),
                scene_ev: scene_ev[i],
                series: series_of[i],
                cam_skin: if camera_awb { a.cam_skin } else { None },
                gain: 2f32.powf(a.exposure.ev),
            })
            .collect();
        light_graph::solve(&nodes)
    };
    let wbs: Vec<white_balance::WbResult> = (0..n)
        .map(|i| {
            let (m, t) = solved.light[i];
            white_balance::from_illuminant(m, t, solved.confidence[i], items[i].3.frame_wb.estimates.clone())
        })
        .collect();

    // Группы освещения — по уже согласованному свету: только для правок, экспозиции и шумодава.
    let gi: Vec<grouping::GroupItem> = (0..n)
        .map(|i| grouping::GroupItem {
            time: items[i].0.capture_seconds(),
            flash: items[i].0.flash_fired(),
            temp: (1e6 / solved.light[i].0) as f32,
            tint: solved.light[i].1 as f32,
            scene_ev: scene_ev[i],
        })
        .collect();
    let groups = grouping::assign_groups(&gi);
    let n_groups = groups.iter().copied().max().map(|g| g + 1).unwrap_or(0);
    if std::env::var_os("WB_DEBUG").is_some() {
        for g in 0..n_groups {
            let m: Vec<usize> = (0..n).filter(|&i| groups[i] == g).collect();
            let t = median_f32(m.iter().map(|&i| gi[i].temp).collect());
            let e = median_f32(m.iter().filter_map(|&i| gi[i].scene_ev).collect());
            eprintln!("группа {}: {} кадр., свет ≈ {:.0} K, сцена ≈ {:.1} EV, первый {}", g + 1, m.len(), t, e, m.first().map(|&i| i).unwrap_or(0));
        }
    }

    // Яркость и кожа каждого кадра при его ББ.
    let measures: Vec<Measure> = items
        .par_iter()
        .zip(&wbs)
        .map(|((_, small, cam, a), wb)| measure(small, cam, &a.faces, wb.temp, wb.tint, a.cam_y50))
        .collect();
    let fe: Vec<exposure::FrameExpo> = (0..n)
        .map(|i| exposure::FrameExpo {
            key: measures[i].key,
            p995: measures[i].p995,
            face_luma: measures[i].skin.map(|s| s.luma),
            ev100: ev100(&items[i].0),
            flash: items[i].0.flash_fired(),
            hist: measures[i].hist.clone(),
            cam_ev: measures[i].cam_ev,
        })
        .collect();

    let mut exps: Vec<Option<exposure::ExposureResult>> = vec![None; n];
    let mut shifts: Vec<Option<SkinShift>> = vec![None; n_groups];
    let mut group_wb: Vec<(f32, f32)> = vec![(5200.0, 0.0); n_groups];
    for g in 0..n_groups {
        let members: Vec<usize> = (0..n).filter(|&i| groups[i] == g).collect();
        if members.is_empty() {
            continue;
        }
        let r = exposure::group(&members.iter().map(|&i| fe[i].clone()).collect::<Vec<_>>());
        for (&i, e) in members.iter().zip(r) {
            exps[i] = Some(e);
        }
        let skin_all: Vec<skin::SkinStats> = members.iter().filter_map(|&i| measures[i].skin).collect();
        shifts[g] = skin::group_correction(&skin_all);
        group_wb[g] = (median_f32(members.iter().map(|&i| wbs[i].temp).collect()), median_f32(members.iter().map(|&i| wbs[i].tint).collect()));
    }

    // Одна яркость людей на всю съёмку: тёмные уголки подтягиваются к залу.
    let mut exps: Vec<exposure::ExposureResult> = exps.into_iter().map(|e| e.expect("экспозиция кадра")).collect();
    exposure::balance_shoot(&mut exps, &fe, &series_of, &solved.edges);

    for (i, (_, _, _, a)) in items.iter_mut().enumerate() {
        let g = groups[i];
        a.group = g;
        a.subgroup = None;
        a.wb = wbs[i].clone();
        a.scene = series_of[i];
        a.scene_len = series_len[i];
        a.group_wb = Some(group_wb[g]);
        a.exposure = exps[i].clone();
        a.skin = measures[i].skin;
        a.skin_shift = shifts[g];
        a.white = measures[i].white;
    }
    groups
}
