//! Декодирование RAW (через rawler) в простой и независимый от формата вид.

use anyhow::{Context, Result, anyhow, bail};
use rawler::decoders::RawDecodeParams;
use rawler::imgop::xyz::Illuminant;
use rawler::rawimage::RawPhotometricInterpretation;
use rawler::rawsource::RawSource;
use rawler::RawImageData;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const RAW_EXTENSIONS: &[&str] = &[
    "cr3", "cr2", "crw", "nef", "nrw", "arw", "srf", "sr2", "dng", "rw2", "orf", "pef", "raf", "srw", "iiq", "3fr",
    "erf", "kdc", "dcr", "mef", "mos", "mrw", "x3f",
];

pub fn is_raw_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| RAW_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Метаданные снимка, которые нужны для анализа и для EXIF в выходном JPEG.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhotoMeta {
    pub make: String,
    pub model: String,
    pub lens: Option<String>,
    pub iso: Option<u32>,
    /// Выдержка как дробь (числитель, знаменатель).
    pub exposure_time: Option<(u32, u32)>,
    pub fnumber: Option<(u32, u32)>,
    pub focal_length: Option<(u32, u32)>,
    pub exposure_bias: Option<(i32, i32)>,
    pub flash: Option<u16>,
    /// "YYYY:MM:DD HH:MM:SS"
    pub datetime_original: Option<String>,
    pub subsec_original: Option<String>,
    pub offset_time_original: Option<String>,
    pub artist: Option<String>,
    pub copyright: Option<String>,
    pub serial_number: Option<String>,
    pub orientation: u16,
}

impl PhotoMeta {
    pub fn flash_fired(&self) -> bool {
        self.flash.map(|f| f & 1 == 1).unwrap_or(false)
    }

    /// Время съёмки в секундах (от условной эпохи), для сортировки и поиска серий.
    pub fn capture_seconds(&self) -> Option<f64> {
        let s = self.datetime_original.as_deref()?;
        let b = s.as_bytes();
        if b.len() < 19 {
            return None;
        }
        let num = |a: usize, n: usize| -> Option<i64> { s.get(a..a + n)?.trim().parse::<i64>().ok() };
        let (y, mo, d) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
        let (h, mi, se) = (num(11, 2)?, num(14, 2)?, num(17, 2)?);
        let days = days_from_civil(y, mo, d);
        let mut t = days as f64 * 86400.0 + (h * 3600 + mi * 60 + se) as f64;
        if let Some(sub) = self.subsec_original.as_deref() {
            let sub = sub.trim();
            if !sub.is_empty() && sub.bytes().all(|c| c.is_ascii_digit()) {
                if let Ok(v) = sub.parse::<f64>() {
                    t += v / 10f64.powi(sub.len() as i32);
                }
            }
        }
        Some(t)
    }

    pub fn exposure_time_secs(&self) -> Option<f64> {
        self.exposure_time.filter(|r| r.1 != 0).map(|(n, d)| n as f64 / d as f64)
    }

    pub fn fnumber_f(&self) -> Option<f64> {
        self.fnumber.filter(|r| r.1 != 0).map(|(n, d)| n as f64 / d as f64)
    }
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Цветовые данные камеры: матрицы XYZ -> камера для источников A и D65 (как в DNG).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraColor {
    pub cm_a: Option<[f32; 9]>,
    pub cm_d65: [f32; 9],
}

/// Байеровский кадр, обрезанный до полезной области.
pub struct RawFrame {
    pub width: usize,
    pub height: usize,
    /// Значения сенсора, построчно.
    pub data: Vec<u16>,
    /// Цвет (0=R, 1=G, 2=B) в позиции ((y & 1) * 2 + (x & 1)).
    pub cfa: [u32; 4],
    /// Уровень чёрного для тех же четырёх позиций.
    pub black: [f32; 4],
    pub white: f32,
    /// Баланс белого камеры: множители R, G, B (G = 1).
    pub asshot_wb: Option<[f32; 3]>,
    pub color: CameraColor,
    pub meta: PhotoMeta,
}

fn rat_u(r: &rawler::formats::tiff::Rational) -> (u32, u32) {
    (r.n, r.d)
}

fn rat_s(r: &rawler::formats::tiff::SRational) -> (i32, i32) {
    (r.n, r.d)
}

fn flat9(v: &[f32]) -> Option<[f32; 9]> {
    if v.len() < 9 {
        return None;
    }
    let mut m = [0f32; 9];
    m.copy_from_slice(&v[..9]);
    Some(m)
}

pub fn read_meta(path: &Path) -> Result<PhotoMeta> {
    let src = RawSource::new(path).with_context(|| format!("не удалось открыть {}", path.display()))?;
    let decoder = rawler::get_decoder(&src).map_err(|e| anyhow!("формат не поддерживается: {e}"))?;
    let params = RawDecodeParams::default();
    let md = decoder.raw_metadata(&src, &params).map_err(|e| anyhow!("метаданные: {e}"))?;
    Ok(meta_from(&md))
}

fn meta_from(md: &rawler::decoders::RawMetadata) -> PhotoMeta {
    let ex = &md.exif;
    PhotoMeta {
        make: md.make.clone(),
        model: md.model.clone(),
        lens: md
            .lens
            .as_ref()
            .map(|l| l.lens_name.clone())
            .or_else(|| ex.lens_model.clone())
            .filter(|s| !s.trim().is_empty()),
        iso: ex.iso_speed_ratings.map(|v| v as u32).or(ex.iso_speed).or(ex.recommended_exposure_index),
        exposure_time: ex.exposure_time.as_ref().map(rat_u),
        fnumber: ex.fnumber.as_ref().map(rat_u),
        focal_length: ex.focal_length.as_ref().map(rat_u),
        exposure_bias: ex.exposure_bias.as_ref().map(rat_s),
        flash: ex.flash,
        datetime_original: ex.date_time_original.clone().or_else(|| ex.create_date.clone()),
        subsec_original: ex.sub_sec_time_original.clone(),
        offset_time_original: ex.offset_time_original.clone(),
        artist: ex.artist.clone(),
        copyright: ex.copyright.clone(),
        serial_number: ex.serial_number.clone(),
        orientation: ex.orientation.unwrap_or(1),
    }
}

pub fn decode(path: &Path) -> Result<RawFrame> {
    let src = RawSource::new(path).with_context(|| format!("не удалось открыть {}", path.display()))?;
    let decoder = rawler::get_decoder(&src).map_err(|e| anyhow!("формат не поддерживается: {e}"))?;
    let params = RawDecodeParams::default();
    let raw = decoder.raw_image(&src, &params, false).map_err(|e| anyhow!("ошибка декодирования: {e}"))?;
    let md = decoder.raw_metadata(&src, &params).map_err(|e| anyhow!("метаданные: {e}"))?;
    let mut meta = meta_from(&md);
    if meta.orientation == 1 {
        meta.orientation = orientation_code(raw.orientation);
    }

    if raw.cpp != 1 {
        bail!("поддерживаются только байеровские RAW (cpp={})", raw.cpp);
    }
    let cfa_pat = match &raw.photometric {
        RawPhotometricInterpretation::Cfa(cfg) => cfg.cfa.clone(),
        _ => bail!("неподдерживаемая фотометрия RAW"),
    };
    if cfa_pat.width != 2 || cfa_pat.height != 2 {
        bail!("матрица {} не байеровская (X-Trans пока не поддерживается)", cfa_pat.name);
    }

    let full_w = raw.width;
    let full_h = raw.height;
    let crop = raw.crop_area.or(raw.active_area);
    let (cx, cy, cw, ch) = match crop {
        Some(r) => (r.p.x, r.p.y, r.d.w, r.d.h),
        None => (0, 0, full_w, full_h),
    };
    let cw = cw.min(full_w - cx) & !1;
    let ch = ch.min(full_h - cy) & !1;

    let src_data: Vec<u16> = match raw.data {
        RawImageData::Integer(ref v) => v.clone(),
        RawImageData::Float(ref v) => {
            let wl = raw.whitelevel.0.first().copied().unwrap_or(65535) as f32;
            let scale = if wl <= 1.5 { 65535.0 } else { 1.0 };
            v.iter().map(|x| (x * scale).clamp(0.0, 65535.0) as u16).collect()
        }
    };
    let mut data = Vec::with_capacity(cw * ch);
    for y in 0..ch {
        let row = (cy + y) * full_w + cx;
        data.extend_from_slice(&src_data[row..row + cw]);
    }

    let mut cfa = [0u32; 4];
    for i in 0..4 {
        let (y, x) = (i / 2, i % 2);
        let c = cfa_pat.color_at(cy + y, cx + x);
        cfa[i] = match c {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => bail!("неизвестный цвет фильтра {c}"),
        };
    }
    let bl = &raw.blacklevel;
    let mut black = [0f32; 4];
    for i in 0..4 {
        let (y, x) = (i / 2, i % 2);
        let bw = bl.width.max(1);
        let bh = bl.height.max(1);
        let idx = ((cy + y) % bh) * bw + ((cx + x) % bw);
        let r = bl.levels.get(idx).or(bl.levels.first());
        black[i] = r.map(|r| if r.d == 0 { 0.0 } else { r.n as f32 / r.d as f32 }).unwrap_or(0.0);
    }
    let white = raw.whitelevel.0.iter().copied().min().unwrap_or(65535) as f32;

    let wb = raw.wb_coeffs;
    let asshot_wb = if wb.iter().take(3).all(|v| v.is_finite() && *v > 0.0) {
        Some([wb[0] / wb[1], 1.0, wb[2] / wb[1]])
    } else {
        None
    };

    let cm_d65 = raw
        .color_matrix
        .get(&Illuminant::D65)
        .and_then(|m| flat9(m))
        .or_else(|| raw.color_matrix.values().next().and_then(|m| flat9(m)))
        .or_else(|| {
            let x = raw.xyz_to_cam;
            Some([x[0][0], x[0][1], x[0][2], x[1][0], x[1][1], x[1][2], x[2][0], x[2][1], x[2][2]])
        })
        .unwrap();
    let cm_a = raw.color_matrix.get(&Illuminant::A).and_then(|m| flat9(m));

    Ok(RawFrame {
        width: cw,
        height: ch,
        data,
        cfa,
        black,
        white,
        asshot_wb,
        color: CameraColor { cm_a, cm_d65 },
        meta,
    })
}

fn orientation_code(o: rawler::Orientation) -> u16 {
    use rawler::Orientation::*;
    match o {
        Normal | Unknown => 1,
        HorizontalFlip => 2,
        Rotate180 => 3,
        VerticalFlip => 4,
        Transpose => 5,
        Rotate90 => 6,
        Transverse => 7,
        Rotate270 => 8,
    }
}

/// Средний JPEG камеры из CR3 (бокс PRVW, 1620×1080): лежит в начале файла и декодируется за
/// миллисекунды — для миниатюр отбора и как эталон яркости камеры. None — не CR3 или бокса нет.
pub fn camera_preview_fast(path: &Path) -> Option<image::DynamicImage> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = Vec::with_capacity(512 * 1024);
    (&mut f).take(512 * 1024).read_to_end(&mut head).ok()?;
    if head.get(4..12)? != b"ftypcrx " {
        return None;
    }
    let i = head.windows(4).position(|w| w == b"PRVW")?;
    // 'PRVW', 4 байта нулей, u16 1, u16 ширина, u16 высота, u16 1, u32 размер JPEG, JPEG.
    let size = u32::from_be_bytes(head.get(i + 16..i + 20)?.try_into().ok()?) as usize;
    if !(1000..=16_000_000).contains(&size) {
        return None;
    }
    let mut jpeg = head.get(i + 20..)?.to_vec();
    if jpeg.len() < size {
        let mut rest = vec![0u8; size - jpeg.len()];
        f.read_exact(&mut rest).ok()?;
        jpeg.extend_from_slice(&rest);
    }
    jpeg.truncate(size);
    if jpeg.get(0..2)? != [0xFF, 0xD8] {
        return None;
    }
    image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg).ok()
}

/// JPEG камеры для сравнения: сначала быстрый средний (CR3), иначе полный встроенный.
pub fn camera_jpeg(path: &Path) -> Result<image::DynamicImage> {
    match camera_preview_fast(path) {
        Some(img) => Ok(img),
        None => embedded_preview(path),
    }
}

/// Встроенное в RAW превью (JPEG камеры) — для быстрых миниатюр и режима «до».
pub fn embedded_preview(path: &Path) -> Result<image::DynamicImage> {
    let src = RawSource::new(path)?;
    let decoder = rawler::get_decoder(&src).map_err(|e| anyhow!("{e}"))?;
    let params = RawDecodeParams::default();
    if let Ok(Some(img)) = decoder.preview_image(&src, &params) {
        return Ok(img);
    }
    if let Ok(Some(img)) = decoder.full_image(&src, &params) {
        return Ok(img);
    }
    if let Ok(Some(img)) = decoder.thumbnail_image(&src, &params) {
        return Ok(img);
    }
    bail!("в файле нет встроенного превью")
}

/// Уменьшенное линейное изображение в цветах камеры (без баланса белого), 0..1.
/// Каждый пиксель — среднее блока f×f байеровских ячеек 2×2.
#[derive(Clone)]
pub struct LinearImage {
    pub width: usize,
    pub height: usize,
    /// RGB построчно.
    pub data: Vec<f32>,
}

impl LinearImage {
    pub fn px(&self, x: usize, y: usize) -> [f32; 3] {
        let i = (y * self.width + x) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

pub fn superpixel(frame: &RawFrame, max_side: usize) -> LinearImage {
    use rayon::prelude::*;
    let cells_w = frame.width / 2;
    let cells_h = frame.height / 2;
    let f = cells_w.max(cells_h).div_ceil(max_side).max(1);
    let ow = cells_w / f;
    let oh = cells_h / f;
    let mut out = vec![0f32; ow * oh * 3];
    let mut scale = [0f32; 4];
    for i in 0..4 {
        scale[i] = 1.0 / (frame.white - frame.black[i]).max(1.0);
    }
    out.par_chunks_mut(ow * 3).enumerate().for_each(|(oy, row)| {
        for ox in 0..ow {
            let mut acc = [0f32; 3];
            let mut cnt = [0f32; 3];
            for by in 0..f {
                for bx in 0..f {
                    let y0 = (oy * f + by) * 2;
                    let x0 = (ox * f + bx) * 2;
                    for i in 0..4 {
                        let (dy, dx) = (i / 2, i % 2);
                        let v = frame.data[(y0 + dy) * frame.width + x0 + dx] as f32;
                        let c = frame.cfa[i] as usize;
                        acc[c] += ((v - frame.black[i]) * scale[i]).clamp(0.0, 1.0);
                        cnt[c] += 1.0;
                    }
                }
            }
            for c in 0..3 {
                row[ox * 3 + c] = acc[c] / cnt[c].max(1.0);
            }
        }
    });
    LinearImage { width: ow, height: oh, data: out }
}
