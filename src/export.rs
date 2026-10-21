//! Запись JPEG: EXIF из исходного RAW и компактный ICC-профиль sRGB.

use crate::develop::Rgba;
use crate::raw::PhotoMeta;
use anyhow::{Context, Result, anyhow};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use std::io::BufWriter;
use std::path::Path;
use std::sync::OnceLock;

pub fn encode_jpeg(img: &Rgba, quality: u8, meta: Option<&PhotoMeta>) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((img.width * img.height) as usize / 3);
    {
        let mut enc = Encoder::new(&mut out, quality);
        configure(&mut enc, quality, img, meta)?;
        enc.encode(&img.data, img.width as u16, img.height as u16, ColorType::Rgba)
            .map_err(|e| anyhow!("JPEG: {e}"))?;
    }
    Ok(out)
}

pub fn write_jpeg(path: &Path, img: &Rgba, quality: u8, meta: &PhotoMeta) -> Result<()> {
    let file = std::fs::File::create(path).with_context(|| format!("не удалось создать {}", path.display()))?;
    let mut enc = Encoder::new(BufWriter::new(file), quality);
    configure(&mut enc, quality, img, Some(meta))?;
    enc.encode(&img.data, img.width as u16, img.height as u16, ColorType::Rgba)
        .map_err(|e| anyhow!("JPEG: {e}"))?;
    Ok(())
}

fn configure<W: jpeg_encoder::JfifWrite>(enc: &mut Encoder<W>, quality: u8, img: &Rgba, meta: Option<&PhotoMeta>) -> Result<()> {
    enc.set_sampling_factor(if quality >= 97 { SamplingFactor::R_4_4_4 } else { SamplingFactor::R_4_2_0 });
    enc.set_optimized_huffman_tables(true);
    enc.add_icc_profile(srgb_icc()).map_err(|e| anyhow!("ICC: {e}"))?;
    if let Some(m) = meta {
        enc.add_exif_metadata(&build_exif(m, img.width, img.height)).map_err(|e| anyhow!("EXIF: {e}"))?;
    }
    Ok(())
}

// ---------- EXIF (TIFF little-endian) ----------

enum Val {
    Ascii(String),
    Short(u16),
    Long(u32),
    Rational(u32, u32),
    SRational(i32, i32),
    Undefined(Vec<u8>),
}

impl Val {
    fn encode(&self) -> (u16, u32, Vec<u8>) {
        match self {
            Val::Ascii(s) => {
                let mut b = s.as_bytes().to_vec();
                b.push(0);
                (2, b.len() as u32, b)
            }
            Val::Short(v) => (3, 1, v.to_le_bytes().to_vec()),
            Val::Long(v) => (4, 1, v.to_le_bytes().to_vec()),
            Val::Rational(n, d) => {
                let mut b = n.to_le_bytes().to_vec();
                b.extend_from_slice(&d.to_le_bytes());
                (5, 1, b)
            }
            Val::SRational(n, d) => {
                let mut b = n.to_le_bytes().to_vec();
                b.extend_from_slice(&d.to_le_bytes());
                (10, 1, b)
            }
            Val::Undefined(v) => (7, v.len() as u32, v.clone()),
        }
    }
}

/// Записывает IFD начиная с `base` (смещение от начала TIFF). Возвращает байты IFD с данными.
fn write_ifd(mut entries: Vec<(u16, Val)>, base: u32) -> Vec<u8> {
    entries.sort_by_key(|e| e.0);
    let n = entries.len() as u32;
    let table_len = 2 + 12 * n + 4;
    let mut data_off = base + table_len;
    let mut table = Vec::new();
    let mut data = Vec::new();
    table.extend_from_slice(&(n as u16).to_le_bytes());
    for (tag, v) in entries {
        let (typ, count, bytes) = v.encode();
        table.extend_from_slice(&tag.to_le_bytes());
        table.extend_from_slice(&typ.to_le_bytes());
        table.extend_from_slice(&count.to_le_bytes());
        if bytes.len() <= 4 {
            let mut inline = bytes.clone();
            inline.resize(4, 0);
            table.extend_from_slice(&inline);
        } else {
            table.extend_from_slice(&data_off.to_le_bytes());
            data.extend_from_slice(&bytes);
            if data.len() % 2 == 1 {
                data.push(0);
            }
            data_off = base + table_len + data.len() as u32;
        }
    }
    table.extend_from_slice(&0u32.to_le_bytes());
    table.extend_from_slice(&data);
    table
}

pub fn build_exif(m: &PhotoMeta, width: u32, height: u32) -> Vec<u8> {
    let mut exif: Vec<(u16, Val)> = vec![
        (0x9000, Val::Undefined(b"0232".to_vec())),
        (0xA001, Val::Short(1)),
        (0xA002, Val::Long(width)),
        (0xA003, Val::Long(height)),
    ];
    if let Some((n, d)) = m.exposure_time {
        exif.push((0x829A, Val::Rational(n, d)));
    }
    if let Some((n, d)) = m.fnumber {
        exif.push((0x829D, Val::Rational(n, d)));
    }
    if let Some(iso) = m.iso {
        exif.push((0x8827, Val::Short(iso.min(65535) as u16)));
    }
    if let Some(dt) = &m.datetime_original {
        exif.push((0x9003, Val::Ascii(dt.clone())));
        exif.push((0x9004, Val::Ascii(dt.clone())));
    }
    if let Some(o) = &m.offset_time_original {
        exif.push((0x9011, Val::Ascii(o.clone())));
    }
    if let Some((n, d)) = m.exposure_bias {
        exif.push((0x9204, Val::SRational(n, d)));
    }
    if let Some(f) = m.flash {
        exif.push((0x9209, Val::Short(f)));
    }
    if let Some((n, d)) = m.focal_length {
        exif.push((0x920A, Val::Rational(n, d)));
    }
    if let Some(s) = &m.subsec_original {
        exif.push((0x9291, Val::Ascii(s.clone())));
    }
    if let Some(s) = &m.serial_number {
        exif.push((0xA431, Val::Ascii(s.clone())));
    }
    if let Some(l) = &m.lens {
        exif.push((0xA434, Val::Ascii(l.clone())));
    }

    let mut ifd0: Vec<(u16, Val)> = vec![
        (0x010F, Val::Ascii(m.make.clone())),
        (0x0110, Val::Ascii(m.model.clone())),
        (0x0112, Val::Short(1)),
        (0x011A, Val::Rational(300, 1)),
        (0x011B, Val::Rational(300, 1)),
        (0x0128, Val::Short(2)),
        (0x0131, Val::Ascii(format!("PhotoStab {}", env!("CARGO_PKG_VERSION")))),
    ];
    if let Some(dt) = &m.datetime_original {
        ifd0.push((0x0132, Val::Ascii(dt.clone())));
    }
    if let Some(a) = &m.artist {
        ifd0.push((0x013B, Val::Ascii(a.clone())));
    }
    if let Some(c) = &m.copyright {
        ifd0.push((0x8298, Val::Ascii(c.clone())));
    }
    // Размер IFD0 нужен до записи, чтобы знать, где начнётся Exif IFD: считаем в два прохода.
    ifd0.push((0x8769, Val::Long(0)));
    let probe = write_ifd(clone_entries(&ifd0), 8);
    let exif_off = 8 + probe.len() as u32;
    if let Some(e) = ifd0.iter_mut().find(|e| e.0 == 0x8769) {
        e.1 = Val::Long(exif_off);
    }

    let mut out = b"II*\0".to_vec();
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&write_ifd(ifd0, 8));
    out.extend_from_slice(&write_ifd(exif, exif_off));
    out
}

fn clone_entries(e: &[(u16, Val)]) -> Vec<(u16, Val)> {
    e.iter()
        .map(|(t, v)| {
            let v2 = match v {
                Val::Ascii(s) => Val::Ascii(s.clone()),
                Val::Short(x) => Val::Short(*x),
                Val::Long(x) => Val::Long(*x),
                Val::Rational(a, b) => Val::Rational(*a, *b),
                Val::SRational(a, b) => Val::SRational(*a, *b),
                Val::Undefined(b) => Val::Undefined(b.clone()),
            };
            (*t, v2)
        })
        .collect()
}

// ---------- ICC sRGB (v2, ~2.6 КБ) ----------

fn s15(v: f64) -> [u8; 4] {
    ((v * 65536.0).round() as i32).to_be_bytes()
}

fn xyz_tag(x: f64, y: f64, z: f64) -> Vec<u8> {
    let mut t = b"XYZ \0\0\0\0".to_vec();
    t.extend_from_slice(&s15(x));
    t.extend_from_slice(&s15(y));
    t.extend_from_slice(&s15(z));
    t
}

pub fn srgb_icc() -> &'static [u8] {
    static ICC: OnceLock<Vec<u8>> = OnceLock::new();
    ICC.get_or_init(|| {
        let desc_text = b"sRGB IEC61966-2.1 (PhotoStab)\0";
        let mut desc = b"desc\0\0\0\0".to_vec();
        desc.extend_from_slice(&(desc_text.len() as u32).to_be_bytes());
        desc.extend_from_slice(desc_text);
        desc.extend_from_slice(&[0u8; 4 + 4 + 2 + 1 + 67]);
        let mut cprt = b"text\0\0\0\0".to_vec();
        cprt.extend_from_slice(b"No copyright, use freely\0");
        let mut curv = b"curv\0\0\0\0".to_vec();
        let n = 1024u32;
        curv.extend_from_slice(&n.to_be_bytes());
        for i in 0..n {
            let v = crate::color::srgb_decode(i as f32 / (n - 1) as f32);
            curv.extend_from_slice(&((v * 65535.0).round() as u16).to_be_bytes());
        }
        let tags: Vec<(&[u8; 4], Vec<u8>)> = vec![
            (b"desc", desc),
            (b"cprt", cprt),
            (b"wtpt", xyz_tag(0.9642, 1.0, 0.8249)),
            (b"rXYZ", xyz_tag(0.4360747, 0.2225045, 0.0139322)),
            (b"gXYZ", xyz_tag(0.3850649, 0.7168786, 0.0971045)),
            (b"bXYZ", xyz_tag(0.1430804, 0.0606169, 0.7141733)),
            (b"rTRC", curv),
        ];
        let count = tags.len() as u32 + 2; // gTRC и bTRC ссылаются на ту же кривую
        let mut offset = 128 + 4 + 12 * count;
        let mut table = Vec::new();
        let mut body = Vec::new();
        let mut trc = (0u32, 0u32);
        for (sig, data) in &tags {
            table.extend_from_slice(*sig);
            table.extend_from_slice(&offset.to_be_bytes());
            table.extend_from_slice(&(data.len() as u32).to_be_bytes());
            if *sig == b"rTRC" {
                trc = (offset, data.len() as u32);
            }
            body.extend_from_slice(data);
            while body.len() % 4 != 0 {
                body.push(0);
            }
            offset = 128 + 4 + 12 * count + body.len() as u32;
        }
        for sig in [b"gTRC", b"bTRC"] {
            table.extend_from_slice(sig);
            table.extend_from_slice(&trc.0.to_be_bytes());
            table.extend_from_slice(&trc.1.to_be_bytes());
        }
        let total = 128 + 4 + table.len() + body.len();
        let mut h = Vec::with_capacity(total);
        h.extend_from_slice(&(total as u32).to_be_bytes());
        h.extend_from_slice(b"none");
        h.extend_from_slice(&0x0210_0000u32.to_be_bytes());
        h.extend_from_slice(b"mntrRGB XYZ ");
        h.extend_from_slice(&[0x07, 0xE0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]); // 2016-01-01
        h.extend_from_slice(b"acsp");
        h.extend_from_slice(&[0u8; 4 * 6]); // платформа, флаги, производитель, модель, атрибуты(8)
        h.extend_from_slice(&0u32.to_be_bytes()); // intent
        h.extend_from_slice(&s15(0.9642));
        h.extend_from_slice(&s15(1.0));
        h.extend_from_slice(&s15(0.8249));
        h.resize(128, 0);
        h.extend_from_slice(&count.to_be_bytes());
        h.extend_from_slice(&table);
        h.extend_from_slice(&body);
        h
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_parses_back() {
        let m = PhotoMeta {
            make: "Canon".into(),
            model: "EOS M50 Mark II".into(),
            iso: Some(3200),
            exposure_time: Some((1, 125)),
            fnumber: Some((28, 10)),
            datetime_original: Some("2026:10:18 19:30:05".into()),
            lens: Some("EF-M15-45mm f/3.5-6.3 IS STM".into()),
            ..Default::default()
        };
        let tiff = build_exif(&m, 6000, 4000);
        let mut data = b"Exif\0\0".to_vec();
        data.extend_from_slice(&tiff);
        assert_eq!(&tiff[0..4], b"II*\0");
        assert!(tiff.len() < 1000);
        // Проверяем, что Exif-указатель ведёт внутрь блока.
        let n0 = u16::from_le_bytes([tiff[8], tiff[9]]) as usize;
        let mut found = false;
        for i in 0..n0 {
            let e = 10 + i * 12;
            let tag = u16::from_le_bytes([tiff[e], tiff[e + 1]]);
            if tag == 0x8769 {
                let off = u32::from_le_bytes([tiff[e + 8], tiff[e + 9], tiff[e + 10], tiff[e + 11]]) as usize;
                assert!(off < tiff.len());
                let n1 = u16::from_le_bytes([tiff[off], tiff[off + 1]]);
                assert!(n1 >= 8);
                found = true;
            }
        }
        assert!(found);
    }

    #[test]
    fn icc_size_consistent() {
        let icc = srgb_icc();
        let size = u32::from_be_bytes([icc[0], icc[1], icc[2], icc[3]]) as usize;
        assert_eq!(size, icc.len());
        assert_eq!(&icc[36..40], b"acsp");
    }
}
