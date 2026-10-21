//! Значок exe: рисуется тем же кодом, что и значок окна, упаковывается в .ico и вшивается в ресурсы.

use std::io::Write;
use std::path::PathBuf;

include!("src/icon_draw.rs");

fn ico(sizes: &[u32]) -> Vec<u8> {
    let mut images: Vec<Vec<u8>> = Vec::new();
    for &s in sizes {
        let rgba = icon_rgba(s);
        let mut b = Vec::new();
        // BITMAPINFOHEADER: высота удвоена (цвет + маска).
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&(s as i32).to_le_bytes());
        b.extend_from_slice(&((s * 2) as i32).to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&32u16.to_le_bytes());
        b.extend_from_slice(&[0u8; 24]);
        for y in (0..s).rev() {
            for x in 0..s {
                let i = ((y * s + x) * 4) as usize;
                b.extend_from_slice(&[rgba[i + 2], rgba[i + 1], rgba[i], rgba[i + 3]]);
            }
        }
        let mask_row = (s.div_ceil(32) * 4) as usize;
        b.extend(std::iter::repeat_n(0u8, mask_row * s as usize));
        images.push(b);
    }
    let mut out = Vec::new();
    out.extend_from_slice(&[0, 0, 1, 0]);
    out.extend_from_slice(&(sizes.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * sizes.len() as u32;
    for (s, img) in sizes.iter().zip(&images) {
        let d = if *s >= 256 { 0u8 } else { *s as u8 };
        out.extend_from_slice(&[d, d, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&(img.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += img.len() as u32;
    }
    for img in images {
        out.extend_from_slice(&img);
    }
    out
}

fn main() {
    println!("cargo:rerun-if-changed=src/icon_draw.rs");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ico_path = out.join("photostab.ico");
    std::fs::write(&ico_path, ico(&[16, 24, 32, 48, 64, 128, 256])).unwrap();
    let rc_path = out.join("photostab.rc");
    let mut rc = std::fs::File::create(&rc_path).unwrap();
    writeln!(rc, "1 ICON \"{}\"", ico_path.display().to_string().replace('\\', "\\\\")).unwrap();
    drop(rc);
    embed_resource::compile(&rc_path, embed_resource::NONE).manifest_optional().unwrap();
}
