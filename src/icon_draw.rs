// Общий рисунок значка PhotoStab: подключается в окне программы и в build.rs.

/// Значок: тёмная скруглённая плашка с «диафрагмой». Тот же рисунок идёт в exe (build.rs).
#[allow(dead_code)]
pub fn icon_rgba(size: u32) -> Vec<u8> {
    let mut px = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    let r_corner = s * 0.22;
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            // Скруглённый квадрат.
            let dx = (fx - s / 2.0).abs() - (s / 2.0 - r_corner);
            let dy = (fy - s / 2.0).abs() - (s / 2.0 - r_corner);
            let d_box = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() - r_corner + dx.max(dy).min(0.0);
            let a_box = (0.5 - d_box).clamp(0.0, 1.0);
            // Кольцо объектива и «зрачок».
            let rr = ((fx - s / 2.0).powi(2) + (fy - s / 2.0).powi(2)).sqrt();
            let ring = (1.0 - ((rr - s * 0.30).abs() - s * 0.055).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            let pupil = (0.5 - (rr - s * 0.16)).clamp(0.0, 1.0);
            let (bg, fg, accent) = ([46.0, 46.0, 46.0], [225.0, 225.0, 225.0], [240.0, 196.0, 60.0]);
            let mut c = bg;
            for k in 0..3 {
                c[k] = c[k] * (1.0 - ring) + fg[k] * ring;
                c[k] = c[k] * (1.0 - pupil) + accent[k] * pupil;
            }
            let i = ((y * size + x) * 4) as usize;
            px[i] = c[0] as u8;
            px[i + 1] = c[1] as u8;
            px[i + 2] = c[2] as u8;
            px[i + 3] = (a_box * 255.0) as u8;
        }
    }
    px
}

