//! Точки лица: MediaPipe Face Mesh V2 (478 точек, Apache-2.0) в ONNX — на видеокарте через
//! ONNX Runtime + DirectML, запасной путь — tract на процессоре.
//! По ним считаются открытость глаз, рот и поворот головы — для автоотбора кадров.

use super::faces::{Face, Rgb8};
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex, OnceLock};
use tract_onnx::prelude::*;

static MODEL_BYTES: &[u8] = include_bytes!("../../models/face_landmarks.onnx");
const SIDE: usize = 256;

fn plan() -> Result<Arc<TypedRunnableModel>> {
    static P: OnceLock<std::result::Result<Arc<TypedRunnableModel>, String>> = OnceLock::new();
    P.get_or_init(|| {
        let build = || -> TractResult<Arc<TypedRunnableModel>> {
            tract_onnx::onnx()
                .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))?
                .with_input_fact(0, f32::fact([1, SIDE, SIDE, 3]).into())?
                .into_optimized()?
                .into_runnable_with_options(&RunOptions::default())
        };
        build().map_err(|e| format!("{e:?}"))
    })
    .clone()
    .map_err(|e| anyhow!("Face Mesh: {e}"))
}

fn gpu_net() -> Option<&'static Mutex<ort::session::Session>> {
    static S: OnceLock<Option<Mutex<ort::session::Session>>> = OnceLock::new();
    if crate::nn::cpu_only() {
        return None;
    }
    S.get_or_init(|| {
        let side = SIDE as i64;
        match crate::nn::with_input_shape(MODEL_BYTES, &[1, side, side, 3]).and_then(|m| crate::nn::gpu_session(&m)) {
            Ok(s) => Some(Mutex::new(s)),
            Err(e) => {
                eprintln!("Face Mesh на видеокарте недоступен ({e}), считаю на процессоре");
                None
            }
        }
    })
    .as_ref()
}

fn run_net(input: &[f32]) -> Result<crate::nn::Outputs> {
    if let Some(net) = gpu_net() {
        match crate::nn::run(net, [1, SIDE, SIDE, 3], input) {
            Ok(o) => return Ok(o),
            Err(e) => eprintln!("Face Mesh на видеокарте: {e}, считаю на процессоре"),
        }
    }
    let t = tract_ndarray::Array4::from_shape_vec((1, SIDE, SIDE, 3), input.to_vec()).map_err(|e| anyhow!("Face Mesh: {e}"))?;
    let outputs = plan()?.run(tvec!(Tensor::from(t).into())).map_err(|e| anyhow!("Face Mesh: {e}"))?;
    outputs
        .iter()
        .map(|o| {
            let v = o.to_plain_array_view::<f32>().map_err(|e| anyhow!("Face Mesh: {e}"))?;
            Ok((v.shape().to_vec(), v.iter().copied().collect()))
        })
        .collect()
}

/// Точки лица в координатах исходного изображения (z — относительная глубина).
#[derive(Clone, Debug)]
pub struct Mesh {
    pub pts: Vec<[f32; 3]>,
    pub presence: f32,
}

/// Квадратная область вокруг лица, повёрнутая так, чтобы глаза были горизонтальны (как в MediaPipe).
struct Roi {
    cx: f32,
    cy: f32,
    side: f32,
    cos: f32,
    sin: f32,
}

impl Roi {
    fn new(f: &Face) -> Self {
        let (cx, cy) = f.center();
        let [re, le, ..] = f.landmarks;
        // Правый глаз человека — слева в кадре.
        let ang = (le[1] - re[1]).atan2(le[0] - re[0]);
        Roi { cx, cy, side: f.w.max(f.h) * 1.5, cos: ang.cos(), sin: ang.sin() }
    }
    /// Координаты в кадре 0..SIDE → координаты изображения.
    fn to_image(&self, u: f32, v: f32) -> (f32, f32) {
        let x = (u / SIDE as f32 - 0.5) * self.side;
        let y = (v / SIDE as f32 - 0.5) * self.side;
        (self.cx + self.cos * x - self.sin * y, self.cy + self.sin * x + self.cos * y)
    }
}

fn sample(img: &Rgb8, fx: f32, fy: f32) -> [f32; 3] {
    let x = (fx - 0.5).clamp(0.0, (img.width - 1) as f32);
    let y = (fy - 0.5).clamp(0.0, (img.height - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(img.width - 1), (y0 + 1).min(img.height - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let p = |xx: usize, yy: usize, c: usize| img.data[(yy * img.width + xx) * 3 + c] as f32;
    let mut out = [0f32; 3];
    for c in 0..3 {
        let a = p(x0, y0, c) * (1.0 - tx) + p(x1, y0, c) * tx;
        let b = p(x0, y1, c) * (1.0 - tx) + p(x1, y1, c) * tx;
        out[c] = (a * (1.0 - ty) + b * ty) / 255.0;
    }
    out
}

pub fn run(img: &Rgb8, face: &Face) -> Result<Mesh> {
    let roi = Roi::new(face);
    let mut input = vec![0f32; SIDE * SIDE * 3];
    for v in 0..SIDE {
        for u in 0..SIDE {
            let (x, y) = roi.to_image(u as f32 + 0.5, v as f32 + 0.5);
            input[(v * SIDE + u) * 3..][..3].copy_from_slice(&sample(img, x, y));
        }
    }
    let mut pts = Vec::new();
    let mut presence = 1.0f32;
    let mut got_presence = false;
    for (_, flat) in run_net(&input)? {
        let n = flat.len();
        if n == 478 * 3 {
            pts = flat
                .chunks_exact(3)
                .map(|p| {
                    let (x, y) = roi.to_image(p[0], p[1]);
                    [x, y, p[2] * roi.side / SIDE as f32]
                })
                .collect();
        } else if n == 1 && !got_presence {
            // Первый скалярный выход — логит присутствия лица.
            let logit = flat[0];
            presence = 1.0 / (1.0 + (-logit).exp());
            got_presence = true;
        }
    }
    if pts.len() != 478 {
        return Err(anyhow!("Face Mesh: неожиданный выход"));
    }
    Ok(Mesh { pts, presence })
}

fn d(m: &Mesh, a: usize, b: usize) -> f32 {
    let (p, q) = (m.pts[a], m.pts[b]);
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

/// Метрики лица по сетке.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct FaceMetrics {
    /// Отношение высоты к ширине глаза (EAR) для правого и левого глаза человека.
    pub ear_right: f32,
    pub ear_left: f32,
    /// Открытость рта (высота / ширина).
    pub mouth: f32,
    /// Поворот головы: 0 — анфас, ±1 — профиль.
    pub yaw: f32,
    pub presence: f32,
}

pub fn metrics(m: &Mesh) -> FaceMetrics {
    let ear_r = (d(m, 160, 144) + d(m, 158, 153)) / (2.0 * d(m, 33, 133)).max(1e-3);
    let ear_l = (d(m, 385, 380) + d(m, 387, 373)) / (2.0 * d(m, 362, 263)).max(1e-3);
    let mouth = d(m, 13, 14) / d(m, 78, 308).max(1e-3);
    let (a, b) = (d(m, 1, 234), d(m, 1, 454));
    let yaw = (a - b) / (a + b).max(1e-3);
    FaceMetrics { ear_right: ear_r, ear_left: ear_l, mouth, yaw, presence: m.presence }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_loads_and_runs() {
        let data = vec![128u8; 300 * 300 * 3];
        let img = Rgb8 { width: 300, height: 300, data: &data };
        let f = Face { x: 100.0, y: 100.0, w: 100.0, h: 100.0, score: 0.9, landmarks: [[130.0, 140.0], [170.0, 140.0], [150.0, 160.0], [135.0, 180.0], [165.0, 180.0]] };
        let m = run(&img, &f).unwrap();
        assert_eq!(m.pts.len(), 478);
        // На сером квадрате лица нет.
        assert!(m.presence < 0.5, "presence {}", m.presence);
    }
}
