//! Детекция лиц нейросетью YuNet (OpenCV Zoo, MIT): на видеокарте через ONNX Runtime + DirectML,
//! запасной путь — tract на процессоре.

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tract_onnx::prelude::*;

static MODEL_BYTES: &[u8] = include_bytes!("../../models/face_detection_yunet_2023mar.onnx");

/// Лицо в координатах исходного (анализируемого) изображения.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Face {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    /// Правый глаз, левый глаз, кончик носа, правый и левый уголки рта (с точки зрения человека).
    pub landmarks: [[f32; 2]; 5],
}

impl Face {
    pub fn area(&self) -> f32 {
        self.w * self.h
    }
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
    /// То же лицо на картинке другого размера (масштаб `s`).
    pub fn scaled(&self, s: f32) -> Face {
        Face { x: self.x * s, y: self.y * s, w: self.w * s, h: self.h * s, score: self.score, landmarks: self.landmarks.map(|p| [p[0] * s, p[1] * s]) }
    }
}

/// Изображение для детектора: sRGB 8 бит, RGB построчно.
pub struct Rgb8<'a> {
    pub width: usize,
    pub height: usize,
    pub data: &'a [u8],
}

const DETECT_LONG: usize = 1024;
const SCORE_MIN: f32 = 0.62;
const NMS_IOU: f32 = 0.3;

type Plan = Arc<TypedRunnableModel>;
type GpuNet = Option<Arc<Mutex<ort::session::Session>>>;

/// Сессии на видеокарте по размеру входа (обычно два: горизонтальный и вертикальный кадр).
/// `None` — для этого размера DirectML не поднялся, считаем на процессоре.
fn gpu_net(w: usize, h: usize) -> GpuNet {
    static S: OnceLock<Mutex<HashMap<(usize, usize), GpuNet>>> = OnceLock::new();
    if crate::nn::cpu_only() {
        return None;
    }
    let mut map = S.get_or_init(Default::default).lock().unwrap();
    map.entry((w, h))
        .or_insert_with(|| {
            let built = crate::nn::with_input_shape(MODEL_BYTES, &[1, 3, h as i64, w as i64]).and_then(|m| crate::nn::gpu_session(&m));
            match built {
                Ok(s) => Some(Arc::new(Mutex::new(s))),
                Err(e) => {
                    eprintln!("YuNet на видеокарте недоступен ({e}), считаю на процессоре");
                    None
                }
            }
        })
        .clone()
}

fn run_net(input: &[f32], w: usize, h: usize) -> Result<crate::nn::Outputs> {
    if let Some(net) = gpu_net(w, h) {
        match crate::nn::run(&net, [1, 3, h, w], input) {
            Ok(o) => return Ok(o),
            Err(e) => eprintln!("YuNet на видеокарте: {e}, считаю на процессоре"),
        }
    }
    let t = tract_ndarray::Array4::from_shape_vec((1, 3, h, w), input.to_vec()).map_err(|e| anyhow!("YuNet: {e}"))?;
    let outputs = plan_for(w, h)?.run(tvec!(Tensor::from(t).into())).map_err(|e| anyhow!("YuNet: {e}"))?;
    outputs
        .iter()
        .map(|o| {
            let v = o.to_plain_array_view::<f32>().map_err(|e| anyhow!("YuNet: {e}"))?;
            Ok((v.shape().to_vec(), v.iter().copied().collect()))
        })
        .collect()
}

fn plans() -> &'static Mutex<HashMap<(usize, usize), Plan>> {
    static P: OnceLock<Mutex<HashMap<(usize, usize), Plan>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

fn plan_for(w: usize, h: usize) -> Result<Plan> {
    if let Some(p) = plans().lock().unwrap().get(&(w, h)) {
        return Ok(p.clone());
    }
    let mut model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))
        .map_err(|e| anyhow!("YuNet: {e}"))?;
    // В модели записаны формы промежуточных тензоров для входа 640×640 — сбрасываем их,
    // чтобы tract вывел формы заново под наш размер.
    for node in model.nodes_mut() {
        for o in node.outputs.iter_mut() {
            o.fact = InferenceFact::default();
        }
    }
    let model = model
        .with_input_fact(0, f32::fact([1, 3, h, w]).into())
        .map_err(|e| anyhow!("YuNet: {e}"))?
        .into_optimized()
        .map_err(|e| anyhow!("YuNet: {e:?}"))?
        .into_runnable_with_options(&RunOptions::default())
        .map_err(|e| anyhow!("YuNet: {e}"))?;
    plans().lock().unwrap().insert((w, h), model.clone());
    Ok(model)
}

fn iou(a: &Face, b: &Face) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    inter / (a.area() + b.area() - inter).max(1e-6)
}

/// Билинейная выборка пикселя RGB8 (координаты в пикселях, центры на +0.5).
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
        out[c] = a * (1.0 - ty) + b * ty;
    }
    out
}

pub fn detect(img: &Rgb8) -> Result<Vec<Face>> {
    let scale = DETECT_LONG as f32 / img.width.max(img.height) as f32;
    let scale = scale.min(1.0);
    let sw = ((img.width as f32 * scale).round() as usize).max(32);
    let sh = ((img.height as f32 * scale).round() as usize).max(32);
    let (pw, ph) = (sw.next_multiple_of(32), sh.next_multiple_of(32));
    let inv = 1.0 / scale;
    // YuNet ждёт BGR 0..255 в NCHW; поле за пределами кадра — нули.
    let plane = pw * ph;
    let mut input = vec![0f32; 3 * plane];
    for y in 0..sh {
        for x in 0..sw {
            let px = sample(img, (x as f32 + 0.5) * inv, (y as f32 + 0.5) * inv);
            for c in 0..3 {
                input[c * plane + y * pw + x] = px[2 - c];
            }
        }
    }
    let outputs = run_net(&input, pw, ph)?;

    // Выходы различаем по форме: [1,N,1] — cls и obj (порядок не важен, score = sqrt(cls*obj)),
    // [1,N,4] — bbox, [1,N,10] — ключевые точки. N однозначно задаёт шаг сетки.
    let mut by_n: HashMap<usize, (Vec<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>)> = HashMap::new();
    for (shape, data) in outputs {
        if shape.len() != 3 {
            continue;
        }
        let e = by_n.entry(shape[1]).or_insert((Vec::new(), None, None));
        match shape[2] {
            1 => e.0.push(data),
            4 => e.1 = Some(data),
            10 => e.2 = Some(data),
            _ => {}
        }
    }

    let mut faces = Vec::new();
    for stride in [8usize, 16, 32] {
        let cols = pw / stride;
        let rows = ph / stride;
        let n = cols * rows;
        let Some((scores, Some(bbox), Some(kps))) = by_n.get(&n) else { continue };
        if scores.len() < 2 {
            continue;
        }
        let s = stride as f32;
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                let score = (scores[0][i].clamp(0.0, 1.0) * scores[1][i].clamp(0.0, 1.0)).sqrt();
                if score < SCORE_MIN {
                    continue;
                }
                let cx = (c as f32 + bbox[i * 4]) * s;
                let cy = (r as f32 + bbox[i * 4 + 1]) * s;
                let w = bbox[i * 4 + 2].exp() * s;
                let h = bbox[i * 4 + 3].exp() * s;
                let mut lm = [[0f32; 2]; 5];
                for k in 0..5 {
                    lm[k] = [(kps[i * 10 + 2 * k] + c as f32) * s * inv, (kps[i * 10 + 2 * k + 1] + r as f32) * s * inv];
                }
                faces.push(Face { x: (cx - w / 2.0) * inv, y: (cy - h / 2.0) * inv, w: w * inv, h: h * inv, score, landmarks: lm });
            }
        }
    }
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Face> = Vec::new();
    for f in faces {
        if f.w < 8.0 || f.h < 8.0 {
            continue;
        }
        if keep.iter().all(|k| iou(k, &f) < NMS_IOU) {
            keep.push(f);
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_runs_on_blank_image() {
        let data = vec![128u8; 600 * 400 * 3];
        let img = Rgb8 { width: 600, height: 400, data: &data };
        let faces = detect(&img).unwrap();
        assert!(faces.is_empty());
    }
}
