//! Пакетная обработка: анализ всех кадров, согласование по сценам, проявка и экспорт.

use crate::analysis::{self, Analysis};
use crate::develop::{self, DevelopParams};
use crate::export;
use crate::gpu::Gpu;
use crate::raw::{self, CameraColor, LinearImage, PhotoMeta};
use anyhow::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Item {
    pub path: PathBuf,
    pub meta: PhotoMeta,
    pub small: LinearImage,
    pub cam: CameraColor,
    pub analysis: Analysis,
    pub group: usize,
}

/// Анализ всех файлов параллельно (CPU) и согласование ББ по сценам.
pub fn analyze_all(files: &[PathBuf], progress: &(dyn Fn(usize, usize) + Sync)) -> (Vec<Item>, Vec<(PathBuf, String)>) {
    let done = AtomicUsize::new(0);
    let results: Vec<Result<(PathBuf, PhotoMeta, LinearImage, CameraColor, Analysis), (PathBuf, String)>> = files
        .par_iter()
        .map(|p| {
            let r = crate::prof::time("0 декодирование RAW", || raw::decode(p)).map(|frame| {
                let cam = crate::prof::time("0б JPEG камеры", || raw::camera_preview_fast(p));
                let (small, a) = analysis::analyze(&frame, cam.as_ref());
                (p.clone(), frame.meta, small, frame.color, a)
            });
            progress(done.fetch_add(1, Ordering::Relaxed) + 1, files.len());
            r.map_err(|e| (p.clone(), e.to_string()))
        })
        .collect();
    let mut ok = Vec::new();
    let mut failed = Vec::new();
    for r in results {
        match r {
            Ok(v) => ok.push(v),
            Err(e) => failed.push(e),
        }
    }
    let mut tmp: Vec<(PhotoMeta, LinearImage, CameraColor, Analysis)> =
        ok.iter().map(|(_, m, s, c, a)| (m.clone(), s.clone(), c.clone(), a.clone())).collect();
    let groups = crate::prof::time("12 согласование групп", || analysis::harmonize(&mut tmp));
    let mut items: Vec<Item> = ok
        .into_iter()
        .zip(tmp)
        .zip(groups)
        .map(|((ok, (_, _, _, a)), g)| Item { path: ok.0, meta: ok.1, small: ok.2, cam: ok.3, analysis: a, group: g })
        .collect();

    // Автоотбор: серии и лучшие кадры.
    let inputs: Vec<analysis::culling::CullInput> = items
        .iter()
        .map(|it| {
            let a = &it.analysis;
            let area = (a.width * a.height).max(1) as f32;
            analysis::culling::CullInput {
                time: it.meta.capture_seconds(),
                landscape: a.width >= a.height,
                thumb: a.thumb.clone(),
                faces: a
                    .faces
                    .iter()
                    .zip(&a.face_metrics)
                    .zip(&a.face_sharp)
                    .map(|((f, m), s)| (f.area() / area, *m, *s))
                    .collect(),
                sharpness: a.sharpness,
                clarity: a.clarity,
                face_clarity: best_face_clarity(a),
                key: a.exposure.key,
            }
        })
        .collect();
    for (it, r) in items.iter_mut().zip(analysis::culling::cull(&inputs)) {
        it.analysis.cull = r;
    }
    (items, failed)
}

/// Чёткость самого чёткого из заметных лиц (не меньше четверти самого крупного): если в фокусе
/// хоть одно заметное лицо, кадр не смазан. 0 — лиц нет или не оценено.
pub fn best_face_clarity(a: &Analysis) -> f32 {
    let biggest = a.faces.iter().map(|f| f.area()).fold(0.0f32, f32::max);
    a.faces
        .iter()
        .zip(&a.face_clarity)
        .filter(|(f, c)| f.area() >= 0.25 * biggest && **c > 0.0)
        .map(|(_, c)| *c)
        .fold(0.0f32, f32::max)
}

pub struct ExportOptions {
    pub quality: u8,
    pub long_side: Option<u32>,
    /// Имена файлов: `None` — как у RAW, `Some(основа)` — «основа_001.jpg», «основа_002.jpg»… по
    /// порядку кадров.
    pub sequence: Option<String>,
}

/// Имя кадра `i` (с нуля) из `n` при нумерации: «основа_001.jpg», не меньше трёх цифр.
pub fn sequence_name(base: &str, i: usize, n: usize) -> String {
    let width = n.to_string().len().max(3);
    format!("{base}_{:0width$}.jpg", i + 1)
}

/// Проявка на GPU и запись JPEG. Декод и кодирование идут параллельно на CPU.
pub fn export_all(
    gpu: &Gpu,
    jobs: &[(PathBuf, DevelopParams)],
    out_dir: &Path,
    opts: &ExportOptions,
    progress: &(dyn Fn(usize, usize, &Path) + Sync),
) -> Vec<(PathBuf, Result<PathBuf>)> {
    let done = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().expect("пул потоков");
    pool.install(|| {
        jobs.par_iter()
            .enumerate()
            .map(|(i, (path, params))| {
                let r = (|| -> Result<PathBuf> {
                    let frame = raw::decode(path)?;
                    let recipe = develop::recipe(params, &frame.color, frame.meta.orientation, opts.long_side);
                    let img = gpu.develop_raw(&frame, &recipe)?;
                    drop(frame.data);
                    let name = match &opts.sequence {
                        Some(base) => sequence_name(base, i, jobs.len()),
                        None => path.file_stem().unwrap_or_default().to_string_lossy().to_string() + ".jpg",
                    };
                    let out = out_dir.join(name);
                    export::write_jpeg(&out, &img, opts.quality, &frame.meta)?;
                    Ok(out)
                })();
                progress(done.fetch_add(1, Ordering::Relaxed) + 1, jobs.len(), path);
                (path.clone(), r)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_names_sort_in_shooting_order() {
        assert_eq!(sequence_name("CU October 2026", 0, 81), "CU October 2026_001.jpg");
        assert_eq!(sequence_name("CU October 2026", 80, 81), "CU October 2026_081.jpg");
        assert_eq!(sequence_name("Свадьба", 9, 1200), "Свадьба_0010.jpg");
    }
}
