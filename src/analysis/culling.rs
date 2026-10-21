//! Автоотбор: разбивка на серии (похожие кадры подряд) и выбор 1–2 лучших в каждой.
//! Лучший кадр — с открытыми глазами у всех заметных лиц, резким лицом и без отвернувшихся.

use super::landmarks::FaceMetrics;
use serde::{Deserialize, Serialize};

/// Данные кадра для отбора.
#[derive(Clone, Debug)]
pub struct CullInput {
    pub time: Option<f64>,
    pub landscape: bool,
    /// Нормированная миниатюра яркости (для сравнения кадров).
    pub thumb: Vec<f32>,
    /// Заметные лица: (доля площади кадра, метрики, резкость лица).
    pub faces: Vec<(f32, FaceMetrics, f32)>,
    /// Резкость самого резкого участка кадра.
    pub sharpness: f32,
    /// Абсолютная чёткость кадра и главного лица (≈2 — резко, ≈1 — размыто; 0 — неизвестно).
    pub clarity: f32,
    pub face_clarity: f32,
    /// Лог-средняя яркость (для поиска технического брака).
    pub key: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CullResult {
    pub series: usize,
    pub series_len: usize,
    pub score: f32,
    pub pick: bool,
    /// Почему кадр не выбран (или на что обратить внимание).
    pub reasons: Vec<String>,
    /// Худшая открытость глаз среди заметных лиц (0..1).
    pub eyes: Option<f32>,
}

/// Пороги «размыто» подобраны по 100%-кропам двух реальных съёмок (команда `photostab blur`):
/// кадр целиком — если даже самый чёткий его участок ниже `BLUR_FRAME`; люди — если самое
/// чёткое из заметных лиц ниже `BLUR_FACE`, а по-настоящему резкого в кадре тоже нет
/// (иначе фокус намеренно на другом — на картине, на руках).
pub const BLUR_FRAME: f32 = 1.72;
pub const BLUR_FACE: f32 = 1.40;
const FACE_FRAME_SHARP: f32 = 2.3;

pub fn is_blurred(f: &CullInput) -> bool {
    let face = f.face_clarity > 0.0 && f.face_clarity < BLUR_FACE && f.clarity < FACE_FRAME_SHARP;
    let frame = f.clarity > 0.0 && f.clarity < BLUR_FRAME;
    face || frame
}

const SERIES_GAP_S: f64 = 8.0;
const SERIES_SIM: f32 = 0.72;
const REPEAT_SIM: f32 = 0.9;
const REPEAT_GAP_S: f64 = 90.0;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Корреляция двух миниатюр (−1..1).
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / a.len() as f32
}

/// Открытость глаз лица 0..1 (у повёрнутой головы смотрим на глаз, обращённый к камере).
pub fn eye_open(m: &FaceMetrics) -> f32 {
    let e = if m.yaw.abs() < 0.35 { m.ear_left.min(m.ear_right) } else { m.ear_left.max(m.ear_right) };
    smoothstep(0.09, 0.19, e)
}

fn face_terms(f: &CullInput) -> (Option<f32>, f32) {
    if f.faces.is_empty() {
        return (None, 1.0);
    }
    let biggest = f.faces.iter().map(|x| x.0).fold(0.0, f32::max);
    let main: Vec<&(f32, FaceMetrics, f32)> = f.faces.iter().filter(|x| x.0 >= biggest * 0.25).collect();
    let eyes: Vec<f32> = main.iter().map(|x| eye_open(&x.1)).collect();
    let min = eyes.iter().copied().fold(1.0, f32::min);
    let mean = eyes.iter().sum::<f32>() / eyes.len() as f32;
    let look = main.iter().map(|x| 1.0 - 0.3 * smoothstep(0.45, 0.8, x.1.yaw.abs())).fold(1.0, f32::min);
    (Some(min), (0.75 * min + 0.25 * mean) * look)
}

fn face_sharp(f: &CullInput) -> Option<f32> {
    let biggest = f.faces.iter().map(|x| x.0).fold(0.0, f32::max);
    let v: Vec<f32> = f.faces.iter().filter(|x| x.0 >= biggest * 0.25).map(|x| x.2).collect();
    (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
}

/// Кадр для разбивки на серии: время, ориентация, миниатюра.
pub struct SeriesItem<'a> {
    pub time: Option<f64>,
    pub landscape: bool,
    pub thumb: &'a [f32],
}

/// Серии: подряд снятые похожие кадры (номера кадров в порядке съёмки).
pub fn series(items: &[SeriesItem]) -> Vec<Vec<usize>> {
    let n = items.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| match (items[a].time, items[b].time) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        _ => a.cmp(&b),
    });
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (k, &i) in order.iter().enumerate() {
        let same = k > 0 && {
            let p = order[k - 1];
            let gap = match (items[p].time, items[i].time) {
                (Some(a), Some(b)) => b - a,
                _ => 0.0,
            };
            let sim = similarity(items[p].thumb, items[i].thumb);
            items[p].landscape == items[i].landscape
                && ((gap <= SERIES_GAP_S && sim >= SERIES_SIM) || (gap <= REPEAT_GAP_S && sim >= REPEAT_SIM))
        };
        if !same {
            members.push(Vec::new());
        }
        members.last_mut().unwrap().push(i);
    }
    members
}

/// Разбивка на серии и выбор лучших.
pub fn cull(items: &[CullInput]) -> Vec<CullResult> {
    let n = items.len();
    let si: Vec<SeriesItem> = items.iter().map(|it| SeriesItem { time: it.time, landscape: it.landscape, thumb: &it.thumb }).collect();
    let members = series(&si);

    let mut out: Vec<CullResult> = vec![CullResult::default(); n];
    for (s, m) in members.iter().enumerate() {
        let max_sharp = m.iter().map(|&i| items[i].sharpness).fold(1e-6, f32::max);
        let max_face_sharp = m.iter().filter_map(|&i| face_sharp(&items[i])).fold(1e-6, f32::max);
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for &i in m {
            let it = &items[i];
            let (eyes, face) = face_terms(it);
            let sharp_rel = match face_sharp(it) {
                Some(fs) => fs / max_face_sharp,
                None => it.sharpness / max_sharp,
            };
            let sharp_term = sharp_rel.clamp(0.0, 1.0).powf(0.7);
            let junk = it.key < 0.007 && it.faces.is_empty();
            let blurred = is_blurred(it);
            let score = if junk || blurred { 0.0 } else { face * sharp_term };
            let mut reasons = Vec::new();
            if junk {
                reasons.push("технический брак (почти чёрный кадр)".to_string());
            }
            if blurred {
                reasons.push("размыто".to_string());
            }
            if let Some(e) = eyes {
                if e < 0.5 {
                    reasons.push("закрыты глаза".into());
                }
            }
            if m.len() > 1 && sharp_rel < 0.7 && !blurred {
                reasons.push("смазано по сравнению с серией".into());
            }
            if it.faces.iter().any(|f| f.1.yaw.abs() > 0.6) && face < 0.8 {
                reasons.push("отвернулся".into());
            }
            out[i] = CullResult { series: s, series_len: m.len(), score, pick: false, reasons, eyes };
            scored.push((i, score));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let (best, best_score) = scored[0];
        if best_score > 0.0 {
            out[best].pick = true;
        }
        // Второй кадр: почти так же хорош и заметно отличается.
        if m.len() >= 3 {
            if let Some(&(second, s2)) = scored.iter().skip(1).find(|(j, _)| similarity(&items[best].thumb, &items[*j].thumb) < 0.93) {
                if s2 >= best_score * 0.85 && s2 > 0.3 {
                    out[second].pick = true;
                }
            }
        }
        for &(i, _) in &scored {
            if !out[i].pick && out[i].reasons.is_empty() {
                out[i].reasons.push("дубль серии".into());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(ear: f32) -> FaceMetrics {
        FaceMetrics { ear_right: ear, ear_left: ear, mouth: 0.05, yaw: 0.0, presence: 1.0 }
    }

    fn frame(t: f64, ear: Option<f32>, sharp: f32) -> CullInput {
        CullInput {
            time: Some(t),
            landscape: true,
            thumb: vec![1.0, -1.0, 1.0, -1.0],
            faces: ear.map(|e| vec![(0.02, metrics(e), sharp)]).unwrap_or_default(),
            sharpness: sharp,
            clarity: 1.9,
            face_clarity: 0.0,
            key: 0.05,
        }
    }

    #[test]
    fn picks_open_eyes_in_burst() {
        let items = vec![frame(0.0, Some(0.28), 1.0), frame(0.5, Some(0.03), 1.0), frame(1.0, Some(0.27), 0.6)];
        let r = cull(&items);
        assert!(r.iter().all(|x| x.series == 0));
        assert!(r[0].pick);
        assert!(!r[1].pick);
        assert!(r[1].reasons.iter().any(|s| s.contains("глаза")));
        assert!(!r[2].pick);
    }

    #[test]
    fn blurred_single_frame_is_rejected() {
        let mut a = frame(0.0, Some(0.28), 1.0);
        a.clarity = 1.05;
        a.face_clarity = 1.02;
        let r = cull(&[a]);
        assert!(!r[0].pick);
        assert!(r[0].reasons.iter().any(|s| s == "размыто"), "{:?}", r[0].reasons);
    }

    #[test]
    fn different_scenes_are_separate_series() {
        let mut a = frame(0.0, None, 1.0);
        let mut b = frame(30.0, None, 1.0);
        a.thumb = vec![1.0, 1.0, -1.0, -1.0];
        b.thumb = vec![-1.0, 1.0, 1.0, -1.0];
        let r = cull(&[a, b]);
        assert_ne!(r[0].series, r[1].series);
        assert!(r[0].pick && r[1].pick);
    }
}
