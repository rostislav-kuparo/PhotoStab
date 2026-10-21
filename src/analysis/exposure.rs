//! Автоэкспозиция: замер по сцене с весом к центру, защита светов, приоритет лиц.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExposureResult {
    pub ev: f32,
    /// Лог-средняя яркость сцены (линейная) до коррекции.
    pub key: f32,
    pub p995: f32,
    pub face_luma: Option<f32>,
    pub reason: String,
}

/// Ключ сцены (лог-средняя линейная яркость до кривой), выше которого светлый кадр с выбитыми
/// светами можно слегка притушить.
const TARGET_KEY: f32 = 0.06;
/// Ключ, ниже которого кадр без людей явно недоэкспонирован (ошибка, а не тёмный зал).
const KEY_OK_LO: f32 = 0.02;
/// Доля исправления по сцене: не вытягиваем «в серое» намеренно тёмные сцены.
const SCENE_STRENGTH: f32 = 0.7;
/// Яркость кожи (линейный Y, до кривой), при которой лица у камеры выглядят нормально. Диапазон
/// широкий: внутри него экспозицию камеры не трогаем (так задумал фотограф, и разные типы кожи
/// остаются разными), вне его тянем только до ближайшей границы. Нижняя граница подобрана по
/// реальным съёмкам: в тёмном зале нормальные лица у камеры — около 0.05.
const FACE_OK_LO: f32 = 0.035;
const FACE_OK_HI: f32 = 0.30;
/// Контровой свет: лицо намного темнее сцены — тянем его хотя бы до этой доли ключа сцены.
const FACE_VS_KEY: f32 = 0.3;
const EV_LIMIT: f32 = 2.5;

pub fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Яркостные характеристики кадра (до экспокоррекции) для согласования по группе.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameExpo {
    pub key: f32,
    pub p995: f32,
    pub face_luma: Option<f32>,
    pub ev100: Option<f32>,
    pub flash: bool,
    /// Гистограмма яркости (доли пикселей) по шкале log2 — для защиты светов.
    #[serde(default)]
    pub hist: Vec<f32>,
    /// Поправка, при которой кадр в среднем так же светел, как JPEG камеры (None — JPEG нет).
    #[serde(default)]
    pub cam_ev: Option<f32>,
}

const HIST_LO: f32 = -12.0;
const HIST_HI: f32 = 1.0;
const HIST_BINS: usize = 130;
/// Сколько кадра можно «довести до белого» ради общей экспозиции группы.
const NEW_WHITE_MAX: f32 = 0.04;

pub fn histogram(lin: &[[f32; 3]]) -> Vec<f32> {
    let mut h = vec![0f32; HIST_BINS];
    let mut n = 0f32;
    for c in lin.iter().step_by(3) {
        let l = luma(*c).max(1e-6).log2();
        let b = (((l - HIST_LO) / (HIST_HI - HIST_LO)) * HIST_BINS as f32).floor().clamp(0.0, (HIST_BINS - 1) as f32) as usize;
        h[b] += 1.0;
        n += 1.0;
    }
    h.iter_mut().for_each(|v| *v /= n.max(1.0));
    h
}

/// Порог «ярких» тонов (линейная яркость): выше него базовая кривая уже сильно сжимает контраст.
const BRIGHT: f32 = 0.8;
/// Выше этого — уже пересвечено (окна, лампы): такие участки не спасти и не учитываем.
const CLIPPED: f32 = 0.98;
/// Если ярких, но не пересвеченных тонов больше этой доли — кадр слегка затемняем.
const OVERBRIGHT_MAX: f32 = 0.2;

/// Доля кадра с яркостью в [lo, hi).
fn frac_between(hist: &[f32], lo: f32, hi: f32) -> f32 {
    if hist.is_empty() || lo >= hi {
        return 0.0;
    }
    let (lo, hi) = (lo.max(1e-9).log2(), hi.log2());
    let bw = (HIST_HI - HIST_LO) / HIST_BINS as f32;
    hist.iter()
        .enumerate()
        .filter(|(i, _)| {
            let c = HIST_LO + (*i as f32 + 0.5) * bw;
            c >= lo && c < hi
        })
        .map(|(_, v)| v)
        .sum()
}

/// Доля кадра, которая уйдёт в яркие тона только из-за осветления на `ev`.
fn new_white(hist: &[f32], ev: f32) -> f32 {
    if ev <= 0.0 {
        return 0.0;
    }
    frac_between(hist, BRIGHT * 2f32.powf(-ev), BRIGHT)
}

/// Предел экспокоррекции кадра по светам. Положительный — сколько можно осветлить, не выбив
/// новые большие яркие участки; отрицательный — кадр уже перегружен яркими тонами
/// (пятно прожектора, светлая стена во весь кадр) и его надо слегка притушить.
fn highlight_cap(hist: &[f32]) -> f32 {
    if hist.is_empty() {
        return EV_LIMIT;
    }
    let bright_at = |ev: f32| frac_between(hist, BRIGHT * 2f32.powf(-ev), CLIPPED);
    if bright_at(0.0) > OVERBRIGHT_MAX {
        let (mut a, mut b) = (-1.0f32, 0.0f32);
        for _ in 0..20 {
            let m = 0.5 * (a + b);
            if bright_at(m) > OVERBRIGHT_MAX { b = m } else { a = m }
        }
        return a;
    }
    if new_white(hist, EV_LIMIT) <= NEW_WHITE_MAX {
        return EV_LIMIT;
    }
    let (mut a, mut b) = (0.0f32, EV_LIMIT);
    for _ in 0..20 {
        let m = 0.5 * (a + b);
        if new_white(hist, m) <= NEW_WHITE_MAX { a = m } else { b = m }
    }
    a
}

/// Насколько можно подправить отдельный кадр ради лица (EV), чтобы не ломать единство группы.
const FACE_NUDGE_MAX: f32 = 0.6;

/// Поправка, возвращающая яркость лица `fy` в нормальный диапазон (0 — уже в нём).
/// `key` — ключ сцены того же кадра (или медиана по группе): для контрового света.
fn face_fix(fy: f32, key: f32) -> f32 {
    let lo = FACE_OK_LO.max(FACE_VS_KEY * key).min(FACE_OK_HI * 0.8);
    if fy < lo {
        (lo / fy.max(1e-4)).log2()
    } else if fy > FACE_OK_HI {
        (FACE_OK_HI / fy).log2()
    } else {
        0.0
    }
}

/// Среднее без крайних `trim` с каждой стороны: средняя яркость группы, как у камеры,
/// но без влияния единичных странных кадров.
fn trimmed_mean(mut v: Vec<f32>, trim: f32) -> f32 {
    v.sort_by(f32::total_cmp);
    let k = ((v.len() as f32) * trim) as usize;
    let core = &v[k..v.len() - k];
    if core.is_empty() {
        return median(v);
    }
    core.iter().sum::<f32>() / core.len() as f32
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let k = v.len() / 2;
    v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    v[k]
}

/// Лог-средняя яркость с весом к центру, 99.5-й процентиль и гистограмма.
pub fn frame_stats(lin: &[[f32; 3]], w: usize, h: usize) -> (f32, f32, Vec<f32>) {
    let r = estimate(lin, w, h, None);
    (r.key, r.p995, histogram(lin))
}

/// Общая экспозиция для группы кадров, снятых при одном свете.
///
/// Без вспышки яркость сцены одинакова, а различия в кадрах — от настроек камеры. Кадры
/// переводятся в общую шкалу по EV100, по группе выбирается одна поправка (по лицам, если есть,
/// иначе по сцене), а каждый кадр получает её плюс разницу своих настроек. Так один и тот же
/// человек при одном свете выходит одинаково светлым во всех кадрах.
pub fn group(frames: &[FrameExpo]) -> Vec<ExposureResult> {
    if frames.is_empty() {
        return Vec::new();
    }
    let scene_referred = frames.iter().all(|f| !f.flash && f.ev100.is_some());
    let offsets: Vec<f32> = if scene_referred {
        let r = median(frames.iter().map(|f| f.ev100.unwrap()).collect());
        frames.iter().map(|f| f.ev100.unwrap() - r).collect()
    } else {
        vec![0.0; frames.len()]
    };
    let k = |i: usize| 2f32.powf(offsets[i]);
    let faces: Vec<f32> = frames.iter().enumerate().filter_map(|(i, f)| f.face_luma.map(|y| y * k(i))).collect();
    let key = median(frames.iter().enumerate().map(|(i, f)| f.key * k(i)).collect());
    let p995 = median(frames.iter().enumerate().map(|(i, f)| f.p995 * k(i)).collect());

    // За основу — яркость камеры: медиана по группе поправок «как JPEG камеры», приведённых
    // к общей шкале. Дальше правим только явные ошибки камеры.
    let cams: Vec<f32> = frames.iter().enumerate().filter_map(|(i, f)| f.cam_ev.map(|e| e - offsets[i])).collect();
    let cam = if cams.is_empty() { 0.0 } else { trimmed_mean(cams, 0.15) };
    let kc = 2f32.powf(cam);
    let (fix, mut reason) = if !faces.is_empty() {
        let fy = median(faces.clone()) * kc;
        let face_key = median(frames.iter().enumerate().filter(|(_, f)| f.face_luma.is_some()).map(|(i, f)| f.key * k(i)).collect()) * kc;
        let ev = face_fix(fy, face_key);
        if ev.abs() < 0.01 {
            (0.0, format!("как у камеры ({:+.2} EV); лица в норме ({} кадр.)", cam, faces.len()))
        } else {
            (ev, format!("как у камеры ({:+.2} EV), лица ({} кадр.) {:+.2} EV", cam, faces.len(), ev))
        }
    } else {
        let ev = scene_ev(key * kc, p995 * kc);
        if ev.abs() < 0.01 {
            (0.0, format!("как у камеры ({:+.2} EV)", cam))
        } else {
            (ev, format!("как у камеры ({:+.2} EV), сцена {:+.2} EV", cam, ev))
        }
    };
    let base = cam + fix;
    if !scene_referred && frames.len() > 1 {
        reason.push_str(" (вспышка: по кадрам)");
    }

    frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut ev = if scene_referred || frames.len() == 1 {
                base + offsets[i]
            } else {
                // Со вспышкой свет зависит от расстояния: считаем по кадру, но тянем к группе.
                let c = f.cam_ev.unwrap_or(0.0);
                let g = 2f32.powf(c);
                let own = c + estimate_from(f.key * g, f.p995 * g, f.face_luma.map(|y| y * g)).ev;
                0.5 * own + 0.5 * base
            };
            let mut r = reason.clone();
            if let Some(fy) = f.face_luma.filter(|v| *v > 1e-4) {
                // Лицо этого кадра всё равно вне нормы (другой угол к свету) — полшага к границе.
                let g = 2f32.powf(ev);
                let nudge = (0.5 * face_fix(fy * g, f.key * g)).clamp(-FACE_NUDGE_MAX, FACE_NUDGE_MAX);
                if nudge.abs() > 0.01 {
                    ev += nudge;
                    r.push_str(&format!(", лицо в кадре {:+.2}", nudge));
                }
            }
            // Защита светов кадра: общая поправка группы не должна выбивать новые большие
            // области в белое (уже белые окна и лампы не в счёт). Темнее камеры из-за светов
            // не делаем: белую точку теперь задаёт само содержимое кадра.
            let mut cap = highlight_cap(&f.hist).max(0.0);
            // Кадр и так светлый целиком (прожектор, светлая стена) — не осветляем.
            if f.key >= 0.2 {
                cap = cap.min(0.0);
            }
            if ev > cap {
                r.push_str(&format!(", защита светов {:+.2}", cap - ev));
                ev = cap;
            }
            ExposureResult { ev: ev.clamp(-EV_LIMIT, EV_LIMIT), key: f.key, p995: f.p995, face_luma: f.face_luma, reason: r }
        })
        .collect()
}

/// Доля, на которую кадр подтягивается к общей яркости людей съёмки: лица / только сцена.
const BALANCE_FACES: f32 = 0.75;
const BALANCE_SCENE: f32 = 0.5;
/// Предел поправки ради общей яркости (EV).
const BALANCE_UP: f32 = 1.5;
/// Цель — яркость хорошо освещённых людей съёмки (этот процентиль), а не медиана: в съёмке с
/// тёмными уголками медиана сама «тёмная», и подтягивание к ней затемняло бы нормальные кадры.
const BALANCE_TARGET_Q: f32 = 0.65;
/// Вес кадра без лиц при усреднении яркости серии (лица — главное).
const BALANCE_KEY_WEIGHT: f32 = 0.25;
/// Вес соседней серии при сглаживании.
const BALANCE_NEIGHBOUR: f32 = 0.5;

/// Одна яркость людей на всю съёмку. После групп («как у камеры», с переводом по EV100) тёмные
/// уголки остаются тёмными: свет в разных местах зала разный. Чтобы съёмка выглядела снятой при
/// одном свете, яркость людей тёмных серий подтягивается к хорошо освещённым людям съёмки — на
/// 3/4 (по лицам) или наполовину (если лиц нет), так что атмосфера места немного остаётся.
/// Только вверх: нормальные кадры камера уже проэкспонировала как надо, их не трогаем. Яркость объекта — лица, а без них ключ сцены (в масштабе лиц). Поправка одна на
/// серию (кадры подряд в одном месте) и сглажена с соседними сериями — внутри серии ничего не
/// «дышит». Светлые участки защищены так же, как в группе.
/// `series` — серия каждого кадра, `edges` — связи соседних кадров (i, j, вес).
pub fn balance_shoot(res: &mut [ExposureResult], frames: &[FrameExpo], series: &[usize], edges: &[(usize, usize, f32)]) {
    let n = frames.len();
    if n < 3 || res.len() != n || series.len() != n {
        return;
    }
    // Ключ сцены в масштабе лиц: типичное отношение «лицо / ключ» по кадрам, где есть оба.
    let ratio = median(frames.iter().filter_map(|f| f.face_luma.filter(|y| *y > 1e-4).map(|y| (y / f.key.max(1e-4)).log2())).collect());
    let to_face = if ratio.is_finite() { ratio } else { 0.0 };
    let subject: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let f = &frames[i];
            match f.face_luma.filter(|y| *y > 1e-4) {
                Some(y) => (y.log2() + res[i].ev, 1.0),
                None => (f.key.max(1e-4).log2() + to_face + res[i].ev, BALANCE_KEY_WEIGHT),
            }
        })
        .collect();
    let target = {
        let faces: Vec<f32> = subject.iter().filter(|s| s.1 >= 1.0).map(|s| s.0).collect();
        let mut v = if faces.len() >= 3 { faces } else { subject.iter().map(|s| s.0).collect() };
        v.sort_by(f32::total_cmp);
        v[((v.len() - 1) as f32 * BALANCE_TARGET_Q).round() as usize]
    };
    // Средняя яркость каждой серии и её доля лиц.
    let ns = series.iter().copied().max().map(|s| s + 1).unwrap_or(0);
    let mut acc = vec![(0f32, 0f32); ns];
    let mut with_faces = vec![(0usize, 0usize); ns];
    for i in 0..n {
        let (v, w) = subject[i];
        acc[series[i]].0 += v * w;
        acc[series[i]].1 += w;
        with_faces[series[i]].1 += 1;
        if w >= 1.0 {
            with_faces[series[i]].0 += 1;
        }
    }
    // Сглаживание с соседними сериями (по связям кадров).
    let mut nb = vec![(0f32, 0f32); ns];
    for &(i, j, w) in edges {
        let (a, b) = (series[i], series[j]);
        if a == b {
            continue;
        }
        let k = BALANCE_NEIGHBOUR * w;
        if acc[b].1 > 0.0 {
            nb[a].0 += k * acc[b].0 / acc[b].1;
            nb[a].1 += k;
        }
        if acc[a].1 > 0.0 {
            nb[b].0 += k * acc[a].0 / acc[a].1;
            nb[b].1 += k;
        }
    }
    for i in 0..n {
        let s = series[i];
        let (sum, w) = acc[s];
        if w <= 0.0 {
            continue;
        }
        let own = sum / w;
        let level = (own + nb[s].0) / (1.0 + nb[s].1);
        let (nf, nt) = with_faces[s];
        let k = if 2 * nf >= nt { BALANCE_FACES } else { BALANCE_SCENE };
        let mut c = (k * (target - level)).clamp(0.0, BALANCE_UP);
        if c > 0.0 {
            // Защита светов: не выбивать новые большие светлые участки.
            let cap = highlight_cap(&frames[i].hist).max(res[i].ev);
            c = c.min((cap - res[i].ev).max(0.0));
            // Лицо не делаем светлее нормы.
            if let Some(y) = frames[i].face_luma.filter(|y| *y > 1e-4) {
                c = c.min((FACE_OK_HI / (y * 2f32.powf(res[i].ev))).log2().max(0.0));
            }
        }
        if c.abs() >= 0.05 {
            res[i].ev = (res[i].ev + c).clamp(-EV_LIMIT, EV_LIMIT);
            res[i].reason.push_str(&format!(", общая яркость людей {c:+.2}"));
        }
    }
}

/// Поправка «по сцене»: явно недоэкспонированные кадры осветляем (частично), остальное
/// оставляем как у камеры; светлые кадры затемняем только при пересвете.
fn scene_ev(key: f32, p995: f32) -> f32 {
    let p995 = p995.max(1e-4);
    if key < KEY_OK_LO {
        let ev = ((KEY_OK_LO / key.max(1e-4)).log2() * SCENE_STRENGTH).min(2.0);
        ev.min((1.3 / p995).log2().max(0.0))
    } else if key >= TARGET_KEY {
        let over = (p995 / 1.0).log2();
        if over > 0.0 { -(over * 0.7).min(1.5) } else { 0.0 }
    } else {
        0.0
    }
}

fn estimate_from(key: f32, p995: f32, face_luma: Option<f32>) -> ExposureResult {
    let mut ev = scene_ev(key, p995);
    if let Some(fy) = face_luma.filter(|v| *v > 1e-4) {
        ev = face_fix(fy, key);
    }
    ExposureResult { ev: ev.clamp(-EV_LIMIT, EV_LIMIT), key, p995, face_luma, reason: String::new() }
}

/// `lin` — линейный sRGB после ББ (без экспокоррекции), `face_luma` — медиана яркости кожи.
pub fn estimate(lin: &[[f32; 3]], w: usize, h: usize, face_luma: Option<f32>) -> ExposureResult {
    let mut sum_w = 0f64;
    let mut sum_l = 0f64;
    let mut ys: Vec<f32> = Vec::with_capacity(lin.len() / 4 + 1);
    for y in (0..h).step_by(2) {
        let dy = y as f32 / h as f32 - 0.5;
        for x in (0..w).step_by(2) {
            let dx = x as f32 / w as f32 - 0.5;
            let wgt = (-(dx * dx + dy * dy) / (2.0 * 0.32 * 0.32)).exp() as f64;
            let l = luma(lin[y * w + x]).max(0.0);
            sum_w += wgt;
            sum_l += wgt * ((l + 1e-4) as f64).ln();
            ys.push(l);
        }
    }
    if ys.is_empty() {
        return ExposureResult { ev: 0.0, key: 0.0, p995: 0.0, face_luma, reason: "пустое изображение".into() };
    }
    let key = (sum_l / sum_w).exp() as f32;
    let k = (ys.len() as f32 * 0.995) as usize;
    let k = k.min(ys.len() - 1);
    ys.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    let p995 = ys[k];

    let mut ev = scene_ev(key, p995);
    let mut reason = format!("сцена: {:+.2} EV", ev);
    // Светам можно слегка уйти за 1.0 — их мягко сожмёт «плечо» кривой.
    let hi_limit = (1.3 / p995.max(1e-4)).log2();

    if let Some(fy) = face_luma.filter(|v| *v > 1e-4) {
        let g = 2f32.powf(ev);
        let fix = face_fix(fy * g, key * g);
        if fix.abs() > 0.01 {
            let push = fix;
            // Ради лица разрешаем подрезать небольшие света, но не больше чем на 1 EV сверх лимита.
            let limit = if push > 0.0 { hi_limit + 1.0 } else { f32::MAX };
            ev = (ev + push).min(limit.max(ev));
            reason.push_str(&format!(", лица: {:+.2} EV", push));
        } else {
            reason.push_str(", лица в норме");
        }
    }
    ExposureResult { ev: ev.clamp(-EV_LIMIT, EV_LIMIT), key, p995, face_luma, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: f32, w: usize, h: usize) -> Vec<[f32; 3]> {
        vec![[v, v, v]; w * h]
    }

    #[test]
    fn underexposed_scene_brightened() {
        let img = flat(0.004, 100, 80);
        let r = estimate(&img, 100, 80, None);
        assert!(r.ev > 1.0, "{r:?}");
    }

    #[test]
    fn normal_scene_untouched() {
        for k in [0.025, TARGET_KEY, 0.15] {
            let img = flat(k, 100, 80);
            let r = estimate(&img, 100, 80, None);
            assert!(r.ev.abs() < 0.05, "{k}: {r:?}");
        }
    }

    #[test]
    fn camera_normal_faces_are_left_alone() {
        // Тёмный зал, лица у камеры нормальные (≈0.05): экспозицию камеры не трогаем.
        let f = |key: f32, face: Option<f32>| FrameExpo { key, p995: key * 6.0, face_luma: face, ev100: Some(6.3), flash: false, hist: vec![], cam_ev: None };
        let r = group(&[f(0.06, Some(0.05)), f(0.04, Some(0.055)), f(0.05, None)]);
        assert!(r.iter().all(|x| x.ev.abs() < 0.01), "{r:?}");
    }

    #[test]
    fn group_keeps_same_light_consistent() {
        // Ручной режим, один свет, но вся съёмка недоэкспонирована: кадр с белой стеной и кадр
        // с тёмным фоном получают одинаковую поправку — относительная яркость сцен сохраняется.
        let f = |key: f32, face: Option<f32>| FrameExpo { key, p995: key * 6.0, face_luma: face, ev100: Some(6.3), flash: false, hist: vec![], cam_ev: None };
        let r = group(&[f(0.03, Some(0.016)), f(0.008, Some(0.016)), f(0.012, None)]);
        assert!((r[0].ev - r[2].ev).abs() < 0.01, "{r:?}");
        assert!((r[0].ev - r[1].ev).abs() < 0.05, "{r:?}");
        assert!(r[0].ev > 0.5, "{r:?}");
    }

    #[test]
    fn group_protects_bright_frame() {
        // Тёмный зал (лица требуют +1 EV) и кадр с пятном прожектора: пятно не должно уйти в белое.
        let dark: Vec<[f32; 3]> = vec![[0.03, 0.03, 0.03]; 1000];
        let mut spot = dark.clone();
        for p in spot.iter_mut().take(400) {
            *p = [0.6, 0.6, 0.6];
        }
        let a = FrameExpo { key: 0.03, p995: 0.1, face_luma: Some(0.016), ev100: Some(6.3), flash: false, hist: histogram(&dark), cam_ev: None };
        let b = FrameExpo { key: 0.15, p995: 0.6, face_luma: None, ev100: Some(6.3), flash: false, hist: histogram(&spot), cam_ev: None };
        let r = group(&[a, b]);
        assert!(r[0].ev > 0.9, "{r:?}");
        assert!(r[1].ev < 0.7, "{r:?}");
    }

    #[test]
    fn group_compensates_camera_settings() {
        // Тот же свет, но второй кадр снят на 1 EV темнее (EV100 больше) — получит +1 EV.
        let a = FrameExpo { key: 0.06, p995: 0.4, face_luma: None, ev100: Some(6.0), flash: false, hist: vec![], cam_ev: None };
        let b = FrameExpo { key: 0.03, p995: 0.2, face_luma: None, ev100: Some(7.0), flash: false, hist: vec![], cam_ev: None };
        let r = group(&[a, b]);
        assert!((r[1].ev - r[0].ev - 1.0).abs() < 0.01, "{r:?}");
    }

    #[test]
    fn dark_corner_is_pulled_towards_the_hall() {
        // Две серии одной группы: в зале лица нормальные, в тёмном углу под лампой — на 1.6 EV темнее.
        let f = |face: f32| FrameExpo { key: face * 0.8, p995: face * 4.0, face_luma: Some(face), ev100: Some(6.0), flash: false, hist: vec![], cam_ev: None };
        let frames: Vec<FrameExpo> = [0.12, 0.12, 0.12, 0.12, 0.04, 0.04].iter().map(|&y| f(y)).collect();
        let mut res: Vec<ExposureResult> = frames.iter().map(|x| ExposureResult { ev: 0.0, key: x.key, p995: x.p995, face_luma: x.face_luma, reason: String::new() }).collect();
        let series = [0, 0, 0, 0, 1, 1];
        balance_shoot(&mut res, &frames, &series, &[(3, 4, 0.1)]);
        assert!(res[0].ev.abs() < 0.1, "зал почти не трогаем: {res:?}");
        let gap = (0.12f32 / 0.04).log2();
        assert!(res[4].ev > 0.6 * gap && res[4].ev < 0.85 * gap, "угол подтянут на ~3/4: {res:?}");
        assert_eq!(res[4].ev, res[5].ev, "серия получает одну поправку");
        // Съёмка, где тёмных серий больше половины: светлая серия всё равно не затемняется.
        let frames: Vec<FrameExpo> = [0.12, 0.12, 0.04, 0.04, 0.04, 0.04].iter().map(|&y| f(y)).collect();
        let mut res: Vec<ExposureResult> = frames.iter().map(|x| ExposureResult { ev: 0.0, key: x.key, p995: x.p995, face_luma: x.face_luma, reason: String::new() }).collect();
        balance_shoot(&mut res, &frames, &[0, 0, 1, 1, 2, 2], &[]);
        assert!(res[0].ev.abs() < 1e-6 && res[1].ev.abs() < 1e-6, "светлое не темнеет: {res:?}");
    }

    #[test]
    fn dark_face_on_bright_background() {
        // Контровой свет: фон яркий, лицо тёмное — лицо важнее.
        let img = flat(0.3, 100, 80);
        let r = estimate(&img, 100, 80, Some(0.04));
        assert!(r.ev > 0.5, "{r:?}");
    }
}
