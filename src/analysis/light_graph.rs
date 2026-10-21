//! Свет всей съёмки как одна задача.
//!
//! У каждого кадра свой источник света (майред, оттенок). Данные кадра — его энергия на сетке:
//! оценки по серым пикселям и (слабее) кожа лиц. Кадры, снятые друг за другом, связаны: разница
//! их света штрафуется, но не больше порога — если данные уверенно говорят о другом свете
//! (человек отошёл от окна к лампе), связь «рвётся». Поэтому кадр без серых пикселей берёт свет
//! соседей, серия выглядит единой, а настоящая смена света сохраняется. Идея — как у
//! согласования коллекций фото (HaCohen et al. 2013, Park et al. 2016): общая модель и одна
//! оптимизация вместо правки каждого кадра по отдельности.
//!
//! Затем «одно место — один свет»: кадры, снятые в пределах полутора минут при той же яркости
//! сцены, получают общий свет (взвешенную медиану). При смешанном свете (лампы над столиками,
//! дверь с дневным светом) серые поверхности разных ракурсов «голосуют» за разные источники, и
//! без этого одно место выходит то оранжевым, то холодным.
//!
//! Кадры идут цепочкой по времени (отдельно — со вспышкой), поэтому задача решается точно:
//! динамическим программированием по цепочке, где переход «свет соседа → свет кадра» считается
//! через преобразование расстояний (Felzenszwalb & Huttenlocher) для усечённого квадрата.
//! Серия не может «расколоться» посередине из-за того, что у части кадров чуть больше
//! серых пикселей другого источника.
//!
//! Где есть люди, свет кадра определяют они. Серые поверхности (стены, экран, окна) часто
//! освещены не тем светом, что лица: стена у окна — дневным, человек — тёплой лампой. Свет,
//! «правильный» для стен, делает людей зеленоватыми. Поэтому второй проход: из первого
//! решения берётся эталон кожи всей съёмки (на автобалансе — тон кожи у камеры), и кадр с лицами
//! получает сильное условие — кожа близка к эталону (с допуском: разные люди разные). Кадры без
//! лиц берут свет места, где лица есть. Детектор лиц находит и лица на картинах, рисунках и
//! плакатах, поэтому в расчёт идут только живые: лицо, жёлто-зелёное при свете стен, — это краска
//! или цветной луч сцены, а не кожа (`live_skin`).
//!
//! Общий вид задаёт один на всю съёмку сдвиг (`anchor`): кожа людей в среднем по съёмке
//! должна выглядеть так же, как на JPEG камеры (тепло и насыщенно, привычно глазу), но без
//! скачков камеры от кадра к кадру. Нет JPEG камеры — кожа приводится к естественному тону, а
//! оттенок тянется к медиане автобаланса камеры. Сдвиг одинаков у всех кадров, поэтому
//! согласованность он не ломает.

use super::white_balance::{self, FrameWb};
use crate::color::{self, M3};
use crate::raw::CameraColor;
use rayon::prelude::*;

/// Сетка поиска: майред (≈ 11900–2000 K) × оттенок.
const M_LO: f64 = 84.0;
const M_HI: f64 = 500.0;
const DM: f64 = 4.0;
const T_LO: f64 = -50.0;
const T_HI: f64 = 50.0;
const DT: f64 = 2.0;

/// Связь соседних кадров: масштабы разницы света, сила и порог «разрыва» (в квадрате масштабов).
const PAIR_MIRED: f64 = 15.0;
const PAIR_TINT: f64 = 5.0;
const PAIR_LAMBDA: f64 = 0.5;
const PAIR_CAP: f64 = 3.0;
/// В одной серии (тот же план, кадры подряд) свет почти наверняка один: связь сильнее, разрыв
/// дороже — иначе при смешанном свете серия «плывёт» вслед за долями источников в кадре.
const PAIR_LAMBDA_SERIES: f64 = 3.0;
const PAIR_CAP_SERIES: f64 = 10.0;
/// Время, за которое связь соседних кадров ослабевает в e раз.
const EDGE_TIME_S: f64 = 60.0;
/// Разница яркости сцены (EV), при которой связь ослабевает в e раз: улица и зал — разный свет.
const EDGE_EV: f64 = 1.5;
/// Слабая тяга оттенка каждого кадра к медиане автобаланса камеры.
const TINT_PRIOR: f64 = 0.3;
/// Доля веса кожи в данных кадра (основа — серые пиксели; кожа на плакатах, в смешанном свете
/// и у разных людей разная, поэтому покадрово она только подсказка).
const SKIN_SHARE: f64 = 0.35;
/// Вес запрета на неправдоподобный тон кожи.
const SKIN_VETO_WEIGHT: f64 = 1.0;
/// Вспышка: свет ≈ 5500 K (с примесью окружающего — поэтому не жёстко).
const FLASH_MIRED: f64 = 1e6 / 5500.0;
const FLASH_WEIGHT: f64 = 1.5;
/// Кожа к эталону: допуск (Oklab, разные люди разные — внутри него без штрафа), масштаб и вес
/// условия (как у серых пикселей), доля насыщенности в расстоянии, «нормальный» размер лица
/// (√площадь × уверенность, пиксели анализа) для полного веса и тон кожи по умолчанию (≈ 40°,
/// как у JPEG Canon на автобалансе), если JPEG камеры нет.
const SKIN_TOL: f64 = 0.005;
const SKIN_SIGMA: f64 = 0.008;
const SKIN_TARGET_WEIGHT: f64 = 12.0;
const SKIN_CHROMA_SHARE: f64 = 0.5;
const SKIN_FACE_REF: f32 = 120.0;
const SKIN_DEFAULT_HUE: f64 = 0.70;
/// Живая кожа. Лицо подсказывает свет, только если при свете серых поверхностей кадра его кожа
/// не жёлто-зелёная. Ошибка в свете по оси «тепло ↔ холод» делает живую кожу розовой, синей или
/// серой, но не зелёной; жёлто-зелёное «лицо» — это картина, рисунок, плакат или человек в
/// цветном луче сцены, и подгонять под него свет всего кадра нельзя. Мера — насколько цвет лица
/// ушёл за луч `LIVE_HUE` (градусы Oklab) в сторону жёлто-зелёного, в единицах Oklab: (полный вес,
/// ноль), между ними — плавно. Почти серое лицо далеко уйти не может — его тон ничего не значит.
const LIVE_HUE: f32 = 68.0;
const LIVE_GREEN: (f32, f32) = (0.004, 0.012);
/// Во сколько раз кадр с лицами весомее при «одно место — один свет» (при полном весе лица).
const PLACE_FACE_BOOST: f64 = 3.0;
/// С какой надёжности лиц кадр оставляет свой свет (по людям), а не берёт свет места.
const KEEP_FACE_RELIABILITY: f32 = 0.25;
/// «Одно место»: окно по времени, спад веса соседей и допуск по яркости сцены (EV).
const PLACE_WINDOW_S: f64 = 90.0;
const PLACE_DECAY_S: f64 = 60.0;
const PLACE_EV: f32 = 1.5;
/// Если со светом места кожа кадра уходит за правдоподобный тон сильнее этого — лицо освещено
/// иначе, чем место, и кадр оставляет свой свет.
const PLACE_SKIN_VETO: f64 = 1.0;
/// Пределы общего сдвига и тяга сдвига температуры к нулю (на кадр с лицами).
const ANCHOR_MIRED: f64 = 30.0;
const ANCHOR_TINT: f64 = 10.0;
const ANCHOR_TINT_PULL: f64 = 0.5;
/// Тяга общего сдвига: на кадр с лицами, в единицах (майред, оттенок).
const ANCHOR_SKIN_PRIOR: f64 = 0.05;
const ANCHOR_SKIN_MIRED: f64 = 10.0;
const ANCHOR_SKIN_TINT: f64 = 4.0;
/// Сколько кадров с лицами нужно, чтобы кожа задавала общий сдвиг температуры.
const ANCHOR_SKIN_MIN: usize = 8;
/// Подбор общего сдвига по коже на JPEG камеры: пределы (майред: теплее / холоднее, оттенок),
/// масштаб разницы цвета кожи (Oklab), доля оси «зелёный ↔ пурпурный» (по ней камера держит
/// лёгкий пурпур — повторяем его наполовину) и слабая тяга к нулю.
const CAM_ANCHOR_WARM: f64 = 80.0;
const CAM_ANCHOR_COOL: f64 = 30.0;
const CAM_ANCHOR_TINT: f64 = 12.0;
const CAM_SKIN_SIGMA: f64 = 0.02;
const CAM_SKIN_MAGENTA: f64 = 0.5;
const CAM_ANCHOR_PRIOR: f64 = 0.02;

pub struct Node<'a> {
    pub frame: &'a FrameWb,
    pub cam: &'a CameraColor,
    pub time: Option<f64>,
    pub flash: bool,
    pub scene_ev: Option<f32>,
    /// Серия автоотбора (кадры подряд в одном месте).
    pub series: usize,
    /// Цвет кожи на JPEG камеры (Oklab a, b), если есть лица и JPEG.
    pub cam_skin: Option<[f32; 2]>,
    /// Усиление экспозиции кадра (для сравнения с камерой на готовой картинке).
    pub gain: f32,
}

pub struct Solved {
    /// Источник каждого кадра (майред, оттенок) — уже с общим сдвигом.
    pub light: Vec<(f64, f64)>,
    /// Насколько кадр определён своими данными (0..1).
    pub confidence: Vec<f32>,
    /// Связи соседних кадров: (i, j, вес).
    pub edges: Vec<(usize, usize, f32)>,
    /// Общий сдвиг (майред, оттенок) — для отладки и тестов.
    #[allow(dead_code)]
    pub anchor: (f64, f64),
}

struct Grid {
    nm: usize,
    nt: usize,
}

impl Grid {
    fn new() -> Self {
        Grid { nm: ((M_HI - M_LO) / DM) as usize + 1, nt: ((T_HI - T_LO) / DT) as usize + 1 }
    }
    fn len(&self) -> usize {
        self.nm * self.nt
    }
    fn at(&self, c: usize) -> (f64, f64) {
        (M_LO + (c / self.nt) as f64 * DM, T_LO + (c % self.nt) as f64 * DT)
    }
}

/// Множители ББ и матрица камеры в точке (майред, оттенок).
fn colour_at(cam: &CameraColor, m: f64, t: f64) -> ([f32; 3], M3) {
    (cam.multipliers(1e6 / m, t), cam.cam_to_srgb(1e6 / m, t))
}

/// Энергия данных кадра без кожи: оценки света, тяга оттенка (если камера его знает), вспышка.
fn base_energy(f: &FrameWb, flash: bool, tint_ref: Option<f64>, m: f64, t: f64) -> f64 {
    let mut e = 0.0;
    for k in &f.estimates {
        let dm = (m - 1e6 / k.temp as f64) / 25.0;
        let dt = (t - k.tint as f64) / 10.0;
        e += k.weight as f64 * (1.0 + dm * dm + dt * dt).ln();
    }
    if let Some(r) = tint_ref {
        e += TINT_PRIOR * ((t - r) / 10.0).powi(2);
    }
    if flash {
        let dm = (m - FLASH_MIRED) / 30.0;
        let dt = (t - tint_ref.unwrap_or(0.0)) / 10.0;
        e += FLASH_WEIGHT * (1.0 + dm * dm + dt * dt).ln();
    }
    e
}

/// Штраф кожи кадра: мягкая подсказка (вес как в `white_balance::solve`, умноженный на долю
/// кожи) и жёсткий запрет на неправдоподобный тон. Если люди освещены не тем светом, что серые
/// поверхности (лицо у окна, стол под лампой), главное — лицо: синяя кожа хуже жёлтой стены.
fn skin_energy(f: &FrameWb, mul: &[f32; 3], mat: &M3) -> f64 {
    if f.skin.is_empty() {
        return 0.0;
    }
    SKIN_SHARE * (1.2 + (f.skin.len() as f64 * 0.3).min(1.0)) * white_balance::skin_penalty_with(mul, mat, &f.skin)
        + SKIN_VETO_WEIGHT * white_balance::skin_veto_with(mul, mat, &f.skin)
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// Связь соседних по времени кадров.
#[derive(Clone, Copy, Debug)]
struct Link {
    /// Вес по времени и яркости сцены (0..1).
    w: f64,
    /// Сила связи и порог разрыва.
    lambda: f64,
    cap: f64,
}

/// Цепочки кадров по времени (без вспышки и со вспышкой — отдельно) и связи между соседями
/// в цепочке. Где связь слишком слабая (долгий перерыв, улица ↔ зал), цепочка рвётся.
fn build_chains(nodes: &[Node]) -> Vec<(Vec<usize>, Vec<Link>)> {
    let mut chains = Vec::new();
    for flash in [false, true] {
        let mut order: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].flash == flash).collect();
        order.sort_by(|&a, &b| nodes[a].time.unwrap_or(f64::MAX).total_cmp(&nodes[b].time.unwrap_or(f64::MAX)).then(a.cmp(&b)));
        let mut cur: Vec<usize> = Vec::new();
        let mut links: Vec<Link> = Vec::new();
        for &j in &order {
            if let Some(&i) = cur.last() {
                let (a, b) = (&nodes[i], &nodes[j]);
                let link = match (a.time, b.time) {
                    (Some(ta), Some(tb)) => {
                        let mut w = (-(tb - ta).abs() / EDGE_TIME_S).exp();
                        let same = a.series == b.series;
                        if same {
                            w = w.max(0.8);
                        }
                        if flash {
                            w *= 0.5;
                        }
                        if let (Some(x), Some(y)) = (a.scene_ev, b.scene_ev) {
                            w *= (-((x - y) as f64 / EDGE_EV).powi(2)).exp();
                        }
                        (w > 0.02).then_some(if same {
                            Link { w, lambda: PAIR_LAMBDA_SERIES, cap: PAIR_CAP_SERIES }
                        } else {
                            Link { w, lambda: PAIR_LAMBDA, cap: PAIR_CAP }
                        })
                    }
                    _ => None,
                };
                match link {
                    Some(l) => links.push(l),
                    None => {
                        chains.push((std::mem::take(&mut cur), std::mem::take(&mut links)));
                    }
                }
            }
            cur.push(j);
        }
        if !cur.is_empty() {
            chains.push((cur, links));
        }
    }
    chains
}

/// Преобразование расстояний (1D): out[p] = min_q f[q] + k·(p − q)² (Felzenszwalb & Huttenlocher).
fn dt1d(f: &[f64], k: f64) -> Vec<f64> {
    let n = f.len();
    let mut out = vec![0.0; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0f64; n + 1];
    let mut j = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let s = |q: usize, p: usize| ((f[q] / k + (q * q) as f64) - (f[p] / k + (p * p) as f64)) / (2.0 * (q as f64 - p as f64));
    for q in 1..n {
        let mut x = s(q, v[j]);
        while x <= z[j] {
            j -= 1;
            x = s(q, v[j]);
        }
        j += 1;
        v[j] = q;
        z[j] = x;
        z[j + 1] = f64::INFINITY;
    }
    j = 0;
    for (p, o) in out.iter_mut().enumerate() {
        while z[j + 1] < p as f64 {
            j += 1;
        }
        let d = p as f64 - v[j] as f64;
        *o = k * d * d + f[v[j]];
    }
    out
}

/// Сообщение по связи: m(x) = min_y h(y) + λw·min(d²(x, y), cap).
fn message(h: &[f64], g: &Grid, l: Link) -> Vec<f64> {
    let lw = l.lambda * l.w;
    let (km, kt) = (lw * (DM / PAIR_MIRED).powi(2), lw * (DT / PAIR_TINT).powi(2));
    let mut a = h.to_vec();
    // По оттенку (строки: одна температура).
    for r in 0..g.nm {
        let row = dt1d(&a[r * g.nt..(r + 1) * g.nt], kt);
        a[r * g.nt..(r + 1) * g.nt].copy_from_slice(&row);
    }
    // По температуре (столбцы).
    let mut col = vec![0f64; g.nm];
    for c in 0..g.nt {
        for r in 0..g.nm {
            col[r] = a[r * g.nt + c];
        }
        let d = dt1d(&col, km);
        for r in 0..g.nm {
            a[r * g.nt + c] = d[r];
        }
    }
    let floor = h.iter().copied().fold(f64::MAX, f64::min) + lw * l.cap;
    let mn = a.iter().copied().fold(f64::MAX, f64::min).min(floor);
    a.iter().map(|v| v.min(floor) - mn).collect()
}

/// Точное решение для цепочки: для каждого кадра — лучшая ячейка сетки при оптимальном свете
/// остальных кадров цепочки (прямой и обратный проход).
fn solve_chain(chain: &[usize], links: &[Link], energies: &[Vec<f32>], g: &Grid) -> Vec<usize> {
    let k = chain.len();
    let e = |t: usize| -> Vec<f64> { energies[chain[t]].iter().map(|&v| v as f64).collect() };
    let mut fwd: Vec<Vec<f64>> = Vec::with_capacity(k);
    fwd.push(vec![0.0; g.len()]);
    for t in 1..k {
        let h: Vec<f64> = e(t - 1).iter().zip(&fwd[t - 1]).map(|(a, b)| a + b).collect();
        fwd.push(message(&h, g, links[t - 1]));
    }
    let mut bwd: Vec<Vec<f64>> = vec![vec![0.0; g.len()]; k];
    for t in (0..k.saturating_sub(1)).rev() {
        let h: Vec<f64> = e(t + 1).iter().zip(&bwd[t + 1]).map(|(a, b)| a + b).collect();
        bwd[t] = message(&h, g, links[t]);
    }
    (0..k)
        .map(|t| {
            let et = e(t);
            (0..g.len()).min_by(|&a, &b| (et[a] + fwd[t][a] + bwd[t][a]).total_cmp(&(et[b] + fwd[t][b] + bwd[t][b]))).unwrap_or(0)
        })
        .collect()
}

pub fn solve(input: &[Node]) -> Solved {
    let n = input.len();
    if n == 0 {
        return Solved { light: Vec::new(), confidence: Vec::new(), edges: Vec::new(), anchor: (0.0, 0.0) };
    }
    let g = Grid::new();
    let cells: Vec<(f64, f64)> = (0..g.len()).map(|c| g.at(c)).collect();

    // Оттенок автобаланса камеры по всей съёмке — её «нейтраль» по оси зелёный ↔ пурпурный.
    let tint_ref = median(input.iter().filter(|nd| !nd.flash).filter_map(|nd| nd.frame.prior_tint).map(|t| t as f64).collect());

    // Свет подсказывают только живые лица: картины, рисунки и лица в цветном луче сцены — нет.
    // Если живых меньше половины веса лиц, цвет кожи на JPEG камеры тоже не про людей.
    let live: Vec<FrameWb> = input.par_iter().map(|nd| live_skin(nd, tint_ref, &cells)).collect();
    let nodes: Vec<Node> = input
        .iter()
        .zip(&live)
        .map(|(nd, f)| {
            let total: f32 = nd.frame.skin.iter().map(|s| s.weight).sum();
            let alive: f32 = f.skin.iter().map(|s| s.weight).sum();
            Node { frame: f, cam_skin: nd.cam_skin.filter(|_| alive >= 0.5 * total), ..*nd }
        })
        .collect();
    let nodes = &nodes[..];

    // Таблицы цвета камеры на сетке (обычно камера одна).
    let mut cams: Vec<(&CameraColor, Vec<([f32; 3], M3)>)> = Vec::new();
    let cam_of: Vec<usize> = nodes
        .iter()
        .map(|nd| match cams.iter().position(|(c, _)| c.cm_d65 == nd.cam.cm_d65 && c.cm_a == nd.cam.cm_a) {
            Some(k) => k,
            None => {
                let tab: Vec<([f32; 3], M3)> = cells.par_iter().map(|&(m, t)| colour_at(nd.cam, m, t)).collect();
                cams.push((nd.cam, tab));
                cams.len() - 1
            }
        })
        .collect();

    // Энергия данных каждого кадра на сетке (минимум — ноль).
    let energies: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let nd = &nodes[i];
            let tab = &cams[cam_of[i]].1;
            let mut e: Vec<f32> = cells
                .iter()
                .zip(tab)
                .map(|(&(m, t), (mul, mat))| (base_energy(nd.frame, nd.flash, tint_ref, m, t) + skin_energy(nd.frame, mul, mat)) as f32)
                .collect();
            let mn = e.iter().copied().fold(f32::MAX, f32::min);
            e.iter_mut().for_each(|v| *v -= mn);
            e
        })
        .collect();

    // Решение по цепочкам.
    let chains = build_chains(nodes);
    let mut x = vec![0usize; n];
    let mut nbs: Vec<Vec<(usize, Link)>> = vec![Vec::new(); n];
    let mut edges = Vec::new();
    let solved: Vec<Vec<usize>> = chains.par_iter().map(|(c, l)| solve_chain(c, l, &energies, &g)).collect();
    for ((c, l), s) in chains.iter().zip(solved) {
        for (k, &i) in c.iter().enumerate() {
            x[i] = s[k];
        }
        for (k, link) in l.iter().enumerate() {
            let (i, j) = (c[k], c[k + 1]);
            nbs[i].push((j, *link));
            nbs[j].push((i, *link));
            edges.push((i, j, link.w as f32));
        }
    }

    // Уточнение внутри ячейки по точной энергии (соседи — в найденных ячейках).
    let pair = |m: f64, t: f64, mj: f64, tj: f64, l: Link| l.lambda * l.w * (((m - mj) / PAIR_MIRED).powi(2) + ((t - tj) / PAIR_TINT).powi(2)).min(l.cap);
    let refined: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let nd = &nodes[i];
            let (m0, t0) = cells[x[i]];
            let mut best = (f64::MAX, m0, t0);
            for a in -8..=8 {
                for b in -8..=8 {
                    let (m, t) = (m0 + a as f64 * DM / 8.0, t0 + b as f64 * DT / 8.0);
                    let mut e = base_energy(nd.frame, nd.flash, tint_ref, m, t);
                    if !nd.frame.skin.is_empty() {
                        let (mul, mat) = colour_at(nd.cam, m, t);
                        e += skin_energy(nd.frame, &mul, &mat);
                    }
                    for &(j, l) in &nbs[i] {
                        e += pair(m, t, cells[x[j]].0, cells[x[j]].1, l);
                    }
                    if e < best.0 {
                        best = (e, m, t);
                    }
                }
            }
            (best.1, best.2)
        })
        .collect();

    let confidence: Vec<f32> = nodes
        .iter()
        .map(|nd| {
            let w: f32 = nd.frame.estimates.iter().map(|e| e.weight).sum();
            ((w / 3.0).min(1.0) + if nd.frame.skin.is_empty() { 0.0 } else { 0.2 }).min(1.0)
        })
        .collect();

    let refined = place_consensus(nodes, &refined, &confidence);

    // Общий сдвиг (одинаков для всех кадров).
    // Оттенок по камере: медиана нашей оценки — на полпути к медиане автобаланса камеры. Камера
    // по этой оси надёжна, но сама держит лёгкий пурпур (на серых её JPEG ≈ +15 по Oklab ×1000):
    // повторять его целиком — кожа краснеет, кадры под лампами уходят в пурпур.
    let ours_t = median((0..n).filter(|&i| !nodes[i].flash).map(|i| refined[i].1).collect());
    let dt_cam = match (tint_ref, ours_t) {
        (Some(c), Some(o)) => (ANCHOR_TINT_PULL * (c - o)).clamp(-ANCHOR_TINT, ANCHOR_TINT),
        _ => 0.0,
    };
    // Кожа всех лиц съёмки (сотни лиц сглаживают различия между людьми). Если есть JPEG камеры —
    // общий сдвиг, при котором кожа в среднем такая же, как у камеры. Иначе — естественный тон
    // кожи по модели, с тягой к «оттенку камеры» и к нулю по температуре. Двумерно: розоватую
    // кожу правит оттенок, а не лишнее тепло на всём кадре.
    let skin_nodes: Vec<usize> = (0..n).filter(|&i| !nodes[i].flash && !nodes[i].frame.skin.is_empty()).collect();
    let cam_nodes: Vec<usize> = skin_nodes.iter().copied().filter(|&i| nodes[i].cam_skin.is_some()).collect();
    let (dm, dt) = if cam_nodes.len() >= ANCHOR_SKIN_MIN {
        match_camera_skin(nodes, &refined, &cam_nodes, dt_cam)
    } else if skin_nodes.len() >= ANCHOR_SKIN_MIN {
        let k = skin_nodes.len() as f64 * ANCHOR_SKIN_PRIOR;
        let cost = |dm: f64, dt: f64| -> f64 {
            let s: f64 = skin_nodes
                .par_iter()
                .map(|&i| {
                    let (m, t) = (refined[i].0 + dm, refined[i].1 + dt);
                    let (mul, mat) = colour_at(nodes[i].cam, m, t);
                    white_balance::skin_penalty_with(&mul, &mat, &nodes[i].frame.skin)
                })
                .sum();
            s + k * ((dm / ANCHOR_SKIN_MIRED).powi(2) + ((dt - dt_cam) / ANCHOR_SKIN_TINT).powi(2))
        };
        let mut best = (cost(0.0, dt_cam), 0.0, dt_cam);
        let mut a = -ANCHOR_MIRED;
        while a <= ANCHOR_MIRED + 1e-9 {
            let mut b = -ANCHOR_TINT;
            while b <= ANCHOR_TINT + 1e-9 {
                let c = cost(a, b);
                if c < best.0 {
                    best = (c, a, b);
                }
                b += 1.0;
            }
            a += 2.0;
        }
        (best.1, best.2)
    } else {
        (0.0, dt_cam)
    };
    if std::env::var_os("WB_DEBUG").is_some() {
        eprintln!(
            "свет съёмки: {} кадров, {} цепочек, {} связей; оттенок камеры {:?}, наш {:?} → тяга {:+.1}; кожа ({} кадр., с JPEG камеры {}) → сдвиг {:+.0} майред, {:+.1} оттенок",
            n,
            chains.len(),
            edges.len(),
            tint_ref.map(|v| (v * 10.0).round() / 10.0),
            ours_t.map(|v| (v * 10.0).round() / 10.0),
            dt_cam,
            skin_nodes.len(),
            cam_nodes.len(),
            dm,
            dt
        );
    }
    // Второй проход: где есть люди, свет определяют они (эталон кожи из первого прохода).
    let refined = match skin_target(nodes, &refined, (dm, dt), &skin_nodes) {
        Some(target) => {
            let people = solve_for_people(nodes, &g, &cells, &energies, &chains, &nbs, &refined, (dm, dt), tint_ref, &target);
            let boost: Vec<f64> = (0..n).map(|i| 1.0 + PLACE_FACE_BOOST * face_reliability(nodes[i].frame) as f64).collect();
            if std::env::var_os("WB_DEBUG").is_some() {
                let moved: Vec<f64> = (0..n).map(|i| (people[i].0 - refined[i].0).abs() + (people[i].1 - refined[i].1).abs()).collect();
                eprintln!(
                    "кожа: эталон тон {:.0}°, насыщенность {:.3}; кадров с лицами {}, свет сдвинулся у {} (больше 5 майред/оттенка)",
                    target.hue.to_degrees(),
                    target.chroma,
                    skin_nodes.len(),
                    moved.iter().filter(|&&d| d > 5.0).count()
                );
            }
            // Кадры с надёжными лицами оставляют свой свет — его определили люди; свет места
            // (с перевесом таких кадров) получают кадры без лиц.
            let keep: Vec<bool> = (0..n).map(|i| face_reliability(nodes[i].frame) >= KEEP_FACE_RELIABILITY).collect();
            place_consensus_weighted(nodes, &people, &confidence, &boost, &keep)
        }
        None => refined,
    };
    let light = refined.iter().map(|&(m, t)| ((m + dm).clamp(M_LO, M_HI), t + dt)).collect();
    Solved { light, confidence, edges, anchor: (dm, dt) }
}

/// Эталон кожи съёмки на готовой картинке: тон и насыщенность.
#[derive(Clone, Copy, Debug)]
struct SkinTarget {
    hue: f64,
    chroma: f64,
}

/// Насколько лицам кадра можно верить (0..1): размер и уверенность детектора самого крупного
/// лица. Не сумма: десяток мелких лиц зрителей — ненадёжный ориентир, свет там решают стены.
fn face_reliability(f: &FrameWb) -> f32 {
    (f.skin.iter().map(|s| s.weight).fold(0.0f32, f32::max) / SKIN_FACE_REF).min(1.0)
}

/// Эталон кожи: тон — как у камеры на автобалансе (медиана по съёмке) или ≈ 40°, насыщенность —
/// медиана нашей кожи после первого прохода (свою насыщенность не трогаем).
fn skin_target(nodes: &[Node], light: &[(f64, f64)], anchor: (f64, f64), skin_nodes: &[usize]) -> Option<SkinTarget> {
    if skin_nodes.len() < ANCHOR_SKIN_MIN {
        return None;
    }
    let chroma = median(
        skin_nodes
            .par_iter()
            .map(|&i| {
                let lut = color::ToneLut::new(nodes[i].gain);
                let s = rendered_skin(&nodes[i], &lut, light[i].0 + anchor.0 - white_balance::WARMTH_MIRED, light[i].1 + anchor.1);
                (s[0] as f64).hypot(s[1] as f64)
            })
            .collect(),
    )?;
    let cam: Vec<f64> = skin_nodes.iter().filter_map(|&i| nodes[i].cam_skin).map(|c| (c[1] as f64).atan2(c[0] as f64)).collect();
    let hue = if cam.len() >= ANCHOR_SKIN_MIN { median(cam)? } else { SKIN_DEFAULT_HUE };
    Some(SkinTarget { hue, chroma })
}

/// Штраф кожи кадра за отход от эталона при итоговом ББ (майред, оттенок — уже с общим сдвигом).
fn skin_target_energy(nd: &Node, lut: &color::ToneLut, m: f64, t: f64, target: &SkinTarget) -> f64 {
    let s = rendered_skin(nd, lut, m - white_balance::WARMTH_MIRED, t);
    let (a, b) = (s[0] as f64, s[1] as f64);
    let chroma = a.hypot(b);
    if chroma < 0.01 {
        return SKIN_TARGET_WEIGHT * 4.0;
    }
    let mut dh = b.atan2(a) - target.hue;
    while dh > std::f64::consts::PI {
        dh -= 2.0 * std::f64::consts::PI;
    }
    while dh < -std::f64::consts::PI {
        dh += 2.0 * std::f64::consts::PI;
    }
    let eh = target.chroma * dh;
    let ec = SKIN_CHROMA_SHARE * (chroma - target.chroma);
    let excess = ((eh * eh + ec * ec).sqrt() - SKIN_TOL).max(0.0) / SKIN_SIGMA;
    SKIN_TARGET_WEIGHT * face_reliability(nd.frame) as f64 * (1.0 + excess * excess).ln()
}

/// Второй проход: к данным кадров с лицами добавляется условие «кожа как эталон», решение
/// по цепочкам и уточнение — как в первом проходе. Возвращает свет без общего сдвига.
#[allow(clippy::too_many_arguments)]
fn solve_for_people(
    nodes: &[Node],
    g: &Grid,
    cells: &[(f64, f64)],
    energies: &[Vec<f32>],
    chains: &[(Vec<usize>, Vec<Link>)],
    nbs: &[Vec<(usize, Link)>],
    first: &[(f64, f64)],
    anchor: (f64, f64),
    tint_ref: Option<f64>,
    target: &SkinTarget,
) -> Vec<(f64, f64)> {
    let n = nodes.len();
    let (dm, dt) = anchor;
    let energies2: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let nd = &nodes[i];
            if nd.frame.skin.is_empty() || nd.flash {
                return energies[i].clone();
            }
            let lut = color::ToneLut::new(nd.gain);
            let mut e: Vec<f32> = cells
                .iter()
                .zip(&energies[i])
                .map(|(&(m, t), &base)| base + skin_target_energy(nd, &lut, m + dm, t + dt, target) as f32)
                .collect();
            let mn = e.iter().copied().fold(f32::MAX, f32::min);
            e.iter_mut().for_each(|v| *v -= mn);
            e
        })
        .collect();
    let mut x = vec![0usize; n];
    let solved: Vec<Vec<usize>> = chains.par_iter().map(|(c, l)| solve_chain(c, l, &energies2, g)).collect();
    for ((c, _), s) in chains.iter().zip(solved) {
        for (k, &i) in c.iter().enumerate() {
            x[i] = s[k];
        }
    }
    let pair = |m: f64, t: f64, mj: f64, tj: f64, l: Link| l.lambda * l.w * (((m - mj) / PAIR_MIRED).powi(2) + ((t - tj) / PAIR_TINT).powi(2)).min(l.cap);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let nd = &nodes[i];
            if nd.frame.skin.is_empty() || nd.flash {
                // Без лиц — как в первом проходе, но соседи могли сдвинуться: уточняем от них.
                let (m0, t0) = cells[x[i]];
                if (m0 - first[i].0).abs() < DM && (t0 - first[i].1).abs() < DT {
                    return first[i];
                }
            }
            let lut = color::ToneLut::new(nd.gain);
            let (m0, t0) = cells[x[i]];
            let mut best = (f64::MAX, m0, t0);
            for a in -8..=8 {
                for b in -8..=8 {
                    let (m, t) = (m0 + a as f64 * DM / 8.0, t0 + b as f64 * DT / 8.0);
                    let mut e = base_energy(nd.frame, nd.flash, tint_ref, m, t);
                    if !nd.frame.skin.is_empty() {
                        let (mul, mat) = colour_at(nd.cam, m, t);
                        e += skin_energy(nd.frame, &mul, &mat);
                        if !nd.flash {
                            e += skin_target_energy(nd, &lut, m + dm, t + dt, target);
                        }
                    }
                    for &(j, l) in &nbs[i] {
                        e += pair(m, t, cells[x[j]].0, cells[x[j]].1, l);
                    }
                    if e < best.0 {
                        best = (e, m, t);
                    }
                }
            }
            (best.1, best.2)
        })
        .collect()
}

fn weighted_median(mut v: Vec<(f64, f64)>) -> f64 {
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = v.iter().map(|x| x.1).sum();
    let mut acc = 0.0;
    for &(val, w) in &v {
        acc += w;
        if acc >= 0.5 * total {
            return val;
        }
    }
    v.last().map(|x| x.0).unwrap_or(0.0)
}

/// Одно место — один свет: свет кадра заменяется взвешенной медианой света кадров, снятых рядом
/// по времени при той же вспышке и похожей яркости сцены. Вес — уверенность кадра и близость
/// по времени. Медиана сохраняет чёткую смену места и убирает одиночные «уехавшие» оценки.
fn place_consensus(nodes: &[Node], light: &[(f64, f64)], conf: &[f32]) -> Vec<(f64, f64)> {
    place_consensus_weighted(nodes, light, conf, &vec![1.0; nodes.len()], &vec![false; nodes.len()])
}

/// То же с дополнительным весом кадров (кадры с лицами задают свет места); `keep` — кадры,
/// которые оставляют свой свет.
fn place_consensus_weighted(nodes: &[Node], light: &[(f64, f64)], conf: &[f32], boost: &[f64], keep: &[bool]) -> Vec<(f64, f64)> {
    let mut order: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].time.is_some()).collect();
    order.sort_by(|&a, &b| nodes[a].time.unwrap().total_cmp(&nodes[b].time.unwrap()).then(a.cmp(&b)));
    let mut out = light.to_vec();
    for (k, &i) in order.iter().enumerate() {
        if keep[i] {
            continue;
        }
        let ti = nodes[i].time.unwrap();
        let near = |j: usize| -> Option<f64> {
            let dt = (nodes[j].time.unwrap() - ti).abs();
            let same_ev = match (nodes[i].scene_ev, nodes[j].scene_ev) {
                (Some(a), Some(b)) => (a - b).abs() <= PLACE_EV,
                _ => true,
            };
            (dt <= PLACE_WINDOW_S && nodes[j].flash == nodes[i].flash && same_ev).then(|| (0.2 + conf[j] as f64) * boost[j] * (-dt / PLACE_DECAY_S).exp())
        };
        let mut ms: Vec<(f64, f64)> = Vec::new();
        let mut ts: Vec<(f64, f64)> = Vec::new();
        let back = order[..k].iter().rev().take_while(|&&j| ti - nodes[j].time.unwrap() <= PLACE_WINDOW_S);
        let fwd = order[k..].iter().take_while(|&&j| nodes[j].time.unwrap() - ti <= PLACE_WINDOW_S);
        for &j in back.chain(fwd) {
            if let Some(w) = near(j) {
                ms.push((light[j].0, w));
                ts.push((light[j].1, w));
            }
        }
        if ms.len() < 2 {
            continue;
        }
        let (m, t) = (weighted_median(ms), weighted_median(ts));
        if !nodes[i].frame.skin.is_empty() {
            let (mul, mat) = colour_at(nodes[i].cam, m, t);
            if white_balance::skin_veto_with(&mul, &mat, &nodes[i].frame.skin) > PLACE_SKIN_VETO {
                continue;
            }
        }
        out[i] = (m, t);
    }
    out
}

/// Цвет одного лица на готовой картинке (Oklab): камера → ББ → матрица → экспозиция → кривая →
/// обратно в линейный — так же, как меряется кожа на JPEG камеры.
fn skin_lab(s: &white_balance::SkinSample, mul: &[f32; 3], mat: &M3, lut: &color::ToneLut, gain: f32) -> [f32; 3] {
    let c = [s.cam[0] * mul[0], s.cam[1] * mul[1], s.cam[2] * mul[2]];
    let c = color::mulv(mat, [c[0] as f64, c[1] as f64, c[2] as f64]);
    let r = lut.apply([c[0].max(0.0) as f32, c[1].max(0.0) as f32, c[2].max(0.0) as f32], gain);
    color::oklab(r.map(color::srgb_decode))
}

/// Цвет кожи кадра на готовой картинке (Oklab a, b) при ББ (майред, оттенок).
fn rendered_skin(nd: &Node, lut: &color::ToneLut, m: f64, t: f64) -> [f32; 2] {
    let (mul, mat) = colour_at(nd.cam, m, t);
    let (mut a, mut b, mut wsum) = (0f32, 0f32, 0f32);
    for s in &nd.frame.skin {
        let lab = skin_lab(s, &mul, &mat, lut, nd.gain);
        a += s.weight * lab[1];
        b += s.weight * lab[2];
        wsum += s.weight;
    }
    [a / wsum.max(1e-6), b / wsum.max(1e-6)]
}

/// Насколько цвет похож на живую кожу (0..1), см. `LIVE_*`.
fn liveness(lab: [f32; 3]) -> f32 {
    let past = lab[1].hypot(lab[2]) * (lab[2].atan2(lab[1]) - LIVE_HUE.to_radians()).sin();
    let (full, zero) = LIVE_GREEN;
    ((past - zero) / (full - zero)).clamp(0.0, 1.0)
}

/// Данные кадра, где у лиц остался только вес живой кожи (картины и лица в цветном луче
/// выпадают). Судим при свете серых поверхностей самого кадра, без кожи. Нет оценок света —
/// судить не по чему, лица остаются как есть.
fn live_skin(nd: &Node, tint_ref: Option<f64>, cells: &[(f64, f64)]) -> FrameWb {
    let mut f = nd.frame.clone();
    if f.skin.is_empty() || f.estimates.is_empty() {
        return f;
    }
    let (_, m, t) = cells
        .iter()
        .map(|&(m, t)| (base_energy(nd.frame, nd.flash, tint_ref, m, t), m, t))
        .fold((f64::MAX, 0.0, 0.0), |a, b| if b.0 < a.0 { b } else { a });
    let (mul, mat) = colour_at(nd.cam, m - white_balance::WARMTH_MIRED, t);
    let lut = color::ToneLut::new(nd.gain);
    f.skin.retain_mut(|s| {
        let w = liveness(skin_lab(s, &mul, &mat, &lut, nd.gain));
        s.weight *= w;
        w > 0.01
    });
    f
}

/// Служебное: тон кожи кадра (градусы) и насыщенность «по модели» — при ББ (температура, оттенок)
/// и усилении экспозиции, как его видит решение по свету.
pub(crate) fn model_skin(f: &FrameWb, cam: &CameraColor, temp: f32, tint: f32, gain: f32) -> Option<(f32, f32)> {
    if f.skin.is_empty() {
        return None;
    }
    let nd = Node { frame: f, cam, time: None, flash: false, scene_ev: None, series: 0, cam_skin: None, gain };
    let lut = color::ToneLut::new(gain);
    let s = rendered_skin(&nd, &lut, 1e6 / temp as f64, tint as f64);
    Some((s[1].atan2(s[0]).to_degrees(), s[0].hypot(s[1])))
}

/// Общий сдвиг (майред, оттенок), при котором кожа людей в среднем по съёмке выглядит так же,
/// как на JPEG камеры. Робастно: кадры с «чужой» кожей (плакат, контровой свет) не перевешивают.
fn match_camera_skin(nodes: &[Node], refined: &[(f64, f64)], idx: &[usize], dt_cam: f64) -> (f64, f64) {
    let ax = color::oklab_axes();
    let luts: Vec<color::ToneLut> = idx.iter().map(|&i| color::ToneLut::new(nodes[i].gain)).collect();
    let k = idx.len() as f64 * CAM_ANCHOR_PRIOR;
    let cost = |dm: f64, dt: f64| -> f64 {
        let s: f64 = (0..idx.len())
            .into_par_iter()
            .map(|j| {
                let i = idx[j];
                let (m, t) = (refined[i].0 + dm - white_balance::WARMTH_MIRED, refined[i].1 + dt);
                let o = rendered_skin(&nodes[i], &luts[j], m, t);
                let c = nodes[i].cam_skin.unwrap();
                let (da, db) = ((o[0] - c[0]) as f64, (o[1] - c[1]) as f64);
                let dw = da * ax.warm[0] as f64 + db * ax.warm[1] as f64;
                let dg = da * ax.magenta[0] as f64 + db * ax.magenta[1] as f64;
                (1.0 + (dw * dw + CAM_SKIN_MAGENTA * dg * dg) / (CAM_SKIN_SIGMA * CAM_SKIN_SIGMA)).ln()
            })
            .sum();
        s + k * ((dm / 30.0).powi(2) + ((dt - dt_cam) / 6.0).powi(2))
    };
    let mut best = (cost(0.0, dt_cam), 0.0, dt_cam);
    let mut a = -CAM_ANCHOR_WARM;
    while a <= CAM_ANCHOR_COOL + 1e-9 {
        let mut b = -CAM_ANCHOR_TINT;
        while b <= CAM_ANCHOR_TINT + 1e-9 {
            let c = cost(a, b);
            if c < best.0 {
                best = (c, a, b);
            }
            b += 2.0;
        }
        a += 4.0;
    }
    let (m0, t0) = (best.1, best.2);
    for i in -4..=4 {
        for j in -2..=2 {
            let (a, b) = (m0 + i as f64, t0 + j as f64 * 0.5);
            if a < -CAM_ANCHOR_WARM || a > CAM_ANCHOR_COOL || b.abs() > CAM_ANCHOR_TINT {
                continue;
            }
            let c = cost(a, b);
            if c < best.0 {
                best = (c, a, b);
            }
        }
    }
    (best.1, best.2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::white_balance::WbEstimate;

    fn cam() -> CameraColor {
        CameraColor {
            cm_a: Some([1.2356, -0.4649, -0.0678, -0.391, 1.1644, 0.2577, -0.0455, 0.1934, 0.7348]),
            cm_d65: [1.0463, -0.2173, -0.1437, -0.4856, 1.2635, 0.2482, -0.1216, 0.2915, 0.7237],
        }
    }

    fn frame(est: &[(f32, f32)], weight: f32) -> FrameWb {
        FrameWb { estimates: est.iter().map(|&(temp, w)| WbEstimate { method: "т".into(), temp, tint: 0.0, weight: weight * w }).collect(), skin: vec![], prior_tint: Some(0.0) }
    }

    fn run(frames: &[FrameWb], times: &[f64], series: &[usize]) -> Vec<(f64, f64)> {
        run_ev(frames, times, series, &vec![6.0; frames.len()])
    }

    fn run_ev(frames: &[FrameWb], times: &[f64], series: &[usize], ev: &[f32]) -> Vec<(f64, f64)> {
        let cam = cam();
        let nodes: Vec<Node> = frames
            .iter()
            .zip(times)
            .zip(series)
            .zip(ev)
            .map(|(((f, &t), &s), &e)| Node { frame: f, cam: &cam, time: Some(t), flash: false, scene_ev: Some(e), series: s, cam_skin: None, gain: 1.0 })
            .collect();
        solve(&nodes).light
    }

    #[test]
    fn distance_transform_matches_brute_force() {
        let f = [3.0, 0.5, 7.0, 2.0, 9.0, 0.0, 4.0];
        let k = 0.7;
        let d = dt1d(&f, k);
        for p in 0..f.len() {
            let b = (0..f.len()).map(|q| f[q] + k * (p as f64 - q as f64).powi(2)).fold(f64::MAX, f64::min);
            assert!((d[p] - b).abs() < 1e-9, "{p}: {} vs {b}", d[p]);
        }
    }

    #[test]
    fn empty_frame_takes_light_of_neighbours() {
        // Серия при 3200 K, в середине кадр без серых пикселей (только слабая чужая оценка).
        let mut f: Vec<FrameWb> = (0..6).map(|_| frame(&[(3200.0, 1.0)], 3.0)).collect();
        f[3] = frame(&[(5000.0, 1.0)], 0.15);
        let times: Vec<f64> = (0..6).map(|i| i as f64 * 5.0).collect();
        let l = run(&f, &times, &[0, 1, 2, 3, 4, 5]);
        let k = 1e6 / 3200.0;
        assert!(l.iter().all(|x| (x.0 - k).abs() < 8.0), "{l:?}");
    }

    #[test]
    fn noisy_series_is_evened_out() {
        // Один свет, оценки по кадрам гуляют ±12 майред — после согласования разброс меньше.
        let base = 1e6 / 4000.0;
        let offs = [-12.0, 9.0, -6.0, 12.0, -9.0, 6.0, 0.0];
        let f: Vec<FrameWb> = offs.iter().map(|o| frame(&[((1e6 / (base + o)) as f32, 1.0)], 3.0)).collect();
        let times: Vec<f64> = (0..offs.len()).map(|i| i as f64 * 3.0).collect();
        let l = run(&f, &times, &[0; 7]);
        let spread = l.iter().map(|x| x.0).fold(f64::MIN, f64::max) - l.iter().map(|x| x.0).fold(f64::MAX, f64::min);
        assert!(spread < 14.0, "разброс {spread:.1} майред: {l:?}");
    }

    #[test]
    fn real_change_of_light_is_kept() {
        // Пять кадров у окна, потом пять под лампой (у окна сцена светлее) — уверенные данные,
        // связь должна «порваться».
        let mut f: Vec<FrameWb> = (0..5).map(|_| frame(&[(5200.0, 1.0)], 3.0)).collect();
        f.extend((0..5).map(|_| frame(&[(2900.0, 1.0)], 3.0)));
        let times: Vec<f64> = (0..10).map(|i| i as f64 * 4.0).collect();
        let ev = [8.0, 8.0, 8.0, 8.0, 8.0, 5.5, 5.5, 5.5, 5.5, 5.5];
        let l = run_ev(&f, &times, &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], &ev);
        assert!((l[0].0 - 1e6 / 5200.0).abs() < 10.0 && (l[4].0 - 1e6 / 5200.0).abs() < 10.0, "{l:?}");
        assert!((l[5].0 - 1e6 / 2900.0).abs() < 10.0 && (l[9].0 - 1e6 / 2900.0).abs() < 10.0, "{l:?}");
    }

    #[test]
    fn one_place_gets_one_light_from_any_angle() {
        // Столики под лампами (≈3000 K): с одного ракурса в кадре дверь с дневным светом, и серые
        // уверенно «голосуют» за 5200 K. Минута съёмки, та же яркость сцены — свет у всех один.
        let mut f: Vec<FrameWb> = (0..6).map(|_| frame(&[(3000.0, 1.0)], 3.0)).collect();
        f.extend((0..4).map(|_| frame(&[(5200.0, 1.0)], 3.0)));
        f.extend((0..5).map(|_| frame(&[(3000.0, 1.0)], 3.0)));
        let times: Vec<f64> = (0..15).map(|i| i as f64 * 5.0).collect();
        let series: Vec<usize> = (0..15).map(|i| i / 5).collect();
        let l = run(&f, &times, &series);
        let k = 1e6 / 3000.0;
        assert!(l.iter().all(|x| (x.0 - k).abs() < 10.0), "{:?}", l.iter().map(|x| (1e6 / x.0) as i32).collect::<Vec<_>>());
    }

    #[test]
    fn series_with_two_lights_does_not_split() {
        // Смешанный свет: в каждом кадре серии два источника, доли чуть гуляют. Один свет на всю серию.
        let shares = [0.56, 0.58, 0.55, 0.45, 0.44, 0.46];
        let f: Vec<FrameWb> = shares.iter().map(|&s| frame(&[(2650.0, s), (3500.0, 1.0 - s)], 3.0)).collect();
        let times: Vec<f64> = (0..6).map(|i| i as f64 * 4.0).collect();
        let l = run(&f, &times, &[0; 6]);
        let spread = l.iter().map(|x| x.0).fold(f64::MIN, f64::max) - l.iter().map(|x| x.0).fold(f64::MAX, f64::min);
        assert!(spread < 6.0, "серия раскололась: {l:?}");
    }

    #[test]
    fn shoot_tone_follows_camera_skin() {
        // Камера показывает кожу теплее, чем наш нейтральный ББ: общий сдвиг уходит в тепло
        // ровно настолько, чтобы кожа в среднем совпала, и одинаков для всех кадров.
        use crate::analysis::white_balance::SkinSample;
        let cam = cam();
        let skin_at = |t: f64| {
            // Кожа (светлая, Macbeth) под источником t в цветах камеры.
            let n = cam.neutral(t, 0.0);
            let refl = [0.45f64, 0.32, 0.25];
            SkinSample { cam: [(refl[0] * n[0] * 0.5) as f32, (refl[1] * n[1] * 0.5) as f32, (refl[2] * n[2] * 0.5) as f32], weight: 1.0 }
        };
        let lights = [3000.0f64, 3200.0, 4000.0, 4500.0, 5200.0, 3100.0, 3600.0, 4200.0, 2900.0, 5000.0];
        let frames: Vec<FrameWb> = lights
            .iter()
            .map(|&t| FrameWb { estimates: vec![WbEstimate { method: "т".into(), temp: t as f32, tint: 0.0, weight: 3.0 }], skin: vec![skin_at(t)], prior_tint: Some(0.0) })
            .collect();
        // «Камера»: та же кожа при ББ на 40 майред теплее источника.
        let lut = color::ToneLut::new(1.0);
        let cam_view: Vec<[f32; 2]> = lights
            .iter()
            .zip(&frames)
            .map(|(&t, f)| {
                let nd = Node { frame: f, cam: &cam, time: None, flash: false, scene_ev: None, series: 0, cam_skin: None, gain: 1.0 };
                rendered_skin(&nd, &lut, 1e6 / t - 40.0, 0.0)
            })
            .collect();
        let nodes: Vec<Node> = frames
            .iter()
            .zip(&cam_view)
            .enumerate()
            .map(|(i, (f, c))| Node { frame: f, cam: &cam, time: Some(i as f64 * 3600.0), flash: false, scene_ev: Some(6.0), series: i, cam_skin: Some(*c), gain: 1.0 })
            .collect();
        let s = solve(&nodes);
        // Итоговый ББ (с общим теплом) — около «источник − 40 майред».
        let total = s.anchor.0 - white_balance::WARMTH_MIRED;
        assert!((total + 40.0).abs() < 6.0, "сдвиг {:?}", s.anchor);
        assert!(s.anchor.1.abs() < 3.0, "оттенок не уводим: {:?}", s.anchor);
    }

    #[test]
    fn people_decide_the_light_when_walls_disagree() {
        // Десять кадров: серые и кожа под одним светом (3200 K, 0). Один кадр (через час, без
        // соседей): серые «говорят» оттенок +12 (стены в другом свете), а кожа освещена как
        // у всех. Кожа должна вернуть его свет к людям, а не к стенам.
        use crate::analysis::white_balance::SkinSample;
        let cam = cam();
        let skin_at = |t: f64, tint: f64| {
            let n = cam.neutral(t, tint);
            let refl = [0.45f64, 0.32, 0.25];
            SkinSample { cam: [(refl[0] * n[0] * 0.5) as f32, (refl[1] * n[1] * 0.5) as f32, (refl[2] * n[2] * 0.5) as f32], weight: 80.0 }
        };
        let mut frames: Vec<FrameWb> = (0..10)
            .map(|_| FrameWb { estimates: vec![WbEstimate { method: "т".into(), temp: 3200.0, tint: 0.0, weight: 3.0 }], skin: vec![skin_at(3200.0, 0.0)], prior_tint: Some(0.0) })
            .collect();
        frames.push(FrameWb { estimates: vec![WbEstimate { method: "т".into(), temp: 3200.0, tint: 12.0, weight: 3.0 }], skin: vec![skin_at(3200.0, 0.0)], prior_tint: Some(0.0) });
        let nodes: Vec<Node> = frames
            .iter()
            .enumerate()
            .map(|(i, f)| Node { frame: f, cam: &cam, time: Some(if i < 10 { i as f64 * 5.0 } else { 7200.0 }), flash: false, scene_ev: Some(6.0), series: i, cam_skin: None, gain: 1.0 })
            .collect();
        let s = solve(&nodes);
        let others = s.light[0].1;
        let odd = s.light[10].1;
        assert!((odd - others).abs() < 6.0, "оттенок кадра {odd:.1} против людей {others:.1}");
    }

    #[test]
    fn living_skin_is_pink_orange_blue_or_gray_but_not_green() {
        let at = |hue: f32, chroma: f32| [0.6, chroma * hue.to_radians().cos(), chroma * hue.to_radians().sin()];
        for (hue, chroma) in [(40.0, 0.07), (65.0, 0.09), (10.0, 0.06), (-30.0, 0.01), (-110.0, 0.06), (120.0, 0.004)] {
            assert_eq!(liveness(at(hue, chroma)), 1.0, "тон {hue}°, насыщенность {chroma}");
        }
        for (hue, chroma) in [(84.0, 0.05), (110.0, 0.05), (93.0, 0.1)] {
            assert_eq!(liveness(at(hue, chroma)), 0.0, "тон {hue}°, насыщенность {chroma}");
        }
    }

    #[test]
    fn painted_faces_do_not_decide_the_light() {
        // Десять кадров с людьми при 3200 K. Отдельный кадр (через час): стены уверенно говорят
        // 4000 K, а «лицо» — жёлто-зелёная картина. Свет кадра остаётся по стенам.
        use crate::analysis::white_balance::SkinSample;
        let cam = cam();
        let face = |t: f64, refl: [f64; 3]| {
            let n = cam.neutral(t, 0.0);
            SkinSample { cam: [(refl[0] * n[0] * 0.5) as f32, (refl[1] * n[1] * 0.5) as f32, (refl[2] * n[2] * 0.5) as f32], weight: 120.0 }
        };
        let painted = face(4000.0, [0.30, 0.38, 0.12]);
        let (mul, mat) = colour_at(&cam, 1e6 / 4000.0 - white_balance::WARMTH_MIRED, 0.0);
        let lut = color::ToneLut::new(1.0);
        assert!(liveness(skin_lab(&painted, &mul, &mat, &lut, 1.0)) < 0.05);
        let mut frames: Vec<FrameWb> = (0..10)
            .map(|_| FrameWb { estimates: vec![WbEstimate { method: "т".into(), temp: 3200.0, tint: 0.0, weight: 3.0 }], skin: vec![face(3200.0, [0.45, 0.32, 0.25])], prior_tint: Some(0.0) })
            .collect();
        frames.push(FrameWb { estimates: vec![WbEstimate { method: "т".into(), temp: 4000.0, tint: 0.0, weight: 3.0 }], skin: vec![painted], prior_tint: Some(0.0) });
        let nodes: Vec<Node> = frames
            .iter()
            .enumerate()
            .map(|(i, f)| Node { frame: f, cam: &cam, time: Some(if i < 10 { i as f64 * 5.0 } else { 7200.0 }), flash: false, scene_ev: Some(6.0), series: i, cam_skin: None, gain: 1.0 })
            .collect();
        let s = solve(&nodes);
        let (dm, dt) = (s.light[10].0 - s.anchor.0 - 1e6 / 4000.0, s.light[10].1 - s.anchor.1);
        assert!(dm.abs() < 4.0 && dt.abs() < 2.0, "картина сдвинула свет: {dm:.1} майред, {dt:.1} оттенок");
    }

    #[test]
    fn anchor_moves_all_frames_equally() {
        // Камера говорит, что «нейтраль» по оттенку на −10; наши оценки — около 0.
        let cam = cam();
        let f: Vec<FrameWb> = [4000.0f32, 3000.0, 5000.0]
            .iter()
            .map(|&t| FrameWb { estimates: vec![WbEstimate { method: "т".into(), temp: t, tint: 0.0, weight: 3.0 }], skin: vec![], prior_tint: Some(-10.0) })
            .collect();
        let nodes: Vec<Node> = f
            .iter()
            .enumerate()
            .map(|(i, fr)| Node { frame: fr, cam: &cam, time: Some(i as f64 * 3600.0), flash: false, scene_ev: Some(6.0), series: i, cam_skin: None, gain: 1.0 })
            .collect();
        let s = solve(&nodes);
        assert!(s.anchor.1 < -3.0, "{:?}", s.anchor);
        // Разница между кадрами по температуре сохранилась.
        let d = |a: usize, b: usize| s.light[a].0 - s.light[b].0;
        assert!((d(1, 0) - (1e6 / 3000.0 - 1e6 / 4000.0)).abs() < 6.0, "{:?}", s.light);
        assert!((d(0, 2) - (1e6 / 4000.0 - 1e6 / 5000.0)).abs() < 6.0, "{:?}", s.light);
    }
}
