//! Группы освещения — «места» съёмки: зал у окна, столики под лампами, панель у экрана, улица,
//! вспышка. Свет каждого кадра решается отдельно (`light_graph`); группы нужны для ручных правок,
//! общей экспозиции и шумодава.
//!
//! Свет в зале обычно меняется плавно (окно → середина → лампы), и по цвету кадры образуют
//! сплошную полосу без провалов: поиск «сгустков» по цвету видит в ней одно место. Поэтому группы
//! строятся по ходу съёмки: хронология режется там, где свет или яркость сцены заметно
//! поменялись (или был долгий перерыв), а куски с одинаковым светом склеиваются — фотограф
//! возвращается в те же места. Совсем маленькие куски присоединяются к ближайшему по свету месту.

#[derive(Clone, Debug)]
pub struct GroupItem {
    /// Время съёмки, секунды.
    pub time: Option<f64>,
    pub flash: bool,
    /// Свет кадра (уже согласованный по съёмке).
    pub temp: f32,
    pub tint: f32,
    /// Яркость сцены, EV при ISO 100 (экспопара + насколько светлым вышел кадр).
    /// Улица днём ≈ 10–15, помещение ≈ 2–8: свет разный, даже если по цвету оценки близки.
    pub scene_ev: Option<f32>,
}

/// Перерыв, после которого начинается другая съёмка (другой день/площадка).
const SESSION_GAP_S: f64 = 2.0 * 3600.0;
/// Масштабы «одного места»: разница света (майред, оттенок) и яркости сцены (EV), при которой
/// кадры ещё считаются снятыми при одном свете. Расстояние больше 1 — другое место.
const PLACE_MIRED: f32 = 30.0;
const PLACE_TINT: f32 = 8.0;
const PLACE_EV: f32 = 2.5;
/// Перерыв, после которого начинается новый кусок даже при похожем свете.
const PLACE_GAP_S: f64 = 180.0;
/// Место из меньшего числа кадров (или меньшей доли съёмки) отдельной группой не становится:
/// для правок мелкие группы только мешают, а свет каждого кадра и так решается сам.
const PLACE_MIN: usize = 6;
const PLACE_MIN_SHARE: f32 = 0.025;

type Pt = [f32; 3];

fn point(it: &GroupItem) -> Pt {
    [1e6 / it.temp.max(1.0), it.tint, it.scene_ev.unwrap_or(f32::NAN)]
}

/// Расстояние между светом двух мест (в масштабах «одного места»).
fn dist(a: &Pt, b: &Pt) -> f32 {
    let dm = (a[0] - b[0]) / PLACE_MIRED;
    let dt = (a[1] - b[1]) / PLACE_TINT;
    let de = if a[2].is_finite() && b[2].is_finite() { (a[2] - b[2]) / PLACE_EV } else { 0.0 };
    (dm * dm + dt * dt + de * de).sqrt()
}

fn median(mut v: Vec<f32>) -> f32 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Типичный свет набора кадров (медианы).
fn centre(pts: &[Pt], idx: &[usize]) -> Pt {
    [0, 1, 2].map(|k| median(idx.iter().map(|&i| pts[i][k]).collect()))
}

/// Места внутри одной сессии и одного вида света (со вспышкой или без): номера мест для `idx`
/// (кадры в порядке времени).
fn places(items: &[GroupItem], idx: &[usize]) -> Vec<usize> {
    let pts: Vec<Pt> = items.iter().map(point).collect();
    // 1. Куски хронологии: кадр, свет которого далеко от типичного света текущего куска
    //    (или после долгого перерыва), начинает новый кусок.
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for &i in idx {
        let new = match runs.last() {
            None => true,
            Some(r) => {
                let last = *r.last().unwrap();
                let gap = match (items[last].time, items[i].time) {
                    (Some(a), Some(b)) => (b - a).abs() > PLACE_GAP_S,
                    _ => false,
                };
                gap || dist(&pts[i], &centre(&pts, r)) > 1.0
            }
        };
        if new {
            runs.push(vec![i]);
        } else {
            runs.last_mut().unwrap().push(i);
        }
    }
    // 2. Склейка кусков с одинаковым светом (возвращение в то же место): каждый раз сливаем
    //    самую близкую пару, пока она ближе 1.
    let mut groups: Vec<Vec<usize>> = runs;
    loop {
        let centres: Vec<Pt> = groups.iter().map(|g| centre(&pts, g)).collect();
        let mut best: Option<(f32, usize, usize)> = None;
        for a in 0..groups.len() {
            for b in a + 1..groups.len() {
                let d = dist(&centres[a], &centres[b]);
                if d <= 1.0 && best.is_none_or(|x| d < x.0) {
                    best = Some((d, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else { break };
        let moved = groups.remove(b);
        groups[a].extend(moved);
    }
    // 3. Совсем маленькие места — к ближайшему по свету большому месту.
    let centres: Vec<Pt> = groups.iter().map(|g| centre(&pts, g)).collect();
    let min = PLACE_MIN.max((idx.len() as f32 * PLACE_MIN_SHARE).round() as usize);
    let big: Vec<usize> = (0..groups.len()).filter(|&g| groups[g].len() >= min).collect();
    let mut label_of = vec![0usize; items.len()];
    for (g, members) in groups.iter().enumerate() {
        let target = if members.len() >= min || big.is_empty() {
            g
        } else {
            *big.iter().min_by(|&&a, &&b| dist(&centres[g], &centres[a]).total_cmp(&dist(&centres[g], &centres[b]))).unwrap()
        };
        for &i in members {
            label_of[i] = target;
        }
    }
    idx.iter().map(|&i| label_of[i]).collect()
}

/// Номер группы (места) для каждого кадра; группы нумеруются по первому кадру во времени.
pub fn assign_groups(items: &[GroupItem]) -> Vec<usize> {
    let n = items.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| match (items[a].time, items[b].time) {
        (Some(x), Some(y)) => x.total_cmp(&y).then(a.cmp(&b)),
        _ => a.cmp(&b),
    });
    // Сессии: большой перерыв во времени — другая съёмка.
    let mut session = vec![0usize; n];
    let mut s = 0;
    for w in order.windows(2) {
        if let (Some(a), Some(b)) = (items[w[0]].time, items[w[1]].time) {
            if b - a > SESSION_GAP_S {
                s += 1;
            }
        }
        session[w[1]] = s;
    }
    let mut raw_label = vec![(0usize, false, 0usize); n];
    for sess in 0..=s {
        for flash in [false, true] {
            let idx: Vec<usize> = order.iter().copied().filter(|&i| session[i] == sess && items[i].flash == flash).collect();
            if idx.is_empty() {
                continue;
            }
            for (k, l) in places(items, &idx).into_iter().enumerate() {
                raw_label[idx[k]] = (sess, flash, l);
            }
        }
    }
    // Перенумерация по порядку появления во времени.
    let mut map: Vec<((usize, bool, usize), usize)> = Vec::new();
    let mut out = vec![0usize; n];
    for &i in &order {
        let key = raw_label[i];
        let id = match map.iter().find(|(k, _)| *k == key) {
            Some((_, id)) => *id,
            None => {
                map.push((key, map.len()));
                map.len() - 1
            }
        };
        out[i] = id;
    }
    out
}

pub fn plan_name(plan: u8) -> &'static str {
    match plan {
        2 => "крупные планы",
        1 => "средние планы",
        _ => "общие планы",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(time: f64, temp: f32, flash: bool) -> GroupItem {
        GroupItem { time: Some(time), flash, temp, tint: 0.0, scene_ev: Some(7.0) }
    }

    #[test]
    fn same_light_joins_even_when_interleaved() {
        let items = vec![
            item(0.0, 3000.0, false),
            item(5.0, 3080.0, false),
            item(9.0, 5200.0, false), // у окна
            item(12.0, 3020.0, false),
            item(20.0, 5400.0, false), // снова у окна
            item(25.0, 5300.0, false),
            item(30.0, 5450.0, true), // вспышка
            item(40.0, 2950.0, false),
        ];
        let g = assign_groups(&items);
        assert_eq!(g[0], g[1]);
        assert_eq!(g[0], g[3]);
        assert_eq!(g[0], g[7]);
        assert_eq!(g[2], g[4]);
        assert_eq!(g[2], g[5]);
        assert_ne!(g[0], g[2]);
        assert_ne!(g[5], g[6]);
    }

    #[test]
    fn lone_odd_frame_joins_a_place() {
        // Кадр с проектором: свет «пурпурный», но кадр один — своей группы не получает.
        let mut items: Vec<GroupItem> = (0..6).map(|i| item(i as f64 * 10.0, 4200.0, false)).collect();
        items.push(GroupItem { time: Some(70.0), flash: false, temp: 3000.0, tint: 52.0, scene_ev: Some(7.0) });
        let g = assign_groups(&items);
        assert!(g.iter().all(|&x| x == g[0]), "{g:?}");
    }

    /// Простой детерминированный шум для тестов: равномерный в [-1, 1].
    fn noise(i: usize, salt: u32) -> f32 {
        let mut x = (i as u32).wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(2_246_822_519);
        x ^= x >> 15;
        x = x.wrapping_mul(2_246_822_519);
        x ^= x >> 13;
        (x % 20_001) as f32 / 10_000.0 - 1.0
    }

    #[test]
    fn street_splits_from_indoor_with_overlapping_colour() {
        // Как в реальной съёмке: в помещении 600 кадров (свет 4300–4700 K, сцена 2.7–5.7 EV),
        // потом 46 кадров на улице (5000–5200 K, 9.8 EV). По цвету близко, по яркости — нет.
        let mut items = Vec::new();
        for i in 0..600 {
            let mut it = item(i as f64 * 5.0, 4500.0 + 100.0 * (noise(i, 1) + noise(i, 7)), false);
            it.tint = -20.0 + 2.0 * noise(i, 2);
            it.scene_ev = Some(4.2 + 0.75 * (noise(i, 3) + noise(i, 8)));
            items.push(it);
        }
        for i in 0..46 {
            let mut it = item(3100.0 + i as f64 * 6.0, 5100.0 + 100.0 * noise(i, 4), false);
            it.tint = -19.0 + 2.0 * noise(i, 5);
            it.scene_ev = Some(9.8 + 0.4 * noise(i, 6));
            items.push(it);
        }
        let g = assign_groups(&items);
        assert!(g[..600].iter().all(|&x| x == g[0]), "помещение разбилось: {:?}", &g[..600]);
        assert!(g[600..].iter().all(|&x| x == g[600]), "улица разбилась: {:?}", &g[600..]);
        assert_ne!(g[0], g[600], "улица слилась с помещением");
    }

    #[test]
    fn places_along_a_continuum_of_light_are_separated() {
        // Как на «CU October»: свет меняется плавно (зал 4500 K → панель 3300 K → лампы 2950 K),
        // есть переходные кадры, по цвету — сплошная полоса. Фотограф ходит по местам.
        let mut items = Vec::new();
        let mut t = 0.0;
        let mut add = |temp: f32, n: usize, items: &mut Vec<GroupItem>| {
            for k in 0..n {
                items.push(item(t, temp + 40.0 * noise(items.len() + k, 3), false));
                t += 8.0;
            }
        };
        add(4500.0, 12, &mut items); // зал
        add(3900.0, 2, &mut items); // проход
        add(3300.0, 15, &mut items); // панель
        add(2950.0, 10, &mut items); // лампы
        add(4500.0, 8, &mut items); // снова зал
        add(3300.0, 6, &mut items); // снова панель
        let g = assign_groups(&items);
        let (hall, panel, lamps) = (g[0], g[16], g[30]);
        assert!(hall != panel && panel != lamps && hall != lamps, "{g:?}");
        assert!(g[39..47].iter().all(|&x| x == hall), "возвращение в зал: {g:?}");
        assert!(g[47..].iter().all(|&x| x == panel), "возвращение к панели: {g:?}");
        assert!(g[14..29].iter().all(|&x| x == panel) && g[29..39].iter().all(|&x| x == lamps), "{g:?}");
    }

    #[test]
    fn content_noise_does_not_split() {
        // Один свет, свет кадров гуляет ±150 K.
        let temps = [3100.0, 3250.0, 2980.0, 3180.0, 3050.0, 3300.0, 3020.0];
        let items: Vec<GroupItem> = temps.iter().enumerate().map(|(i, &t)| item(i as f64 * 10.0, t, false)).collect();
        let g = assign_groups(&items);
        assert!(g.iter().all(|&x| x == g[0]), "{g:?}");
    }
}
