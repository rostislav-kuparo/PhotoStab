//! Простой профилировщик стадий (включается переменной окружения PHOTOSTAB_PROFILE).

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn table() -> &'static Mutex<BTreeMap<&'static str, (f64, u64)>> {
    static T: OnceLock<Mutex<BTreeMap<&'static str, (f64, u64)>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("PHOTOSTAB_PROFILE").is_ok())
}

/// Выполнить `f` и учесть время под именем `name`.
pub fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let dt = t.elapsed().as_secs_f64();
    let mut m = table().lock().unwrap();
    let e = m.entry(name).or_insert((0.0, 0));
    e.0 += dt;
    e.1 += 1;
    r
}

/// Сводка: суммарное процессорное время по стадиям (мс на кадр).
pub fn report() {
    if !enabled() {
        return;
    }
    let m = table().lock().unwrap();
    let total: f64 = m.values().map(|v| v.0).sum();
    eprintln!("стадия                      мс/вызов   вызовов   доля");
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    for (k, (t, n)) in v {
        eprintln!("{:<26} {:>9.1} {:>9} {:>6.1}%", k, t * 1000.0 / *n as f64, n, 100.0 * t / total.max(1e-9));
    }
}
