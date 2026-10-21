//! Недавние проекты (папки со съёмками) для главного меню: путь, когда открывали, сколько
//! кадров и сколько отобрано, обложка. Хранится в `%LOCALAPPDATA%\PhotoStab\history.json`,
//! обложки — маленькие JPEG рядом, в `covers\`. В папки съёмок ничего не пишется.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX: usize = 40;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub path: String,
    pub name: String,
    /// Когда открывали последний раз (секунды Unix).
    pub opened: u64,
    #[serde(default)]
    pub count: usize,
    #[serde(default)]
    pub picked: usize,
    /// Имя файла обложки в `covers\`.
    #[serde(default)]
    pub cover: Option<String>,
}

/// Чтение-изменение-запись истории не должны пересекаться между потоками.
static LOCK: Mutex<()> = Mutex::new(());

fn dir() -> PathBuf {
    std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir()).join("PhotoStab")
}

fn file() -> PathBuf {
    dir().join("history.json")
}

pub fn covers_dir() -> PathBuf {
    dir().join("covers")
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Один и тот же путь, как бы его ни написали (регистр, слэши, хвостовой разделитель).
fn same(a: &str, b: &Path) -> bool {
    let norm = |s: &str| s.trim_end_matches(['\\', '/']).replace('/', "\\").to_lowercase();
    norm(a) == norm(&b.to_string_lossy())
}

/// Имя файла обложки по пути папки (FNV-1a).
fn cover_name(path: &Path) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.to_string_lossy().to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}.jpg")
}

pub fn load() -> Vec<Entry> {
    std::fs::read_to_string(file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn store(v: &[Entry]) {
    let _ = std::fs::create_dir_all(dir());
    if let Ok(t) = serde_json::to_string_pretty(v) {
        let _ = std::fs::write(file(), t);
    }
}

/// Изменить (или создать) запись папки и поднять её в начало списка.
fn update(folder: &Path, f: impl FnOnce(&mut Entry)) {
    let _g = LOCK.lock();
    let mut v = load();
    let mut e = match v.iter().position(|e| same(&e.path, folder)) {
        Some(i) => v.remove(i),
        None => Entry {
            path: folder.to_string_lossy().to_string(),
            name: folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| folder.to_string_lossy().to_string()),
            opened: now(),
            count: 0,
            picked: 0,
            cover: None,
        },
    };
    f(&mut e);
    v.insert(0, e);
    v.truncate(MAX);
    store(&v);
}

/// Папка открыта.
pub fn opened(folder: &Path) {
    update(folder, |e| e.opened = now());
}

/// Анализ закончен или изменился отбор: сколько кадров, сколько отобрано и обложка.
pub fn counts(folder: &Path, count: usize, picked: usize, cover_jpeg: Option<Vec<u8>>) {
    let cover = cover_jpeg.and_then(|bytes| {
        let name = cover_name(folder);
        let _ = std::fs::create_dir_all(covers_dir());
        std::fs::write(covers_dir().join(&name), bytes).ok().map(|_| name)
    });
    update(folder, |e| {
        e.count = count;
        e.picked = picked;
        if cover.is_some() {
            e.cover = cover;
        }
    });
}

/// Убрать папку из списка (сама папка и снимки не трогаются).
pub fn remove(path: &str) {
    let _g = LOCK.lock();
    let mut v = load();
    v.retain(|e| {
        let keep = !same(&e.path, Path::new(path));
        if !keep {
            if let Some(c) = &e.cover {
                let _ = std::fs::remove_file(covers_dir().join(c));
            }
        }
        keep
    });
    store(&v);
}

/// Безопасное имя файла обложки (только то, что мы сами выдаём).
pub fn cover_file(name: &str) -> Option<PathBuf> {
    let ok = name.len() == 20 && name.ends_with(".jpg") && name[..16].bytes().all(|b| b.is_ascii_hexdigit());
    ok.then(|| covers_dir().join(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_compare_loosely() {
        assert!(same("E:\\Arbeit\\Shoot\\", Path::new("e:/arbeit/shoot")));
        assert!(!same("E:\\Arbeit\\Shoot2", Path::new("E:\\Arbeit\\Shoot")));
    }

    #[test]
    fn cover_names_are_safe() {
        let n = cover_name(Path::new("E:\\Arbeit\\Shoot"));
        assert!(cover_file(&n).is_some());
        assert!(cover_file("..\\history.json").is_none());
    }
}
