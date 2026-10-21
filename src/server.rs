//! Веб-интерфейс: сессия папки со снимками, API и встроенные страницы.
//!
//! Правки хранятся на двух уровнях: группа освещения (ББ и экспозиция для всех её кадров —
//! так съёмка остаётся единым целым) и отдельный кадр (тонкая экспозиция, угол, отбор).
//! Правки сохраняются в `photostab.json` рядом со снимками.

use crate::analysis::{self, Analysis};
use crate::batch;
use crate::develop::{self, DevelopParams};
use crate::export;
use crate::gpu::Gpu;
use crate::history;
use crate::raw::{self, CameraColor, LinearImage, PhotoMeta};
use anyhow::{Context, Result, anyhow};
use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, Path as AxPath, Query, State};
use axum::http::{StatusCode, Uri, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

#[derive(rust_embed::RustEmbed)]
#[folder = "web/"]
struct Assets;

const SAVE_FILE: &str = "photostab.json";
/// Версия автосвета. Правки света групп — поправка к автосвету своей версии: после смены
/// алгоритма они уже не про то (например, −19 по оттенку исправляло зелень старой версии), поэтому
/// при открытии старых правок свет сбрасывается к новому авто, а прежние значения хранятся в файле.
const AUTOLIGHT: u32 = 3;
const THUMB_SIDE: u32 = 420;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhotoOverride {
    /// Дополнительная экспозиция кадра (EV) поверх групповой.
    pub exposure: Option<f32>,
    pub angle: Option<f32>,
    pub pick: Option<bool>,
    /// false — выключить коррекцию кожи для кадра.
    pub skin: Option<bool>,
    /// Сила восстановления пересвета 0..1.
    #[serde(default)]
    pub recovery: Option<f32>,
}

impl PhotoOverride {
    fn any(&self) -> bool {
        self.exposure.is_some() || self.angle.is_some() || self.pick.is_some() || self.skin.is_some() || self.recovery.is_some()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroupOverride {
    /// Старый формат: абсолютные температура и оттенок для всей группы. При открытии папки
    /// переводятся в сдвиг (`dmired`, `dtint`) — у сцен теперь свой свет.
    pub temp: Option<f32>,
    pub tint: Option<f32>,
    /// Сдвиг света всей группы: температура в майредах и оттенок. Сцены сохраняют разницу между собой.
    #[serde(default)]
    pub dmired: Option<f32>,
    #[serde(default)]
    pub dtint: Option<f32>,
    /// Дополнительная экспозиция всей группы (EV).
    pub exposure: Option<f32>,
    /// Шумодав группы: выключен / сила яркостного и цветового (0..1). Не задано — авто по ISO.
    #[serde(default)]
    pub nr_off: Option<bool>,
    #[serde(default)]
    pub nr_luma: Option<f32>,
    #[serde(default)]
    pub nr_chroma: Option<f32>,
    /// Тон и цвет места: контраст, света, тени (−1..1), насыщенность (1 — как есть), резкость.
    #[serde(default)]
    pub contrast: Option<f32>,
    #[serde(default)]
    pub highlights: Option<f32>,
    #[serde(default)]
    pub shadows: Option<f32>,
    #[serde(default)]
    pub saturation: Option<f32>,
    #[serde(default)]
    pub sharpen: Option<f32>,
    /// Правка света, сделанная при старой версии автосвета (не применяется, хранится на память).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_before: Option<LightBefore>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LightBefore {
    pub autolight: u32,
    pub temp: Option<f32>,
    pub tint: Option<f32>,
    pub dmired: Option<f32>,
    pub dtint: Option<f32>,
}

impl GroupOverride {
    fn any(&self) -> bool {
        self.temp.is_some()
            || self.tint.is_some()
            || self.dmired.is_some()
            || self.dtint.is_some()
            || self.exposure.is_some()
            || self.nr_off.is_some()
            || self.nr_luma.is_some()
            || self.nr_chroma.is_some()
            || self.contrast.is_some()
            || self.highlights.is_some()
            || self.shadows.is_some()
            || self.saturation.is_some()
            || self.sharpen.is_some()
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Saved {
    /// Версия автосвета, при которой сделаны правки (0 — до появления поля).
    #[serde(default)]
    autolight: u32,
    photos: HashMap<String, PhotoOverride>,
    /// Ключ — имя первого кадра группы.
    groups: HashMap<String, GroupOverride>,
}

/// Уменьшенный кадр (ориентация показа) в 16 битах — компактно для больших съёмок.
struct Small {
    w: usize,
    h: usize,
    data: Vec<u16>,
}

impl Small {
    fn from(img: &LinearImage) -> Self {
        Small { w: img.width, h: img.height, data: img.data.iter().map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect() }
    }
    fn to_linear(&self) -> LinearImage {
        LinearImage { width: self.w, height: self.h, data: self.data.iter().map(|v| *v as f32 / 65535.0).collect() }
    }
}

struct Photo {
    path: PathBuf,
    name: String,
    meta: PhotoMeta,
    cam: CameraColor,
    small: Small,
    analysis: Analysis,
    ovr: PhotoOverride,
    version: u64,
}

#[derive(Clone, Serialize)]
#[serde(tag = "kind")]
enum Status {
    Idle,
    Analyzing { done: usize, total: usize },
    Exporting { done: usize, total: usize },
}

#[derive(Default)]
struct Session {
    folder: Option<PathBuf>,
    photos: Vec<Photo>,
    groups: HashMap<usize, GroupOverride>,
    failed: Vec<(String, String)>,
    status: Option<Status>,
    last_export: Option<Value>,
    thumbs: HashMap<usize, (u64, Vec<u8>)>,
    started: Option<Instant>,
    /// Файлы папки (известны сразу, до конца анализа) — для миниатюр раздела «Отбор».
    files: Vec<PathBuf>,
    /// Миниатюры JPEG камеры по имени кадра.
    cam_thumbs: HashMap<String, Vec<u8>>,
    /// Номер сессии (папки): в адресах картинок, чтобы кэш браузера не путал кадры разных папок.
    sid: u64,
    /// Готовые превью и JPEG камеры: при листании соседние кадры отдаются сразу.
    previews: VecDeque<(String, Arc<Vec<u8>>)>,
    /// Полноразмерные JPEG камеры (для зума) — их немного, они крупные.
    fulls: VecDeque<(String, Arc<Vec<u8>>)>,
    /// Сообщение пользователю при открытии папки (например, сброс старых правок света).
    notice: Option<String>,
}

const PREVIEW_CACHE: usize = 80;
const FULL_CACHE: usize = 4;

fn lookup(q: &VecDeque<(String, Arc<Vec<u8>>)>, key: &str) -> Option<Arc<Vec<u8>>> {
    q.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

fn remember(q: &mut VecDeque<(String, Arc<Vec<u8>>)>, key: String, v: Arc<Vec<u8>>, cap: usize) {
    q.retain(|(k, _)| *k != key);
    q.push_back((key, v));
    while q.len() > cap {
        q.pop_front();
    }
}

/// Полный кадр до шумодава и резкости — источник плиток зума.
struct MidFrame {
    id: usize,
    version: u64,
    px: Vec<u32>,
    w: u32,
    h: u32,
    sw: u32,
    sh: u32,
    recipe: develop::Recipe,
}

pub struct App {
    gpu: Arc<Gpu>,
    s: RwLock<Session>,
    tx: broadcast::Sender<String>,
    loupe_cache: std::sync::Mutex<Option<(PathBuf, Arc<raw::RawFrame>)>>,
    /// Кадры для зума (текущий и предыдущий) и замок их построения.
    zoom: std::sync::Mutex<VecDeque<Arc<MidFrame>>>,
    zoom_build: std::sync::Mutex<()>,
}

type Shared = Arc<App>;

impl App {
    fn emit(&self, v: Value) {
        let _ = self.tx.send(v.to_string());
    }
}

fn effective(p: &Photo, g: &GroupOverride) -> DevelopParams {
    let a = &p.analysis;
    let mut d = analysis::auto_params(a);
    if let Some(t) = g.temp {
        d.temp = t;
    }
    if let Some(t) = g.tint {
        d.tint = t;
    }
    // Сдвиг группы ложится на свет сцены: в майредах — так он одинаково заметен и в тёплых, и в холодных.
    if let Some(dm) = g.dmired {
        d.temp = shift_temp(d.temp, dm);
    }
    if let Some(dt) = g.dtint {
        d.tint += dt;
    }
    d.exposure_ev += g.exposure.unwrap_or(0.0) + p.ovr.exposure.unwrap_or(0.0);
    d.contrast = g.contrast.unwrap_or(0.0);
    d.highlights = g.highlights.unwrap_or(0.0);
    d.shadows = g.shadows.unwrap_or(0.0);
    if let Some(v) = g.saturation {
        d.saturation = v;
    }
    if let Some(v) = g.sharpen {
        d.sharpen = v;
    }
    if let Some(a) = p.ovr.angle {
        d.angle_deg = a;
    }
    if p.ovr.skin == Some(false) {
        d.skin = None;
    }
    if let Some(r) = p.ovr.recovery {
        d.recovery = r;
    }
    d.denoise = denoise_for(p, g, d.exposure_ev);
    crate::raw_denoise::apply(&mut d, p.analysis.noise);
    d
}

/// Шумодав кадра: ручные значения группы или авто по «эффективному ISO».
fn denoise_for(p: &Photo, g: &GroupOverride, ev: f32) -> Option<crate::denoise::DenoiseParams> {
    if g.nr_off == Some(true) {
        return None;
    }
    let auto = crate::denoise::auto_strength(p.meta.iso, ev).unwrap_or(crate::denoise::DenoiseParams { luma: 0.0, chroma: 0.0 });
    let d = crate::denoise::DenoiseParams { luma: g.nr_luma.unwrap_or(auto.luma), chroma: g.nr_chroma.unwrap_or(auto.chroma) };
    d.active().then_some(d)
}

fn picked(p: &Photo) -> bool {
    p.ovr.pick.unwrap_or(p.analysis.cull.pick)
}

// ---------- запуск ----------

/// Веб-режим: сервер + браузер.
pub fn run(gpu: Gpu, open: Option<PathBuf>) -> Result<()> {
    run_with(gpu, open, None)
}

/// Сервер. Если передан ready, порт отправляется туда (окно приложения), браузер не открывается.
pub fn run_with(gpu: Gpu, open: Option<PathBuf>, ready: Option<std::sync::mpsc::Sender<u16>>) -> Result<()> {
    let desktop = ready.is_some();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let (tx, _) = broadcast::channel(256);
        let app: Shared = Arc::new(App {
            gpu: Arc::new(gpu),
            s: RwLock::new(Session::default()),
            tx,
            loupe_cache: std::sync::Mutex::new(None),
            zoom: std::sync::Mutex::new(VecDeque::new()),
            zoom_build: std::sync::Mutex::new(()),
        });
        crate::denoise::warm_up();
        if let Some(dir) = open {
            start_analysis(app.clone(), dir);
        }
        let router = Router::new()
            .route("/api/state", get(api_state))
            .route("/api/events", get(api_events))
            .route("/api/open", post(api_open))
            .route("/api/pick-folder", post(api_pick_folder))
            .route("/api/upload", post(api_upload).layer(DefaultBodyLimit::disable()))
            .route("/api/photo/{id}/thumb", get(api_thumb))
            .route("/api/photo/{id}/preview", get(api_preview))
            .route("/api/photo/{id}/tile", get(api_tile))
            .route("/api/photo/{id}/warm", post(api_warm))
            .route("/api/photo/{id}", post(api_photo))
            .route("/api/group/{gid}", post(api_group))
            .route("/api/export", post(api_export))
            .route("/api/reveal", post(api_reveal))
            .route("/api/history", get(api_history))
            .route("/api/history/remove", post(api_history_remove))
            .route("/api/history/cover/{file}", get(api_history_cover))
            .route("/api/cam/{name}", get(api_cam))
            .fallback(get(asset))
            .with_state(app);
        let mut listener = None;
        let first: u16 = std::env::var("PHOTOSTAB_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8080);
        for port in first..first.saturating_add(20) {
            if let Ok(l) = tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
                listener = Some((l, port));
                break;
            }
        }
        let (listener, port) = listener.ok_or_else(|| anyhow!("нет свободного порта {first}–{}", first.saturating_add(19)))?;
        let url = format!("http://127.0.0.1:{port}");
        if let Some(tx) = ready {
            let _ = tx.send(port);
        } else {
            println!("PhotoStab: {url}  (закройте это окно, чтобы выйти)");
        }
        if std::env::var("PHOTOSTAB_NO_BROWSER").is_err() && !desktop {
            let mut cmd = std::process::Command::new("cmd");
            cmd.args(["/C", "start", "", &url]);
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW — без мигания консоли
            }
            let _ = cmd.spawn();
        }
        axum::serve(listener, router).await?;
        Ok(())
    })
}

// ---------- статика ----------

async fn asset(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match Assets::get(path) {
        Some(f) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            ([(header::CONTENT_TYPE, mime.as_ref().to_string()), (header::CACHE_CONTROL, "no-cache".into())], f.data.into_owned())
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "не найдено").into_response(),
    }
}

fn err(e: impl std::fmt::Display) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))).into_response()
}

// ---------- состояние ----------

fn state_json(s: &Session) -> Value {
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, p) in s.photos.iter().enumerate() {
        groups.entry(p.analysis.group).or_default().push(i);
    }
    let mut gids: Vec<usize> = groups.keys().copied().collect();
    gids.sort();
    let groups_json: Vec<Value> = gids
        .iter()
        .map(|g| {
            let m = &groups[g];
            let a = &s.photos[m[0]].analysis;
            let o = s.groups.get(g).cloned().unwrap_or_default();
            let flash = s.photos[m[0]].meta.flash_fired();
            // Авто-сила шумодава для группы (медиана по кадрам) — для показа на ползунках.
            let mut autos: Vec<(f32, f32)> = m
                .iter()
                .map(|&i| {
                    let p = &s.photos[i];
                    let ev = p.analysis.exposure.ev + o.exposure.unwrap_or(0.0);
                    crate::denoise::auto_strength(p.meta.iso, ev).map(|d| (d.luma, d.chroma)).unwrap_or((0.0, 0.0))
                })
                .collect();
            autos.sort_by(|x, y| x.0.total_cmp(&y.0));
            let auto_nr = autos[autos.len() / 2];
            let sub = a.subgroup;
            // Свет группы (фон для сцен) и он же со сдвигом пользователя — для кружка и списка.
            let gwb = a.group_wb.unwrap_or((a.wb.temp, a.wb.tint));
            let shifted = {
                let t = o.dmired.map(|dm| shift_temp(gwb.0, dm)).unwrap_or(gwb.0);
                (o.temp.unwrap_or(t), o.tint.unwrap_or(gwb.1 + o.dtint.unwrap_or(0.0)))
            };
            json!({
                "id": g,
                "count": m.len(),
                "flash": flash,
                "parent": sub.map(|x| x.0),
                "plan": sub.map(|x| analysis::grouping::plan_name(x.1)),
                "auto_temp": gwb.0, "auto_tint": gwb.1,
                "temp": shifted.0, "tint": shifted.1,
                "exposure": o.exposure.unwrap_or(0.0),
                "contrast": o.contrast.unwrap_or(0.0),
                "highlights": o.highlights.unwrap_or(0.0),
                "shadows": o.shadows.unwrap_or(0.0),
                "saturation": o.saturation.unwrap_or(1.0),
                "sharpen": o.sharpen.unwrap_or(DevelopParams::default().sharpen),
                "tone_set": [o.contrast.is_some(), o.highlights.is_some(), o.shadows.is_some(), o.saturation.is_some(), o.sharpen.is_some()],
                "nr_off": o.nr_off.unwrap_or(false),
                "nr_luma": o.nr_luma.unwrap_or(auto_nr.0),
                "nr_chroma": o.nr_chroma.unwrap_or(auto_nr.1),
                "nr_luma_set": o.nr_luma.is_some(),
                "nr_chroma_set": o.nr_chroma.is_some(),
                "temp_set": o.temp.is_some() || o.dmired.is_some(), "tint_set": o.tint.is_some() || o.dtint.is_some(), "exposure_set": o.exposure.is_some(),
                "confidence": a.wb.confidence,
                "overridden": o.any(),
                "cover": m.iter().copied().find(|&i| picked(&s.photos[i])).unwrap_or(m[0]),
                "picked": m.iter().filter(|&&i| picked(&s.photos[i])).count(),
            })
        })
        .collect();
    let photos: Vec<Value> = s
        .photos
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let a = &p.analysis;
            let g = s.groups.get(&a.group).cloned().unwrap_or_default();
            let e = effective(p, &g);
            let out = out_dims(p, e.angle_deg);
            json!({
                "id": i,
                "name": p.name,
                "time": p.meta.datetime_original,
                "iso": p.meta.iso,
                "group": a.group,
                "series": a.cull.series,
                "series_len": a.cull.series_len,
                "pick": picked(p),
                "auto_pick": a.cull.pick,
                "score": a.cull.score,
                "reasons": a.cull.reasons,
                "eyes": a.cull.eyes,
                "faces": a.faces.len(),
                "w": a.width, "h": a.height, "full_w": a.full_w, "full_h": a.full_h,
                "out_w": out.0, "out_h": out.1,
                "temp": e.temp, "tint": e.tint,
                "auto_temp": a.wb.temp, "auto_tint": a.wb.tint,
                "scene": a.scene, "scene_len": a.scene_len,
                "ev": e.exposure_ev,
                "auto_ev": a.exposure.ev,
                "ev_reason": a.exposure.reason,
                "photo_exposure": p.ovr.exposure.unwrap_or(0.0),
                "angle": e.angle_deg,
                "auto_angle": if a.horizon.apply { a.horizon.angle } else { 0.0 },
                "horizon_found": a.horizon.angle,
                "horizon_conf": a.horizon.confidence,
                "skin": e.skin.is_some(),
                "skin_auto": a.skin_shift.is_some(),
                "recovery": e.recovery,
                "auto_recovery": a.recovery,
                "recovery_set": p.ovr.recovery.is_some(),
                "exposure_set": p.ovr.exposure.is_some(),
                "angle_set": p.ovr.angle.is_some(),
                "nr": e.denoise.map(|d| json!({ "luma": d.luma, "chroma": d.chroma })),
                "edited": p.ovr.any(),
                "v": p.version,
            })
        })
        .collect();
    let files: Vec<String> = s.files.iter().map(|p| p.file_stem().unwrap_or_default().to_string_lossy().to_string()).collect();
    json!({
        "folder": s.folder,
        "sid": s.sid,
        "files": files,
        "status": s.status.clone().unwrap_or(Status::Idle),
        "groups": groups_json,
        "photos": photos,
        "failed": s.failed,
        "last_export": s.last_export,
        "notice": s.notice,
    })
}

async fn api_state(State(app): State<Shared>) -> Response {
    let s = app.s.read().unwrap();
    let mut v = state_json(&s);
    v["gpu"] = json!(app.gpu.name);
    Json(v).into_response()
}

async fn api_events(State(app): State<Shared>) -> impl IntoResponse {
    let rx = app.tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|m| m.ok().map(|s| Ok::<Event, std::convert::Infallible>(Event::default().data(s))));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ---------- открытие папки и анализ ----------

#[derive(Deserialize)]
struct OpenReq {
    path: String,
}

async fn api_open(State(app): State<Shared>, Json(r): Json<OpenReq>) -> Response {
    let p = PathBuf::from(r.path.trim().trim_matches('"'));
    if !p.is_dir() {
        return err(format!("папка не найдена: {}", p.display()));
    }
    if matches!(app.s.read().unwrap().status, Some(Status::Analyzing { .. } | Status::Exporting { .. })) {
        return err("идёт обработка, дождитесь окончания");
    }
    start_analysis(app.clone(), p);
    Json(json!({ "ok": true })).into_response()
}

/// Окно программы (HWND) — владелец системных диалогов; 0 — окна нет (режим браузера).
pub static MAIN_HWND: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);

/// Окно-владелец для диалога: диалог открывается поверх PhotoStab и не прячется за ним.
struct Owner(std::num::NonZeroIsize);

impl raw_window_handle::HasWindowHandle for Owner {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let h = raw_window_handle::Win32WindowHandle::new(self.0);
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Win32(h)) })
    }
}

impl raw_window_handle::HasDisplayHandle for Owner {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(raw_window_handle::DisplayHandle::windows())
    }
}

/// Обычный проводник Windows в режиме выбора папки.
async fn api_pick_folder(State(app): State<Shared>) -> Response {
    let start = app
        .s
        .read()
        .unwrap()
        .folder
        .as_ref()
        .and_then(|f| f.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(dirs_pictures);
    let r = tokio::task::spawn_blocking(move || {
        let mut d = rfd::FileDialog::new().set_title("Папка со съёмкой (RAW)").set_directory(start);
        if let Some(h) = std::num::NonZeroIsize::new(MAIN_HWND.load(std::sync::atomic::Ordering::Relaxed)) {
            d = d.set_parent(&Owner(h));
        }
        d.pick_folder()
    })
    .await;
    match r {
        Ok(path) => Json(json!({ "path": path })).into_response(),
        Err(_) => err("не удалось открыть выбор папки"),
    }
}

async fn api_upload(State(app): State<Shared>, mut mp: Multipart) -> Response {
    let base = dirs_pictures().join("PhotoStab").join(chrono_like_stamp());
    if let Err(e) = tokio::fs::create_dir_all(&base).await {
        return err(e);
    }
    let mut count = 0;
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return err(e),
        };
        let Some(name) = field.file_name().map(|s| s.to_string()) else { continue };
        let name = Path::new(&name).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if !raw::is_raw_file(Path::new(&name)) {
            continue;
        }
        let mut field = field;
        let mut file = match tokio::fs::File::create(base.join(&name)).await {
            Ok(f) => f,
            Err(e) => return err(e),
        };
        loop {
            match field.chunk().await {
                Ok(Some(bytes)) => {
                    if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut file, &bytes).await {
                        return err(e);
                    }
                }
                Ok(None) => break,
                Err(e) => return err(e),
            }
        }
        count += 1;
    }
    if count == 0 {
        return err("среди файлов нет RAW");
    }
    start_analysis(app.clone(), base.clone());
    Json(json!({ "ok": true, "folder": base, "count": count })).into_response()
}

fn dirs_pictures() -> PathBuf {
    std::env::var("USERPROFILE").map(|h| PathBuf::from(h).join("Pictures")).unwrap_or_else(|_| PathBuf::from("."))
}

fn chrono_like_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("upload_{secs}")
}

fn list_raws(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| raw::is_raw_file(p)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn start_analysis(app: Shared, folder: PathBuf) {
    let files = list_raws(&folder);
    {
        let mut s = app.s.write().unwrap();
        *s = Session {
            folder: Some(folder.clone()),
            status: Some(Status::Analyzing { done: 0, total: files.len() }),
            started: Some(Instant::now()),
            files: files.clone(),
            sid: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0),
            ..Default::default()
        };
    }
    // Номера кадров в новой папке начинаются заново — кадры зума от прошлой не годятся.
    app.zoom.lock().unwrap().clear();
    *app.loupe_cache.lock().unwrap() = None;
    history::opened(&folder);
    app.emit(json!({ "type": "state" }));
    std::thread::spawn(move || {
        let last = std::sync::Mutex::new(Instant::now());
        let (items, failed) = batch::analyze_all(&files, &|done, total| {
            if let Ok(mut s) = app.s.write() {
                s.status = Some(Status::Analyzing { done, total });
            }
            let mut l = last.lock().unwrap();
            if l.elapsed().as_millis() > 150 || done == total {
                *l = Instant::now();
                app.emit(json!({ "type": "progress", "kind": "analyze", "done": done, "total": total }));
            }
        });
        let saved: Saved = std::fs::read_to_string(folder.join(SAVE_FILE))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let mut photos: Vec<Photo> = items
            .into_iter()
            .map(|it| {
                let name = it.path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                let ovr = saved.photos.get(&name).cloned().unwrap_or_default();
                Photo {
                    small: Small::from(&it.small),
                    name,
                    path: it.path,
                    meta: it.meta,
                    cam: it.cam,
                    analysis: it.analysis,
                    ovr,
                    version: 1,
                }
            })
            .collect();
        photos.sort_by(|a, b| {
            a.meta.capture_seconds().unwrap_or(0.0).total_cmp(&b.meta.capture_seconds().unwrap_or(0.0)).then(a.name.cmp(&b.name))
        });
        let mut groups = HashMap::new();
        let mut light_reset = false;
        for (key, mut o) in saved.groups {
            if let Some(p) = photos.iter().find(|p| p.name == key) {
                if saved.autolight < AUTOLIGHT {
                    light_reset |= reset_light(&mut o, saved.autolight);
                } else {
                    migrate_light(&mut o, p.analysis.group_wb.unwrap_or((p.analysis.wb.temp, p.analysis.wb.tint)));
                }
                groups.insert(p.analysis.group, o);
            }
        }
        {
            let mut s = app.s.write().unwrap();
            if s.folder.as_deref() != Some(folder.as_path()) {
                return;
            }
            s.photos = photos;
            s.groups = groups;
            s.notice = light_reset.then(|| {
                "Автосвет обновлён: прошлые правки температуры и оттенка групп были поправкой к старому автосвету и сброшены. Экспозиция и шумодав сохранены.".to_string()
            });
            s.failed = failed.into_iter().map(|(p, e)| (p.display().to_string(), e)).collect();
            s.status = Some(Status::Idle);
        }
        app.emit(json!({ "type": "state" }));
        update_history(&app, true);
    });
}

/// Температура со сдвигом в майредах.
fn shift_temp(t: f32, dmired: f32) -> f32 {
    (1e6 / (1e6 / t.max(1.0) as f64 + dmired as f64).max(60.0)).clamp(1800.0, 15000.0) as f32
}

/// Правка света от старой версии автосвета: убрать (новый авто — без неё) и сохранить на память.
/// true — было что сбрасывать.
fn reset_light(o: &mut GroupOverride, version: u32) -> bool {
    if o.temp.is_none() && o.tint.is_none() && o.dmired.is_none() && o.dtint.is_none() {
        return false;
    }
    o.light_before = Some(LightBefore { autolight: version, temp: o.temp.take(), tint: o.tint.take(), dmired: o.dmired.take(), dtint: o.dtint.take() });
    true
}

/// Старая абсолютная правка света группы → сдвиг относительно авто-света группы.
fn migrate_light(o: &mut GroupOverride, group_wb: (f32, f32)) {
    if let Some(t) = o.temp.take() {
        o.dmired = Some(o.dmired.unwrap_or(0.0) + (1e6 / t - 1e6 / group_wb.0.max(1.0)));
    }
    if let Some(t) = o.tint.take() {
        o.dtint = Some(o.dtint.unwrap_or(0.0) + (t - group_wb.1));
    }
}

/// Число кадров и отобранных — в историю проектов; `cover` — заодно обновить обложку.
fn update_history(app: &App, cover: bool) {
    let (folder, count, picked, cover_path) = {
        let s = app.s.read().unwrap();
        let Some(folder) = s.folder.clone() else { return };
        let picked: Vec<&Photo> = s.photos.iter().filter(|p| picked(p)).collect();
        let cover_path = picked.first().copied().or(s.photos.first()).map(|p| p.path.clone());
        (folder, s.photos.len(), picked.len(), cover_path)
    };
    let jpeg = if cover { cover_path.and_then(|p| cam_jpeg(&p, 360).ok()) } else { None };
    history::counts(&folder, count, picked, jpeg);
}

/// JPEG камеры: быстрый средний из CR3 (или встроенный), в ориентации показа, длинная сторона `long`.
/// `long == 0` — полный размер (для зума).
fn cam_jpeg(path: &Path, long: u32) -> Result<Vec<u8>> {
    let o = raw::read_meta(path).map(|m| m.orientation).unwrap_or(1);
    let img = if long > 0 && long <= 1700 { raw::camera_jpeg(path)? } else { raw::embedded_preview(path)? };
    let img = match o {
        3 => img.rotate180(),
        6 => img.rotate90(),
        8 => img.rotate270(),
        _ => img,
    };
    let img = if long == 0 { img.to_rgba8() } else { img.resize(long, long, image::imageops::FilterType::Triangle).to_rgba8() };
    let r = develop::Rgba { width: img.width(), height: img.height(), data: img.into_raw() };
    export::encode_jpeg(&r, if long == 0 { 90 } else { 85 }, None)
}

#[derive(Deserialize)]
struct CamQ {
    w: Option<u32>,
    full: Option<u8>,
}

/// Миниатюра/просмотр JPEG камеры по имени кадра (раздел «Отбор» и сторона «Камера»).
async fn api_cam(State(app): State<Shared>, AxPath(name): AxPath<String>, Query(q): Query<CamQ>) -> Response {
    let full = q.full == Some(1);
    let long = if full { 0 } else { q.w.unwrap_or(THUMB_SIDE).clamp(120, 2400) };
    let small = !full && long <= 480;
    let key = format!("c:{name}:{long}");
    let path = {
        let s = app.s.read().unwrap();
        if small {
            if let Some(b) = s.cam_thumbs.get(&name) {
                return jpeg_response(b.clone());
            }
        } else if let Some(b) = lookup(if full { &s.fulls } else { &s.previews }, &key) {
            return jpeg_response((*b).clone());
        }
        s.files.iter().find(|p| p.file_stem().map(|x| x.to_string_lossy() == name.as_str()).unwrap_or(false)).cloned()
    };
    let Some(path) = path else { return err("нет такого кадра") };
    match tokio::task::spawn_blocking(move || cam_jpeg(&path, long)).await {
        Ok(Ok(bytes)) => {
            let mut s = app.s.write().unwrap();
            if small {
                s.cam_thumbs.insert(name, bytes.clone());
            } else if full {
                remember(&mut s.fulls, key, Arc::new(bytes.clone()), FULL_CACHE);
            } else {
                remember(&mut s.previews, key, Arc::new(bytes.clone()), PREVIEW_CACHE);
            }
            drop(s);
            jpeg_response(bytes)
        }
        Ok(Err(e)) => err(e),
        Err(e) => err(e),
    }
}

async fn api_history() -> Response {
    let list: Vec<Value> = history::load()
        .into_iter()
        .map(|e| {
            let exists = Path::new(&e.path).is_dir();
            json!({
                "path": e.path, "name": e.name, "opened": e.opened, "count": e.count, "picked": e.picked,
                "cover": e.cover.map(|c| format!("/api/history/cover/{c}")), "exists": exists,
            })
        })
        .collect();
    Json(json!({ "projects": list })).into_response()
}

async fn api_history_remove(Json(r): Json<OpenReq>) -> Response {
    history::remove(&r.path);
    Json(json!({ "ok": true })).into_response()
}

async fn api_history_cover(AxPath(file): AxPath<String>) -> Response {
    match history::cover_file(&file).and_then(|p| std::fs::read(p).ok()) {
        Some(b) => jpeg_response(b),
        None => (StatusCode::NOT_FOUND, "нет обложки").into_response(),
    }
}

fn save(s: &Session) {
    let Some(folder) = &s.folder else { return };
    let mut saved = Saved { autolight: AUTOLIGHT, ..Default::default() };
    for p in &s.photos {
        if p.ovr.any() {
            saved.photos.insert(p.name.clone(), p.ovr.clone());
        }
    }
    for (g, o) in &s.groups {
        if let Some(p) = s.photos.iter().find(|p| p.analysis.group == *g) {
            saved.groups.insert(p.name.clone(), o.clone());
        }
    }
    if let Ok(t) = serde_json::to_string_pretty(&saved) {
        let _ = std::fs::write(folder.join(SAVE_FILE), t);
    }
    let picked_n = s.photos.iter().filter(|p| picked(p)).count();
    history::counts(folder, s.photos.len(), picked_n, None);
}

// ---------- правки ----------

/// Поле из JSON-патча: отсутствует — не трогаем, null — сброс к авто, число/bool — значение.
fn patch_f32(v: &Value, key: &str, slot: &mut Option<f32>) {
    if let Some(x) = v.get(key) {
        *slot = x.as_f64().map(|f| f as f32);
    }
}

fn patch_bool(v: &Value, key: &str, slot: &mut Option<bool>) {
    if let Some(x) = v.get(key) {
        *slot = x.as_bool();
    }
}

async fn api_photo(State(app): State<Shared>, AxPath(id): AxPath<usize>, Json(v): Json<Value>) -> Response {
    {
        let mut s = app.s.write().unwrap();
        let Some(p) = s.photos.get_mut(id) else { return err("нет такого кадра") };
        let look = |o: &PhotoOverride| (o.exposure, o.angle, o.skin, o.recovery);
        let before = look(&p.ovr);
        if v.get("reset").and_then(|x| x.as_bool()) == Some(true) {
            p.ovr = PhotoOverride::default();
        }
        patch_f32(&v, "exposure", &mut p.ovr.exposure);
        patch_f32(&v, "angle", &mut p.ovr.angle);
        patch_bool(&v, "pick", &mut p.ovr.pick);
        patch_bool(&v, "skin", &mut p.ovr.skin);
        patch_f32(&v, "recovery", &mut p.ovr.recovery);
        if p.ovr.pick == Some(p.analysis.cull.pick) {
            p.ovr.pick = None;
        }
        // Отбор картинку не меняет — превью и плитки остаются в кэше.
        if look(&p.ovr) != before {
            p.version += 1;
        }
        save(&s);
    }
    app.emit(json!({ "type": "state" }));
    Json(json!({ "ok": true })).into_response()
}

async fn api_group(State(app): State<Shared>, AxPath(gid): AxPath<usize>, Json(v): Json<Value>) -> Response {
    {
        let mut s = app.s.write().unwrap();
        let o = s.groups.entry(gid).or_default();
        if v.get("reset").and_then(|x| x.as_bool()) == Some(true) {
            *o = GroupOverride::default();
        }
        patch_f32(&v, "temp", &mut o.temp);
        patch_f32(&v, "tint", &mut o.tint);
        patch_f32(&v, "dmired", &mut o.dmired);
        patch_f32(&v, "dtint", &mut o.dtint);
        patch_f32(&v, "exposure", &mut o.exposure);
        patch_bool(&v, "nr_off", &mut o.nr_off);
        patch_f32(&v, "nr_luma", &mut o.nr_luma);
        patch_f32(&v, "nr_chroma", &mut o.nr_chroma);
        patch_f32(&v, "contrast", &mut o.contrast);
        patch_f32(&v, "highlights", &mut o.highlights);
        patch_f32(&v, "shadows", &mut o.shadows);
        patch_f32(&v, "saturation", &mut o.saturation);
        patch_f32(&v, "sharpen", &mut o.sharpen);
        for p in s.photos.iter_mut().filter(|p| p.analysis.group == gid) {
            p.version += 1;
        }
        save(&s);
    }
    app.emit(json!({ "type": "state" }));
    Json(json!({ "ok": true })).into_response()
}

// ---------- изображения ----------

#[derive(Deserialize, Clone)]
struct PreviewQ {
    w: Option<u32>,
    before: Option<u8>,
    /// 0 — без шумодава (сравнение «без шумодава»).
    nr: Option<u8>,
    /// 1 — черновое превью, пока тянут ползунок: параметры из запроса, без шумодава и без кэша.
    live: Option<u8>,
    temp: Option<f32>,
    tint: Option<f32>,
    gexp: Option<f32>,
    pexp: Option<f32>,
    rec: Option<f32>,
    angle: Option<f32>,
    /// Тон и цвет места (живое превью): контраст, света, тени, насыщенность, резкость.
    con: Option<f32>,
    hi: Option<f32>,
    sh: Option<f32>,
    sat: Option<f32>,
    shp: Option<f32>,
}

/// Размеры результата в полном разрешении (после поворота и автокропа).
fn out_dims(p: &Photo, angle: f32) -> (u32, u32) {
    let a = &p.analysis;
    let (fw, fh) = (a.full_w as u32, a.full_h as u32);
    let o = p.meta.orientation;
    let (sw, sh) = if matches!(o, 5..=8) { (fh, fw) } else { (fw, fh) };
    let g = develop::geometry(sw.max(1), sh.max(1), o, angle, None);
    (g.dw, g.dh)
}

fn jpeg_response(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "max-age=3600")], bytes).into_response()
}

fn render(app: &App, id: usize, long: u32, before: bool, with_nr: bool, live: Option<&PreviewQ>) -> Result<Vec<u8>> {
    let (lin, cam, params, path, orientation) = {
        let s = app.s.read().unwrap();
        let p = s.photos.get(id).context("нет такого кадра")?;
        let g = s.groups.get(&p.analysis.group).cloned().unwrap_or_default();
        let lin = if before { None } else { Some(p.small.to_linear()) };
        let mut params = effective(p, &g);
        if let Some(q) = live {
            if let Some(t) = q.temp {
                params.temp = t;
            }
            if let Some(t) = q.tint {
                params.tint = t;
            }
            if q.gexp.is_some() || q.pexp.is_some() {
                let base = analysis::auto_params(&p.analysis).exposure_ev;
                params.exposure_ev = base + q.gexp.or(g.exposure).unwrap_or(0.0) + q.pexp.or(p.ovr.exposure).unwrap_or(0.0);
            }
            if let Some(r) = q.rec {
                params.recovery = r.clamp(0.0, 1.0);
            }
            if let Some(a) = q.angle {
                params.angle_deg = a;
            }
            if let Some(v) = q.con {
                params.contrast = v.clamp(-1.0, 1.0);
            }
            if let Some(v) = q.hi {
                params.highlights = v.clamp(-1.0, 1.0);
            }
            if let Some(v) = q.sh {
                params.shadows = v.clamp(-1.0, 1.0);
            }
            if let Some(v) = q.sat {
                params.saturation = v.clamp(0.0, 2.0);
            }
            if let Some(v) = q.shp {
                params.sharpen = v.clamp(0.0, 1.5);
            }
        }
        // Миниатюры — без шумодава (скорость); превью просмотра — с ним, в масштабе экрана.
        if !with_nr {
            params.denoise = None;
        }
        (lin, p.cam.clone(), params, p.path.clone(), p.meta.orientation)
    };
    if before {
        let img = if long <= 1700 { raw::camera_jpeg(&path)? } else { raw::embedded_preview(&path)? };
        let img = match orientation {
            3 => img.rotate180(),
            6 => img.rotate90(),
            8 => img.rotate270(),
            _ => img,
        };
        let img = img.resize(long, long, image::imageops::FilterType::Triangle).to_rgba8();
        let r = develop::Rgba { width: img.width(), height: img.height(), data: img.into_raw() };
        return export::encode_jpeg(&r, 88, None);
    }
    let lin = lin.unwrap();
    let r = develop::recipe(&params, &cam, 1, Some(long));
    let img = app.gpu.develop_linear(&lin, &r)?;
    export::encode_jpeg(&img, 88, None)
}

async fn api_thumb(State(app): State<Shared>, AxPath(id): AxPath<usize>) -> Response {
    let ver = {
        let s = app.s.read().unwrap();
        let Some(p) = s.photos.get(id) else { return err("нет такого кадра") };
        if let Some((v, bytes)) = s.thumbs.get(&id) {
            if *v == p.version {
                return jpeg_response(bytes.clone());
            }
        }
        p.version
    };
    let a = app.clone();
    match tokio::task::spawn_blocking(move || render(&a, id, THUMB_SIDE, false, false, None)).await {
        Ok(Ok(bytes)) => {
            app.s.write().unwrap().thumbs.insert(id, (ver, bytes.clone()));
            jpeg_response(bytes)
        }
        Ok(Err(e)) => err(e),
        Err(e) => err(e),
    }
}

async fn api_preview(State(app): State<Shared>, AxPath(id): AxPath<usize>, Query(q): Query<PreviewQ>) -> Response {
    let a = app.clone();
    let long = q.w.unwrap_or(1600).clamp(200, 2400);
    let before = q.before == Some(1);
    let live = q.live == Some(1);
    let nr = !live && q.nr != Some(0);
    let key = if live {
        None
    } else {
        let s = app.s.read().unwrap();
        let v = if before { 0 } else { s.photos.get(id).map(|p| p.version).unwrap_or(0) };
        let k = format!("p:{id}:{v}:{long}:{}:{}", before as u8, nr as u8);
        if let Some(b) = lookup(&s.previews, &k) {
            return jpeg_response((*b).clone());
        }
        Some(k)
    };
    match tokio::task::spawn_blocking(move || render(&a, id, long, before, nr, if live { Some(&q) } else { None })).await {
        Ok(Ok(bytes)) if live => ([(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")], bytes).into_response(),
        Ok(Ok(bytes)) => {
            if let Some(k) = key {
                remember(&mut app.s.write().unwrap().previews, k, Arc::new(bytes.clone()), PREVIEW_CACHE);
            }
            jpeg_response(bytes)
        }
        Ok(Err(e)) => err(e),
        Err(e) => err(e),
    }
}

// ---------- лупа 100% ----------

/// Плитки зума: 512×512 в полном разрешении результата.
const TILE: u32 = 512;
/// Запас вокруг плитки: шумодаву и резкости нужен контекст, чтобы не было швов.
const TILE_PAD: u32 = 64;

#[derive(Deserialize)]
struct TileQ {
    x: u32,
    y: u32,
    /// Уровень: 0 — 100%, 1 — 50% (плитка 512 покрывает 1024 пикселя кадра).
    l: Option<u8>,
    /// 0 — без шумодава.
    nr: Option<u8>,
}

/// Полный кадр до шумодава для зума: из кэша или проявка (RAW держим в памяти — его обычно
/// и смотрят).
fn mid_frame(app: &App, id: usize) -> Result<Arc<MidFrame>> {
    let (path, params, version) = {
        let s = app.s.read().unwrap();
        let p = s.photos.get(id).context("нет такого кадра")?;
        let g = s.groups.get(&p.analysis.group).cloned().unwrap_or_default();
        (p.path.clone(), effective(p, &g), p.version)
    };
    let find = || app.zoom.lock().unwrap().iter().find(|m| m.id == id && m.version == version).cloned();
    if let Some(m) = find() {
        return Ok(m);
    }
    // Одновременные запросы плиток одного кадра ждут одну проявку, а не делают свою.
    let _building = app.zoom_build.lock().unwrap();
    if let Some(m) = find() {
        return Ok(m);
    }
    let frame = {
        let mut cache = app.loupe_cache.lock().unwrap();
        match cache.as_ref() {
            Some((p, f)) if *p == path => f.clone(),
            _ => {
                let f = Arc::new(raw::decode(&path)?);
                *cache = Some((path.clone(), f.clone()));
                f
            }
        }
    };
    let recipe = develop::recipe(&params, &frame.color, frame.meta.orientation, None);
    let (px, w, h) = app.gpu.develop_mid(&frame, &recipe)?;
    let m = Arc::new(MidFrame { id, version, px, w, h, sw: frame.width as u32, sh: frame.height as u32, recipe });
    let mut z = app.zoom.lock().unwrap();
    z.retain(|x| x.id != id);
    z.push_back(m.clone());
    while z.len() > 2 {
        z.pop_front();
    }
    Ok(m)
}

fn tile(app: &App, id: usize, q: &TileQ) -> Result<Vec<u8>> {
    let m = mid_frame(app, id)?;
    let k = if q.l == Some(1) { 2 } else { 1 };
    let span = TILE * k;
    let (x0, y0) = (q.x * span, q.y * span);
    if x0 >= m.w || y0 >= m.h {
        return Err(anyhow!("плитка за краем кадра"));
    }
    let (tw, th) = (span.min(m.w - x0), span.min(m.h - y0));
    let (ax, ay) = (x0.saturating_sub(TILE_PAD), y0.saturating_sub(TILE_PAD));
    let (bx, by) = ((x0 + tw + TILE_PAD).min(m.w), (y0 + th + TILE_PAD).min(m.h));
    let (aw, ah) = (bx - ax, by - ay);
    let mut crop: Vec<u32> = Vec::with_capacity((aw * ah) as usize);
    for y in ay..by {
        let s = (y * m.w + ax) as usize;
        crop.extend_from_slice(&m.px[s..s + aw as usize]);
    }
    if q.nr != Some(0) {
        if let Some(p) = m.recipe.denoise.filter(|p| p.active()) {
            let region = ((x0 - ax) as usize, (y0 - ay) as usize, tw as usize, th as usize);
            crate::denoise::denoise_packed(&mut crop, aw as usize, ah as usize, p, Some(region))?;
        }
    }
    let img = app.gpu.finish_crop(&crop, aw, ah, &m.recipe, m.sw, m.sh)?;
    // Внутренняя часть плитки; на уровне 50% — с усреднением 2×2.
    let (ow, oh) = (tw.div_ceil(k), th.div_ceil(k));
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = [0u32; 4];
            let mut n = 0;
            for dy in 0..k {
                for dx in 0..k {
                    let (x, y) = (ox * k + dx, oy * k + dy);
                    if x < tw && y < th {
                        let i = (((y0 - ay + y) * aw + (x0 - ax + x)) * 4) as usize;
                        for c in 0..4 {
                            acc[c] += img.data[i + c] as u32;
                        }
                        n += 1;
                    }
                }
            }
            let o = ((oy * ow + ox) * 4) as usize;
            for c in 0..4 {
                out[o + c] = (acc[c] / n.max(1)) as u8;
            }
        }
    }
    export::encode_jpeg(&develop::Rgba { width: ow, height: oh, data: out }, 90, None)
}

async fn api_tile(State(app): State<Shared>, AxPath(id): AxPath<usize>, Query(q): Query<TileQ>) -> Response {
    let a = app.clone();
    match tokio::task::spawn_blocking(move || tile(&a, id, &q)).await {
        Ok(Ok(bytes)) => jpeg_response(bytes),
        Ok(Err(e)) => err(e),
        Err(e) => err(e),
    }
}

/// Подготовить кадр к зуму заранее (пока его смотрят целиком): RAW и проявка — в фоне.
async fn api_warm(State(app): State<Shared>, AxPath(id): AxPath<usize>) -> Response {
    tokio::task::spawn_blocking(move || {
        let _ = mid_frame(&app, id);
    });
    Json(json!({ "ok": true })).into_response()
}

// ---------- экспорт ----------

#[derive(Deserialize)]
struct ExportReq {
    only_picked: bool,
    quality: Option<u8>,
    long_side: Option<u32>,
    dir: Option<String>,
}

/// Папка экспорта, если своя не указана, — внутри папки съёмки.
const EXPORT_DIR: &str = "Edited";

/// Основа имён экспорта — имя папки съёмки: «CU October 2026_001.jpg».
fn export_base(folder: &Path) -> String {
    folder.file_name().map(|n| n.to_string_lossy().trim().to_string()).filter(|n| !n.is_empty()).unwrap_or_else(|| "photo".into())
}

async fn api_export(State(app): State<Shared>, Json(r): Json<ExportReq>) -> Response {
    let (jobs, out, base) = {
        let s = app.s.read().unwrap();
        if !matches!(s.status, Some(Status::Idle) | None) {
            return err("идёт обработка, дождитесь окончания");
        }
        let Some(folder) = s.folder.clone() else { return err("сначала откройте папку") };
        let jobs: Vec<(PathBuf, DevelopParams)> = s
            .photos
            .iter()
            .filter(|p| !r.only_picked || picked(p))
            .map(|p| (p.path.clone(), effective(p, &s.groups.get(&p.analysis.group).cloned().unwrap_or_default())))
            .collect();
        let out = r.dir.clone().filter(|d| !d.trim().is_empty()).map(PathBuf::from).unwrap_or_else(|| folder.join(EXPORT_DIR));
        (jobs, out, export_base(&folder))
    };
    if jobs.is_empty() {
        return err("нет кадров для экспорта");
    }
    if let Err(e) = std::fs::create_dir_all(&out) {
        return err(e);
    }
    {
        let mut s = app.s.write().unwrap();
        s.status = Some(Status::Exporting { done: 0, total: jobs.len() });
    }
    app.emit(json!({ "type": "state" }));
    let opts = batch::ExportOptions {
        quality: r.quality.unwrap_or(92).clamp(60, 100),
        long_side: r.long_side.filter(|v| *v > 0),
        sequence: Some(base),
    };
    std::thread::spawn(move || {
        let t = Instant::now();
        let gpu = app.gpu.clone();
        let results = batch::export_all(&gpu, &jobs, &out, &opts, &|done, total, _| {
            if let Ok(mut s) = app.s.write() {
                s.status = Some(Status::Exporting { done, total });
            }
            app.emit(json!({ "type": "progress", "kind": "export", "done": done, "total": total }));
        });
        let errors: Vec<String> =
            results.iter().filter_map(|(p, r)| r.as_ref().err().map(|e| format!("{}: {e}", p.display()))).collect();
        let summary = json!({
            "dir": out,
            "count": results.len() - errors.len(),
            "errors": errors,
            "seconds": t.elapsed().as_secs_f32(),
        });
        {
            let mut s = app.s.write().unwrap();
            s.status = Some(Status::Idle);
            s.last_export = Some(summary.clone());
        }
        app.emit(json!({ "type": "exported", "summary": summary }));
        app.emit(json!({ "type": "state" }));
    });
    Json(json!({ "ok": true })).into_response()
}

async fn api_reveal(Json(r): Json<OpenReq>) -> Response {
    let _ = std::process::Command::new("explorer").arg(&r.path).spawn();
    Json(json!({ "ok": true })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_names_come_from_shoot_folder() {
        assert_eq!(export_base(Path::new(r"D:\Shoots\CU October 2026")), "CU October 2026");
        assert_eq!(export_base(Path::new(r"D:\Shoots\CU October 2026\")), "CU October 2026");
        assert_eq!(export_base(Path::new(r"E:\")), "photo");
    }

    #[test]
    fn light_edits_of_old_autolight_are_kept_aside() {
        let mut o = GroupOverride { dtint: Some(-7.8), exposure: Some(0.3), ..Default::default() };
        assert!(reset_light(&mut o, 0));
        assert_eq!((o.dmired, o.dtint, o.temp, o.tint), (None, None, None, None));
        assert_eq!(o.exposure, Some(0.3), "экспозиция остаётся");
        assert_eq!(o.light_before.as_ref().and_then(|l| l.dtint), Some(-7.8));
        let mut plain = GroupOverride { exposure: Some(0.3), ..Default::default() };
        assert!(!reset_light(&mut plain, 0), "без правок света нечего сбрасывать");
        // В файле поле версии и сохранённая правка переживают запись и чтение.
        let saved = Saved { autolight: AUTOLIGHT, photos: HashMap::new(), groups: [("IMG_1".to_string(), o)].into() };
        let back: Saved = serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        assert_eq!(back.autolight, AUTOLIGHT);
        assert!(back.groups["IMG_1"].light_before.is_some());
        let old: Saved = serde_json::from_str(r#"{"photos":{},"groups":{}}"#).unwrap();
        assert_eq!(old.autolight, 0);
    }

    #[test]
    fn old_absolute_light_becomes_group_shift() {
        let mut o = GroupOverride { tint: Some(-19.0), temp: Some(4000.0), ..Default::default() };
        migrate_light(&mut o, (4278.0, -3.6));
        assert_eq!((o.temp, o.tint), (None, None));
        assert!((o.dtint.unwrap() - (-15.4)).abs() < 1e-4);
        // Группа вернётся ровно к тому, что пользователь выставил.
        assert!((shift_temp(4278.0, o.dmired.unwrap()) - 4000.0).abs() < 0.5);
        // Тот же сдвиг на тёплой сцене — в майредах, поэтому в кельвинах меньше.
        let warm = shift_temp(3000.0, o.dmired.unwrap());
        assert!(warm < 3000.0 && warm > 2850.0, "{warm}");
    }
}
