//! Отдельное окно приложения: интерфейс открывается в собственном окне Windows (WebView2),
//! без браузера. Внутри — тот же локальный сервер, что и в веб-режиме.

use crate::gpu::Gpu;
use crate::server;
use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;
use std::time::Duration;
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::window::{Icon, WindowBuilder};
use wry::{WebContext, WebViewBuilder};

// Рисунок значка общий с build.rs (значок exe).
include!("icon_draw.rs");

pub fn run(open: Option<PathBuf>) -> Result<()> {
    let gpu = Gpu::new()?;
    let (tx, rx) = std::sync::mpsc::channel::<u16>();
    std::thread::spawn(move || {
        if let Err(e) = server::run_with(gpu, open, Some(tx)) {
            eprintln!("сервер: {e}");
        }
    });
    let port = rx.recv_timeout(Duration::from_secs(30)).context("интерфейс не запустился")?;

    let event_loop = EventLoop::new();
    let icon = Icon::from_rgba(icon_rgba(64), 64, 64).ok();
    let window = WindowBuilder::new()
        .with_title("PhotoStab")
        .with_inner_size(LogicalSize::new(1440.0, 900.0))
        .with_min_inner_size(LogicalSize::new(900.0, 600.0))
        .with_window_icon(icon)
        .with_maximized(true)
        .build(&event_loop)
        .map_err(|e| anyhow!("окно: {e}"))?;
    {
        use tao::platform::windows::WindowExtWindows;
        server::MAIN_HWND.store(window.hwnd(), std::sync::atomic::Ordering::Relaxed);
    }
    // Данные WebView2 — в профиле пользователя, а не рядом с программой.
    let data_dir = std::env::var("LOCALAPPDATA")
        .map(|d| PathBuf::from(d).join("PhotoStab").join("webview"))
        .unwrap_or_else(|_| std::env::temp_dir().join("PhotoStab-webview"));
    let mut ctx = WebContext::new(Some(data_dir));
    let webview = WebViewBuilder::new_with_web_context(&mut ctx)
        .with_url(format!("http://127.0.0.1:{port}"))
        .with_background_color((42, 42, 42, 255))
        .with_devtools(cfg!(debug_assertions))
        .build(&window)
        .map_err(|e| anyhow!("WebView2: {e}. Установите Microsoft Edge WebView2 Runtime."))?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let _keep = &webview;
        if let Event::WindowEvent { event: WindowEvent::CloseRequested, .. } = event {
            *control_flow = ControlFlow::Exit;
        }
    });
}
