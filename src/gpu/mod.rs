//! GPU-конвейер проявки на wgpu (DX12/Vulkan): демозаик, цвет, геометрия, резкость.

use crate::develop::{Recipe, Rgba, geometry};
use crate::raw::{LinearImage, RawFrame};
use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use std::sync::Mutex;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DemosaicParams {
    w: u32,
    h: u32,
    _a: u32,
    _b: u32,
    cfa: [u32; 4],
    black: [f32; 4],
    inv_range: [f32; 4],
    mul: [f32; 4],
    clip: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorParams {
    w: u32,
    h: u32,
    lut_n: u32,
    lut_m: u32,
    m0: [f32; 4],
    m1: [f32; 4],
    m2: [f32; 4],
    mul: [f32; 4],
    gain: f32,
    sat: f32,
    lut_max: f32,
    skin_on: f32,
    skin: [f32; 4],
    skin2: [f32; 4],
    skin_ell: [[f32; 4]; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransformParams {
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    a: [f32; 4],
    b: [f32; 4],
    scale: f32,
    sharpen: f32,
    _p0: f32,
    _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HighlightParams {
    w: u32,
    h: u32,
    bw: u32,
    bh: u32,
    bs: u32,
    _a: u32,
    _b: u32,
    _c: u32,
    mul: [f32; 4],
    thr: [f32; 4],
}

/// Заполнение пустот карты цветности (push-pull): каждый блок получает цвет света от ближайших
/// блоков, где были яркие необрезанные пиксели. Возвращает нормированную цветность (r+g+b = 1).
fn fill_chroma(stats: &[[f32; 4]], bw: usize, bh: usize) -> Vec<[f32; 4]> {
    let mut levels: Vec<(Vec<[f32; 4]>, usize, usize)> = Vec::new();
    // Уровень 0: (r, g, b, вес) — суммы цвета ярких пикселей.
    let base: Vec<[f32; 4]> = stats.iter().map(|s| if s[3] > 1e-3 { *s } else { [0.0; 4] }).collect();
    levels.push((base, bw, bh));
    while levels.last().map(|l| l.1 > 1 || l.2 > 1).unwrap_or(false) {
        let (prev, pw, ph) = levels.last().unwrap();
        let (nw, nh) = (pw.div_ceil(2), ph.div_ceil(2));
        let mut next = vec![[0f32; 4]; nw * nh];
        for y in 0..*ph {
            for x in 0..*pw {
                let s = prev[y * pw + x];
                let d = &mut next[(y / 2) * nw + x / 2];
                for k in 0..4 {
                    d[k] += s[k];
                }
            }
        }
        levels.push((next, nw, nh));
    }
    // Сверху вниз: пустые блоки берут цвет у родителя.
    let norm = |s: [f32; 4]| -> [f32; 4] {
        let t = s[0] + s[1] + s[2];
        if s[3] > 0.0 && t > 1e-6 { [s[0] / t, s[1] / t, s[2] / t, 1.0] } else { [0.0; 4] }
    };
    let top = norm(levels.last().unwrap().0[0]);
    let mut parent: Vec<[f32; 4]> = vec![if top[3] > 0.0 { top } else { [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0] }];
    let mut pw = 1usize;
    for li in (0..levels.len() - 1).rev() {
        let (lv, w, h) = &levels[li];
        let mut cur = vec![[0f32; 4]; w * h];
        for y in 0..*h {
            for x in 0..*w {
                let own = norm(lv[y * w + x]);
                cur[y * w + x] = if own[3] > 0.0 { own } else { parent[(y / 2) * pw + x / 2] };
            }
        }
        parent = cur;
        pw = *w;
    }
    // Сильное сглаживание (σ ≈ 4 блока) и подмешивание общего цвета света: без ступенек.
    let sigma = 4.0f32;
    let r = 10isize;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let blur = |src: &Vec<[f32; 4]>, horiz: bool| -> Vec<[f32; 4]> {
        let mut out = vec![[0f32; 4]; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let (mut acc, mut ws) = ([0f32; 3], 0f32);
                for (j, kw) in k.iter().enumerate() {
                    let o = j as isize - r;
                    let (sx, sy) = if horiz { (x as isize + o, y as isize) } else { (x as isize, y as isize + o) };
                    if sx < 0 || sy < 0 || sx >= bw as isize || sy >= bh as isize {
                        continue;
                    }
                    let s = src[sy as usize * bw + sx as usize];
                    for c in 0..3 {
                        acc[c] += kw * s[c];
                    }
                    ws += kw;
                }
                out[y * bw + x] = [acc[0] / ws, acc[1] / ws, acc[2] / ws, 1.0];
            }
        }
        out
    };
    let smooth = blur(&blur(&parent, true), false);
    let global = if top[3] > 0.0 { top } else { [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0] };
    smooth
        .iter()
        .map(|s| {
            let m = [0.7 * s[0] + 0.3 * global[0], 0.7 * s[1] + 0.3 * global[1], 0.7 * s[2] + 0.3 * global[2]];
            let t = (m[0] + m[1] + m[2]).max(1e-6);
            [m[0] / t, m[1] / t, m[2] / t, 1.0]
        })
        .collect()
}

struct FullBufs {
    pixels: u64,
    raw: wgpu::Buffer,
    cfa: wgpu::Buffer,
    vh: wgpu::Buffer,
    pq: wgpu::Buffer,
    rr: wgpu::Buffer,
    gg: wgpu::Buffer,
    bb: wgpu::Buffer,
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    prepare_cfa: wgpu::ComputePipeline,
    rcd_dir: wgpu::ComputePipeline,
    rcd_green: wgpu::ComputePipeline,
    rcd_rb_nongreen: wgpu::ComputePipeline,
    rcd_rb_green: wgpu::ComputePipeline,
    prepare_rgb: wgpu::ComputePipeline,
    color: wgpu::ComputePipeline,
    transform: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    hl_stats: wgpu::ComputePipeline,
    hl_reconstruct: wgpu::ComputePipeline,
    full: Mutex<Option<FullBufs>>,
    pub name: String,
}

const STORAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);

impl Gpu {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("видеокарта не найдена: {e}"))?;
        let info = adapter.get_info();
        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("photostab"),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .context("не удалось открыть видеокарту")?;

        let module = |src: &str, label: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let demosaic = module(include_str!("demosaic.wgsl"), "demosaic");
        let colorm = module(include_str!("color.wgsl"), "color");
        let transformm = module(include_str!("transform.wgsl"), "transform");
        let highlightsm = module(include_str!("highlights.wgsl"), "highlights");
        let pipe = |m: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            prepare_cfa: pipe(&demosaic, "prepare_cfa"),
            rcd_dir: pipe(&demosaic, "rcd_dir"),
            rcd_green: pipe(&demosaic, "rcd_green"),
            rcd_rb_nongreen: pipe(&demosaic, "rcd_rb_nongreen"),
            rcd_rb_green: pipe(&demosaic, "rcd_rb_green"),
            prepare_rgb: pipe(&colorm, "prepare_rgb"),
            color: pipe(&colorm, "color"),
            transform: pipe(&transformm, "transform"),
            finish: pipe(&transformm, "finish"),
            hl_stats: pipe(&highlightsm, "hl_stats"),
            hl_reconstruct: pipe(&highlightsm, "hl_reconstruct"),
            name: format!("{} ({:?})", info.name, info.backend),
            device,
            queue,
            full: Mutex::new(None),
        })
    }

    fn storage(&self, size: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16).next_multiple_of(4),
            usage: STORAGE,
            mapped_at_creation: false,
        })
    }

    fn uniform<T: Pod>(&self, v: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(v),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn bind(&self, p: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
        let layout = p.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries })
    }

    fn pass(enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, x: u32, y: u32) {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(p);
        cp.set_bind_group(0, bg, &[]);
        cp.dispatch_workgroups(x, y, 1);
    }

    fn groups_1d(n: u64) -> (u32, u32) {
        let groups = n.div_ceil(256);
        let x = groups.min(65535);
        (x as u32, groups.div_ceil(x).max(1) as u32)
    }

    fn read_back(&self, buf: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("GPU: {e}"))?;
        rx.recv()?.map_err(|e| anyhow!("GPU: {e}"))?;
        let data = slice.get_mapped_range().map_err(|e| anyhow!("GPU: {e}"))?.to_vec();
        staging.unmap();
        Ok(data)
    }

    fn color_params(&self, w: u32, h: u32, r: &Recipe) -> ColorParams {
        let m = &r.matrix;
        let sk = r.skin.unwrap_or(crate::develop::SkinShift {
            hue_center: 0.0,
            half_width: 1.0,
            shift: 0.0,
            chroma_scale: 1.0,
            chroma_min: 0.02,
            chroma_max: 0.2,
        });
        ColorParams {
            w,
            h,
            lut_n: crate::color::SHOULDER_LUT as u32,
            lut_m: crate::color::CONTRAST_LUT as u32,
            m0: [m[0][0], m[0][1], m[0][2], 0.0],
            m1: [m[1][0], m[1][1], m[1][2], 0.0],
            m2: [m[2][0], m[2][1], m[2][2], 0.0],
            mul: [r.mul[0], r.mul[1], r.mul[2], r.clip],
            gain: r.gain,
            sat: r.saturation,
            lut_max: r.lut_max,
            skin_on: if r.skin.is_some() { 1.0 } else { 0.0 },
            skin: [sk.hue_center, sk.half_width, sk.shift, sk.chroma_scale],
            skin2: [sk.chroma_min, sk.chroma_max, r.skin_n as f32, 0.0],
            skin_ell: r.skin_ell,
        }
    }

    /// Цвет + геометрия + резкость; на входе rr/gg/bb (камера после ББ), packed — рабочий буфер.
    fn finish_pipeline(
        &self,
        enc: &mut wgpu::CommandEncoder,
        sw: u32,
        sh: u32,
        r: &Recipe,
        rgb: [&wgpu::Buffer; 3],
        packed: &wgpu::Buffer,
        mid: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        keep: &mut Vec<wgpu::BindGroup>,
    ) -> (u32, u32) {
        let cpar = self.uniform(&self.color_params(sw, sh, r));
        let lut = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lut"),
            contents: bytemuck::cast_slice(&r.luts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bg = self.bind(&self.color, &[(0, &cpar), (1, rgb[0]), (2, rgb[1]), (3, rgb[2]), (4, &lut), (5, packed)]);
        let (gx, gy) = Self::groups_1d(sw as u64 * sh as u64);
        Self::pass(enc, &self.color, &bg, gx, gy);
        keep.push(bg);

        let g = geometry(sw, sh, r.orientation, r.angle_deg, r.out_long);
        let tpar = self.uniform(&TransformParams {
            sw,
            sh,
            dw: g.dw,
            dh: g.dh,
            a: [g.a[0], g.a[1], g.a[2], 0.0],
            b: [g.b[0], g.b[1], g.b[2], 0.0],
            scale: g.scale,
            sharpen: r.sharpen * (1.0 / g.scale).sqrt().max(0.5),
            _p0: 0.0,
            _p1: 0.0,
        });
        let wg = (g.dw.div_ceil(16), g.dh.div_ceil(16));
        let bg = self.bind(&self.transform, &[(0, &tpar), (1, packed), (2, mid)]);
        Self::pass(enc, &self.transform, &bg, wg.0, wg.1);
        keep.push(bg);
        (g.dw, g.dh)
    }

    /// Резкость и упаковка в RGBA8 (после возможного шумодава).
    fn finish_pass(&self, enc: &mut wgpu::CommandEncoder, dw: u32, dh: u32, r: &Recipe, sw: u32, sh: u32, mid: &wgpu::Buffer, dst: &wgpu::Buffer) {
        let g = geometry(sw, sh, r.orientation, r.angle_deg, r.out_long);
        let tpar = self.uniform(&TransformParams {
            sw,
            sh,
            dw,
            dh,
            a: [0.0; 4],
            b: [0.0; 4],
            scale: g.scale,
            sharpen: r.sharpen * (1.0 / g.scale).sqrt().max(0.5),
            _p0: 0.0,
            _p1: 0.0,
        });
        let bg = self.bind(&self.finish, &[(0, &tpar), (2, mid), (3, dst)]);
        Self::pass(enc, &self.finish, &bg, dw.div_ceil(16), dh.div_ceil(16));
    }

    /// Цвет → геометрия → [шумодав на CPU-стороне] → резкость. Возвращает размеры результата в `dst`.
    #[allow(clippy::too_many_arguments)]
    fn run_tail(
        &self,
        mut enc: wgpu::CommandEncoder,
        sw: u32,
        sh: u32,
        r: &Recipe,
        rgb: [&wgpu::Buffer; 3],
        packed: &wgpu::Buffer,
        mid: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        nr_region: Option<crate::denoise::Region>,
    ) -> Result<(u32, u32)> {
        let mut keep = Vec::new();
        let (dw, dh) = self.finish_pipeline(&mut enc, sw, sh, r, rgb, packed, mid, dst, &mut keep);
        if let Some(p) = r.denoise.filter(|p| p.active()) {
            self.queue.submit(Some(enc.finish()));
            let bytes = self.read_back(mid, dw as u64 * dh as u64 * 4)?;
            let mut px: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
            drop(bytes);
            crate::denoise::denoise_packed(&mut px, dw as usize, dh as usize, p, nr_region)?;
            self.queue.write_buffer(mid, 0, bytemuck::cast_slice(&px));
            enc = self.device.create_command_encoder(&Default::default());
        }
        self.finish_pass(&mut enc, dw, dh, r, sw, sh, mid, dst);
        self.queue.submit(Some(enc.finish()));
        Ok((dw, dh))
    }

    /// Восстановление пересвета в плоскостях камеры (после ББ, без обрезки). `bs` — размер блока
    /// карты цветности в пикселях.
    fn reconstruct(&self, rgb: [&wgpu::Buffer; 3], w: u32, h: u32, r: &Recipe, bs: u32) -> Result<()> {
        if r.recovery <= 0.0 {
            return Ok(());
        }
        let (bw, bh) = (w.div_ceil(bs), h.div_ceil(bs));
        let hp = self.uniform(&HighlightParams {
            w,
            h,
            bw,
            bh,
            bs,
            _a: 0,
            _b: 0,
            _c: 0,
            mul: [r.mul[0], r.mul[1], r.mul[2], crate::develop::recovery_white(r.recovery)],
            thr: [0.98, 0.0, 0.0, 0.0],
        });
        let n = (bw * bh) as u64 * 16;
        let stats = self.storage(n, "hl-stats");
        let mut enc = self.device.create_command_encoder(&Default::default());
        let bg = self.bind(&self.hl_stats, &[(0, &hp), (1, rgb[0]), (2, rgb[1]), (3, rgb[2]), (4, &stats)]);
        Self::pass(&mut enc, &self.hl_stats, &bg, bw, bh);
        self.queue.submit(Some(enc.finish()));
        let raw = self.read_back(&stats, n)?;
        let s: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        let map = fill_chroma(s, bw as usize, bh as usize);
        let chroma = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hl-chroma"),
            contents: bytemuck::cast_slice(&map),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        let bg = self.bind(&self.hl_reconstruct, &[(0, &hp), (1, rgb[0]), (2, rgb[1]), (3, rgb[2]), (5, &chroma)]);
        Self::pass(&mut enc, &self.hl_reconstruct, &bg, w.div_ceil(16), h.div_ceil(16));
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    /// Полная проявка байеровского кадра.
    pub fn develop_raw(&self, frame: &RawFrame, r: &Recipe) -> Result<Rgba> {
        self.develop_raw_ext(frame, r, None)
    }

    /// То же, но шумодав только в заданной области результата (для лупы 100%).
    pub fn develop_raw_ext(&self, frame: &RawFrame, r: &Recipe, nr_region: Option<crate::denoise::Region>) -> Result<Rgba> {
        let (w, h) = (frame.width as u32, frame.height as u32);
        let guard = self.demosaic_locked(frame, r.mul, r.clip, r.raw_nr.as_ref())?;
        let b = guard.as_ref().unwrap();
        self.reconstruct([&b.rr, &b.gg, &b.bb], w, h, r, 32)?;
        let enc = self.device.create_command_encoder(&Default::default());
        // cfa больше не нужен — в нём упакованный результат цвета; vh/pq — под геометрию.
        let (dw, dh) = self.run_tail(enc, w, h, r, [&b.rr, &b.gg, &b.bb], &b.cfa, &b.vh, &b.pq, nr_region)?;
        let data = self.read_back(&b.pq, dw as u64 * dh as u64 * 4)?;
        Ok(Rgba { width: dw, height: dh, data })
    }

    /// Полный кадр до шумодава и резкости (цвет + геометрия, упаковка 10:10:10) — для зума:
    /// из него плитки 100% режутся без повторной проявки всего кадра.
    pub fn develop_mid(&self, frame: &RawFrame, r: &Recipe) -> Result<(Vec<u32>, u32, u32)> {
        let (w, h) = (frame.width as u32, frame.height as u32);
        let guard = self.demosaic_locked(frame, r.mul, r.clip, r.raw_nr.as_ref())?;
        let b = guard.as_ref().unwrap();
        self.reconstruct([&b.rr, &b.gg, &b.bb], w, h, r, 32)?;
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut keep = Vec::new();
        let (dw, dh) = self.finish_pipeline(&mut enc, w, h, r, [&b.rr, &b.gg, &b.bb], &b.cfa, &b.vh, &b.pq, &mut keep);
        self.queue.submit(Some(enc.finish()));
        let bytes = self.read_back(&b.vh, dw as u64 * dh as u64 * 4)?;
        Ok((bytemuck::cast_slice(&bytes).to_vec(), dw, dh))
    }

    /// Резкость и 8 бит для куска кадра из `develop_mid` (после шумодава на нём).
    /// `sw`, `sh` — размеры исходного кадра: от них зависит сила резкости, как при полной проявке.
    pub fn finish_crop(&self, px: &[u32], cw: u32, ch: u32, r: &Recipe, sw: u32, sh: u32) -> Result<Rgba> {
        let n = cw as u64 * ch as u64 * 4;
        let mid = self.storage(n, "crop-mid");
        let dst = self.storage(n, "crop-dst");
        self.queue.write_buffer(&mid, 0, bytemuck::cast_slice(px));
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.finish_pass(&mut enc, cw, ch, r, sw, sh, &mid, &dst);
        self.queue.submit(Some(enc.finish()));
        let data = self.read_back(&dst, n)?;
        Ok(Rgba { width: cw, height: ch, data })
    }

    /// Только демозаик: плоскости R, G, B в цветах камеры после ББ (для тестов и анализа).
    #[allow(dead_code)]
    pub fn demosaic_planes(&self, frame: &RawFrame, mul: [f32; 3], clip: f32) -> Result<[Vec<f32>; 3]> {
        let guard = self.demosaic_locked(frame, mul, clip, None)?;
        let b = guard.as_ref().unwrap();
        let n = (frame.width * frame.height) as u64 * 4;
        let conv = |v: Vec<u8>| -> Vec<f32> { bytemuck::cast_slice(&v).to_vec() };
        Ok([conv(self.read_back(&b.rr, n)?), conv(self.read_back(&b.gg, n)?), conv(self.read_back(&b.bb, n)?)])
    }

    fn demosaic_locked(
        &self,
        frame: &RawFrame,
        mul: [f32; 3],
        clip: f32,
        raw_nr: Option<&crate::raw_denoise::RawNr>,
    ) -> Result<std::sync::MutexGuard<'_, Option<FullBufs>>> {
        let (w, h) = (frame.width as u32, frame.height as u32);
        let n = w as u64 * h as u64;
        // Шумодав в RAW — до того, как занять буферы видеокарты (сеть работает на своей сессии).
        let cleaned = match raw_nr {
            Some(nr) => Some(crate::raw_denoise::run(frame, nr)?),
            None => None,
        };
        let src: &[u16] = cleaned.as_deref().unwrap_or(&frame.data);
        let mut guard = self.full.lock().map_err(|_| anyhow!("GPU занят"))?;
        if guard.as_ref().map(|b| b.pixels < n).unwrap_or(true) {
            *guard = None;
            *guard = Some(FullBufs {
                pixels: n,
                raw: self.storage(n * 2, "raw"),
                cfa: self.storage(n * 4, "cfa"),
                vh: self.storage(n * 4, "vh"),
                pq: self.storage(n * 4, "pq"),
                rr: self.storage(n * 4, "rr"),
                gg: self.storage(n * 4, "gg"),
                bb: self.storage(n * 4, "bb"),
            });
        }
        let b = guard.as_ref().unwrap();
        let bytes: &[u8] = bytemuck::cast_slice(src);
        if bytes.len() % 4 == 0 {
            self.queue.write_buffer(&b.raw, 0, bytes);
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(bytes.len().next_multiple_of(4), 0);
            self.queue.write_buffer(&b.raw, 0, &padded);
        }

        let mut mul4 = [1f32; 4];
        let mut inv = [1f32; 4];
        for i in 0..4 {
            mul4[i] = mul[frame.cfa[i] as usize];
            inv[i] = 1.0 / (frame.white - frame.black[i]).max(1.0);
        }
        let dpar = self.uniform(&DemosaicParams {
            w,
            h,
            _a: 0,
            _b: 0,
            cfa: frame.cfa,
            black: frame.black,
            inv_range: inv,
            mul: mul4,
            clip: [clip, 0.0, 0.0, 0.0],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        let wg = (w.div_ceil(16), h.div_ceil(16));
        let steps: [(&wgpu::ComputePipeline, Vec<(u32, &wgpu::Buffer)>); 5] = [
            (&self.prepare_cfa, vec![(0, &dpar), (1, &b.raw), (2, &b.cfa)]),
            (&self.rcd_dir, vec![(0, &dpar), (2, &b.cfa), (3, &b.vh), (4, &b.pq)]),
            (&self.rcd_green, vec![(0, &dpar), (2, &b.cfa), (3, &b.vh), (6, &b.gg)]),
            (&self.rcd_rb_nongreen, vec![(0, &dpar), (2, &b.cfa), (4, &b.pq), (5, &b.rr), (6, &b.gg), (7, &b.bb)]),
            (&self.rcd_rb_green, vec![(0, &dpar), (3, &b.vh), (5, &b.rr), (6, &b.gg), (7, &b.bb)]),
        ];
        for (p, entries) in steps.iter() {
            let bg = self.bind(p, entries);
            Self::pass(&mut enc, p, &bg, wg.0, wg.1);
        }
        self.queue.submit(Some(enc.finish()));
        drop(steps);
        Ok(guard)
    }

    /// Быстрая проявка уменьшенного линейного кадра (превью и живые ползунки).
    pub fn develop_linear(&self, img: &LinearImage, r: &Recipe) -> Result<Rgba> {
        let (w, h) = (img.width as u32, img.height as u32);
        let n = w as u64 * h as u64;
        let mut planes = [vec![0f32; n as usize], vec![0f32; n as usize], vec![0f32; n as usize]];
        for (i, px) in img.data.chunks_exact(3).enumerate() {
            planes[0][i] = px[0];
            planes[1][i] = px[1];
            planes[2][i] = px[2];
        }
        let mk = |v: &Vec<f32>| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(v),
                usage: STORAGE,
            })
        };
        let (rr, gg, bb) = (mk(&planes[0]), mk(&planes[1]), mk(&planes[2]));
        let packed = self.storage(n * 4, "packed");
        let mid = self.storage(n * 4, "mid");
        let dst = self.storage(n * 4, "dst");

        let mut enc = self.device.create_command_encoder(&Default::default());
        let cpar = self.uniform(&self.color_params(w, h, r));
        let bg = self.bind(&self.prepare_rgb, &[(0, &cpar), (1, &rr), (2, &gg), (3, &bb)]);
        let (gx, gy) = Self::groups_1d(n);
        Self::pass(&mut enc, &self.prepare_rgb, &bg, gx, gy);
        if r.recovery > 0.0 {
            self.queue.submit(Some(enc.finish()));
            self.reconstruct([&rr, &gg, &bb], w, h, r, 8)?;
            enc = self.device.create_command_encoder(&Default::default());
        }
        let (dw, dh) = self.run_tail(enc, w, h, r, [&rr, &gg, &bb], &packed, &mid, &dst, None)?;
        let data = self.read_back(&dst, dw as u64 * dh as u64 * 4)?;
        Ok(Rgba { width: dw, height: dh, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::{CameraColor, PhotoMeta};

    /// Синтетическая сцена: плавные градиенты, резкие границы и тонкие линии.
    fn scene(x: usize, y: usize, w: usize, h: usize) -> [f32; 3] {
        let fx = x as f32 / w as f32;
        let fy = y as f32 / h as f32;
        let mut c = [0.15 + 0.5 * fx, 0.25 + 0.3 * (fy * 6.0).sin().abs(), 0.6 - 0.4 * fx * fy];
        if x > w / 2 && y < h / 2 {
            c = [0.7, 0.35, 0.2];
        }
        if (x as i32 - y as i32).abs() < 2 {
            c = [0.05, 0.05, 0.05];
        }
        c
    }

    fn mosaic(w: usize, h: usize) -> (RawFrame, Vec<[f32; 3]>) {
        let cfa = [0u32, 1, 1, 2];
        let (black, white) = (512.0f32, 16383.0f32);
        let mut data = vec![0u16; w * h];
        let mut truth = vec![[0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let c = scene(x, y, w, h);
                truth[y * w + x] = c;
                let ch = cfa[(y & 1) * 2 + (x & 1)] as usize;
                data[y * w + x] = (black + c[ch] * (white - black)).round() as u16;
            }
        }
        let frame = RawFrame {
            width: w,
            height: h,
            data,
            cfa,
            black: [black; 4],
            white,
            asshot_wb: None,
            color: CameraColor { cm_a: None, cm_d65: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] },
            meta: PhotoMeta::default(),
        };
        (frame, truth)
    }

    #[test]
    fn rcd_reconstructs_synthetic_scene() {
        let Ok(gpu) = Gpu::new() else {
            eprintln!("GPU недоступен — тест пропущен");
            return;
        };
        let (w, h) = (320, 240);
        let (frame, truth) = mosaic(w, h);
        let planes = gpu.demosaic_planes(&frame, [1.0, 1.0, 1.0], 1.0).unwrap();
        for c in 0..3 {
            let mut se = 0.0f64;
            let mut n = 0.0f64;
            for y in 6..h - 6 {
                for x in 6..w - 6 {
                    let d = (planes[c][y * w + x] - truth[y * w + x][c]) as f64;
                    se += d * d;
                    n += 1.0;
                }
            }
            let psnr = 10.0 * (1.0 / (se / n)).log10();
            println!("канал {c}: PSNR {psnr:.1} дБ");
            assert!(psnr > 30.0, "канал {c}: PSNR {psnr:.1}");
        }
        // Плавная область должна восстанавливаться почти точно.
        let i = 60 * w + 40;
        for c in 0..3 {
            assert!((planes[c][i] - truth[i][c]).abs() < 0.01, "канал {c}: {} vs {}", planes[c][i], truth[i][c]);
        }
    }

    #[test]
    fn full_pipeline_neutral_gray() {
        let Ok(gpu) = Gpu::new() else { return };
        let (w, h) = (64, 48);
        let mut frame = mosaic(w, h).0;
        let (black, white) = (512.0f32, 16383.0f32);
        for v in frame.data.iter_mut() {
            *v = (black + 0.18 * (white - black)) as u16;
        }
        let p = crate::develop::DevelopParams { temp: 6504.0, tint: 0.0, sharpen: 0.0, ..Default::default() };
        let mut r = crate::develop::recipe(&p, &frame.color, 1, None);
        r.mul = [1.0, 1.0, 1.0];
        r.matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let img = gpu.develop_raw(&frame, &r).unwrap();
        assert_eq!((img.width, img.height), (64, 48));
        let px = &img.data[(24 * 64 + 32) * 4..(24 * 64 + 32) * 4 + 4];
        let expect = (crate::color::render_gray(0.18, 1.0) * 255.0).round() as i32;
        for c in 0..3 {
            assert!((px[c] as i32 - expect).abs() <= 2, "{px:?} vs {expect}");
        }
    }

    #[test]
    fn crop_of_mid_matches_full_develop() {
        let Ok(gpu) = Gpu::new() else { return };
        let frame = mosaic(64, 48).0;
        let p = crate::develop::DevelopParams { temp: 5000.0, tint: 0.0, sharpen: 0.35, ..Default::default() };
        let r = crate::develop::recipe(&p, &frame.color, 1, None);
        let full = gpu.develop_raw(&frame, &r).unwrap();
        let (mid, dw, dh) = gpu.develop_mid(&frame, &r).unwrap();
        assert_eq!((dw, dh), (full.width, full.height));
        let (x0, y0, cw, ch) = (8usize, 6usize, 40usize, 30usize);
        let crop: Vec<u32> = (y0..y0 + ch).flat_map(|y| mid[y * dw as usize + x0..y * dw as usize + x0 + cw].to_vec()).collect();
        let img = gpu.finish_crop(&crop, cw as u32, ch as u32, &r, 64, 48).unwrap();
        // Внутри запаса (резкость смотрит на соседей) кусок совпадает с полной проявкой.
        for y in 3..ch - 3 {
            for x in 3..cw - 3 {
                let a = &img.data[(y * cw + x) * 4..(y * cw + x) * 4 + 3];
                let i = ((y0 + y) * dw as usize + x0 + x) * 4;
                let b = &full.data[i..i + 3];
                for c in 0..3 {
                    assert!((a[c] as i32 - b[c] as i32).abs() <= 1, "({x},{y}) {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn skin_correction_stays_inside_face_mask() {
        let Ok(gpu) = Gpu::new() else { return };
        // Весь кадр «телесного» цвета (как дерево или стена), лицо — только слева.
        let (w, h) = (64usize, 48usize);
        let c = [0.40f32, 0.25, 0.18];
        let img = LinearImage { width: w, height: h, data: c.repeat(w * h) };
        let lab = crate::color::oklab(c);
        let shift = crate::develop::SkinShift {
            hue_center: lab[2].atan2(lab[1]),
            half_width: 0.6,
            shift: 0.2,
            chroma_scale: 0.7,
            chroma_min: 0.0,
            chroma_max: 0.5,
        };
        let mut mask = crate::develop::SkinMask::default();
        mask.n = 1;
        mask.e[0] = [0.25, 0.5, 0.2, 0.4];
        let render = |skin: bool| {
            let p = crate::develop::DevelopParams {
                temp: 6504.0,
                sharpen: 0.0,
                skin: skin.then_some(shift),
                skin_mask: skin.then_some(mask),
                ..Default::default()
            };
            let cam = CameraColor { cm_a: None, cm_d65: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
            let mut r = crate::develop::recipe(&p, &cam, 1, None);
            r.mul = [1.0, 1.0, 1.0];
            r.matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            gpu.develop_linear(&img, &r).unwrap()
        };
        let (with, without) = (render(true), render(false));
        let at = |im: &Rgba, x: usize, y: usize| im.data[(y * w + x) * 4..(y * w + x) * 4 + 3].to_vec();
        assert_ne!(at(&with, 16, 24), at(&without, 16, 24), "в маске лица коррекция должна работать");
        assert_eq!(at(&with, 52, 24), at(&without, 52, 24), "вне лица кадр трогать нельзя");
    }
}
