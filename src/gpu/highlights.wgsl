// Восстановление пересвеченных участков: если обрезан не каждый канал, недостающие каналы
// достраиваются из живых по локальной цветности ярких необрезанных пикселей вокруг.

struct HP {
    w: u32,
    h: u32,
    bw: u32,        // размер карты цветности в блоках
    bh: u32,
    bs: u32,        // размер блока в пикселях
    _a: u32,
    _b: u32,
    _c: u32,
    mul: vec4<f32>, // xyz — уровни обрезки каналов (множители ББ), w — уровень «белого» после восстановления
    thr: vec4<f32>, // x — доля уровня обрезки, выше которой канал считаем обрезанным
}

@group(0) @binding(0) var<uniform> hp: HP;
@group(0) @binding(1) var<storage, read_write> rr: array<f32>;
@group(0) @binding(2) var<storage, read_write> gg: array<f32>;
@group(0) @binding(3) var<storage, read_write> bb: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> chroma: array<vec4<f32>>;

var<workgroup> part: array<vec4<f32>, 256>;

fn saturated_at(x: i32, y: i32) -> bool {
    let cx = u32(clamp(x, 0, i32(hp.w) - 1));
    let cy = u32(clamp(y, 0, i32(hp.h) - 1));
    let i = cy * hp.w + cx;
    return any(vec3<f32>(rr[i], gg[i], bb[i]) >= hp.mul.xyz * hp.thr.x);
}

// Цвет света по краю пересвеченных зон: яркие необрезанные пиксели рядом с обрезанными
// (там та же стена под тем же светом). Вес растёт с яркостью — тёмные цветные предметы
// почти не влияют. Один блок — одна рабочая группа.
@compute @workgroup_size(16, 16)
fn hl_stats(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = vec4<f32>(0.0);
    let per = max(hp.bs / 16u, 1u);
    let clipv = hp.mul.xyz * hp.thr.x;
    let msum = hp.mul.x + hp.mul.y + hp.mul.z;
    let d = i32(max(hp.bs / 8u, 2u));
    for (var dy = 0u; dy < per; dy++) {
        for (var dx = 0u; dx < per; dx++) {
            let x = wid.x * hp.bs + lid.x * per + dx;
            let y = wid.y * hp.bs + lid.y * per + dy;
            if (x < hp.w && y < hp.h) {
                let i = y * hp.w + x;
                let c = vec3<f32>(rr[i], gg[i], bb[i]);
                let s = (c.x + c.y + c.z) / msum;
                if (all(c < clipv * 0.97) && s > 0.3) {
                    let xi = i32(x);
                    let yi = i32(y);
                    let near = saturated_at(xi - d, yi) || saturated_at(xi + d, yi) || saturated_at(xi, yi - d) ||
                        saturated_at(xi, yi + d) || saturated_at(xi - d, yi - d) || saturated_at(xi + d, yi + d) ||
                        saturated_at(xi + d, yi - d) || saturated_at(xi - d, yi + d);
                    if (near) {
                        let wgt = s * s * s * s;
                        acc += vec4<f32>(c * wgt, wgt);
                    }
                }
            }
        }
    }
    part[li] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) {
            part[li] += part[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        stats[wid.y * hp.bw + wid.x] = part[0];
    }
}

fn chroma_at(x: u32, y: u32) -> vec3<f32> {
    let fx = clamp((f32(x) + 0.5) / f32(hp.bs) - 0.5, 0.0, f32(hp.bw - 1u));
    let fy = clamp((f32(y) + 0.5) / f32(hp.bs) - 0.5, 0.0, f32(hp.bh - 1u));
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, hp.bw - 1u);
    let y1 = min(y0 + 1u, hp.bh - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let a = mix(chroma[y0 * hp.bw + x0].xyz, chroma[y0 * hp.bw + x1].xyz, tx);
    let b = mix(chroma[y1 * hp.bw + x0].xyz, chroma[y1 * hp.bw + x1].xyz, tx);
    return mix(a, b, ty);
}

fn fully_clipped(x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(hp.w) - 1));
    let cy = u32(clamp(y, 0, i32(hp.h) - 1));
    let i = cy * hp.w + cx;
    return select(0.0, 1.0, all(vec3<f32>(rr[i], gg[i], bb[i]) >= hp.mul.xyz * hp.thr.x));
}

@compute @workgroup_size(16, 16)
fn hl_reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hp.w || gid.y >= hp.h) { return; }
    let i = gid.y * hp.w + gid.x;
    var c = vec3<f32>(rr[i], gg[i], bb[i]);
    let sat = c >= hp.mul.xyz * hp.thr.x;
    if (!any(sat)) { return; }
    let white = hp.mul.w;
    let ch = max(chroma_at(gid.x, gid.y), vec3<f32>(1e-4));
    var est: vec3<f32>;
    if (all(sat)) {
        // Обрезано всё: минимальный цвет света, который выше всех уровней обрезки.
        est = ch * max(max(c.x / ch.x, c.y / ch.y), c.z / ch.z);
    } else {
        let live = select(vec3<f32>(1.0), vec3<f32>(0.0), sat);
        est = ch * (dot(c, live) / max(dot(ch, live), 1e-5));
    }
    // Обрезанный канал не может быть темнее уровня обрезки.
    c = select(c, max(c, est), sat);
    // Внутри сплошного пересвета (лампы, окна) — честно белое; на границе — плавный переход,
    // чтобы одиночные полностью обрезанные пиксели не давали белую «крошку».
    let x = i32(gid.x);
    let y = i32(gid.y);
    var f = 0.0;
    for (var dy = -2; dy <= 2; dy += 2) {
        for (var dx = -2; dx <= 2; dx += 2) {
            f += fully_clipped(x + dx, y + dy);
        }
    }
    f = f / 9.0;
    c = mix(c, vec3<f32>(white), smoothstep(0.55, 1.0, f));
    c = min(c, vec3<f32>(white));
    rr[i] = c.x;
    gg[i] = c.y;
    bb[i] = c.z;
}
