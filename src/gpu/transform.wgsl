// Геометрия (ориентация EXIF, поворот горизонта, кроп, масштаб) с фильтром Ланцоша
// и финальная лёгкая резкость с упаковкой в RGBA8.

struct TP {
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    a: vec4<f32>,      // src_x = a.x * ox + a.y * oy + a.z   (координаты центров пикселей +0.5)
    b: vec4<f32>,      // src_y = b.x * ox + b.y * oy + b.z
    scale: f32,        // растяжение ядра (>= 1 при уменьшении)
    sharpen: f32,
    _p0: f32,
    _p1: f32,
}

@group(0) @binding(0) var<uniform> tp: TP;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> mid: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

const PI: f32 = 3.14159265;

fn unpack10(v: u32) -> vec3<f32> {
    return vec3<f32>(f32(v & 1023u), f32((v >> 10u) & 1023u), f32((v >> 20u) & 1023u)) / 1023.0;
}

fn pack10(c: vec3<f32>) -> u32 {
    let q = vec3<u32>(round(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0));
    return q.x | (q.y << 10u) | (q.z << 20u);
}

fn lanczos3(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-5) { return 1.0; }
    if (ax >= 3.0) { return 0.0; }
    let px = PI * x;
    return 3.0 * sin(px) * sin(px / 3.0) / (px * px);
}

@compute @workgroup_size(16, 16)
fn transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tp.dw || gid.y >= tp.dh) { return; }
    let ox = f32(gid.x) + 0.5;
    let oy = f32(gid.y) + 0.5;
    let fx = tp.a.x * ox + tp.a.y * oy + tp.a.z - 0.5;
    let fy = tp.b.x * ox + tp.b.y * oy + tp.b.z - 0.5;
    let s = max(tp.scale, 1.0);
    let rad = 3.0 * s;
    let x0 = i32(ceil(fx - rad));
    let x1 = i32(floor(fx + rad));
    let y0 = i32(ceil(fy - rad));
    let y1 = i32(floor(fy + rad));
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var yy = y0; yy <= y1; yy++) {
        let wy = lanczos3((f32(yy) - fy) / s);
        if (wy == 0.0) { continue; }
        let cy = u32(clamp(yy, 0, i32(tp.sh) - 1));
        for (var xx = x0; xx <= x1; xx++) {
            let w = lanczos3((f32(xx) - fx) / s) * wy;
            let cx = u32(clamp(xx, 0, i32(tp.sw) - 1));
            acc += w * unpack10(src[cy * tp.sw + cx]);
            wsum += w;
        }
    }
    mid[gid.y * tp.dw + gid.x] = pack10(acc / max(wsum, 1e-6));
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn M(x: i32, y: i32) -> vec3<f32> {
    let cx = u32(clamp(x, 0, i32(tp.dw) - 1));
    let cy = u32(clamp(y, 0, i32(tp.dh) - 1));
    return unpack10(mid[cy * tp.dw + cx]);
}

@compute @workgroup_size(16, 16)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tp.dw || gid.y >= tp.dh) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    var c = M(x, y);
    if (tp.sharpen > 0.0) {
        let blur = (4.0 * luma(c)
            + 2.0 * (luma(M(x - 1, y)) + luma(M(x + 1, y)) + luma(M(x, y - 1)) + luma(M(x, y + 1)))
            + luma(M(x - 1, y - 1)) + luma(M(x + 1, y - 1)) + luma(M(x - 1, y + 1)) + luma(M(x + 1, y + 1))) / 16.0;
        let d = luma(c) - blur;
        let t = smoothstep(0.004, 0.015, abs(d));
        c = clamp(c + vec3<f32>(tp.sharpen * d * t), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let q = vec3<u32>(round(c * 255.0));
    dst[gid.y * tp.dw + gid.x] = q.x | (q.y << 8u) | (q.z << 16u) | (255u << 24u);
}
