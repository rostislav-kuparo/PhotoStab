// Цвет и тон: камера (после ББ) -> линейный sRGB -> экспозиция -> кожа/насыщенность (Oklab)
// -> мягкое плечо светов -> OETF sRGB + базовая кривая. Результат упаковывается в 10:10:10.

struct CP {
    w: u32,
    h: u32,
    lut_n: u32,
    lut_m: u32,
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    mul: vec4<f32>,      // xyz — множители ББ (только для prepare_rgb), w — отсечка
    gain: f32,
    sat: f32,
    lut_max: f32,
    skin_on: f32,
    skin: vec4<f32>,     // x — центр тона (рад), y — полуширина окна (рад), z — сдвиг тона (рад), w — множитель насыщенности
    skin2: vec4<f32>,    // x — мин. насыщенность кожи, y — макс., z — число эллипсов кожи
    skin_ell: array<vec4<f32>, 8>, // эллипсы «лицо + шея» в долях кадра: центр x, y, полуоси x, y
}

@group(0) @binding(0) var<uniform> cp: CP;
@group(0) @binding(1) var<storage, read_write> rr: array<f32>;
@group(0) @binding(2) var<storage, read_write> gg: array<f32>;
@group(0) @binding(3) var<storage, read_write> bb: array<f32>;
@group(0) @binding(4) var<storage, read> lut: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<u32>;

const PI: f32 = 3.14159265;

@compute @workgroup_size(256)
fn prepare_rgb(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= cp.w * cp.h) { return; }
    rr[i] = min(max(rr[i], 0.0) * cp.mul.x, cp.mul.w);
    gg[i] = min(max(gg[i], 0.0) * cp.mul.y, cp.mul.w);
    bb[i] = min(max(bb[i], 0.0) * cp.mul.z, cp.mul.w);
}

fn cbrt(x: f32) -> f32 {
    return select(pow(x, 1.0 / 3.0), 0.0, x <= 0.0);
}

fn to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = cbrt(0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b);
    let m = cbrt(0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b);
    let s = cbrt(0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b);
    return vec3<f32>(
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}

fn from_oklab(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

fn lut_sample(off: u32, n: u32, t: f32) -> f32 {
    let f = clamp(t, 0.0, 1.0) * f32(n - 1u);
    let i0 = u32(floor(f));
    let i1 = min(i0 + 1u, n - 1u);
    return mix(lut[off + i0], lut[off + i1], f - f32(i0));
}

fn shoulder(x: f32) -> f32 {
    return lut_sample(0u, cp.lut_n, sqrt(max(x, 0.0) / cp.lut_max));
}

fn contrast(x: f32) -> f32 {
    return lut_sample(cp.lut_n, cp.lut_m, sqrt(clamp(x, 0.0, 1.0)));
}

fn wrap_angle(a: f32) -> f32 {
    var x = a;
    if (x > PI) { x -= 2.0 * PI; }
    if (x < -PI) { x += 2.0 * PI; }
    return x;
}

// Вес «здесь кожа» по положению пикселя: 1 внутри эллипсов лиц, плавно к 0 у их края.
fn skin_place(i: u32) -> f32 {
    let u = (f32(i % cp.w) + 0.5) / f32(cp.w);
    let v = (f32(i / cp.w) + 0.5) / f32(cp.h);
    var best = 0.0;
    let n = u32(cp.skin2.z);
    for (var k = 0u; k < min(n, 8u); k++) {
        let e = cp.skin_ell[k];
        let dx = (u - e.x) / max(e.z, 1e-4);
        let dy = (v - e.y) / max(e.w, 1e-4);
        best = max(best, 1.0 - smoothstep(0.55, 1.0, dx * dx + dy * dy));
    }
    return best;
}

fn pack10(c: vec3<f32>) -> u32 {
    let q = vec3<u32>(round(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0));
    return q.x | (q.y << 10u) | (q.z << 20u);
}

@compute @workgroup_size(256)
fn color(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= cp.w * cp.h) { return; }
    let cam = vec3<f32>(max(rr[i], 0.0), max(gg[i], 0.0), max(bb[i], 0.0));
    var rgb = vec3<f32>(dot(cp.m0.xyz, cam), dot(cp.m1.xyz, cam), dot(cp.m2.xyz, cam)) * cp.gain;
    rgb = max(rgb, vec3<f32>(0.0));

    if (cp.skin_on > 0.5 || abs(cp.sat - 1.0) > 0.001) {
        let lab = to_oklab(rgb);
        var ch = length(lab.yz);
        var hue = atan2(lab.z, lab.y);
        if (cp.skin_on > 0.5 && ch > 1e-5) {
            let dh = abs(wrap_angle(hue - cp.skin.x));
            let wh = 1.0 - smoothstep(cp.skin.y * 0.6, cp.skin.y, dh);
            let wc = smoothstep(cp.skin2.x * 0.5, cp.skin2.x, ch) * (1.0 - smoothstep(cp.skin2.y, cp.skin2.y * 1.6, ch));
            let wl = smoothstep(0.12, 0.25, lab.x) * (1.0 - smoothstep(0.93, 0.99, lab.x));
            let wgt = wh * wc * wl * skin_place(i);
            hue = hue + cp.skin.z * wgt;
            ch = ch * mix(1.0, cp.skin.w, wgt);
        }
        ch = ch * cp.sat;
        rgb = max(from_oklab(vec3<f32>(lab.x, ch * cos(hue), ch * sin(hue))), vec3<f32>(0.0));
    }

    let n = max(rgb.r, max(rgb.g, rgb.b));
    if (n > 1e-6) {
        rgb = rgb * (shoulder(n) / n);
    }
    let e = vec3<f32>(contrast(rgb.r), contrast(rgb.g), contrast(rgb.b));
    outp[i] = pack10(e);
}
