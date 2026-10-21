// Подготовка байеровских данных и демозаик RCD (Ratio Corrected Demosaicing, Luis Sanz Rodríguez).
// Каждый этап алгоритма — отдельный проход, чтобы на GPU не было гонок данных.

struct P {
    w: u32,
    h: u32,
    _a: u32,
    _b: u32,
    cfa: vec4<u32>,        // цвет (0 R, 1 G, 2 B) в позиции (y&1)*2 + (x&1)
    black: vec4<f32>,
    inv_range: vec4<f32>,  // 1 / (white - black)
    mul: vec4<f32>,        // множитель баланса белого для позиции
    clip: vec4<f32>,       // x = уровень отсечки после ББ
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> cfa: array<f32>;
@group(0) @binding(3) var<storage, read_write> vh: array<f32>;
@group(0) @binding(4) var<storage, read_write> pq: array<f32>;
@group(0) @binding(5) var<storage, read_write> rr: array<f32>;
@group(0) @binding(6) var<storage, read_write> gg: array<f32>;
@group(0) @binding(7) var<storage, read_write> bb: array<f32>;

const EPS: f32 = 1e-5;
const EPSSQ: f32 = 1e-10;

// Зеркальное отражение сохраняет чётность координаты, а значит и цвет фильтра.
fn mirror(i: i32, n: i32) -> i32 {
    var x = i;
    if (x < 0) { x = -x; }
    if (x >= n) { x = 2 * (n - 1) - x; }
    return clamp(x, 0, n - 1);
}

fn idx(x: i32, y: i32) -> u32 {
    return u32(mirror(y, i32(p.h))) * p.w + u32(mirror(x, i32(p.w)));
}

fn color_at(x: i32, y: i32) -> u32 {
    return p.cfa[u32((y & 1) * 2 + (x & 1))];
}

fn C(x: i32, y: i32) -> f32 { return cfa[idx(x, y)]; }
fn G(x: i32, y: i32) -> f32 { return gg[idx(x, y)]; }
fn VH(x: i32, y: i32) -> f32 { return vh[idx(x, y)]; }
fn PQ(x: i32, y: i32) -> f32 { return pq[idx(x, y)]; }
fn RR(x: i32, y: i32) -> f32 { return rr[idx(x, y)]; }
fn BB(x: i32, y: i32) -> f32 { return bb[idx(x, y)]; }

@compute @workgroup_size(16, 16)
fn prepare_cfa(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let i = gid.y * p.w + gid.x;
    let word = raw[i >> 1u];
    let v = f32((word >> ((i & 1u) * 16u)) & 0xFFFFu);
    let pos = (gid.y & 1u) * 2u + (gid.x & 1u);
    let x = clamp((v - p.black[pos]) * p.inv_range[pos], 0.0, 1.0) * p.mul[pos];
    cfa[i] = min(x, p.clip.x);
}

fn sq(x: f32) -> f32 { return x * x; }

fn hpf_v(x: i32, y: i32) -> f32 {
    return sq((C(x, y - 3) - C(x, y - 1) - C(x, y + 1) + C(x, y + 3)) - 3.0 * (C(x, y - 2) + C(x, y + 2)) + 6.0 * C(x, y));
}
fn hpf_h(x: i32, y: i32) -> f32 {
    return sq((C(x - 3, y) - C(x - 1, y) - C(x + 1, y) + C(x + 3, y)) - 3.0 * (C(x - 2, y) + C(x + 2, y)) + 6.0 * C(x, y));
}
fn hpf_p(x: i32, y: i32) -> f32 {
    return sq((C(x - 3, y - 3) - C(x - 1, y - 1) - C(x + 1, y + 1) + C(x + 3, y + 3)) - 3.0 * (C(x - 2, y - 2) + C(x + 2, y + 2)) + 6.0 * C(x, y));
}
fn hpf_q(x: i32, y: i32) -> f32 {
    return sq((C(x + 3, y - 3) - C(x + 1, y - 1) - C(x - 1, y + 1) + C(x - 3, y + 3)) - 3.0 * (C(x + 2, y - 2) + C(x - 2, y + 2)) + 6.0 * C(x, y));
}

// Шаг 1 и 4.0–4.1: направленность (вертикаль/горизонталь и диагонали P/Q).
@compute @workgroup_size(16, 16)
fn rcd_dir(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * p.w + gid.x;
    let v_stat = max(EPSSQ, hpf_v(x, y - 1) + hpf_v(x, y) + hpf_v(x, y + 1));
    let h_stat = max(EPSSQ, hpf_h(x - 1, y) + hpf_h(x, y) + hpf_h(x + 1, y));
    vh[i] = v_stat / (v_stat + h_stat);
    if (color_at(x, y) != 1u) {
        let p_stat = max(EPSSQ, hpf_p(x - 1, y - 1) + hpf_p(x, y) + hpf_p(x + 1, y + 1));
        let q_stat = max(EPSSQ, hpf_q(x + 1, y - 1) + hpf_q(x, y) + hpf_q(x - 1, y + 1));
        pq[i] = p_stat / (p_stat + q_stat);
    } else {
        pq[i] = 0.5;
    }
}

fn lpf(x: i32, y: i32) -> f32 {
    return C(x, y)
        + 0.5 * (C(x, y - 1) + C(x, y + 1) + C(x - 1, y) + C(x + 1, y))
        + 0.25 * (C(x - 1, y - 1) + C(x + 1, y - 1) + C(x - 1, y + 1) + C(x + 1, y + 1));
}

fn refine(c: f32, n: f32) -> f32 {
    return select(c, n, abs(0.5 - c) < abs(0.5 - n));
}

// Шаг 3: зелёный в позициях R и B.
@compute @workgroup_size(16, 16)
fn rcd_green(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * p.w + gid.x;
    let cfai = C(x, y);
    if (color_at(x, y) == 1u) {
        gg[i] = cfai;
        return;
    }
    let n_grad = EPS + abs(C(x, y - 1) - C(x, y + 1)) + abs(cfai - C(x, y - 2)) + abs(C(x, y - 1) - C(x, y - 3)) + abs(C(x, y - 2) - C(x, y - 4));
    let s_grad = EPS + abs(C(x, y - 1) - C(x, y + 1)) + abs(cfai - C(x, y + 2)) + abs(C(x, y + 1) - C(x, y + 3)) + abs(C(x, y + 2) - C(x, y + 4));
    let w_grad = EPS + abs(C(x - 1, y) - C(x + 1, y)) + abs(cfai - C(x - 2, y)) + abs(C(x - 1, y) - C(x - 3, y)) + abs(C(x - 2, y) - C(x - 4, y));
    let e_grad = EPS + abs(C(x - 1, y) - C(x + 1, y)) + abs(cfai - C(x + 2, y)) + abs(C(x + 1, y) - C(x + 3, y)) + abs(C(x + 2, y) - C(x + 4, y));

    let lpfi = lpf(x, y);
    let n_est = C(x, y - 1) * 2.0 * lpfi / (EPS + lpfi + lpf(x, y - 2));
    let s_est = C(x, y + 1) * 2.0 * lpfi / (EPS + lpfi + lpf(x, y + 2));
    let w_est = C(x - 1, y) * 2.0 * lpfi / (EPS + lpfi + lpf(x - 2, y));
    let e_est = C(x + 1, y) * 2.0 * lpfi / (EPS + lpfi + lpf(x + 2, y));

    let v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
    let h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad);

    let vh_n = 0.25 * (VH(x - 1, y - 1) + VH(x + 1, y - 1) + VH(x - 1, y + 1) + VH(x + 1, y + 1));
    let vh_disc = refine(vh[i], vh_n);
    gg[i] = mix(v_est, h_est, vh_disc);
}

// Шаг 4.2: красный в синих позициях и синий в красных.
@compute @workgroup_size(16, 16)
fn rcd_rb_nongreen(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = color_at(x, y);
    if (c == 1u) { return; }
    let i = gid.y * p.w + gid.x;

    let pq_n = 0.25 * (PQ(x - 1, y - 1) + PQ(x + 1, y - 1) + PQ(x - 1, y + 1) + PQ(x + 1, y + 1));
    let pq_disc = refine(pq[i], pq_n);
    let g0 = gg[i];

    let nw_grad = EPS + abs(C(x - 1, y - 1) - C(x + 1, y + 1)) + abs(C(x - 1, y - 1) - C(x - 3, y - 3)) + abs(g0 - G(x - 2, y - 2));
    let ne_grad = EPS + abs(C(x + 1, y - 1) - C(x - 1, y + 1)) + abs(C(x + 1, y - 1) - C(x + 3, y - 3)) + abs(g0 - G(x + 2, y - 2));
    let sw_grad = EPS + abs(C(x + 1, y - 1) - C(x - 1, y + 1)) + abs(C(x - 1, y + 1) - C(x - 3, y + 3)) + abs(g0 - G(x - 2, y + 2));
    let se_grad = EPS + abs(C(x - 1, y - 1) - C(x + 1, y + 1)) + abs(C(x + 1, y + 1) - C(x + 3, y + 3)) + abs(g0 - G(x + 2, y + 2));

    let nw_est = C(x - 1, y - 1) - G(x - 1, y - 1);
    let ne_est = C(x + 1, y - 1) - G(x + 1, y - 1);
    let sw_est = C(x - 1, y + 1) - G(x - 1, y + 1);
    let se_est = C(x + 1, y + 1) - G(x + 1, y + 1);

    let p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad);
    let q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad);
    let other = g0 + mix(p_est, q_est, pq_disc);
    let own = C(x, y);
    if (c == 0u) {
        rr[i] = own;
        bb[i] = other;
    } else {
        bb[i] = own;
        rr[i] = other;
    }
}

fn est_at_green(cn: f32, cs: f32, cw: f32, ce: f32, cn3: f32, cs3: f32, cw3: f32, ce3: f32,
                n1: f32, s1: f32, w1: f32, e1: f32,
                gn: f32, gs: f32, gw: f32, ge: f32, g1: f32, vh_disc: f32) -> f32 {
    let sn_abs = abs(cn - cs);
    let ew_abs = abs(cw - ce);
    let n_grad = n1 + sn_abs + abs(cn - cn3);
    let s_grad = s1 + sn_abs + abs(cs - cs3);
    let w_grad = w1 + ew_abs + abs(cw - cw3);
    let e_grad = e1 + ew_abs + abs(ce - ce3);
    let n_est = cn - gn;
    let s_est = cs - gs;
    let w_est = cw - gw;
    let e_est = ce - ge;
    let v_est = (n_grad * s_est + s_grad * n_est) / (n_grad + s_grad);
    let h_est = (e_grad * w_est + w_grad * e_est) / (e_grad + w_grad);
    return g1 + mix(v_est, h_est, vh_disc);
}

// Шаг 4.3: красный и синий в зелёных позициях.
@compute @workgroup_size(16, 16)
fn rcd_rb_green(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (color_at(x, y) != 1u) { return; }
    let i = gid.y * p.w + gid.x;

    let vh_n = 0.25 * (VH(x - 1, y - 1) + VH(x + 1, y - 1) + VH(x - 1, y + 1) + VH(x + 1, y + 1));
    let vh_disc = refine(vh[i], vh_n);
    let g1 = gg[i];
    let n1 = EPS + abs(g1 - G(x, y - 2));
    let s1 = EPS + abs(g1 - G(x, y + 2));
    let w1 = EPS + abs(g1 - G(x - 2, y));
    let e1 = EPS + abs(g1 - G(x + 2, y));
    let gn = G(x, y - 1);
    let gs = G(x, y + 1);
    let gw = G(x - 1, y);
    let ge = G(x + 1, y);

    rr[i] = est_at_green(RR(x, y - 1), RR(x, y + 1), RR(x - 1, y), RR(x + 1, y),
                         RR(x, y - 3), RR(x, y + 3), RR(x - 3, y), RR(x + 3, y),
                         n1, s1, w1, e1, gn, gs, gw, ge, g1, vh_disc);
    bb[i] = est_at_green(BB(x, y - 1), BB(x, y + 1), BB(x - 1, y), BB(x + 1, y),
                         BB(x, y - 3), BB(x, y + 3), BB(x - 3, y), BB(x + 3, y),
                         n1, s1, w1, e1, gn, gs, gw, ge, g1, vh_disc);
}
