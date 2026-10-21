// PhotoStab — интерфейс. Без сборки и фреймворков.
// Три раздела: Проекты (открыть папку, история) → Отбор (что выбрал алгоритм) → Обработка.
"use strict";

const $ = (id) => document.getElementById(id);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const api = async (url, body) => {
  const r = await fetch(url, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
const store = {
  get(k, d) { try { const v = localStorage.getItem("ps." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("ps." + k, JSON.stringify(v)); } catch { /* без хранилища — просто не запоминаем */ } },
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ---------- иконки (встроенные SVG: программа работает офлайн) ----------

const ICONS = {
  folder: '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h7.5A2.5 2.5 0 0 1 21 9.5v8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
  image: '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="M21 15.5l-4.5-4.5L8 19.5"/>',
  star: '<path d="M12 3.6l2.6 5.3 5.8.8-4.2 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.2-4.1 5.8-.8z"/>',
  "star-fill": '<path fill="currentColor" d="M12 3.6l2.6 5.3 5.8.8-4.2 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.2-4.1 5.8-.8z"/>',
  close: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
  left: '<path d="M14.5 5.5L8 12l6.5 6.5"/>',
  right: '<path d="M9.5 5.5L16 12l-6.5 6.5"/>',
  arrow: '<path d="M5 12h14M13.5 6.5L19 12l-5.5 5.5"/>',
  export: '<path d="M12 15V3.5M7.5 8L12 3.5 16.5 8"/><path d="M5 14v4.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V14"/>',
  fit: '<path d="M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15"/>',
  ruler: '<path d="M3.5 15.5l12-12 5 5-12 12z"/><path d="M7 12l2 2M9.5 9.5l1.5 1.5M12 7l2 2M14.5 4.5l1.5 1.5"/>',
  "cmp-slider": '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><path d="M12 2.5v19"/><circle cx="12" cy="12" r="2" fill="currentColor"/>',
  "cmp-side": '<rect x="2.5" y="5.5" width="8.5" height="13" rx="1.8"/><rect x="13" y="5.5" width="8.5" height="13" rx="1.8"/>',
  "cmp-off": '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/>',
  strip: '<rect x="3" y="4" width="18" height="10" rx="2"/><rect x="3" y="16.5" width="4.5" height="3.5" rx="1"/><rect x="9.75" y="16.5" width="4.5" height="3.5" rx="1"/><rect x="16.5" y="16.5" width="4.5" height="3.5" rx="1"/>',
  panel: '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><path d="M15 4.5v15M17 8.5h2M17 11.5h2M17 14.5h2"/>',
  "size-s": '<rect x="3.5" y="3.5" width="4.5" height="4.5" rx="1"/><rect x="9.75" y="3.5" width="4.5" height="4.5" rx="1"/><rect x="16" y="3.5" width="4.5" height="4.5" rx="1"/><rect x="3.5" y="9.75" width="4.5" height="4.5" rx="1"/><rect x="9.75" y="9.75" width="4.5" height="4.5" rx="1"/><rect x="16" y="9.75" width="4.5" height="4.5" rx="1"/><rect x="3.5" y="16" width="4.5" height="4.5" rx="1"/><rect x="9.75" y="16" width="4.5" height="4.5" rx="1"/><rect x="16" y="16" width="4.5" height="4.5" rx="1"/>',
  "size-m": '<rect x="3.5" y="3.5" width="7.5" height="7.5" rx="1.5"/><rect x="13" y="3.5" width="7.5" height="7.5" rx="1.5"/><rect x="3.5" y="13" width="7.5" height="7.5" rx="1.5"/><rect x="13" y="13" width="7.5" height="7.5" rx="1.5"/>',
  "size-l": '<rect x="3.5" y="5" width="17" height="14" rx="2"/>',
};
function icon(name) {
  return `<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;
}
function paintIcons(root = document) {
  for (const el of $$("[data-icon]", root)) el.innerHTML = icon(el.dataset.icon);
}

// ---------- состояние ----------

let state = null;
let view = "projects";
let cullFilter = "all";
let groupFilter = null;
let current = null; // id кадра в просмотрщике
let viewerMode = "process"; // "cull" — JPEG камеры и отбор, "process" — проявка и правки
let visibleCull = [];
let visibleProc = [];
let lastFolder = null;

// Адреса картинок: номер сессии (папки) в адресе — кэш браузера не путает кадры разных папок.
const sidq = () => `s=${state ? state.sid : 0}`;
const camUrl = (name, w) => `/api/cam/${encodeURIComponent(name)}?${w ? `w=${w}&` : ""}${sidq()}`;
const camFullUrl = (name) => `/api/cam/${encodeURIComponent(name)}?full=1&${sidq()}`;
const thumbUrl = (p) => `/api/photo/${p.id}/thumb?v=${p.v}&${sidq()}`;
// Превью проявляется из уменьшенного кадра (до 1440 px), поэтому ширина одна — и кэш общий.
const PREVIEW_W = 1600;
const CAM_W = 1620;
const previewUrl = (p, nr = true) => `/api/photo/${p.id}/preview?w=${PREVIEW_W}&v=${p.v}${nr ? "" : "&nr=0"}&${sidq()}`;
const tileUrl = (p, l, x, y, nr) => `/api/photo/${p.id}/tile?x=${x}&y=${y}&l=${l}&nr=${nr ? 1 : 0}&v=${p.v}&${sidq()}`;

function toast(html, ms = 6000) {
  const t = $("toast");
  t.innerHTML = html;
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  if (ms) toast.timer = setTimeout(() => t.classList.add("hidden"), ms);
}

function fmtEv(v) { return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) + " EV"; }
function fmtDeg(v) { return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) + "°"; }
function fmtPct(v) { return Math.round(v) + "%"; }
function baseName(path) { return (path || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop(); }

// Приблизительный цвет источника по температуре — для кружка-образца группы.
function kelvinColor(k) {
  const t = k / 100;
  let r = t <= 66 ? 255 : 329.7 * Math.pow(t - 60, -0.1332);
  let g = t <= 66 ? 99.47 * Math.log(t) - 161.1 : 288.1 * Math.pow(t - 60, -0.0755);
  let b = t >= 66 ? 255 : t <= 19 ? 0 : 138.5 * Math.log(t - 10) - 305;
  const c = (v) => Math.max(0, Math.min(255, Math.round(v)));
  return `rgb(${c(r)},${c(g)},${c(b)})`;
}

let noticeShown = null;
async function refresh() {
  state = await api("/api/state");
  // Сообщение при открытии папки — один раз за сессию папки.
  if (state.notice && noticeShown !== state.sid) {
    noticeShown = state.sid;
    toast(state.notice, 12000);
  }
  // Открыли новую папку (в том числе перетаскиванием) — сразу в «Отбор».
  if (state.folder && state.folder !== lastFolder) {
    lastFolder = state.folder;
    if (view === "projects" && started) return setView("cull");
  }
  render();
}

// ---------- разделы и навигация ----------

let started = false;
function setView(v) {
  if (v === "cull" && !(state && state.folder)) v = "projects";
  if (v === "process" && !(state && state.photos.length)) v = state && state.folder ? "cull" : "projects";
  view = v;
  for (const s of ["projects", "cull", "process"]) $("view-" + s).classList.toggle("hidden", s !== v);
  if (v === "projects") loadHistory();
  render();
}

function renderNav() {
  const has = !!(state && state.folder);
  const done = !!(state && state.photos.length);
  for (const b of $$("#nav button")) {
    b.classList.toggle("on", b.dataset.view === view);
    b.disabled = (b.dataset.view === "cull" && !has) || (b.dataset.view === "process" && !done);
  }
  $("nav-badge").textContent = done ? state.photos.filter((p) => p.pick).length : "";
}
for (const b of $$("#nav button")) b.onclick = () => setView(b.dataset.view);

// Оценка оставшегося времени анализа по скорости последних кадров.
let anHist = [];
function etaText(done, total) {
  const now = performance.now();
  anHist.push({ t: now, d: done });
  anHist = anHist.filter((x) => now - x.t < 20000);
  if (anHist.length < 2) return "";
  const a = anHist[0], b = anHist[anHist.length - 1];
  const rate = (b.d - a.d) / Math.max(0.001, (b.t - a.t) / 1000);
  if (rate <= 0 || done >= total) return "";
  const sec = Math.round((total - done) / rate);
  return sec < 60 ? `осталось ≈ ${Math.max(1, sec)} с` : `осталось ≈ ${Math.round(sec / 60)} мин`;
}

function renderStatus() {
  const s = state.status;
  let txt;
  let frac = 0;
  if (s.kind === "Analyzing") {
    txt = `анализ ${s.done} из ${s.total}`;
    frac = s.total ? s.done / s.total : 0;
    $("an-fill").style.width = frac * 100 + "%";
    const eta = etaText(s.done, s.total);
    $("an-count").textContent = `${s.done} из ${s.total} кадров${eta ? " · " + eta : ""}`;
  } else if (s.kind === "Exporting") {
    txt = `экспорт ${s.done} из ${s.total}`;
    frac = s.total ? s.done / s.total : 0;
  } else if (state.photos.length) {
    const picked = state.photos.filter((p) => p.pick).length;
    txt = `${state.photos.length} кадров · ★ ${picked} · ${state.groups.length} групп света`;
  } else {
    txt = state.gpu ? `готово · ${state.gpu}` : "готово";
  }
  if (s.kind !== "Analyzing") anHist = [];
  for (const pill of $$(".status-pill")) {
    pill.querySelector(".st-folder").textContent = baseName(state.folder) || "PhotoStab";
    pill.title = state.folder || "";
    pill.querySelector(".st-text").textContent = txt;
    pill.querySelector(".bar-fill").style.width = frac * 100 + "%";
  }
  const busy = s.kind !== "Idle";
  $("btn-export").disabled = busy || !state.photos.some((p) => p.pick);
  $("to-process").disabled = busy || !state.photos.length;
  $("btn-open").disabled = busy;
}

function render() {
  if (!state) return;
  renderNav();
  renderStatus();
  // Пока открыт просмотр, сетку под ним не перерисовываем: после правки группы у сотен кадров
  // новая версия, и перезагрузка их миниатюр забивала бы браузер (шторка дёргалась).
  if (current !== null) return renderViewer(false);
  if (view === "cull") renderCull();
  if (view === "process") { renderGroups(); renderGrid(); }
}

// ---------- размер плиток: иконки ----------

const SIZES = { s: 170, m: 240, l: 340 };
let size = store.get("tile", "m");
function applySize() {
  document.documentElement.style.setProperty("--tile", (SIZES[size] || 240) + "px");
  for (const b of $$(".sizes button")) b.classList.toggle("on", b.dataset.size === size);
}
for (const box of $$(".sizes")) {
  box.innerHTML = ["s", "m", "l"].map((k) => `<button data-size="${k}" title="${{ s: "Мелкие", m: "Средние", l: "Крупные" }[k]} плитки">${icon("size-" + k)}</button>`).join("");
  for (const b of $$("button", box)) b.onclick = () => { size = b.dataset.size; store.set("tile", size); applySize(); };
}
applySize();

// ---------- 📁 Проекты ----------

async function loadHistory() {
  let list = [];
  try { list = (await api("/api/history")).projects; } catch (e) { toast(e.message); }
  const box = $("history");
  if (!list.length) {
    box.innerHTML = `<div class="empty">Здесь появятся открытые съёмки.</div>`;
    return;
  }
  box.replaceChildren(...list.map((p) => {
    const el = document.createElement("div");
    el.className = "proj" + (p.exists ? "" : " missing");
    el.title = p.path + (p.exists ? "" : " — папка не найдена");
    const when = new Date(p.opened * 1000).toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    const counts = p.count ? `${p.count} кадров · ★ ${p.picked}` : "ещё не разобрана";
    el.innerHTML = `${p.cover ? `<img src="${p.cover}" alt="" loading="lazy">` : `<div class="noimg"></div>`}
      <div class="meta"><div class="nm"></div><div class="sub">${counts} · ${when}</div></div>
      <button class="rm" title="Убрать из списка (снимки не трогаются)">${icon("close")}</button>`;
    el.querySelector(".nm").textContent = p.name;
    el.onclick = () => p.exists ? openFolder(p.path) : toast("Папка не найдена: " + p.path);
    el.querySelector(".rm").onclick = (e) => {
      e.stopPropagation();
      api("/api/history/remove", { path: p.path }).then(loadHistory).catch((x) => toast(x.message));
    };
    return el;
  }));
}

async function openFolder(path) {
  try {
    await api("/api/open", { path });
    groupFilter = null;
    await refresh();
    setView("cull");
  } catch (e) { toast(e.message); }
}
async function pickFolder() {
  try {
    const r = await api("/api/pick-folder", {});
    if (r.path) openFolder(r.path);
  } catch (e) { toast(e.message); }
}
$("btn-open").onclick = pickFolder;
$("btn-path").onclick = () => openFolder($("path-input").value);
$("path-input").addEventListener("keydown", (e) => e.key === "Enter" && openFolder(e.target.value));

// ---------- 👁 Отбор ----------

function renderCull() {
  // Пока идёт анализ — только прогресс по центру, плитки появятся, когда всё будет готово.
  const busy = state.status.kind === "Analyzing";
  $("analysis").classList.toggle("hidden", !busy);
  $("cull-grid").parentElement.classList.toggle("hidden", busy);
  if (busy) {
    for (const b of $$("#cull-filter button")) b.querySelector("span").textContent = "";
    visibleCull = [];
    return;
  }
  const all = state.photos;
  const picked = all.filter((p) => p.pick).length;
  const counts = { all: all.length, picked, rejected: all.length - picked };
  for (const b of $$("#cull-filter button")) {
    b.classList.toggle("on", b.dataset.f === cullFilter);
    b.querySelector("span").textContent = counts[b.dataset.f];
  }
  const list = all.filter((p) => cullFilter === "all" || (cullFilter === "picked" ? p.pick : !p.pick));
  visibleCull = list.map((p) => p.id);
  const grid = $("cull-grid");
  if (!list.length) {
    grid.innerHTML = `<div class="placeholder">${cullFilter === "rejected" ? "Ничего не отбраковано." : "Нет кадров."}</div>`;
    return;
  }
  const existing = new Map($$(".tile", grid).map((el) => [el.dataset.key, el]));
  const frag = document.createDocumentFragment();
  let prevSeries = null;
  for (const p of list) {
    const key = `${state.sid}:${p.name}`;
    let el = existing.get(key);
    if (!el) {
      el = document.createElement("div");
      el.dataset.key = key;
      el.innerHTML = `<img loading="lazy" decoding="async" alt=""><span class="series hidden"></span>
        <button class="star" title="Отобрать (S)">${icon("star-fill")}</button><div class="cap"><span class="nm"></span><span class="why"></span></div>`;
      el.querySelector("img").src = camUrl(p.name);
    }
    el.className = "tile" + (p.pick ? " picked" : " rejected") + (p.series !== prevSeries && p.series_len > 1 ? " series-start" : "");
    prevSeries = p.series;
    const s = el.querySelector(".series");
    s.textContent = p.series_len > 1 ? `серия ×${p.series_len}` : "";
    s.classList.toggle("hidden", p.series_len <= 1);
    el.querySelector(".star").onclick = (e) => { e.stopPropagation(); togglePick(p.id); };
    el.onclick = () => openViewer(p.id, "cull");
    el.querySelector(".nm").textContent = p.name;
    el.querySelector(".why").textContent = p.pick ? "" : (p.reasons[0] || "");
    el.title = p.reasons.join(", ");
    frag.appendChild(el);
  }
  grid.replaceChildren(frag);
}

for (const b of $$("#cull-filter button")) b.onclick = () => { cullFilter = b.dataset.f; renderCull(); };
$("to-process").onclick = () => setView("process");

async function togglePick(id) {
  const p = state.photos[id];
  try { await api(`/api/photo/${id}`, { pick: !p.pick }); } catch (e) { toast(e.message); }
}

// ---------- 🖼 Обработка ----------

function groupLabel(g) {
  if (g.parent !== null && g.parent !== undefined) return `Группа ${g.parent + 1} · ${g.plan}`;
  return `Группа ${g.id + 1}`;
}

function renderGroups() {
  const box = $("groups");
  const picked = state.photos.filter((p) => p.pick);
  const el = (g, sub) => {
    const cover = state.photos[g.cover];
    const d = document.createElement("div");
    d.className = "group" + (sub ? " sub" : "") + (groupFilter === g.id ? " on" : "");
    const name = sub ? g.plan[0].toUpperCase() + g.plan.slice(1) : groupLabel(g);
    d.innerHTML = `
      <img loading="lazy" src="${cover ? thumbUrl(cover) : ""}" alt="">
      <div>
        <div class="g-name">${name}${g.flash ? " · вспышка" : ""}${g.overridden ? " · правка" : ""}</div>
        <div class="g-sub"><span class="swatch" style="background:${kelvinColor(g.temp)}"></span>${Math.round(g.temp)} K · оттенок ${g.tint >= 0 ? "+" : ""}${g.tint.toFixed(0)}</div>
        <div class="g-sub">★ ${g.picked} из ${g.count}</div>
      </div>`;
    d.onclick = () => { groupFilter = g.id; render(); };
    return d;
  };
  const all = document.createElement("div");
  all.className = "group" + (groupFilter === null ? " on" : "");
  all.innerHTML = `<div><div class="g-name">Все отобранные</div><div class="g-sub">★ ${picked.length} из ${state.photos.length}</div></div>`;
  all.onclick = () => { groupFilter = null; render(); };
  const nodes = [all];
  const isSub = (g) => g.parent !== null && g.parent !== undefined;
  for (const g of state.groups.filter((x) => !isSub(x))) {
    nodes.push(el(g, false));
    for (const s of state.groups.filter((x) => isSub(x) && x.parent === g.id)) nodes.push(el(s, true));
  }
  box.replaceChildren(...nodes);
}

function renderGrid() {
  const grid = $("grid");
  const list = state.photos.filter((p) => p.pick && (groupFilter === null || p.group === groupFilter));
  visibleProc = list.map((p) => p.id);
  if (!list.length) {
    grid.innerHTML = `<div class="placeholder">Нет отобранных кадров${groupFilter !== null ? " в этой группе" : ""} — отметьте их звёздочкой в разделе «Отбор».</div>`;
    return;
  }
  const existing = new Map($$(".tile", grid).map((el) => [el.dataset.key, el]));
  const frag = document.createDocumentFragment();
  for (const p of list) {
    const key = `${state.sid}:${p.id}`;
    let el = existing.get(key);
    if (!el) {
      el = document.createElement("div");
      el.dataset.key = key;
      el.innerHTML = `<img loading="lazy" decoding="async" alt=""><span class="edited hidden"></span>
        <button class="star" title="Убрать из отбора (S)">${icon("star-fill")}</button><div class="cap"><span class="nm"></span></div>`;
    }
    el.className = "tile picked";
    const img = el.querySelector("img");
    const src = thumbUrl(p);
    if (img.getAttribute("src") !== src) img.setAttribute("src", src);
    el.querySelector(".edited").classList.toggle("hidden", !p.edited);
    el.querySelector(".star").onclick = (e) => { e.stopPropagation(); togglePick(p.id); };
    el.onclick = () => openViewer(p.id, "process");
    el.querySelector(".nm").textContent = p.name;
    frag.appendChild(el);
  }
  grid.replaceChildren(frag);
}

// =====================================================================================
// Просмотрщик: окно на весь кадр с зумом (превью + резкие плитки), режимы сравнения,
// лента кадров, живое превью при правках и линейка горизонта.
// =====================================================================================

let cmpMode = store.get("cmpMode", "slider"); // "slider" | "side" | "off"
let cmpSrc = "camera"; // "camera" | "nonr"
let splitPos = 0.5;
// Масштаб: fit — вписать; иначе z — экранных точек на пиксель кадра, (cx, cy) — центр в пикселях кадра.
const Z = { fit: true, z: 1, cx: 0, cy: 0 };
let VW = null; // { id, v, layout, W, H, panes: [...] }

const curPhoto = () => (current === null ? null : state.photos[current]);
// Список ленты и листания фиксируется при открытии просмотра: снятая ★ не убирает кадр сразу,
// список обновится, когда просмотр закроют и откроют снова.
let viewList = null;
const listNow = () => viewList || (viewerMode === "cull" ? visibleCull : visibleProc);

function openViewer(id, mode) {
  current = id;
  viewerMode = mode;
  viewList = [...(mode === "cull" ? visibleCull : visibleProc)];
  Z.fit = true;
  stripCentered = null;
  $("viewer").classList.remove("hidden");
  $("viewer").classList.toggle("cull-mode", mode === "cull");
  renderViewer(true);
}

function closeViewer() {
  setRuler(false);
  current = null;
  viewList = null;
  VW = null;
  $("panes").replaceChildren();
  $("viewer").classList.add("hidden");
  render();
}

function goTo(id) {
  if (id === undefined || id === current) return;
  current = id;
  renderViewer(true);
}

function step(d) {
  const list = listNow();
  if (!list.length) return;
  let i = list.indexOf(current);
  // Кадр убрали из списка, пока смотрели — идём от его места в съёмке.
  if (i < 0) i = d > 0 ? list.findIndex((x) => x > current) - 1 : list.findIndex((x) => x > current);
  goTo(list[(((i + d) % list.length) + list.length) % list.length]);
}

function goProcess() {
  closeViewer();
  setView("process");
}

// ---------- раскладка: панели и слои ----------

function layoutKind(p) {
  if (viewerMode === "cull") return "cull";
  if (ruler.on) return "off";
  return cmpMode;
}

function beforeSrc(p) {
  return cmpSrc === "nonr" && p.nr ? "nonr" : "camera";
}

// Мир — кадр в его пикселях; слои внутри мира, общий сдвиг/масштаб задаётся трансформацией.
function makeWorld(kind, p) {
  const el = document.createElement("div");
  el.className = "world";
  const w = { el, kind, tiles: new Map(), hires: false };
  if (kind === "after" || kind === "nonr") {
    w.base = document.createElement("img");
    w.base.className = "base";
    w.base.draggable = false;
    w.tilesEl = document.createElement("div");
    w.tilesEl.className = "tiles";
    el.append(w.base, w.tilesEl);
  } else {
    // JPEG камеры: в «Отборе» он и есть кадр; при сравнении — повёрнут и обрезан как результат.
    w.cam = document.createElement("img");
    w.cam.className = kind === "cam" ? "base" : "cam";
    w.cam.draggable = false;
    el.append(w.cam);
  }
  return w;
}

function makePane(worlds, tag) {
  const el = document.createElement("div");
  el.className = "pane";
  const pane = { el, worlds: [], clip: null };
  for (const w of worlds) { el.append(w.el); pane.worlds.push(w); }
  if (tag) {
    const t = document.createElement("div");
    t.className = "tag tag-l";
    t.textContent = tag;
    el.append(t);
  }
  return pane;
}

function buildPanes(p) {
  const layout = layoutKind(p);
  const bsrc = beforeSrc(p);
  const key = `${layout}:${bsrc}:${viewerMode}`;
  if (VW && VW.key === key && VW.id === p.id) return false;
  const panes = [];
  const box = $("panes");
  if (layout === "cull") {
    panes.push(makePane([makeWorld("cam", p)]));
  } else if (layout === "off") {
    panes.push(makePane([makeWorld("after", p)]));
  } else if (layout === "side") {
    panes.push(makePane([makeWorld(bsrc === "nonr" ? "nonr" : "before", p)], bsrc === "nonr" ? "Без шумодава" : "Камера"));
    panes.push(makePane([makeWorld("after", p)], "PhotoStab"));
  } else {
    // Шторка: результат, поверх — исходник, обрезанный по положению шторки.
    const after = makeWorld("after", p);
    const before = makeWorld(bsrc === "nonr" ? "nonr" : "before", p);
    const pane = makePane([after]);
    const clip = document.createElement("div");
    clip.className = "clip";
    clip.append(before.el);
    pane.worlds.push(before);
    pane.clip = clip;
    const split = document.createElement("div");
    split.className = "split";
    split.innerHTML = "<span></span>";
    const tl = document.createElement("div");
    tl.className = "tag tag-l";
    tl.textContent = bsrc === "nonr" ? "Без шумодава" : "Камера";
    const tr = document.createElement("div");
    tr.className = "tag tag-r";
    tr.textContent = "PhotoStab";
    pane.el.append(clip, split, tl, tr);
    pane.split = split;
    panes.push(pane);
  }
  box.replaceChildren(...panes.map((x) => x.el));
  const prevVW = VW;
  VW = { key, id: p.id, v: null, panes, layout };
  // При смене раскладки на том же кадре сохраняем уже загруженные картинки как заглушки.
  VW.placeholder = prevVW && prevVW.id === p.id ? afterWorlds(prevVW)[0]?.base.src : null;
  placeSplit();
  return true;
}

function allWorlds(vw = VW) { return vw ? vw.panes.flatMap((x) => x.worlds) : []; }
function afterWorlds(vw = VW) { return allWorlds(vw).filter((w) => w.kind === "after"); }

// Картинку в слой — только когда она уже раскодирована: без мигания и пустого кадра.
function loadInto(img, url, done) {
  img.dataset.want = url;
  const im = new Image();
  im.src = url;
  im.decode().then(() => {
    if (img.dataset.want !== url) return;
    img.src = url;
    if (done) done();
  }).catch(() => {});
}

// Источники слоёв для текущего кадра.
function fillWorlds(p, changedPhoto) {
  const W = viewerMode === "cull" ? p.full_w : p.out_w;
  const H = viewerMode === "cull" ? p.full_h : p.out_h;
  VW.W = W;
  VW.H = H;
  for (const w of allWorlds()) {
    w.el.style.width = W + "px";
    w.el.style.height = H + "px";
    if (w.kind === "after" || w.kind === "nonr") {
      if (changedPhoto || VW.v !== p.v) {
        w.tiles.clear();
        w.tilesEl.replaceChildren();
      }
      // Заглушка: при новом кадре — миниатюра (уже в кэше), при правке — то, что уже на экране.
      if (changedPhoto || !w.base.getAttribute("src")) w.base.src = VW.placeholder || thumbUrl(p);
      loadInto(w.base, previewUrl(p, w.kind === "after"), () => {
        w.el.classList.remove("live");
        if (w.kind === "after") prefetchNeighbors();
      });
    } else if (w.kind === "cam") {
      w.hires = false;
      if (changedPhoto || !w.cam.getAttribute("src")) w.cam.src = camUrl(p.name);
      loadInto(w.cam, camUrl(p.name, CAM_W), prefetchNeighbors);
    } else {
      // Исходник при сравнении: кадр камеры, повёрнутый и обрезанный так же, как результат.
      w.hires = false;
      w.cam.style.width = p.full_w + "px";
      w.cam.style.height = p.full_h + "px";
      w.cam.style.left = (W - p.full_w) / 2 + "px";
      w.cam.style.top = (H - p.full_h) / 2 + "px";
      w.cam.style.transform = `rotate(${-p.angle}deg)`;
      if (changedPhoto || !w.cam.getAttribute("src")) w.cam.src = camUrl(p.name);
      loadInto(w.cam, camUrl(p.name, CAM_W));
    }
  }
  VW.placeholder = null;
  VW.v = p.v;
}

// ---------- масштаб и перемещение ----------

function fitZ() {
  if (!VW) return 1;
  let z = Infinity;
  for (const pane of VW.panes) {
    const r = pane.el.getBoundingClientRect();
    z = Math.min(z, r.width / VW.W, r.height / VW.H);
  }
  return z === Infinity ? 1 : z;
}
const zNow = () => (Z.fit ? fitZ() : Z.z);
const isZoomed = () => !Z.fit && Z.z > fitZ() * 1.01;

function paneGeom(pane) {
  const r = pane.el.getBoundingClientRect();
  const z = zNow();
  let cx = Z.fit ? VW.W / 2 : Z.cx;
  let cy = Z.fit ? VW.H / 2 : Z.cy;
  // Кадр не уезжает за край: если он больше окна — держим край у края.
  cx = VW.W * z <= r.width ? VW.W / 2 : clamp(cx, r.width / (2 * z), VW.W - r.width / (2 * z));
  cy = VW.H * z <= r.height ? VW.H / 2 : clamp(cy, r.height / (2 * z), VW.H - r.height / (2 * z));
  return { r, z, tx: r.width / 2 - cx * z, ty: r.height / 2 - cy * z };
}

function applyZoom() {
  if (!VW) return;
  const zoomed = isZoomed();
  for (const pane of VW.panes) {
    const g = paneGeom(pane);
    for (const w of pane.worlds) w.el.style.transform = `translate(${g.tx}px, ${g.ty}px) scale(${g.z})`;
    pane.el.classList.toggle("zoomed", zoomed);
  }
  if (!Z.fit) {
    // Центр после ограничения — чтобы перемещение не «копило» уход за край.
    const g = paneGeom(VW.panes[0]);
    Z.cx = (g.r.width / 2 - g.tx) / g.z;
    Z.cy = (g.r.height / 2 - g.ty) / g.z;
  }
  $("z-100").textContent = Math.round(zNow() * 100) + "%";
  $("z-fit").classList.toggle("on", !zoomed);
  scheduleDetail();
  drawRuler();
}

function zoomAt(pane, clientX, clientY, nz) {
  const g = paneGeom(pane);
  const ix = (clientX - g.r.left - g.tx) / g.z;
  const iy = (clientY - g.r.top - g.ty) / g.z;
  const fz = fitZ();
  nz = clamp(nz, fz, Math.max(2, fz));
  if (nz <= fz * 1.001) {
    Z.fit = true;
  } else {
    Z.fit = false;
    Z.z = nz;
    Z.cx = ix - (clientX - g.r.left - g.r.width / 2) / nz;
    Z.cy = iy - (clientY - g.r.top - g.r.height / 2) / nz;
  }
  applyZoom();
}

function toggleZoom(pane, x, y) {
  if (!VW) return;
  pane = pane || VW.panes[VW.panes.length - 1];
  const r = pane.el.getBoundingClientRect();
  if (isZoomed()) { Z.fit = true; applyZoom(); return; }
  zoomAt(pane, x ?? r.left + r.width / 2, y ?? r.top + r.height / 2, 1);
}

$("z-fit").onclick = () => { Z.fit = true; applyZoom(); };
$("z-100").onclick = () => toggleZoom();

// ---------- резкие плитки и полный JPEG камеры при зуме ----------

let detailTimer = null;
function scheduleDetail() {
  clearTimeout(detailTimer);
  detailTimer = setTimeout(updateDetail, 70);
}

const tileQueue = [];
let tilesInFlight = 0;
function pumpTiles() {
  tileQueue.sort((a, b) => a.d - b.d);
  while (tilesInFlight < 4 && tileQueue.length) {
    const t = tileQueue.shift();
    if (!t.img.isConnected) continue;
    tilesInFlight++;
    const done = () => { tilesInFlight--; pumpTiles(); };
    t.img.onload = () => { t.img.classList.add("ok"); done(); };
    t.img.onerror = done;
    t.img.src = t.url;
  }
}

function updateDetail() {
  if (!VW) return;
  const p = curPhoto();
  if (!p) return;
  const z = zNow();
  for (const pane of VW.panes) {
    const g = paneGeom(pane);
    for (const w of pane.worlds) {
      if (w.kind === "cam" || w.kind === "before") {
        // Полный JPEG камеры — как только средний становится мягче экрана.
        if (!w.hires && z * p.full_w > CAM_W * 1.15) {
          w.hires = true;
          loadInto(w.cam, camFullUrl(p.name));
        }
        continue;
      }
      if (w.el.classList.contains("live")) continue;
      const baseW = w.base.naturalWidth || 1440;
      if (z * VW.W <= baseW * 1.2) continue;
      // Уровень плиток: 100% при сильном зуме, 50% — при умеренном.
      const l = z >= 0.6 ? 0 : 1;
      const span = 512 * (l ? 2 : 1);
      const x0 = Math.max(0, Math.floor((-g.tx / z - span / 2) / span));
      const y0 = Math.max(0, Math.floor((-g.ty / z - span / 2) / span));
      const x1 = Math.min(Math.ceil(VW.W / span) - 1, Math.floor(((g.r.width - g.tx) / z + span / 2) / span));
      const y1 = Math.min(Math.ceil(VW.H / span) - 1, Math.floor(((g.r.height - g.ty) / z + span / 2) / span));
      const cx = (g.r.width / 2 - g.tx) / z;
      const cy = (g.r.height / 2 - g.ty) / z;
      for (let ty = y0; ty <= y1; ty++) {
        for (let tx = x0; tx <= x1; tx++) {
          const key = `${l}:${tx}:${ty}`;
          if (w.tiles.has(key)) continue;
          const img = document.createElement("img");
          img.draggable = false;
          img.decoding = "async";
          img.style.left = tx * span + "px";
          img.style.top = ty * span + "px";
          img.style.width = Math.min(span, VW.W - tx * span) + "px";
          img.style.height = Math.min(span, VW.H - ty * span) + "px";
          img.style.zIndex = l ? 1 : 2;
          w.tilesEl.append(img);
          w.tiles.set(key, img);
          const d = Math.hypot(tx * span + span / 2 - cx, ty * span + span / 2 - cy) + (l ? 0 : 1);
          tileQueue.push({ img, url: tileUrl(p, l, tx, ty, w.kind === "after"), d });
        }
      }
    }
  }
  pumpTiles();
}

// Кадр смотрят — заранее готовим его к зуму на сервере (RAW и проявка в фоне).
let warmTimer = null;
function warmUp(p) {
  clearTimeout(warmTimer);
  if (viewerMode !== "process") return;
  warmTimer = setTimeout(() => { if (curPhoto() === p) api(`/api/photo/${p.id}/warm`, {}).catch(() => {}); }, 700);
}

// Соседние кадры — заранее в кэш браузера: листание без ожидания.
function prefetchNeighbors() {
  const list = listNow();
  const i = list.indexOf(current);
  if (i < 0) return;
  for (const d of [1, 2, -1]) {
    const q = state.photos[list[(i + d + list.length) % list.length]];
    if (!q || q.id === current) continue;
    const urls = viewerMode === "cull" ? [camUrl(q.name, CAM_W)] : [previewUrl(q), cmpMode === "off" ? null : beforeSrc(q) === "nonr" ? previewUrl(q, false) : camUrl(q.name, CAM_W)];
    for (const u of urls) if (u) new Image().src = u;
  }
}

// ---------- шторка, мышь, колесо ----------

function placeSplit() {
  if (!VW) return;
  for (const pane of VW.panes) {
    if (!pane.clip) continue;
    pane.clip.style.clipPath = `inset(0 ${(1 - splitPos) * 100}% 0 0)`;
    pane.split.style.left = `${splitPos * 100}%`;
  }
}
function setSplitAt(pane, clientX) {
  const r = pane.el.getBoundingClientRect();
  splitPos = clamp((clientX - r.left) / r.width, 0, 1);
  placeSplit();
}

(() => {
  const box = $("panes");
  let drag = null;
  const paneOf = (el) => VW && VW.panes.find((x) => x.el === el.closest(".pane"));
  box.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !VW) return;
    const pane = paneOf(e.target);
    if (!pane) return;
    box.setPointerCapture(e.pointerId);
    if (ruler.on) return rulerDown(e);
    drag = { x: e.clientX, y: e.clientY, cx: Z.cx, cy: Z.cy, moved: false, pane, split: !!e.target.closest(".split") };
  });
  box.addEventListener("pointermove", (e) => {
    if (ruler.on) return rulerMove(e);
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.split) return setSplitAt(drag.pane, e.clientX);
    if (isZoomed()) {
      drag.pane.el.classList.add("dragging");
      Z.cx = drag.cx - dx / Z.z;
      Z.cy = drag.cy - dy / Z.z;
      applyZoom();
    } else if (drag.pane.clip && drag.moved) {
      setSplitAt(drag.pane, e.clientX);
    }
  });
  const end = (e) => {
    if (ruler.on) return rulerUp(e);
    if (!drag) return;
    drag.pane.el.classList.remove("dragging");
    // Короткий клик в масштабе «вписать» переставляет шторку.
    if (!drag.moved && !drag.split && drag.pane.clip && !isZoomed()) setSplitAt(drag.pane, e.clientX);
    drag = null;
  };
  box.addEventListener("pointerup", end);
  box.addEventListener("pointercancel", end);
  box.addEventListener("wheel", (e) => {
    if (!VW) return;
    e.preventDefault();
    const pane = paneOf(e.target);
    if (!pane) return;
    const k = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    zoomAt(pane, e.clientX, e.clientY, zNow() * k);
  }, { passive: false });
  box.addEventListener("dblclick", (e) => {
    if (ruler.on) return;
    const pane = paneOf(e.target);
    if (pane) toggleZoom(pane, e.clientX, e.clientY);
  });
  window.addEventListener("resize", () => current !== null && applyZoom());
})();

// ---------- режимы сравнения ----------

function renderTools(p) {
  for (const b of $$("#cmp-mode button")) b.classList.toggle("on", b.dataset.mode === cmpMode);
  $("cmp-src").classList.toggle("hidden", !p.nr || cmpMode === "off");
  for (const b of $$("#cmp-src button")) b.classList.toggle("on", b.dataset.src === beforeSrc(p));
  $("ruler-btn").classList.toggle("on", ruler.on);
}
function setCmpMode(m) {
  cmpMode = m;
  store.set("cmpMode", m);
  if (current !== null) renderViewer(false);
}
for (const b of $$("#cmp-mode button")) b.onclick = () => setCmpMode(b.dataset.mode);
for (const b of $$("#cmp-src button")) b.onclick = () => { cmpSrc = b.dataset.src; if (current !== null) renderViewer(false); };

// ---------- лента и инспектор: можно скрыть ----------

let showStrip = store.get("strip", true);
let showPanel = store.get("panel", true);
function applyChrome() {
  $("viewer").classList.toggle("no-strip", !showStrip);
  $("viewer").classList.toggle("no-panel", !showPanel);
  $("t-strip").classList.toggle("on", showStrip);
  $("t-panel").classList.toggle("on", showPanel);
  // Без инспектора кнопка «закрыть» переезжает в пилюлю.
  $("t-close").classList.toggle("hidden", showPanel);
  // Окно для фото поменяло размер — пересчитать вписывание и плитки.
  if (current !== null) requestAnimationFrame(applyZoom);
}
function toggleStrip() { showStrip = !showStrip; store.set("strip", showStrip); applyChrome(); }
function togglePanel() { showPanel = !showPanel; store.set("panel", showPanel); applyChrome(); }
$("t-strip").onclick = toggleStrip;
$("t-panel").onclick = togglePanel;
$("t-close").onclick = () => closeViewer();
applyChrome();

// ---------- лента кадров ----------

let stripCentered = null;
// Миниатюры ленты грузятся и обновляются только у видимых элементов.
const stripIO = new IntersectionObserver((entries) => {
  for (const e of entries) {
    e.target._vis = e.isIntersecting;
    if (e.isIntersecting) stripSrc(e.target);
  }
}, { root: $("strip-items"), rootMargin: "0px 400px" });
function stripSrc(el) {
  const img = el.querySelector("img");
  if (img.getAttribute("src") !== el.dataset.src) img.setAttribute("src", el.dataset.src);
}
function renderStrip() {
  const list = listNow();
  const box = $("strip-items");
  const existing = new Map($$(".si", box).map((el) => [el.dataset.key, el]));
  const frag = document.createDocumentFragment();
  for (const id of list) {
    const p = state.photos[id];
    const key = `${state.sid}:${viewerMode}:${id}`;
    let el = existing.get(key);
    if (!el) {
      el = document.createElement("div");
      el.dataset.key = key;
      el.innerHTML = `<img alt="" decoding="async"><span class="st">${icon("star-fill")}</span>`;
      el.onclick = () => goTo(id);
      stripIO.observe(el);
    }
    el.dataset.src = viewerMode === "cull" ? camUrl(p.name) : thumbUrl(p);
    if (el._vis || id === current) stripSrc(el);
    el.className = "si" + (p.pick ? " picked" : "") + (viewerMode === "cull" && !p.pick ? " rejected" : "") + (id === current ? " cur" : "");
    el.title = p.name + (p.reasons.length && !p.pick ? " — " + p.reasons.join(", ") : "");
    frag.appendChild(el);
  }
  box.replaceChildren(frag);
  const i = list.indexOf(current);
  const picked = list.filter((id) => state.photos[id].pick).length;
  $("strip-count").innerHTML = `<b>${i >= 0 ? i + 1 : "—"} из ${list.length}</b>${viewerMode === "cull" ? `★ ${picked} отобрано` : "отобранные"}`;
  if (stripCentered !== current) {
    stripCentered = current;
    const el = box.querySelector(".cur");
    if (el) requestAnimationFrame(() => box.scrollTo({ left: el.offsetLeft - box.clientWidth / 2 + el.clientWidth / 2, behavior: "smooth" }));
  }
}
$("strip-items").addEventListener("wheel", (e) => {
  e.preventDefault();
  $("strip-items").scrollLeft += e.deltaY + e.deltaX;
}, { passive: false });

// ---------- инспектор ----------

function renderViewer(full) {
  const p = curPhoto();
  if (!p) return closeViewer();
  const g = state.groups.find((x) => x.id === p.group);
  const cull = viewerMode === "cull";
  const changedPhoto = !VW || VW.id !== p.id;
  if (buildPanes(p) || changedPhoto || VW.v !== p.v) {
    fillWorlds(p, changedPhoto);
    if (changedPhoto) warmUp(p);
  }
  applyZoom();
  renderTools(p);
  renderStrip();

  $("v-name").textContent = p.name;
  $("v-meta").textContent = [p.time, p.iso ? `ISO ${p.iso}` : ""].filter(Boolean).join(" · ");
  const star = $("v-pick");
  star.classList.toggle("on", p.pick);
  star.innerHTML = icon(p.pick ? "star-fill" : "star");
  star.title = p.pick ? "Отобран — убрать из отбора (S)" : "Не отобран — отобрать (S)";
  $("v-reasons").textContent = p.pick ? "" : p.reasons.join(", ");
  $("edit-blocks").classList.toggle("hidden", cull);
  $("cull-block").classList.toggle("hidden", !cull);

  const info = [];
  if (!cull && g) {
    $("g-title").textContent = `Свет кадра · ${groupLabel(g)}`;
    $("g-title").title = "Температура и оттенок — свет этого кадра: найден по серым поверхностям и согласован с соседними кадрами. Правка сдвигает всю группу одинаково.";
    setSlider("g-temp", p.temp);
    setSlider("g-tint", p.tint);
    setSlider("g-exp", g.exposure);
    setSlider("g-con", g.contrast * 100);
    setSlider("g-hi", g.highlights * 100);
    setSlider("g-sh", g.shadows * 100);
    setSlider("g-sat", (g.saturation - 1) * 100);
    setSlider("g-shp", g.sharpen * 100);
    $("g-nr").checked = !g.nr_off;
    setSlider("g-nrl", g.nr_luma * 100);
    setSlider("g-nrc", g.nr_chroma * 100);
    for (const id of ["g-nrl", "g-nrc"]) $(id).disabled = g.nr_off;
    setSlider("p-exp", p.photo_exposure);
    setSlider("p-rec", p.recovery * 100);
    setSlider("p-angle", p.angle);
    $("p-skin").checked = p.skin;
    $("p-skin").disabled = !p.skin_auto;
    // Подсветка ↺ там, где значение изменено вручную.
    const ts = g.tone_set || [];
    const set = { "g-temp": g.temp_set, "g-tint": g.tint_set, "g-exp": g.exposure_set, "g-nrl": g.nr_luma_set,
      "g-con": ts[0], "g-hi": ts[1], "g-sh": ts[2], "g-sat": ts[3], "g-shp": ts[4],
      "g-nrc": g.nr_chroma_set, "p-exp": p.exposure_set, "p-rec": p.recovery_set, "p-angle": p.angle_set };
    for (const b of $$(".rst")) b.classList.toggle("on", !!set[b.dataset.rst]);
    info.push(`Экспозиция: ${fmtEv(p.ev)} (${p.ev_reason})`);
    info.push(p.nr ? `Шумодав: яркость ${Math.round(p.nr.luma * 100)}%, цвет ${Math.round(p.nr.chroma * 100)}%` : "Шумодав: выключен для этого кадра");
    if (p.auto_recovery > 0) info.push(`Пересвет: восстановление ${Math.round(p.auto_recovery * 100)}% по умолчанию`);
    info.push(`Горизонт: найдено ${fmtDeg(p.horizon_found)}, уверенность ${Math.round(p.horizon_conf * 100)}%${p.angle_set ? ", поправлен вручную" : p.auto_angle ? ", исправлено" : ", не поворачивали"}`);
  }
  info.push(`Лица: ${p.faces}${p.eyes !== null && p.eyes !== undefined ? ` · глаза открыты на ${Math.round(p.eyes * 100)}%` : ""}`);
  if (p.series_len > 1) info.push(`Серия из ${p.series_len} кадров, оценка ${(p.score * 100).toFixed(0)}%`);
  $("v-info").innerHTML = info.map((x) => `<div>${x}</div>`).join("");
}

// Ползунки: подпись значения, перетаскивание значения мышью, живое превью, сохранение.
const SLIDERS = {
  // Свет сцены этого кадра; правка сдвигает всю группу (в майредах и по оттенку), сцены сохраняют разницу.
  "g-temp": { fmt: (v) => `${Math.round(v)} K`, step: 10, live: true, save: (v) => ["group", { temp: null, dmired: 1e6 / v - 1e6 / curPhoto().auto_temp }] },
  "g-tint": { fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(1), step: 0.2, live: true, save: (v) => ["group", { tint: null, dtint: v - curPhoto().auto_tint }] },
  "g-exp": { fmt: fmtEv, step: 0.01, live: true, save: (v) => ["group", { exposure: v }] },
  // Тон и цвет места: −100…+100 (резкость 0…100), ноль — как решила программа.
  "g-con": { fmt: fmtSigned, step: 0.5, live: true, save: (v) => ["group", { contrast: v / 100 }] },
  "g-hi": { fmt: fmtSigned, step: 0.5, live: true, save: (v) => ["group", { highlights: v / 100 }] },
  "g-sh": { fmt: fmtSigned, step: 0.5, live: true, save: (v) => ["group", { shadows: v / 100 }] },
  "g-sat": { fmt: fmtSigned, step: 0.5, live: true, save: (v) => ["group", { saturation: 1 + v / 100 }] },
  "g-shp": { fmt: (v) => `${Math.round(v)}`, step: 0.5, live: true, save: (v) => ["group", { sharpen: v / 100 }] },
  "g-nrl": { fmt: fmtPct, step: 0.5, save: (v) => ["group", { nr_luma: v / 100 }] },
  "g-nrc": { fmt: fmtPct, step: 0.5, save: (v) => ["group", { nr_chroma: v / 100 }] },
  "p-exp": { fmt: fmtEv, step: 0.01, live: true, save: (v) => ["photo", { exposure: v }] },
  "p-rec": { fmt: fmtPct, step: 0.5, live: true, save: (v) => ["photo", { recovery: v / 100 }] },
  "p-angle": { fmt: fmtDeg, step: 0.02, live: true, save: (v) => ["photo", { angle: v }] },
};
const groupUrl = () => `/api/group/${state.photos[current].group}`;
function fmtSigned(v) {
  const r = Math.round(v);
  return r > 0 ? `+${r}` : `${r}`;
}

function setSlider(id, value) {
  const el = $(id);
  if (document.activeElement !== el && !scrub.active) el.value = value;
  if (!$(id + "-v").querySelector("input")) $(id + "-v").textContent = SLIDERS[id].fmt(parseFloat(el.value));
}

async function saveSlider(id) {
  const [kind, body] = SLIDERS[id].save(parseFloat($(id).value));
  try {
    await api(kind === "group" ? groupUrl() : `/api/photo/${current}`, body);
    await refresh();
  } catch (e) { toast(e.message); }
  // Значение не изменилось (версия та же) — вернуть полное превью вместо чернового.
  const p = curPhoto();
  if (p && VW && VW.v === p.v && afterWorlds().some((w) => w.el.classList.contains("live"))) fillWorlds(p, false);
}

for (const id of Object.keys(SLIDERS)) {
  const el = $(id);
  el.addEventListener("input", () => {
    $(id + "-v").textContent = SLIDERS[id].fmt(parseFloat(el.value));
    if (SLIDERS[id].live) liveRender();
  });
  // Сохраняем, когда отпустили: пока тянут — только черновое превью.
  el.addEventListener("change", () => { saveSlider(id); el.blur(); });
}
$("g-nr").onchange = (e) => api(groupUrl(), { nr_off: e.target.checked ? null : true }).catch((x) => toast(x.message));
$("p-skin").onchange = (e) => api(`/api/photo/${current}`, { skin: e.target.checked ? null : false }).catch((x) => toast(x.message));

// ↺ — вернуть значение, которое выбрала программа (не JPEG камеры).
const RESET = {
  "g-temp": ["group", ["temp", "dmired"]], "g-tint": ["group", ["tint", "dtint"]], "g-exp": ["group", ["exposure"]],
  "g-nrl": ["group", ["nr_luma"]], "g-nrc": ["group", ["nr_chroma"]],
  "g-con": ["group", ["contrast"]], "g-hi": ["group", ["highlights"]], "g-sh": ["group", ["shadows"]],
  "g-sat": ["group", ["saturation"]], "g-shp": ["group", ["sharpen"]],
  "p-exp": ["photo", ["exposure"]], "p-rec": ["photo", ["recovery"]], "p-angle": ["photo", ["angle"]],
};
function resetValue(id) {
  const [kind, keys] = RESET[id];
  const body = Object.fromEntries(keys.map((k) => [k, null]));
  api(kind === "group" ? groupUrl() : `/api/photo/${current}`, body).catch((e) => toast(e.message));
}
for (const b of $$(".rst")) b.onclick = () => resetValue(b.dataset.rst);
$("g-reset").onclick = () => api(groupUrl(), { reset: true }).catch((e) => toast(e.message));
$("p-reset").onclick = () => api(`/api/photo/${current}`, { reset: true }).catch((e) => toast(e.message));
$("v-pick").onclick = () => togglePick(current);
$("v-close").onclick = closeViewer;
$("v-to-process").onclick = goProcess;
$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);

// Значение тянется мышью, как в DaVinci Resolve: вправо — больше, влево — меньше.
// Shift — точнее, Ctrl — грубее; клик — ввести число; двойной клик — значение программы.
const scrub = { active: false };
for (const id of Object.keys(SLIDERS)) {
  const label = $(id + "-v");
  label.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || label.querySelector("input") || $(id).disabled) return;
    e.preventDefault();
    label.setPointerCapture(e.pointerId);
    Object.assign(scrub, { active: true, id, x: e.clientX, start: parseFloat($(id).value), moved: false, acc: 0, last: e.clientX });
    label.classList.add("active");
  });
  label.addEventListener("pointermove", (e) => {
    if (!scrub.active || scrub.id !== id) return;
    const mult = e.shiftKey ? 0.1 : e.ctrlKey ? 10 : 1;
    scrub.acc += (e.clientX - scrub.last) * SLIDERS[id].step * mult;
    scrub.last = e.clientX;
    if (Math.abs(e.clientX - scrub.x) > 2) scrub.moved = true;
    if (!scrub.moved) return;
    const el = $(id);
    el.value = clamp(scrub.start + scrub.acc, parseFloat(el.min), parseFloat(el.max));
    el.dispatchEvent(new Event("input"));
  });
  label.addEventListener("pointerup", () => {
    if (!scrub.active || scrub.id !== id) return;
    scrub.active = false;
    label.classList.remove("active");
    if (scrub.moved) saveSlider(id);
    else if (!label._clickTimer) label._clickTimer = setTimeout(() => { label._clickTimer = null; editValue(id); }, 230);
  });
  label.addEventListener("dblclick", () => {
    clearTimeout(label._clickTimer);
    label._clickTimer = null;
    resetValue(id);
  });
}

function editValue(id) {
  const label = $(id + "-v");
  const el = $(id);
  const input = document.createElement("input");
  input.type = "text";
  input.value = (+parseFloat(el.value).toFixed(2)).toString();
  label.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = (ok) => {
    if (done) return;
    done = true;
    const v = parseFloat(input.value.replace(",", ".").replace("−", "-"));
    label.textContent = SLIDERS[id].fmt(parseFloat(el.value));
    if (ok && Number.isFinite(v)) {
      el.value = clamp(v, parseFloat(el.min), parseFloat(el.max));
      label.textContent = SLIDERS[id].fmt(parseFloat(el.value));
      saveSlider(id);
    }
  };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

// Черновое превью, пока тянут: параметры прямо в запросе, в работе один запрос — последний побеждает.
const live = { inflight: false, again: false, url: null };
function liveRender() {
  const p = curPhoto();
  if (!p || viewerMode !== "process" || !VW) return;
  if (live.inflight) { live.again = true; return; }
  live.inflight = true;
  const q = {
    temp: $("g-temp").value, tint: $("g-tint").value, gexp: $("g-exp").value,
    pexp: $("p-exp").value, rec: parseFloat($("p-rec").value) / 100, angle: $("p-angle").value,
    con: parseFloat($("g-con").value) / 100, hi: parseFloat($("g-hi").value) / 100, sh: parseFloat($("g-sh").value) / 100,
    sat: 1 + parseFloat($("g-sat").value) / 100, shp: parseFloat($("g-shp").value) / 100,
  };
  const url = `/api/photo/${p.id}/preview?live=1&w=1100&` + Object.entries(q).map(([k, v]) => `${k}=${v}`).join("&");
  fetch(url)
    .then((r) => (r.ok ? r.blob() : null))
    .then(async (b) => {
      if (!b || curPhoto() !== p) return;
      const u = URL.createObjectURL(b);
      const im = new Image();
      im.src = u;
      await im.decode().catch(() => {});
      for (const w of afterWorlds()) {
        w.el.classList.add("live");
        w.base.dataset.want = u;
        w.base.src = u;
      }
      if (live.url) URL.revokeObjectURL(live.url);
      live.url = u;
    })
    .catch(() => {})
    .finally(() => {
      live.inflight = false;
      if (live.again) { live.again = false; liveRender(); }
    });
}

// ---------- линейка горизонта ----------

const ruler = { on: false, a: null, b: null };
function setRuler(on) {
  if (on && viewerMode !== "process") return;
  ruler.on = on;
  ruler.a = ruler.b = null;
  $("viewer").classList.toggle("ruling", on);
  $("ruler-layer").classList.toggle("hidden", !on);
  if (on) toast("Проведите линию вдоль того, что должно быть ровным: пол, стол, рама, колонна. Esc — отмена.", 4000);
  if (current !== null && VW) renderViewer(false);
}
$("ruler-btn").onclick = () => setRuler(!ruler.on);

// Наклон линии: к горизонтали или (если ближе к вертикали) к вертикали, градусы против часовой.
function rulerTilt() {
  if (!ruler.a || !ruler.b) return null;
  const dx = ruler.b.x - ruler.a.x, dy = ruler.b.y - ruler.a.y;
  if (Math.hypot(dx, dy) < 20) return null;
  let a = (Math.atan2(-dy, dx) * 180) / Math.PI;
  if (a > 90) a -= 180;
  if (a <= -90) a += 180;
  return Math.abs(a) <= 45 ? a : a - 90 * Math.sign(a);
}

function drawRuler() {
  const svg = $("ruler-layer");
  if (!ruler.on || !VW) { svg.innerHTML = ""; return; }
  const sr = $("stage").getBoundingClientRect();
  const pr = VW.panes[0].el.getBoundingClientRect();
  const g = paneGeom(VW.panes[0]);
  // Сетка поверх видимой части кадра.
  const x0 = Math.max(pr.left, pr.left + g.tx) - sr.left, y0 = Math.max(pr.top, pr.top + g.ty) - sr.top;
  const x1 = Math.min(pr.right, pr.left + g.tx + VW.W * g.z) - sr.left, y1 = Math.min(pr.bottom, pr.top + g.ty + VW.H * g.z) - sr.top;
  let h = "";
  for (let k = 1; k < 8; k++) {
    const x = x0 + ((x1 - x0) * k) / 8, y = y0 + ((y1 - y0) * k) / 8;
    h += `<line class="grid" x1="${x}" y1="${y0}" x2="${x}" y2="${y1}"/><line class="grid" x1="${x0}" y1="${y}" x2="${x1}" y2="${y}"/>`;
  }
  if (ruler.a && ruler.b) {
    const a = { x: ruler.a.x - sr.left, y: ruler.a.y - sr.top }, b = { x: ruler.b.x - sr.left, y: ruler.b.y - sr.top };
    const t = rulerTilt();
    h += `<line class="rule" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/><circle cx="${a.x}" cy="${a.y}" r="4"/><circle cx="${b.x}" cy="${b.y}" r="4"/>`;
    if (t !== null) h += `<text x="${b.x + 10}" y="${b.y - 10}">${fmtDeg(-t)}</text>`;
  }
  svg.innerHTML = h;
}
function rulerDown(e) { ruler.a = ruler.b = { x: e.clientX, y: e.clientY }; drawRuler(); }
function rulerMove(e) { if (ruler.a) { ruler.b = { x: e.clientX, y: e.clientY }; drawRuler(); } }
function rulerUp() {
  const t = rulerTilt();
  const p = curPhoto();
  if (t === null || !p) { ruler.a = ruler.b = null; drawRuler(); return; }
  // Линия на уже повёрнутом кадре: к текущему повороту добавляем найденный наклон.
  const want = p.angle - t;
  const angle = clamp(want, -10, 10);
  if (angle !== want) toast("Наклон больше 10° — повернул на максимум.", 4000);
  setRuler(false);
  api(`/api/photo/${p.id}`, { angle: +angle.toFixed(2) }).catch((e) => toast(e.message));
}

// ---------- клавиши ----------

document.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t.tagName === "INPUT" && t.type === "text") return;
  if (current === null) {
    const v = { "1": "projects", "2": "cull", "3": "process" }[e.key];
    if (v) setView(v);
    return;
  }
  // На ползунке стрелки двигают его, а не листают.
  if (t.tagName === "INPUT" && t.type === "range" && e.code.startsWith("Arrow")) return;
  const c = e.code;
  if (c === "Escape") {
    if (ruler.on) return setRuler(false);
    if (isZoomed()) { Z.fit = true; return applyZoom(); }
    return closeViewer();
  }
  if (c === "KeyA" || c === "ArrowLeft") { e.preventDefault(); step(-1); }
  else if (c === "KeyD" || c === "ArrowRight") { e.preventDefault(); step(1); }
  else if (c === "KeyS" || c === "KeyP") { e.preventDefault(); togglePick(current); }
  else if (c === "Enter" || c === "NumpadEnter") { if (viewerMode === "cull") { e.preventDefault(); goProcess(); } }
  else if (c === "KeyB" && viewerMode === "process") setCmpMode({ slider: "side", side: "off", off: "slider" }[cmpMode]);
  else if (c === "KeyZ") toggleZoom();
  else if (c === "KeyR" && viewerMode === "process") setRuler(!ruler.on);
  else if (c === "KeyF") toggleStrip();
  else if (c === "KeyI") togglePanel();
});

// ---------- перетаскивание RAW-файлов ----------

let dragDepth = 0;
// Оверлей загрузки — только когда тащат файлы из Проводника, а не картинку внутри окна.
const draggingFiles = (e) => !!e.dataTransfer && [...e.dataTransfer.types].includes("Files");
window.addEventListener("dragstart", (e) => { if (e.target.tagName === "IMG") e.preventDefault(); });
window.addEventListener("dragenter", (e) => { if (!draggingFiles(e)) return; e.preventDefault(); dragDepth++; $("drop-overlay").classList.remove("hidden"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop-overlay").classList.add("hidden"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("drop-overlay").classList.add("hidden");
  const files = [...e.dataTransfer.files];
  if (files.length) upload(files);
});

function upload(files) {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    toast(`Загрузка RAW: ${Math.round((100 * e.loaded) / e.total)}%`, 0);
  };
  xhr.onload = () => {
    $("toast").classList.add("hidden");
    if (xhr.status >= 400) {
      try { toast(JSON.parse(xhr.responseText).error); } catch { toast("Ошибка загрузки"); }
      return;
    }
    refresh().then(() => setView("cull"));
  };
  xhr.onerror = () => toast("Ошибка загрузки");
  xhr.send(fd);
}

// ---------- экспорт ----------

$("btn-export").onclick = () => {
  const n = state.photos.filter((p) => p.pick).length;
  $("x-picked-n").textContent = `(${n} из ${state.photos.length})`;
  $("x-dir").placeholder = (state.folder || "") + "\\Edited";
  const base = baseName(state.folder) || "photo";
  $("x-names").textContent = `Имена: ${base}_001.jpg, ${base}_002.jpg… по порядку съёмки`;
  $("export-dlg").classList.remove("hidden");
};
$("x-cancel").onclick = () => $("export-dlg").classList.add("hidden");
$("x-go").onclick = async () => {
  $("export-dlg").classList.add("hidden");
  try {
    await api("/api/export", {
      only_picked: $("x-picked").checked,
      quality: parseInt($("x-quality").value, 10),
      long_side: parseInt($("x-size").value, 10) || null,
      dir: $("x-dir").value || null,
    });
  } catch (e) { toast(e.message); }
};

function exported(s) {
  const err = s.errors.length ? ` · ошибок: ${s.errors.length}` : "";
  toast(`Готово: ${s.count} JPEG за ${s.seconds.toFixed(0)} с${err} <button class="btn small" id="reveal">Открыть папку</button>`, 0);
  $("reveal").onclick = () => { api("/api/reveal", { path: s.dir }); $("toast").classList.add("hidden"); };
}

// ---------- события сервера ----------

function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.type === "progress" && state) {
      state.status = { kind: ev.kind === "export" ? "Exporting" : "Analyzing", done: ev.done, total: ev.total };
      renderStatus();
    } else if (ev.type === "state") {
      refresh();
    } else if (ev.type === "exported") {
      exported(ev.summary);
    }
  };
  es.onerror = () => { es.close(); setTimeout(connect, 1500); };
}

paintIcons();
refresh()
  .then(() => {
    started = true;
    lastFolder = state.folder;
    setView(state.folder ? "cull" : "projects");
  })
  .then(connect)
  .catch((e) => toast("Сервер недоступен: " + e.message));
