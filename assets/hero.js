/* Hero: live 2-D Monte Carlo of near-infrared light in a layered head.
   Photons random-walk with per-layer reduced scattering (isotropic, similarity relation) and absorption
   (implicit capture + Russian roulette). Fluence is scored per grid cell by track length.
   Sensitivity ("banana") = Φ_source · Φ_detector, the adjoint estimate of the photon measurement density function.
   Optical properties are approximate literature values, for illustration. */
(() => {
  const cv = document.getElementById('brain');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- geometry (mm; head centre at origin, y negative = up) ----------
  const R = 90, R_SKULL = 84, R_CSF = 77, R_CORTEX = 75, R_KILL = 30;
  // irregular sulci, precomputed as a lookup table over arc length along the cortex (0.1 mm steps)
  const ARC = Math.PI * R_CORTEX, SUL = new Float32Array(Math.ceil(2 * ARC * 10) + 1);
  { let seed = 5; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let a = -ARC; a < ARC; a += 8 + rnd() * 9) {
      const depth = 2.5 + rnd() * 6.5, w = 0.9 + rnd() * 0.8;
      for (let u = -4 * w; u <= 4 * w; u += 0.1) {
        const i = Math.round((a + u + ARC) * 10); if (i < 0 || i >= SUL.length) continue;
        SUL[i] = Math.max(SUL[i], depth * Math.exp(-(u * u) / (2 * w * w)));
      }
    }
  }
  const sulcus = a => SUL[Math.min(SUL.length - 1, Math.max(0, Math.round((a + ARC) * 10)))];
  const gmOuter = phi => R_CORTEX - sulcus(phi * R_CORTEX);
  const wmBound = phi => R_CORTEX - 3.2 - 0.8 * sulcus(phi * R_CORTEX) - 1.2 * Math.sin(phi * 9 + 0.7);
  const NAMES = ['scalp', 'skull', 'CSF', 'gray matter', 'white matter'];
  function layer(x, y) {
    const r = Math.hypot(x, y);
    if (r > R) return -1;                     // air: photon left the head
    if (r < R_KILL) return 5;                 // too deep to matter
    if (r > R_SKULL) return 0;
    if (r > R_CSF) return 1;
    const phi = Math.atan2(x, -y);
    if (r > gmOuter(phi)) return 2;
    return r > wmBound(phi) ? 3 : 4;
  }
  // reduced scattering μs' and absorption μa (1/mm)
  const MUS = [1.0, 0.9, 0.08, 1.1, 5.0];
  const MUA = { 760: [0.018, 0.016, 0.004, 0.028, 0.016], 850: [0.016, 0.014, 0.004, 0.021, 0.016] };

  // optodes along the scalp, 10 mm apart: S D S D S D S
  const OPT = [];
  for (let k = -3; k <= 3; k++) OPT.push({ k, phi: k * 10 / R, src: (k & 1) !== 0 });
  let S = 1, D = 4;                           // indices into OPT (k = -2 → src? see below)
  S = OPT.findIndex(o => o.k === -1); D = OPT.findIndex(o => o.k === 2);   // 30 mm pair
  let lambda = 760, view = 'sens';

  // ---------- fluence grids, cached per optode and wavelength ----------
  const GX0 = -92, GX1 = 92, GY0 = -92, GY1 = -26, RES = 0.6;
  const NX = Math.round((GX1 - GX0) / RES), NY = Math.round((GY1 - GY0) / RES);
  const cache = new Map();
  const grid = i => { const key = i + '@' + lambda; if (!cache.has(key)) cache.set(key, { g: new Float32Array(NX * NY), n: 0 }); return cache.get(key); };

  const trails = [], hits = [];
  let photons = 0;

  function launch(o) {
    const r = R - 0.05, x = r * Math.sin(o.phi), y = -r * Math.cos(o.phi);
    const nx = -Math.sin(o.phi), ny = Math.cos(o.phi);               // inward normal
    const t = Math.asin(2 * Math.random() - 1);                       // Lambertian in 2-D
    return [x, y, nx * Math.cos(t) - ny * Math.sin(t), nx * Math.sin(t) + ny * Math.cos(t)];
  }

  function run(i, rec) {
    const G = grid(i), g = G.g, mua = MUA[lambda];
    let [x, y, dx, dy] = launch(OPT[i]), w = 1, tau = -Math.log(Math.random()), steps = 0;
    const path = rec ? [x, y] : null;
    for (let it = 0; it < 5000; it++) {
      const L = layer(x, y);
      if (L < 0 || L === 5) break;
      const d = Math.min(0.5, tau / MUS[L]);
      x += dx * d; y += dy * d; tau -= MUS[L] * d; w *= Math.exp(-mua[L] * d); steps++;
      const cx = ((x - GX0) / RES) | 0, cy = ((y - GY0) / RES) | 0;
      if (cx >= 0 && cx < NX && cy >= 0 && cy < NY) g[cy * NX + cx] += w * d;
      if (tau <= 1e-9) { const a = Math.random() * 6.283185; dx = Math.cos(a); dy = Math.sin(a); tau = -Math.log(Math.random()); }
      if (w < 1e-3) { if (Math.random() < 0.1) w *= 10; else break; }
      if (path && (it & 1)) path.push(x, y);
    }
    G.n++; photons++;
    // a source photon leaving the scalp at the detector counts as detected
    if (path && Math.hypot(x, y) > R) {
      const dphi = Math.atan2(x, -y) - OPT[D].phi;
      if (Math.abs(dphi) * R < 3) hits.push({ path, t: performance.now() });
    }
    return { steps, path };
  }

  // ---------- drawing ----------
  let W, H, Spx, ox, oy, base, heat, heatCtx, img, dpr;
  const P = (x, y) => [ox + x * Spx, oy + y * Spx];

  function setup() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const narrow = W < 820;
    const Rpx = narrow ? Math.min(W * 1.2, H * 0.9) : Math.min(W * 0.36, H * 0.82);
    Spx = Rpx / R;
    ox = narrow ? W / 2 : W * 0.67;
    oy = (narrow ? H * 0.11 : H * 0.17) + Rpx;

    base = document.createElement('canvas'); base.width = cv.width; base.height = cv.height;
    const g = base.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);

    // soft glow behind the head
    const glow = g.createRadialGradient(ox, oy - Rpx * 0.75, Rpx * 0.05, ox, oy - Rpx * 0.75, Rpx * 0.9);
    glow.addColorStop(0, 'rgba(91,140,255,.10)'); glow.addColorStop(1, 'rgba(14,20,48,0)');
    g.fillStyle = glow; g.fillRect(0, 0, W, H);

    const ring = f => { const p = new Path2D(); for (let i = 0; i <= 720; i++) { const phi = -Math.PI + i * Math.PI / 360, r = f(phi); const [a, b] = P(r * Math.sin(phi), -r * Math.cos(phi)); i ? p.lineTo(a, b) : p.moveTo(a, b); } p.closePath(); return p; };
    const scalp = ring(() => R), skull = ring(() => R_SKULL), csf = ring(() => R_CSF), gm = ring(gmOuter), wm = ring(wmBound);
    g.fillStyle = '#3A2440'; g.fill(scalp);
    g.fillStyle = '#2A2F4E'; g.fill(skull);
    // diploë texture in the skull
    g.save(); g.clip(skull); g.fillStyle = 'rgba(160,170,210,.10)';
    let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < 1400; i++) { const phi = (rnd() - .5) * 2.4, r = R_CSF + 1.5 + rnd() * (R_SKULL - R_CSF - 3); const [a, b] = P(r * Math.sin(phi), -r * Math.cos(phi)); g.beginPath(); g.arc(a, b, (.4 + rnd() * 1.1) * Spx * .35, 0, 6.283); g.fill(); }
    g.restore();
    g.fillStyle = '#0F1C3F'; g.fill(csf);
    g.fillStyle = '#35294F'; g.fill(gm);
    g.fillStyle = '#1D2550'; g.fill(wm);
    g.lineWidth = 1;
    g.strokeStyle = 'rgba(233,236,245,.28)'; g.stroke(scalp);
    g.strokeStyle = 'rgba(233,236,245,.12)'; g.stroke(skull); g.stroke(csf);
    g.strokeStyle = 'rgba(201,180,230,.30)'; g.stroke(gm);
    g.strokeStyle = 'rgba(201,207,230,.10)'; g.stroke(wm);

    // layer labels
    if (!narrow) {
      const lp = 0.56, mids = [(R + R_SKULL) / 2, (R_SKULL + R_CSF) / 2, R_CSF - 0.8, gmOuter(lp) - 1.4, wmBound(lp) - 7];
      g.font = '11px "IBM Plex Mono", monospace'; g.textBaseline = 'middle';
      mids.forEach((r, i) => {
        const [a, b] = P(r * Math.sin(lp), -r * Math.cos(lp)), tx = Math.min(a + 70 + i * 6, W - 92), ty = b + (i - 2) * 9;
        if (ty > H - 10) return;
        g.strokeStyle = 'rgba(140,149,181,.45)'; g.beginPath(); g.moveTo(a, b); g.lineTo(tx - 6, ty); g.stroke();
        g.fillStyle = 'rgba(140,149,181,.9)'; g.fillText(NAMES[i], tx, ty);
      });
    }
    heat = document.createElement('canvas'); heat.width = NX; heat.height = NY;
    heatCtx = heat.getContext('2d'); img = heatCtx.createImageData(NX, NY);
    cv.__clip = scalp;
  }

  // colour map: transparent navy → deoxy blue → violet → oxy red → warm white
  const STOPS = [[0, 14, 20, 48, 0], [.3, 91, 140, 255, .45], [.55, 160, 90, 220, .65], [.8, 240, 80, 106, .85], [1, 255, 226, 200, .95]];
  const LUT = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0; while (k < STOPS.length - 2 && t > STOPS[k + 1][0]) k++;
    const a = STOPS[k], b = STOPS[k + 1], u = (t - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 4; c++) LUT[i * 4 + c] = (a[c + 1] + (b[c + 1] - a[c + 1]) * u) * (c === 3 ? 255 : 1);
  }

  function paintHeat() {
    const s = grid(S), d = grid(D), px = img.data;
    const N = NX * NY, val = new Float32Array(N);
    let mx = 0;
    if (view === 'sens') {
      const k = 1 / (Math.max(s.n, 1) * Math.max(d.n, 1));
      for (let i = 0; i < N; i++) { const v = s.g[i] * d.g[i] * k; val[i] = v; if (v > mx) mx = v; }
    } else {
      const k = 1 / Math.max(s.n, 1);
      for (let i = 0; i < N; i++) { const v = s.g[i] * k; val[i] = v; if (v > mx) mx = v; }
    }
    const lmx = Math.log10(mx || 1), imx = 1 / (mx || 1);
    for (let i = 0; i < N; i++) {
      const v = val[i];
      // sensitivity on a power scale (shows the banana), fluence on a 4.5-decade log scale
      let t = view === 'sens' ? Math.pow(v * imx, 0.33) * 1.15 - 0.08 : v > 0 ? (Math.log10(v) - lmx + 4.5) / 4.5 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const j = (t * 255) | 0;
      px[i * 4] = LUT[j * 4]; px[i * 4 + 1] = LUT[j * 4 + 1]; px[i * 4 + 2] = LUT[j * 4 + 2]; px[i * 4 + 3] = LUT[j * 4 + 3];
    }
    heatCtx.putImageData(img, 0, 0);
  }

  function drawOptode(o, i, t) {
    const r = R + 1.6, [x, y] = P(r * Math.sin(o.phi), -r * Math.cos(o.phi));
    const active = i === S || i === D, sz = (active ? 9 : 6.5);
    // short stem, like an optode holder
    const [ex, ey] = P((R + 7) * Math.sin(o.phi), -(R + 7) * Math.cos(o.phi));
    ctx.strokeStyle = 'rgba(140,149,181,.35)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.save(); ctx.translate(x, y); ctx.rotate(o.phi);
    if (o.src) {
      if (i === S) { const p = still ? .6 : .5 + .5 * Math.sin(t * 5); ctx.fillStyle = `rgba(240,80,106,${.15 + .25 * p})`; ctx.beginPath(); ctx.arc(0, 0, 20, 0, 6.283); ctx.fill(); }
      ctx.fillStyle = i === S ? '#F0506A' : 'rgba(240,80,106,.55)'; ctx.beginPath(); ctx.arc(0, 0, sz, 0, 6.283); ctx.fill();
    } else {
      if (i === D) { ctx.strokeStyle = 'rgba(91,140,255,.6)'; ctx.lineWidth = 1.5; ctx.strokeRect(-sz - 5, -sz - 5, 2 * sz + 10, 2 * sz + 10); }
      ctx.fillStyle = i === D ? '#5B8CFF' : 'rgba(91,140,255,.55)'; ctx.fillRect(-sz, -sz, 2 * sz, 2 * sz);
    }
    ctx.restore();
    o.sx = x; o.sy = y;
  }

  const ui = {
    read: document.getElementById('mc-read'),
    lam: document.querySelectorAll('[data-lambda]'),
    view: document.querySelectorAll('[data-view]'),
  };
  ui.lam.forEach(b => b.addEventListener('click', () => { lambda = +b.dataset.lambda; ui.lam.forEach(x => x.setAttribute('aria-pressed', x === b)); kick(); }));
  ui.view.forEach(b => b.addEventListener('click', () => { view = b.dataset.view; ui.view.forEach(x => x.setAttribute('aria-pressed', x === b)); kick(); }));

  let lastPaint = 0, frameNo = 0;
  function frame(now) {
    const t = now / 1000;
    // simulate within a time budget, alternating source and detector
    const t0 = performance.now(), budget = still ? 400 : 7;
    let k = 0;
    while (performance.now() - t0 < budget) {
      const i = (k++ & 1) ? D : S, rec = !still && i === S && Math.random() < 0.06;
      const r = run(i, rec);
      if (rec) { trails.push({ path: r.path, t: now }); if (trails.length > 14) trails.shift(); }
    }
    if (now - lastPaint > 90 || still) { paintHeat(); lastPaint = now; }

    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(base, 0, 0, W, H);
    ctx.save(); ctx.clip(cv.__clip);
    ctx.globalCompositeOperation = 'lighter'; ctx.imageSmoothingEnabled = true;
    const [hx, hy] = P(GX0, GY0);
    ctx.drawImage(heat, hx, hy, NX * RES * Spx, NY * RES * Spx);
    ctx.globalCompositeOperation = 'source-over';
    // random-walk trails
    for (const tr of trails) {
      const age = (now - tr.t) / 2600; if (age > 1) continue;
      ctx.strokeStyle = `rgba(255,214,222,${.35 * (1 - age)})`; ctx.lineWidth = .8; ctx.beginPath();
      for (let j = 0; j < tr.path.length; j += 2) { const [a, b] = P(tr.path[j], tr.path[j + 1]); j ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }
      ctx.stroke();
    }
    // detected photons: drawn brighter, red → blue
    for (let h = hits.length - 1; h >= 0; h--) {
      const age = (now - hits[h].t) / 2200; if (age > 1) { hits.splice(h, 1); continue; }
      const p = hits[h].path, [a0, b0] = P(p[0], p[1]), [a1, b1] = P(p[p.length - 2], p[p.length - 1]);
      const gr = ctx.createLinearGradient(a0, b0, a1, b1);
      gr.addColorStop(0, `rgba(240,80,106,${1 - age})`); gr.addColorStop(1, `rgba(91,140,255,${1 - age})`);
      ctx.strokeStyle = gr; ctx.lineWidth = 1.6; ctx.beginPath();
      for (let j = 0; j < p.length; j += 2) { const [a, b] = P(p[j], p[j + 1]); j ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }
      ctx.stroke();
    }
    ctx.restore();
    OPT.forEach((o, i) => drawOptode(o, i, t));

    if (ui.read && (frameNo++ % 10 === 0 || still)) {
      const sd = Math.abs(OPT[S].k - OPT[D].k) * 10;
      ui.read.textContent = `${photons.toLocaleString('en')} photons · S–D ${sd} mm · ${lambda} nm`;
    }
    if (!still && running) requestAnimationFrame(frame);
  }

  // interaction: click an optode to make it the active source or detector
  function pick(ev) {
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = -1, bd = 18;
    OPT.forEach((o, i) => { const d = Math.hypot(o.sx - x, o.sy - y); if (d < bd) { bd = d; best = i; } });
    return best;
  }
  cv.addEventListener('pointermove', ev => { cv.style.cursor = pick(ev) >= 0 ? 'pointer' : ''; });
  cv.addEventListener('click', ev => { const i = pick(ev); if (i < 0) return; if (OPT[i].src) S = i; else D = i; hits.length = 0; trails.length = 0; kick(); });

  let running = false;
  function kick() { if (still) requestAnimationFrame(frame); else if (!running) { running = true; requestAnimationFrame(frame); } }
  new IntersectionObserver(es => { const vis = es[0].isIntersecting; if (vis) kick(); else running = false; }).observe(cv);
  let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { setup(); kick(); }, 150); });
  setup(); kick();
})();
