/* RAX background: a robot arm drawn as a cloud of blue points, in 3D, picking a lying
   tube, standing it up and dropping it into a rack. The joint angles come from the same
   kind of solution the real arm uses: base yaw toward the target, then two-link IK in
   the arm's vertical plane with the wrist pitch held. */
(function () {
  const cv = document.getElementById("arm");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, dpr = 1;
  const CUBES = document.body.dataset.scene === "cubes";

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);
  resize();

  // ---- geometry, in centimetres; z is up -----------------------------------------
  const L0 = 7, L1 = 17, L2 = 15, L3 = 8;             // base height, links, hand
  const rnd = (() => { let s = 7; return () => ((s = (s * 16807) % 2147483647) / 2147483647); })();

  // WIREFRAME, NOT NOISE: dots spaced evenly along edges and rings, like a scan
  function line(a, b, step, out) {
    const n = Math.max(1, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / step));
    for (let i = 0; i <= n; i++) out.push([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, a[2] + (b[2] - a[2]) * i / n]);
  }
  function boxPts(len, w, d, step = 0.55, rings = 4) {   // a box along x
    const p = [], y = w / 2, z = d / 2;
    const c = [[y, z], [-y, z], [-y, -z], [y, -z]];
    c.forEach(([cy, cz]) => line([0, cy, cz], [len, cy, cz], step, p));
    for (let r = 0; r <= rings; r++) {
      const u = (r / rings) * len;
      for (let k = 0; k < 4; k++) line([u, ...c[k]], [u, ...c[(k + 1) % 4]], step, p);
    }
    return p;
  }
  function cylPts(r, h, n, axis, rings = 4, spokes = 12) {
    const p = [];
    for (let q = 0; q <= rings; q++) {
      const hh = (q / rings) * h;
      for (let k = 0; k < n; k++) {
        const a = (k / n) * Math.PI * 2;
        p.push(axis === "x" ? [hh, r * Math.cos(a), r * Math.sin(a)] : [r * Math.cos(a), r * Math.sin(a), hh]);
      }
    }
    for (let k = 0; k < spokes; k++) {
      const a = (k / spokes) * Math.PI * 2, ca = r * Math.cos(a), sa = r * Math.sin(a);
      line(axis === "x" ? [0, ca, sa] : [ca, sa, 0], axis === "x" ? [h, ca, sa] : [ca, sa, h], 0.6, p);
    }
    return p;
  }
  const BASE = cylPts(4.5, L0, 40, "z", 4, 14);
  const LINK1 = boxPts(L1, 3.6, 3.6, 0.55, 5);
  const LINK2 = boxPts(L2, 3.0, 3.0, 0.55, 5);
  const HANDP = boxPts(L3 * 0.55, 3.4, 5.0, 0.5, 3);
  const JAW = boxPts(L3 * 0.5, 0.8, 2.2, 0.45, 2);
  const TUBEP = cylPts(0.8, 10, 12, "x", 6, 6);
  const CAPP = cylPts(0.95, 1.6, 14, "x", 3, 8);
  const RACK = { x: 17, y: -16, w: 7, d: 11, h: 5 };
  const RACKP = (() => {
    const p = boxPts(RACK.d, RACK.w, RACK.h, 0.6, 4).map(([u, y, z]) => [y, u - RACK.d / 2, z + RACK.h / 2]);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 2; c++)
      for (let k = 0; k < 18; k++) {
        const a = (k / 18) * Math.PI * 2;
        p.push([(c - 0.5) * 3.2 + 0.9 * Math.cos(a), (r - 1.5) * 2.5 + 0.9 * Math.sin(a), RACK.h]);
      }
    return p;
  })();
  const FLOOR = [];
  for (let x = -26; x <= 30; x += 3) for (let y = -26; y <= 26; y += 3) FLOOR.push([x, y, 0]);

  // ---- motion ------------------------------------------------------------------------
  const tubeAt = { x: 20, y: 10 };                   // lying on the table
  const hole = { x: RACK.x - 1.6, y: RACK.y - 1.25 };
  const P = -Math.PI / 2;
  const keys = [                                     // fingertip, hand pitch, grip, tube
    { t: 0.0, p: [14, 2, 16], a: -0.5, g: 1, s: "rest" },
    { t: 1.7, p: [tubeAt.x, tubeAt.y, 12], a: P, g: 1, s: "rest" },
    { t: 2.6, p: [tubeAt.x, tubeAt.y, 12], a: P, g: 1, s: "rest", tw: 1 },
    { t: 3.5, p: [tubeAt.x, tubeAt.y, 2.2], a: P, g: 1, s: "rest" },
    { t: 4.0, p: [tubeAt.x, tubeAt.y, 2.2], a: P, g: 0.2, s: "rest" },
    { t: 4.9, p: [tubeAt.x, tubeAt.y, 12], a: P, g: 0.2, s: "held" },
    { t: 6.0, p: [16, -2, 18], a: 0, g: 0.2, s: "up" },
    { t: 7.4, p: [hole.x - 6, hole.y, 22], a: 0, g: 0.2, s: "up" },
    { t: 8.2, p: [hole.x - 6, hole.y, 15.5], a: 0, g: 0.2, s: "up" },
    { t: 8.7, p: [hole.x - 6, hole.y, 15.5], a: 0, g: 1, s: "drop" },
    { t: 9.8, p: [12, -4, 18], a: -0.5, g: 1, s: "in" },
    { t: 11.4, p: [14, 2, 16], a: -0.5, g: 1, s: "in" },
  ];
  // ---- scene 2: stacking cubes (the platform page) ----------------------------------
  const CUBE = 4, cubeStart = [[22, 12], [27, 3], [21, -4]], stackAt = [13, -15];
  const CUBEP = boxPts(CUBE, CUBE, CUBE, 0.5, 4);
  if (CUBES) {
    keys.length = 0;
    let t = 0;
    const hover = 15, at = (i) => 2.2 + CUBE * i;       // fingertip at a cube's middle
    keys.push({ t, p: [14, 2, 17], a: -0.5, g: 1, held: -1, n: 0 });
    cubeStart.forEach(([x, y], i) => {
      keys.push({ t: t += 1.4, p: [x, y, hover], a: P, g: 1, held: -1, n: i });
      keys.push({ t: t += 0.8, p: [x, y, 2.2], a: P, g: 1, held: -1, n: i });
      keys.push({ t: t += 0.45, p: [x, y, 2.2], a: P, g: 0.35, held: i, n: i });
      keys.push({ t: t += 0.8, p: [x, y, hover + 2], a: P, g: 0.35, held: i, n: i });
      keys.push({ t: t += 1.3, p: [stackAt[0], stackAt[1], hover + 2 + CUBE * i], a: P, g: 0.35, held: i, n: i });
      keys.push({ t: t += 0.8, p: [stackAt[0], stackAt[1], at(i)], a: P, g: 0.35, held: i, n: i });
      keys.push({ t: t += 0.4, p: [stackAt[0], stackAt[1], at(i)], a: P, g: 1, held: -1, n: i + 1 });
      keys.push({ t: t += 0.6, p: [stackAt[0], stackAt[1], hover + CUBE * (i + 1)], a: P, g: 1, held: -1, n: i + 1 });
    });
    keys.push({ t: t += 1.6, p: [14, 2, 17], a: -0.5, g: 1, held: -1, n: 3 });
    keys.push({ t: t += 1.2, p: [14, 2, 17], a: -0.5, g: 1, held: -1, n: 3 });
  }
  const T = keys[keys.length - 1].t;
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  const lerp = (a, b, u) => a + (b - a) * u;
  function sample(t) {
    for (let i = 0; i < keys.length - 1; i++) {
      const A = keys[i], B = keys[i + 1];
      if (t >= A.t && t <= B.t) {
        const u = ease((t - A.t) / (B.t - A.t));
        return { p: A.p.map((v, k) => lerp(v, B.p[k], u)), a: lerp(A.a, B.a, u), g: lerp(A.g, B.g, u),
                 s: B.s === "drop" ? "drop" : A.s, u, roll: B.tw ? u * 0.6 : (A.tw ? 0.6 : 0),
                 held: B.held === -1 && A.held !== -1 ? A.held : (A.held ?? -1),
                 n: Math.min(A.n ?? 0, B.n ?? 0) };
      }
    }
    return { p: keys[0].p, a: keys[0].a, g: 1, s: "rest", u: 0, roll: 0, held: -1, n: 0 };
  }

  function solve(p, a) {                              // base yaw + planar 2-link IK
    const yaw = Math.atan2(p[1], p[0]);
    const r = Math.hypot(p[0], p[1]), z = p[2];
    const wr = r - L3 * Math.cos(a), wz = z - L3 * Math.sin(a) - L0;
    const d = Math.min(Math.hypot(wr, wz), L1 + L2 - 0.01);
    const c2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    let best = null;
    for (const sg of [1, -1]) {                        // keep the elbow up
      const q2 = sg * Math.acos(Math.max(-1, Math.min(1, c2)));
      const q1 = Math.atan2(wz, wr) - Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
      if (!best || Math.sin(q1) > Math.sin(best.q1)) best = { q1, q2 };
    }
    return { yaw, q1: best.q1, q2: best.q2, q3: a - best.q1 - best.q2 };
  }

  // ---- 3D helpers --------------------------------------------------------------------
  const rotZ = (v, t) => { const c = Math.cos(t), s = Math.sin(t); return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]; };
  const rotP = (v, t) => { const c = Math.cos(t), s = Math.sin(t); return [c * v[0] - s * v[2], v[1], s * v[0] + c * v[2]]; };
  const rotX = (v, t) => { const c = Math.cos(t), s = Math.sin(t); return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]; };
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

  let camYaw = -1.05;
  const camPitch = 0.32;
  function project(v) {
    const q = rotZ(v, camYaw);
    const c = Math.cos(camPitch), s = Math.sin(camPitch);
    const y = q[1] * c + q[2] * s, z = -q[1] * s + q[2] * c;
    const depth = 95 + y;
    const f = Math.min(W * 0.62, H * 0.95) * 1.35 / depth;
    const cx = W > 900 ? W * (CUBES ? 0.7 : 0.79) : W * 0.45, cy = H * (W > 900 ? 0.74 : 0.8);
    return [cx + q[0] * f, cy - z * f, depth];
  }

  function frame(ms) {
    const t = reduce ? 1.7 : (ms / 1000) % T;
    if (!reduce) camYaw = -1.05 + Math.sin(ms / 9000) * 0.22;
    ctx.clearRect(0, 0, W, H);
    const s = sample(t);
    const k = solve(s.p, s.a);

    const pts = [];
    const push = (v, a, sz) => { const [x, y, d] = project(v); pts.push([x, y, d, a, sz]); };
    FLOOR.forEach((v) => push(v, 0.14, 1.1));

    const shoulder = [0, 0, L0];
    const place = (local, origin, pitch, alpha = 0.85, sz = 1.5, roll = 0) => {
      local.forEach((lp) => {
        let v = roll ? rotX(lp, roll) : lp;
        v = rotZ(rotP(v, pitch), k.yaw);
        push(add(origin, v), alpha, sz);
      });
    };
    BASE.forEach((v) => push(v, 0.7, 1.4));
    place(LINK1, shoulder, k.q1);
    const elbow = add(shoulder, rotZ(rotP([L1, 0, 0], k.q1), k.yaw));
    place(LINK2, elbow, k.q1 + k.q2);
    const wrist = add(elbow, rotZ(rotP([L2, 0, 0], k.q1 + k.q2), k.yaw));
    const ha = k.q1 + k.q2 + k.q3;
    place(HANDP, wrist, ha, 0.9, 1.5, s.roll);
    const tip = add(wrist, rotZ(rotP([L3, 0, 0], ha), k.yaw));
    const open = 0.9 + 1.6 * s.g;
    [-1, 1].forEach((sg) => {
      const jawO = add(wrist, rotZ(rotP(rotX([L3 * 0.5, sg * open, 0], s.roll), ha), k.yaw));
      place(JAW, jawO, ha, 0.95, 1.6, s.roll);
    });

    const lying = (o) => { TUBEP.forEach((v) => push(add(o, [v[0] - 5, v[1], v[2] + 0.8]), 0.9, 1.5));
                           CAPP.forEach((v) => push(add(o, [v[0] + 5, v[1], v[2] + 0.8]), 1, 1.9)); };
    const standing = (o) => { TUBEP.forEach((v) => push(add(o, [v[1], v[2], v[0] - 10]), 0.9, 1.5));
                              CAPP.forEach((v) => push(add(o, [v[1], v[2], v[0] + 0.3]), 1, 1.9)); };
    if (CUBES) {
      const cube = (o) => CUBEP.forEach((v) => push(add(o, [v[0] - CUBE / 2, v[1], v[2]]), 0.95, 1.5));
      cubeStart.forEach(([x, y], i) => {
        if (i === s.held) cube(add(tip, [0, 0, -2.2]));
        else if (i < s.n) cube([stackAt[0], stackAt[1], CUBE * i + CUBE / 2]);
        else cube([x, y, CUBE / 2]);
      });
    } else {
      if (s.s === "rest") lying([tubeAt.x, tubeAt.y, 0]);
      else if (s.s === "held") lying(add(tip, [0, 0, -1.6]));
      else if (s.s === "up") standing(add(tip, [0, 0, 1.5]));
      else if (s.s === "drop") standing(add(tip, [0, 0, 1.5 - s.u * 6]));
      else standing([hole.x, hole.y, RACK.h + 7.5]);
      RACKP.forEach((v) => push([v[0] + RACK.x, v[1] + RACK.y, v[2]], 0.75, 1.3));
    }

    pts.sort((a, b) => b[2] - a[2]);                  // far points first
    for (const [x, y, d, a, sz] of pts) {
      const fade = Math.max(0.35, Math.min(1, 1.35 - (d - 70) / 60));
      ctx.fillStyle = `rgba(29,78,216,${Math.min(1, a * fade * 1.05).toFixed(3)})`;
      ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz);
    }
    if (!reduce) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
