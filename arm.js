/* A blueprint robot arm that picks a lying tube, stands it up and drops it into a rack.
   Drawn on a full-screen canvas behind the page. Pure 2D: two links, a wrist, a gripper,
   solved with the same two-link inverse kinematics the real arm's planner uses. */
(function () {
  const cv = document.getElementById("arm");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, dpr = 1, S = 1, base = { x: 0, y: 0 };

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // keep the drawing on the right, clear of the headline on wide screens
    S = W > 760 ? Math.min(W * 0.5, H * 1.05) / 900 : Math.min(W * 0.95, H) / 900;
    base = { x: W * (W > 760 ? 0.8 : 0.55), y: H * 0.8 };
  }
  window.addEventListener("resize", resize);
  resize();

  // world units: base at origin, +x right, +y up. Table is y = 0.
  const L1 = 260, L2 = 230, HAND = 95;
  const TUBE = 120;                                    // tube length
  const tubeStart = { x: -330, y: 8 };                 // lying on the table, left of base
  const rack = { x: 250, y: 0, w: 150, h: 90, holes: [ -45, 0, 45 ] };

  // keyframes: fingertip position (x, y), hand angle (rad, -pi/2 = pointing down),
  // grip (0 closed .. 1 open), and what the tube is doing
  const P = -Math.PI / 2;
  const keys = [
    { t: 0.0, x: -40, y: 360, a: -0.6, g: 1, tube: "rest", say: "LOOK · MAP" },
    { t: 1.6, x: -330, y: 200, a: P, g: 1, tube: "rest", say: "LOCATE · PIXEL → TABLE" },
    { t: 2.4, x: -330, y: 200, a: P, g: 1, tube: "rest", say: "ORIENT · TWIST TO SQUARE" },
    { t: 3.4, x: -330, y: 30, a: P, g: 1, tube: "rest", say: "STRAIGHT DOWN · 90°" },
    { t: 3.9, x: -330, y: 30, a: P, g: 0.25, tube: "rest", say: "GRASP · JAWS STOP SHORT = HELD" },
    { t: 4.8, x: -330, y: 230, a: P, g: 0.25, tube: "held", say: "LIFT" },
    { t: 5.9, x: -120, y: 330, a: 0, g: 0.25, tube: "up", say: "HAND LEVEL · TUBE STANDS" },
    { t: 7.4, x: 250, y: 290, a: 0, g: 0.25, tube: "up", say: "CARRY HIGH" },
    { t: 8.2, x: 250, y: 205, a: 0, g: 0.25, tube: "up", say: "OVER THE HOLE" },
    { t: 8.7, x: 250, y: 205, a: 0, g: 1, tube: "drop", say: "RELEASE" },
    { t: 9.6, x: 120, y: 360, a: -0.6, g: 1, tube: "in", say: "VERIFY · CAP IN HOLE" },
    { t: 11.2, x: -40, y: 360, a: -0.6, g: 1, tube: "in", say: "NEXT TUBE" },
  ];
  const T = keys[keys.length - 1].t;

  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  function sample(t) {
    for (let i = 0; i < keys.length - 1; i++) {
      const a = keys[i], b = keys[i + 1];
      if (t >= a.t && t <= b.t) {
        const u = ease((t - a.t) / (b.t - a.t || 1));
        return { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u,
                 a: a.a + (b.a - a.a) * u, g: a.g + (b.g - a.g) * u,
                 tube: b.tube === "drop" ? "drop" : a.tube, u, say: b.say };
      }
    }
    return { ...keys[0], u: 0 };
  }

  function ik(tx, ty, ha) {
    // wrist = fingertip minus the hand, then the classic two-link solution (elbow up)
    const wx = tx - HAND * Math.cos(ha), wy = ty - HAND * Math.sin(ha) - 60;
    let d = Math.hypot(wx, wy);
    d = Math.min(d, L1 + L2 - 1);
    const c2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    // both elbow solutions; keep the one with the elbow HIGHER (elbow up, like the arm)
    let best = null;
    for (const sg of [1, -1]) {
      const q2 = sg * Math.acos(Math.max(-1, Math.min(1, c2)));
      const q1 = Math.atan2(wy, wx) - Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
      const ey = L1 * Math.sin(q1);
      if (!best || ey > best.ey) best = { q1, q2, ey };
    }
    return { q1: best.q1, q2: best.q2, q3: ha - best.q1 - best.q2 };
  }

  const X = (x) => base.x + x * S, Y = (y) => base.y - y * S;
  const COL = { line: "rgba(190,226,255,0.8)", faint: "rgba(190,226,255,0.25)",
                glow: "rgba(127,212,255,0.9)", text: "rgba(190,226,255,0.75)", amber: "#ffd27a" };

  function link(x1, y1, x2, y2, w) {
    const a = Math.atan2(y2 - y1, x2 - x1), nx = -Math.sin(a) * w / 2, ny = Math.cos(a) * w / 2;
    ctx.beginPath();
    ctx.moveTo(X(x1 + nx), Y(y1 + ny)); ctx.lineTo(X(x2 + nx), Y(y2 + ny));
    ctx.lineTo(X(x2 - nx), Y(y2 - ny)); ctx.lineTo(X(x1 - nx), Y(y1 - ny)); ctx.closePath();
    ctx.stroke();
    ctx.save(); ctx.setLineDash([6 * S, 6 * S]); ctx.strokeStyle = COL.faint;
    ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke(); ctx.restore();
  }
  function joint(x, y, r, label, ang) {
    ctx.beginPath(); ctx.arc(X(x), Y(y), r * S, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(x - r * 1.6), Y(y)); ctx.lineTo(X(x + r * 1.6), Y(y));
    ctx.moveTo(X(x), Y(y - r * 1.6)); ctx.lineTo(X(x), Y(y + r * 1.6)); ctx.stroke();
    if (label) {
      ctx.fillStyle = COL.text;
      ctx.fillText(`${label} ${(ang * 180 / Math.PI).toFixed(1)}°`, X(x + r * 2.2), Y(y + r * 1.8));
    }
  }
  function tubeShape(cx, cy, ang, capCol) {
    const dx = Math.cos(ang) * TUBE / 2, dy = Math.sin(ang) * TUBE / 2;
    ctx.save(); ctx.strokeStyle = COL.line; ctx.lineWidth = 1.2;
    link(cx - dx, cy - dy, cx + dx * 0.72, cy + dy * 0.72, 18);
    ctx.fillStyle = capCol; ctx.strokeStyle = capCol;
    const cxp = cx + dx * 0.86, cyp = cy + dy * 0.86;
    ctx.beginPath(); ctx.arc(X(cxp), Y(cyp), 11 * S, 0, Math.PI * 2); ctx.globalAlpha = 0.85; ctx.fill();
    ctx.restore();
  }

  function frame(ms) {
    const t = reduce ? 6.4 : (ms / 1000) % T;
    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 1.3; ctx.strokeStyle = COL.line;
    ctx.font = `${Math.max(10, 12 * S * 1.1)}px "IBM Plex Mono", ui-monospace, monospace`;

    // table + rack
    ctx.strokeStyle = COL.faint; ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke();
    ctx.strokeStyle = COL.line;
    ctx.strokeRect(X(rack.x - rack.w / 2), Y(rack.h), rack.w * S, rack.h * S);
    rack.holes.forEach((h) => { ctx.beginPath(); ctx.ellipse(X(rack.x + h), Y(rack.h), 13 * S, 4 * S, 0, 0, Math.PI * 2); ctx.stroke(); });
    ctx.fillStyle = COL.text; ctx.fillText("RACK · HOLE 1", X(rack.x - rack.w / 2), Y(rack.h + 18));

    const s = sample(t);
    const k = ik(s.x, s.y, s.a);
    const j1 = { x: 0, y: 60 };
    const j2 = { x: j1.x + L1 * Math.cos(k.q1), y: j1.y + L1 * Math.sin(k.q1) };
    const j3 = { x: j2.x + L2 * Math.cos(k.q1 + k.q2), y: j2.y + L2 * Math.sin(k.q1 + k.q2) };
    const ha = k.q1 + k.q2 + k.q3;
    const tip = { x: j3.x + HAND * Math.cos(ha), y: j3.y + HAND * Math.sin(ha) };

    // the tube
    const caps = "#4da3ff";
    if (s.tube === "rest") tubeShape(tubeStart.x + 20, tubeStart.y + 9, 0, caps);        // lying
    else if (s.tube === "held") tubeShape(tip.x, tip.y - 6, 0, caps);                    // lifted, still lying
    else if (s.tube === "up") tubeShape(tip.x, tip.y - TUBE * 0.35, Math.PI / 2, caps);   // standing in the jaws
    else if (s.tube === "drop") tubeShape(tip.x, tip.y - TUBE * 0.35 - s.u * 60, Math.PI / 2, caps);
    else tubeShape(rack.x, rack.h + 18, Math.PI / 2, caps);                             // in the hole

    // the arm
    ctx.strokeStyle = COL.line; ctx.lineWidth = 1.4;
    ctx.strokeRect(X(-70), Y(60), 140 * S, 60 * S);                 // base
    link(j1.x, j1.y, j2.x, j2.y, 44); link(j2.x, j2.y, j3.x, j3.y, 34);
    link(j3.x, j3.y, tip.x, tip.y, 30);
    joint(j1.x, j1.y, 16, "θ1", k.q1); joint(j2.x, j2.y, 13, "θ2", k.q2); joint(j3.x, j3.y, 11, "θ3", k.q3);
    // jaws
    const open = 10 + 26 * s.g, px = -Math.sin(ha), py = Math.cos(ha);
    ctx.beginPath();
    [-1, 1].forEach((sg) => {
      const bx = tip.x + px * open * sg, by = tip.y + py * open * sg;
      ctx.moveTo(X(bx), Y(by)); ctx.lineTo(X(bx + Math.cos(ha) * 30), Y(by + Math.sin(ha) * 30));
    });
    ctx.stroke();

    // the wrist camera's view cone and the target it has locked on
    const cam = { x: j3.x + Math.cos(ha) * 40, y: j3.y + Math.sin(ha) * 40 };
    ctx.save(); ctx.strokeStyle = "rgba(127,212,255,0.35)"; ctx.setLineDash([3 * S, 5 * S]);
    const look = s.tube === "rest" ? { x: tubeStart.x + 65, y: 17 } : { x: rack.x, y: rack.h };
    ctx.beginPath(); ctx.moveTo(X(cam.x), Y(cam.y)); ctx.lineTo(X(look.x - 40), Y(look.y));
    ctx.moveTo(X(cam.x), Y(cam.y)); ctx.lineTo(X(look.x + 40), Y(look.y)); ctx.stroke(); ctx.restore();
    ctx.save(); ctx.strokeStyle = COL.amber; ctx.lineWidth = 1.2;
    const r = 18 * S + Math.sin(ms / 180) * 2;
    ctx.beginPath(); ctx.arc(X(look.x), Y(look.y), r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(look.x) - r - 6, Y(look.y)); ctx.lineTo(X(look.x) - r + 4, Y(look.y));
    ctx.moveTo(X(look.x) + r - 4, Y(look.y)); ctx.lineTo(X(look.x) + r + 6, Y(look.y)); ctx.stroke();
    ctx.restore();

    // a dimension line for the reach
    ctx.save(); ctx.strokeStyle = COL.faint; ctx.fillStyle = COL.text;
    const dy = -34;
    ctx.beginPath(); ctx.moveTo(X(0), Y(dy)); ctx.lineTo(X(tip.x), Y(dy));
    ctx.moveTo(X(0), Y(dy - 8)); ctx.lineTo(X(0), Y(dy + 8));
    ctx.moveTo(X(tip.x), Y(dy - 8)); ctx.lineTo(X(tip.x), Y(dy + 8)); ctx.stroke();
    ctx.fillText(`r = ${(Math.abs(tip.x) / 10).toFixed(1)} cm`, X(tip.x / 2) - 30, Y(dy - 16));
    ctx.restore();

    // the step caption, like a callout on a drawing
    ctx.save(); ctx.fillStyle = COL.amber;
    ctx.font = `600 ${Math.max(11, 13 * S * 1.1)}px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.fillText(`▸ ${s.say || ""}`, X(-360), Y(560)); ctx.restore();

    if (!reduce) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
