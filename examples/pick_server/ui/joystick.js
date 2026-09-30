// Shared analog joysticks -> /jogvec. Used by BOTH the admin and guest pages.
//
// This was duplicated once and the copy was subtly broken: it translated the nub
// by a PERCENTAGE, and CSS translate percentages resolve against the element's
// own size, so the nub crept a few pixels and looked frozen. It also POSTed a
// JSON body while /jogvec reads QUERY ARGS r/th/z, so the arm never moved at all.
// One implementation now, so neither can drift again.
//
// Markup contract (same as the admin page): a .pad/.stick element containing a
// .nub child that is CENTRED BY LAYOUT (flex), never by transform — the transform
// is what we drive, in pixels.
window.RAXStick = (function () {
  const jv = {r: 0, th: 0, z: 0};
  let jvTimer = null;
  let gate = () => true;           // pages can veto sending (e.g. "not my turn")

  function jvSend() {
    if (!gate()) return;
    fetch(`/jogvec?r=${jv.r.toFixed(3)}&th=${jv.th.toFixed(3)}&z=${jv.z.toFixed(3)}`,
          {method: 'POST'}).catch(() => {});
  }
  function jvActive() {
    return Math.abs(jv.r) > 1e-3 || Math.abs(jv.th) > 1e-3 || Math.abs(jv.z) > 1e-3;
  }
  function jvStart() { if (!jvTimer) jvTimer = setInterval(jvSend, 60); }   // ~16 Hz
  function jvMaybeStop() {
    // always send the final zero, or the arm keeps drifting after release
    if (!jvActive() && jvTimer) { jvSend(); clearInterval(jvTimer); jvTimer = null; }
  }

  function makeStick(id, vert, apply) {
    const pad = document.getElementById(id);
    if (!pad) return;
    const nub = pad.querySelector('.nub');
    let on = false, cx = 0, cy = 0, R = 1;
    const dz = v => Math.abs(v) < 0.09 ? 0 : v;                             // deadzone
    function begin(e) {
      on = true;
      const r = pad.getBoundingClientRect();
      cx = r.left + r.width / 2;
      cy = r.top + r.height / 2;
      R = Math.max(12, r.width / 2 - 22);
      pad.setPointerCapture(e.pointerId);
      jvStart(); move(e); e.preventDefault();
    }
    function move(e) {
      if (!on) return;
      let dx = e.clientX - cx, dy = e.clientY - cy;
      if (vert) dx = 0;
      const d = Math.min(Math.hypot(dx, dy), R), a = Math.atan2(dy, dx);
      const kx = vert ? 0 : d * Math.cos(a), ky = d * Math.sin(a);
      nub.style.transform = `translate(${kx}px,${ky}px)`;                   // PIXELS
      apply(dz(kx / R), dz(ky / R));
      e.preventDefault();
    }
    function end() {
      on = false;
      nub.style.transform = 'translate(0,0)';
      apply(0, 0);
      jvMaybeStop();
    }
    pad.addEventListener('pointerdown', begin);
    pad.addEventListener('pointermove', move);
    pad.addEventListener('pointerup', end);
    pad.addEventListener('pointercancel', end);
  }

  return {
    jv,
    /** init({move:'id', lift:'id', enabled:fn}) */
    init(opts) {
      if (opts && typeof opts.enabled === 'function') gate = opts.enabled;
      if (opts && opts.move) makeStick(opts.move, false, (x, y) => { jv.r = -y; jv.th = -x; });
      if (opts && opts.lift) makeStick(opts.lift, true,  (x, y) => { jv.z = -y; });
    },
  };
})();
