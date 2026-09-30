// Shared 3D robot viewer: poses the REAL URDF meshes from /urdf using the live
// link transforms from /geom. Used by BOTH the admin and guest pages - it was
// inline in admin.html until the guest UI needed it too.
(function(){
  const a = document.getElementById('r3d-link'); if(a) a.href = `http://${location.hostname}:9090`;
  const cv = document.getElementById('v3d'); if(!cv) return;
  const ctx = cv.getContext('2d');
  // az 0 = look straight down the robot's forward axis from behind. It was -1.05
  // (-60deg), an off-axis orbit that makes a straight base look turned and a
  // straight wrist look sideways. Drag still rotates; this is only the default.
  let az = 0.0, el = 0.62, geom = {links:[], ee:null, obj:null};
  // Turn the robot 30° left ON the table: yaw the robot + objects about the base
  // vertical (z) axis while the ground grid stays fixed. (Orbiting the camera via
  // `az` rotates the whole scene together, so the robot never turns on the table.)
  // YAW was 0.524 (30deg) to "turn the robot on the table". It rotates the robot
  // AND the object markers, so a cube straight ahead got drawn 30deg to the left
  // and the wrist looked twisted when wrist_roll was actually 0. That made the 3D
  // view disagree with the camera and the map. Back to 0 so the view is truthful.
  // Scene yaw = the base-zero error (shoulder_pan reads ~-14deg while the arm is
  // physically straight). Rotating the WHOLE scene keeps the robot and the object
  // markers consistent with each other; offsetting only the arm pulled them apart.
  const YAW = 0.246, YC = Math.cos(YAW), YS = Math.sin(YAW);
  function yz(p){ return [p[0]*YC - p[1]*YS, p[0]*YS + p[1]*YC, p[2]]; }
  function resize(){ const r = cv.getBoundingClientRect(); const dpr = window.devicePixelRatio||1;
    cv.width = Math.round(r.width*dpr); cv.height = Math.round(r.height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); }
  window.addEventListener('resize', resize); resize();
  // The REAL URDF meshes (so101_new_calib.urdf -> assets/*.stl), fetched once in
  // link-local coords; /geom streams a 4x4 per link. This used to draw a bare
  // polyline through the link ORIGINS, which is why the arm looked like a stick
  // figure and its size/reach couldn't be judged against the cube.
  let mesh = null;
  fetch('/urdf').then(r=>r.json()).then(d=>{
    mesh = (d.links||[]).map(L=>({name:L.name, v:Float32Array.from(L.v), f:Int32Array.from(L.f)}));
    if(!mesh.length) mesh = null;
    draw();
  }).catch(()=>{});
  const LINKCOL = {base_link:[122,134,152], shoulder_link:[100,140,200],
    upper_arm_link:[120,160,215], lower_arm_link:[100,140,200],
    wrist_link:[130,170,220], gripper_link:[190,200,214],
    moving_jaw_so101_v1_link:[225,232,241]};
  // rotate: base-frame point -> screen. z is up. also returns view depth.
  function proj(p){ const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    const rx = -p[0]*sa + p[1]*ca;            // screen right
    const dep =  p[0]*ca + p[1]*sa;           // into-screen (before tilt)
    const uy =  p[2]*ce - dep*se;             // screen up
    const W = cv.clientWidth, H = cv.clientHeight, s = Math.min(W,H)/0.60;
    return [W*0.5 + s*rx, H*0.62 - s*uy, dep*ce + p[2]*se]; }
  function line(a,b,col,w){ const p=proj(a), q=proj(b); ctx.strokeStyle=col; ctx.lineWidth=w||1;
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke(); }
  function drawMeshes(){
    const xf = geom.xf || {}; const tris = [];
    // View direction (into the screen) in BASE coords — depth = dot(p, vd).
    const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    // yaw the view direction WITH the geometry so backface culling stays correct
    const vd0x=ca*ce, vd0y=sa*ce; const vdx=vd0x*YC - vd0y*YS, vdy=vd0x*YS + vd0y*YC, vdz=se;
    for(const L of mesh){
      const T = xf[L.name]; if(!T) continue;
      const nv = L.v.length/3, sx=new Float64Array(nv), sy=new Float64Array(nv), sd=new Float64Array(nv);
      const bx=new Float64Array(nv), by=new Float64Array(nv), bz=new Float64Array(nv);
      for(let i=0;i<nv;i++){
        const x=L.v[3*i], y=L.v[3*i+1], z=L.v[3*i+2];
        const X0 = T[0]*x+T[1]*y+T[2]*z+T[3];      // row-major 4x4
        const Y0 = T[4]*x+T[5]*y+T[6]*z+T[7];
        const Z = T[8]*x+T[9]*y+T[10]*z+T[11];
        const X = X0*YC - Y0*YS, Y = X0*YS + Y0*YC;   // yaw about base z (turn on table)
        bx[i]=X; by[i]=Y; bz[i]=Z;
        const s = proj([X,Y,Z]); sx[i]=s[0]; sy[i]=s[1]; sd[i]=s[2];
      }
      const c = LINKCOL[L.name] || [140,160,190];
      for(let t=0;t<L.f.length;t+=3){
        const a=L.f[t], b=L.f[t+1], q=L.f[t+2];
        // backface/shade from the base-frame normal; cheap lambert
        const ux=bx[b]-bx[a], uy2=by[b]-by[a], uz=bz[b]-bz[a];
        const vx=bx[q]-bx[a], vy=by[q]-by[a], vz=bz[q]-bz[a];
        let nx=uy2*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy2*vx;
        const nl=Math.hypot(nx,ny,nz)||1; nx/=nl; ny/=nl; nz/=nl;
        // BACKFACE CULL. Winding is preserved through decimation now, so the
        // normal is a real outward normal. Drawing both sides (what it did before)
        // paints the INSIDE of the arm on top of the outside — that is what made
        // it look transparent/x-ray, not the triangle budget. Skip faces pointing
        // away from the viewer and only the outer skin remains.
        if(nx*vdx + ny*vdy + nz*vdz >= 0) continue;
        const lam = Math.max(0.34, Math.min(1, 0.42 + 0.58*(0.35*nx - 0.45*ny + 0.82*nz)));
        tris.push([(sd[a]+sd[b]+sd[q])/3, sx[a],sy[a],sx[b],sy[b],sx[q],sy[q],
                   `rgb(${Math.round(c[0]*lam)},${Math.round(c[1]*lam)},${Math.round(c[2]*lam)})`]);
      }
    }
    tris.sort((p,q)=>q[0]-p[0]);                 // painter's: far first
    ctx.lineJoin='round';
    for(const t of tris){
      ctx.fillStyle = t[7]; ctx.strokeStyle = t[7]; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(t[1],t[2]); ctx.lineTo(t[3],t[4]); ctx.lineTo(t[5],t[6]); ctx.closePath();
      ctx.fill();
      ctx.stroke();   // seal the sub-pixel seams between fills — un-stroked
                      // canvas triangles leave hairline gaps that show the dark
                      // background through and make the arm look transparent.
    }
  }
  function draw(){
    const W=cv.clientWidth, H=cv.clientHeight; ctx.clearRect(0,0,W,H);
    // ground grid at z=0
    const g=0.30, n=6; ctx.globalAlpha=0.5;
    for(let i=0;i<=n;i++){ const t=-g+2*g*i/n;
      line([t,-g,0],[t,g,0],'#243040',1); line([-g,t,0],[g,t,0],'#243040',1); }
    ctx.globalAlpha=1;
    // base axes (turn with the robot's base frame)
    line(yz([0,0,0]),yz([0.08,0,0]),'#c9524a',2); line(yz([0,0,0]),yz([0,0.08,0]),'#4a93c9',2); line(yz([0,0,0]),yz([0,0,0.08]),'#4ac275',2);
    if(mesh && geom.xf){
      drawMeshes();
    } else {
      // fallback: link-origin polyline (what this viewer used to be)
      const L=geom.links||[];
      for(let i=0;i+1<L.length;i++) line(yz(L[i]),yz(L[i+1]),'#7f9fd8',4);
      for(const p of L){ const s=proj(yz(p)); ctx.fillStyle='#b9c9ea'; ctx.beginPath(); ctx.arc(s[0],s[1],3.5,0,7); ctx.fill(); }
    }
    if(geom.ee){ const s=proj(yz(geom.ee)); ctx.fillStyle='#e6ecf1'; ctx.beginPath(); ctx.arc(s[0],s[1],4.5,0,7); ctx.fill(); }
    // object cube
    if(geom.obj){ const o=geom.obj, h=(geom.obj_size||0.03)/2;
      const c=[]; for(let dx of [-h,h]) for(let dy of [-h,h]) for(let dz of [-h,h]) c.push(yz([o[0]+dx,o[1]+dy,o[2]+dz]));
      const E=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
      const col = (geom.obj_label==='green')?'#3fc46b':'#e2574c';
      for(const e of E) line(c[e[0]],c[e[1]],col,2);
      const s=proj(yz(o)); ctx.fillStyle=col; ctx.font='11px ui-monospace,Consolas';
      ctx.fillText(`${(geom.obj_label||'obj')}  r=${Math.hypot(o[0],o[1]).toFixed(2)}m`, s[0]+8, s[1]-8); }
    // mapped objects (2D map) — each as its MEASURED box, yawed, resting on the
    // table (z: 0..h). Not a uniform cube any more: a pen draws as a pen.
    const E2=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
    for(const o of (geom.objs2d||[])){
      const hw=(o.w||0.03)/2, hd=(o.d||0.03)/2, ht=o.h||0.03;
      const yw=Math.PI/180*(o.yaw||0), cs=Math.cos(yw), sn=Math.sin(yw);
      const cl=/green/.test(o.label)?'#3fc46b':/blue/.test(o.label)?'#4a93c9':/red/.test(o.label)?'#e2574c':'#e0b040';
      const c=[]; for(const dd of [-hd,hd]) for(const dw of [-hw,hw]) for(const dz of [0,ht])
        c.push(yz([o.x+dd*cs-dw*sn, o.y+dd*sn+dw*cs, dz]));
      for(const e of E2) line(c[e[0]],c[e[1]],cl,1.5);
      const s=proj(yz([o.x,o.y,ht])); ctx.fillStyle=cl; ctx.font='10px ui-monospace,Consolas';
      ctx.fillText(`${o.label.split(' ')[0]}#${o.tag}`, s[0]+7, s[1]-6); }
  }
  let drag=null;
  cv.addEventListener('pointerdown', e=>{ drag=[e.clientX,e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e=>{ if(!drag) return; az -= (e.clientX-drag[0])*0.01; el += (e.clientY-drag[1])*0.01;
    el=Math.max(-0.2,Math.min(1.5,el)); drag=[e.clientX,e.clientY]; draw(); });
  cv.addEventListener('pointerup', ()=>{ drag=null; });
  async function poll(){ try{ geom = await (await fetch('/geom')).json(); }catch(e){} draw(); }
  poll(); setInterval(poll, 180);
})();
