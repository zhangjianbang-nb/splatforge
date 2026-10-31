// WebGL2 renderer for 3D Gaussian Splatting checkpoints.
//
// Loads a 3DGS PLY (the format SplatForge's GaussianModel.to_ply writes) and
// rasterises it: each Gaussian becomes a screen-space ellipse whose 2D
// covariance comes from projecting the 3D covariance through the camera
// Jacobian, composited back-to-front.
//
// The per-splat data lives in float textures and is fetched by gl_InstanceID,
// while a small integer texture holds the depth-sorted permutation.  Re-sorting
// therefore only re-uploads ~4 bytes per splat instead of the whole cloud.

const BLOCK = 1024;            // splats per texture row
const SH_C0 = 0.28209479177387814;

const VERT = `#version 300 es
precision highp float;
precision highp int;

uniform mat4 uView;
uniform mat3 uModelRot;     // scene up axis -> +Y, so the orbit camera is y-up
uniform vec2 uResolution;
uniform vec2 uFocal;        // focal length in pixels (x, y)
uniform vec3 uCamPos;
uniform float uScaleMul;
uniform float uAA;
uniform float uRadius;
uniform int uShDegree;
uniform int uBlock;
uniform highp usampler2D uOrder;
uniform highp sampler2D uSplat;   // 3 texels per splat
uniform highp sampler2D uSh;      // 12 texels per splat

out vec2 vCenter;
out vec3 vInvCov;
out vec4 vColor;

ivec2 origin(int i, int stride) {
  return ivec2((i % uBlock) * stride, i / uBlock);
}

float evalSh(float c[16], vec3 d) {
  float x = d.x, y = d.y, z = d.z;
  float r = 0.28209479177387814 * c[0];
  if (uShDegree > 0) {
    r += -0.4886025119029199 * y * c[1];
    r +=  0.4886025119029199 * z * c[2];
    r += -0.4886025119029199 * x * c[3];
  }
  if (uShDegree > 1) {
    r +=  1.0925484305920792 * x * y * c[4];
    r += -1.0925484305920792 * y * z * c[5];
    r +=  0.31539156525252005 * (2.0 * z * z - x * x - y * y) * c[6];
    r += -1.0925484305920792 * x * z * c[7];
    r +=  0.5462742152960396 * (x * x - y * y) * c[8];
  }
  if (uShDegree > 2) {
    r += -0.5900435899266435 * (y * (3.0 * x * x - y * y)) * c[9];
    r +=  2.890611442640554  * (x * y * z) * c[10];
    r += -0.4570457994644658 * (y * (4.0 * z * z - x * x - y * y)) * c[11];
    r +=  0.3731763325901154 * (z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)) * c[12];
    r += -0.4570457994644658 * (x * (4.0 * z * z - x * x - y * y)) * c[13];
    r +=  1.445305721320277  * (z * (x * x - y * y)) * c[14];
    r += -0.5900435899266435 * (x * (x * x - 3.0 * y * y)) * c[15];
  }
  return r;
}

void main() {
  int si = int(texelFetch(uOrder, origin(gl_InstanceID, 1), 0).r);
  ivec2 o = origin(si, 3);
  vec4 t0 = texelFetch(uSplat, o, 0);
  vec4 t1 = texelFetch(uSplat, o + ivec2(1, 0), 0);
  vec4 t2 = texelFetch(uSplat, o + ivec2(2, 0), 0);

  vec3 pos = uModelRot * t0.xyz;
  float opacity = 1.0 / (1.0 + exp(-t0.w));      // logit -> [0,1]
  vec3 scl = exp(t1.xyz) * uScaleMul;            // log-scale -> metres
  vec4 q = normalize(vec4(t1.w, t2.x, t2.y, t2.z));

  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat3 R = mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
    2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));

  mat3 M = uModelRot * R * mat3(scl.x, 0.0, 0.0, 0.0, scl.y, 0.0, 0.0, 0.0, scl.z);
  mat3 Sigma = M * transpose(M);

  vec4 viewPos = uView * vec4(pos, 1.0);
  if (viewPos.z > -0.001) {                      // behind the camera
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vColor = vec4(0.0); vCenter = vec2(0.0); vInvCov = vec3(1.0, 0.0, 1.0);
    return;
  }

  // Project the 3D covariance: Sigma' = J Sigma J^T, J the projection Jacobian
  // in y-up pixel coordinates.
  float invZ = 1.0 / viewPos.z;
  float negInvZ = -invZ;
  float a = uFocal.x * negInvZ;
  float b = uFocal.y * negInvZ;
  float c = uFocal.x * viewPos.x * invZ * invZ;
  float d = uFocal.y * viewPos.y * invZ * invZ;

  float S00 = Sigma[0][0], S01 = Sigma[0][1], S02 = Sigma[0][2];
  float S11 = Sigma[1][1], S12 = Sigma[1][2], S22 = Sigma[2][2];

  float s00 = a * a * S00 + 2.0 * a * c * S02 + c * c * S22 + uAA;
  float s01 = a * b * S01 + (a * d + b * c) * S02 + b * c * S12 + c * d * S22;
  float s11 = b * b * S11 + 2.0 * b * d * S12 + d * d * S22 + uAA;

  float mid = 0.5 * (s00 + s11);
  float diff = 0.5 * (s00 - s11);
  float disc = sqrt(max(diff * diff + s01 * s01, 0.0));
  float l1 = mid + disc;
  float l2 = max(mid - disc, 1e-6);

  vec2 e1 = (abs(s01) < 1e-9)
    ? ((s00 >= s11) ? vec2(1.0, 0.0) : vec2(0.0, 1.0))
    : normalize(vec2(s01, l1 - s00));
  vec2 e2 = vec2(-e1.y, e1.x);

  float i1 = 1.0 / l1, i2 = 1.0 / l2;
  vec3 invCov = vec3(
    e1.x * e1.x * i1 + e2.x * e2.x * i2,
    e1.x * e1.y * i1 + e2.x * e2.y * i2,
    e1.y * e1.y * i1 + e2.y * e2.y * i2);

  // View-dependent colour from the spherical harmonics.
  vec3 dir = normalize(pos - uCamPos);
  ivec2 so = origin(si, 12);
  float coeffs[16];
  vec3 rgb = vec3(0.0);
  for (int ch = 0; ch < 3; ch++) {
    for (int t = 0; t < 4; t++) {
      vec4 texel = texelFetch(uSh, so + ivec2(ch * 4 + t, 0), 0);
      coeffs[4 * t + 0] = texel.x;
      coeffs[4 * t + 1] = texel.y;
      coeffs[4 * t + 2] = texel.z;
      coeffs[4 * t + 3] = texel.w;
    }
    rgb[ch] = evalSh(coeffs, dir);
  }
  vec3 color = clamp(rgb + 0.5, 0.0, 1.0);

  // Screen-space quad at +-uRadius sigma along the ellipse axes.
  float r1 = min(uRadius * sqrt(l1), 4096.0);
  float r2 = min(uRadius * sqrt(l2), 4096.0);
  vec2 corner = vec2(
    (gl_VertexID == 0 || gl_VertexID == 2) ? -1.0 : 1.0,
    (gl_VertexID < 2) ? -1.0 : 1.0);
  vec2 offset = corner.x * r1 * e1 + corner.y * r2 * e2;
  vec2 centerPx = 0.5 * uResolution
                + vec2(uFocal.x * viewPos.x * negInvZ, uFocal.y * viewPos.y * negInvZ);

  gl_Position = vec4(((centerPx + offset) / uResolution) * 2.0 - 1.0, 0.0, 1.0);
  vCenter = centerPx;
  vInvCov = invCov;
  vColor = vec4(color * opacity, opacity);       // premultiplied
}`;

const FRAG = `#version 300 es
precision highp float;

in vec2 vCenter;
in vec3 vInvCov;
in vec4 vColor;
out vec4 fragColor;

void main() {
  vec2 d = gl_FragCoord.xy - vCenter;
  float power = -0.5 * (vInvCov.x * d.x * d.x
                        + 2.0 * vInvCov.y * d.x * d.y
                        + vInvCov.z * d.y * d.y);
  if (power > 0.0) discard;
  float w = exp(power);
  float alpha = vColor.a * w;
  if (alpha < 0.0039) discard;                   // below 1/255
  fragColor = vec4(vColor.rgb * w, alpha);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader compile failed: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

/** Parse a binary 3DGS PLY. Returns the raw float body plus resolved offsets. */
export function parsePly(buffer) {
  const bytes = new Uint8Array(buffer);
  const headerText = new TextDecoder('ascii').decode(bytes.subarray(0, Math.min(8192, bytes.length)));
  const marker = headerText.indexOf('end_header');
  if (marker < 0) throw new Error('not a PLY: no end_header');
  const bodyStart = bytes.indexOf(0x0a, marker) + 1;      // newline after end_header

  const lines = headerText.slice(0, marker).split('\n').map((l) => l.trim());
  if (!lines[0].startsWith('ply')) throw new Error('not a PLY file');
  if (!lines.some((l) => l.includes('binary_little_endian'))) {
    throw new Error('only binary_little_endian PLY is supported');
  }

  let count = 0;
  const props = [];
  for (const line of lines) {
    if (line.startsWith('element vertex')) count = parseInt(line.split(/\s+/)[2], 10);
    else if (line.startsWith('property') && line.includes('float')) {
      props.push({ name: line.split(/\s+/).pop(), offset: props.length * 4 });
    }
  }
  if (!count || !props.length) throw new Error('no vertex element in PLY');

  const strideBytes = props.length * 4;
  const needed = bodyStart + count * strideBytes;
  if (buffer.byteLength < needed) {
    throw new Error(`PLY truncated: expected ${needed} bytes, got ${buffer.byteLength}`);
  }
  const index = {};
  for (const p of props) index[p.name] = p.offset / 4;

  const body = new Float32Array(buffer.slice(bodyStart, needed));
  return { count, stride: strideBytes / 4, index, body };
}

export class SplatViewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      preserveDrawingBuffer: true,      // lets us grab screenshots
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is unavailable in this browser');
    this.gl = gl;

    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(this.program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('program link failed: ' + gl.getProgramInfoLog(this.program));
    }
    gl.useProgram(this.program);
    this.uni = {};
    for (const n of ['uView', 'uModelRot', 'uResolution', 'uFocal', 'uCamPos', 'uScaleMul',
                     'uAA', 'uRadius', 'uShDegree', 'uBlock', 'uOrder', 'uSplat', 'uSh']) {
      this.uni[n] = gl.getUniformLocation(this.program, n);
    }
    this.modelRot = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

    // Each splat owns 3 texels of core data and 12 of SH coefficients, laid out
    // in rows.  Software rasterisers (SwiftShader) cap textures at 8192, so the
    // row width has to follow the device limit rather than a fixed constant.
    const maxTexels = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    let block = BLOCK;
    while (block * 12 > maxTexels && block > 32) block >>= 1;
    this.block = block;
    gl.uniform1i(this.uni.uBlock, block);
    gl.uniform1i(this.uni.uOrder, 0);
    gl.uniform1i(this.uni.uSplat, 1);
    gl.uniform1i(this.uni.uSh, 2);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.texOrder = this._makeTexture(gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT);
    this.texSplat = this._makeTexture(gl.RGBA32F, gl.RGBA, gl.FLOAT);
    this.texSh = this._makeTexture(gl.RGBA32F, gl.RGBA, gl.FLOAT);

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(1, 1, 1, 1);

    this.options = {
      background: [1, 1, 1],
      scaleMul: 1.0,
      radius: 3.0,
      aa: 0.3,
      shDegree: 3,
      sortMode: 'continuous',       // continuous | release | none
    };

    this.camera = { target: [0, 0, 0], yaw: 0, pitch: 0, distance: 3, fovY: 50 };
    this.count = 0;
    this.positions = null;
    this.view = new Float32Array(16);
    this.depths = null;
    this.orderData = null;
    this.rows = 0;
    this.sortPending = false;
    this.sortInFlight = false;
    this.stats = { fps: 0, sortMs: 0, drawMs: 0, sorted: 0 };
    this._worker = null;
    this._workerFailed = false;
    this._sortGeneration = 0;
    this._lastFrame = performance.now();
    this._frames = 0;
    this._onCameraChange = null;
    this._resizeObserver = null;
    this._installWorker();
    this._installControls();
    this._observeResize();
  }

  _makeTexture(internalFormat, format, type) {
    const { gl } = this;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    tex._fmt = { internalFormat, format, type };
    return tex;
  }

  _upload(tex, width, height, data) {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, tex._fmt.internalFormat, width, height, 0,
                  tex._fmt.format, tex._fmt.type, data);
  }

  // ---------------------------------------------------------------- loading

  /** Load a 3DGS PLY from an ArrayBuffer. `meta` may carry camera framing. */
  async load(buffer, meta = {}) {
    const ply = parsePly(buffer);
    const { count, stride, index, body } = ply;
    const { gl } = this;

    const need = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
                  'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
    for (const n of need) {
      if (!(n in index)) throw new Error(`PLY is missing property "${n}" — not a 3DGS cloud?`);
    }

    const rows = Math.ceil(count / this.block);
    const block = this.block;
    const splat = new Float32Array(block * 3 * 4 * rows);
    const sh = new Float32Array(block * 12 * 4 * rows);
    const positions = new Float32Array(count * 3);

    const hasRest = 'f_rest_0' in index;
    const restBase = hasRest ? index.f_rest_0 : -1;

    for (let i = 0; i < count; i++) {
      const src = i * stride;
      const bx = i % block;
      const by = (i / block) | 0;

      const x = body[src + index.x];
      const y = body[src + index.y];
      const z = body[src + index.z];
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      const a = (by * block * 3 + bx * 3) * 4;
      splat[a + 0] = x;
      splat[a + 1] = y;
      splat[a + 2] = z;
      splat[a + 3] = body[src + index.opacity];
      splat[a + 4] = body[src + index.scale_0];
      splat[a + 5] = body[src + index.scale_1];
      splat[a + 6] = body[src + index.scale_2];
      splat[a + 7] = body[src + index.rot_0];
      splat[a + 8] = body[src + index.rot_1];
      splat[a + 9] = body[src + index.rot_2];
      splat[a + 10] = body[src + index.rot_3];
      splat[a + 11] = 0;

      // SH layout in the texture: [R0..R15, G0..G15, B0..B15]
      const s = (by * block * 12 + bx * 12) * 4;
      for (let k = 0; k < 16; k++) {
        const rest = hasRest ? restBase + (k - 1) : -1;
        sh[s + k] = k === 0 ? body[src + index.f_dc_0]
                            : (hasRest ? body[src + rest] : 0);
        sh[s + 16 + k] = k === 0 ? body[src + index.f_dc_1]
                                 : (hasRest ? body[src + rest + 15] : 0);
        sh[s + 32 + k] = k === 0 ? body[src + index.f_dc_2]
                                 : (hasRest ? body[src + rest + 30] : 0);
      }
    }

    this.count = count;
    this.positions = positions;
    this.rows = rows;
    this.orderData = new Uint32Array(rows * block);
    this.depths = new Float32Array(count);
    for (let i = 0; i < count; i++) this.orderData[i] = i;

    this._upload(this.texSplat, block * 3, rows, splat);
    this._upload(this.texSh, block * 12, rows, sh);
    this._uploadOrder(this.orderData);

    this._frame(meta);
    return { count, rows };
  }

  /** Load straight from a URL (same-origin; used for /ply/<scene>). */
  async loadUrl(url, meta = {}) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
    return this.load(await res.arrayBuffer(), meta);
  }

  /** Fit the camera to the cloud, honouring server-provided framing if given. */
  _frame(meta) {
    const c = this.camera;
    // The viewer's orbit camera is y-up, but a NeRF-synthetic scene is authored
    // z-up; rotate the cloud so the scene's own up axis becomes +Y.
    this.modelRot = rotationBetween(meta?.up ?? [0, 1, 0], [0, 1, 0]);

    if (meta && meta.center && meta.radius) {
      const target = rotate(this.modelRot, meta.target ?? meta.center);
      c.target = target;
      if (meta.initialEye) {
        const eye = rotate(this.modelRot, meta.initialEye);
        const dx = eye[0] - target[0];
        const dy = eye[1] - target[1];
        const dz = eye[2] - target[2];
        c.distance = Math.hypot(dx, dy, dz) || meta.radius * 2.6;
        c.yaw = Math.atan2(dx, dz);
        c.pitch = Math.asin(Math.max(-1, Math.min(1, dy / c.distance)));
      } else {
        c.distance = meta.radius * 2.6;
      }
      if (meta.fovY) c.fovY = meta.fovY;
      return;
    }
    // No metadata (e.g. a dropped file): robust bounds from a sampled subset.
    const p = this.positions;
    const n = this.count;
    const step = Math.max(1, Math.floor(n / 4000));
    const sample = [];
    for (let i = 0; i < n; i += step) {
      sample.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    sample.sort((a, b) => a - b);
    const lo = sample[Math.floor(sample.length * 0.02)];
    const hi = sample[Math.floor(sample.length * 0.98)];
    const extent = hi - lo || 1;
    // Per-axis percentiles via a coarse histogram-free approach: use the global
    // spread and the centroid, which is stable for these scenes.
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
      cx += p[i * 3]; cy += p[i * 3 + 1]; cz += p[i * 3 + 2];
    }
    cx /= n; cy /= n; cz /= n;
    c.target = [cx, cy, cz];
    c.distance = extent * 2.2;
    c.yaw = 0.6;
    c.pitch = 0.25;
  }

  // --------------------------------------------------------------- controls

  _installControls() {
    const canvas = this.canvas;
    let dragging = null;
    let lastX = 0, lastY = 0;

    const onDown = (e) => {
      dragging = (e.button === 0 && !e.shiftKey) ? 'orbit' : 'pan';
      lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture?.(e.pointerId);
    };
    const onMove = (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (dragging === 'orbit') {
        this.camera.yaw -= dx * 0.006;
        this.camera.pitch = Math.max(-1.5533, Math.min(1.5533,
          this.camera.pitch + dy * 0.006));
      } else {
        const scale = this.camera.distance * 0.0016;
        const right = [Math.cos(this.camera.yaw), 0, -Math.sin(this.camera.yaw)];
        const up = [0, 1, 0];
        for (let i = 0; i < 3; i++) {
          this.camera.target[i] += (-dx * scale) * right[i] + (dy * scale) * up[i];
        }
      }
      this._cameraMoved();
    };
    const onUp = (e) => {
      if (dragging && this.options.sortMode === 'release') this.requestSort(true);
      dragging = null;
      canvas.releasePointerCapture?.(e.pointerId);
    };
    const onWheel = (e) => {
      e.preventDefault();
      this.camera.distance *= Math.exp(e.deltaY * 0.0012);
      this.camera.distance = Math.max(0.05, Math.min(1e4, this.camera.distance));
      this._cameraMoved();
    };

    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  _cameraMoved() {
    if (this.options.sortMode === 'continuous') this.requestSort();
    if (this._onCameraChange) this._onCameraChange();
  }

  _observeResize() {
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
    };
    resize();
    this._resizeObserver = new ResizeObserver(resize);
    this._resizeObserver.observe(this.canvas);
    window.addEventListener('resize', resize);
  }

  // ---------------------------------------------------------------- sorting

  _installWorker() {
    try {
      this._worker = new Worker('sort-worker.js');
      this._worker.onmessage = (e) => this._applyOrder(e.data.order, e.data.generation);
      this._worker.onerror = () => {
        this._workerFailed = true;
        this._worker = null;
      };
    } catch (err) {
      this._workerFailed = true;
    }
  }

  requestSort(force = false) {
    if (!this.count) return;
    if (this.options.sortMode === 'none' && !force) return;
    if (this.sortInFlight) { this.sortPending = true; return; }
    this._sortNow();
  }

  _sortNow() {
    if (!this.count) return;
    const t0 = performance.now();
    this._viewMatrix();                 // depths need the current camera
    const view = this.view;
    const p = this.positions;
    const depths = this.depths;
    const n = this.count;
    for (let i = 0; i < n; i++) {
      const j = i * 3;
      depths[i] = view[2] * p[j] + view[6] * p[j + 1] + view[10] * p[j + 2] + view[14];
    }

    if (this._worker && !this._workerFailed) {
      const copy = depths.slice();
      this.sortInFlight = true;
      this._sortGeneration++;
      this._worker.postMessage({ depths: copy, generation: this._sortGeneration },
                               [copy.buffer]);
      this.stats.sortMs = performance.now() - t0;
    } else {
      // Fallback for file:// (workers are blocked there): main-thread radix sort.
      const order = radixSort(depths);
      this._applyOrder(order, this._sortGeneration);
      this.stats.sortMs = performance.now() - t0;
    }
  }

  _applyOrder(order, generation) {
    this.sortInFlight = false;
    if (generation >= this._sortGeneration) {
      this.orderData.set(order);
      this._uploadOrder(this.orderData);
      this.stats.sorted++;
    }
    if (this.sortPending) {
      this.sortPending = false;
      this._sortNow();
    }
  }

  _uploadOrder(data) {
    this._upload(this.texOrder, this.block, this.rows, data);
  }

  // ---------------------------------------------------------------- drawing

  _viewMatrix() {
    const c = this.camera;
    const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
    const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
    const dir = [cp * sy, sp, cp * cy];              // target -> eye
    const eye = [
      c.target[0] + dir[0] * c.distance,
      c.target[1] + dir[1] * c.distance,
      c.target[2] + dir[2] * c.distance,
    ];
    const fwd = [-dir[0], -dir[1], -dir[2]];
    const up = [0, 1, 0];
    let right = [
      fwd[1] * up[2] - fwd[2] * up[1],
      fwd[2] * up[0] - fwd[0] * up[2],
      fwd[0] * up[1] - fwd[1] * up[0],
    ];
    let rl = Math.hypot(...right);
    if (rl < 1e-6) { right = [1, 0, 0]; rl = 1; }
    right = right.map((v) => v / rl);
    const trueUp = [
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0],
    ];
    const m = this.view;
    m[0] = right[0]; m[4] = right[1]; m[8] = right[2];
    m[1] = trueUp[0]; m[5] = trueUp[1]; m[9] = trueUp[2];
    m[2] = -fwd[0]; m[6] = -fwd[1]; m[10] = -fwd[2];
    m[3] = 0; m[7] = 0; m[11] = 0;
    m[12] = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
    m[13] = -(trueUp[0] * eye[0] + trueUp[1] * eye[1] + trueUp[2] * eye[2]);
    m[14] = fwd[0] * eye[0] + fwd[1] * eye[1] + fwd[2] * eye[2];
    m[15] = 1;
    return eye;
  }

  render() {
    const { gl } = this;
    const now = performance.now();
    this._frames++;
    if (now - this._lastFrame > 500) {
      this.stats.fps = (this._frames * 1000) / (now - this._lastFrame);
      this._frames = 0;
      this._lastFrame = now;
    }

    const t0 = performance.now();
    const eye = this._viewMatrix();
    const w = this.canvas.width, h = this.canvas.height;
    const t = Math.tan((this.camera.fovY * Math.PI / 180) / 2);
    const fy = (h / 2) / t;
    const fx = (w / 2) / t;

    gl.viewport(0, 0, w, h);
    const bg = this.options.background;
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (!this.count) { this.stats.drawMs = performance.now() - t0; return; }

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texOrder);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.texSplat);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.texSh);

    gl.uniformMatrix4fv(this.uni.uView, false, this.view);
    gl.uniformMatrix3fv(this.uni.uModelRot, false, this.modelRot);
    gl.uniform2f(this.uni.uResolution, w, h);
    gl.uniform2f(this.uni.uFocal, fx, fy);
    gl.uniform3f(this.uni.uCamPos, eye[0], eye[1], eye[2]);
    gl.uniform1f(this.uni.uScaleMul, this.options.scaleMul);
    gl.uniform1f(this.uni.uAA, this.options.aa);
    gl.uniform1f(this.uni.uRadius, this.options.radius);
    gl.uniform1i(this.uni.uShDegree, this.options.shDegree);

    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    this.stats.drawMs = performance.now() - t0;
  }

  setBackground(rgb) { this.options.background = rgb; }
  setScale(v) { this.options.scaleMul = v; }
  setShDegree(d) { this.options.shDegree = d; }
  setSortMode(m) { this.options.sortMode = m; }

  dispose() {
    if (this._worker) this._worker.terminate();
    this._resizeObserver?.disconnect();
  }
}

/** Rotation (column-major 3x3) taking unit vector `a` onto unit vector `b`. */
export function rotationBetween(a, b) {
  const al = Math.hypot(...a) || 1;
  const A = a.map((v) => v / al);
  const dot = A[0] * b[0] + A[1] * b[1] + A[2] * b[2];
  let axis = [
    A[1] * b[2] - A[2] * b[1],
    A[2] * b[0] - A[0] * b[2],
    A[0] * b[1] - A[1] * b[0],
  ];
  let s = Math.hypot(...axis);

  if (s < 1e-8) {
    if (dot > 0) return new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    // Antiparallel: any 180 degree turn about a perpendicular axis will do.
    axis = Math.abs(A[0]) < 0.9
      ? [0 * A[2] - 0 * A[1], 0 * A[0] - 1 * A[2], 1 * A[1] - 0 * A[0]]
      : [1 * A[2] - 0 * A[1], 0 * A[0] - 0 * A[2], 0 * A[1] - 1 * A[0]];
    const l = Math.hypot(...axis) || 1;
    const x = axis[0] / l, y = axis[1] / l, z = axis[2] / l;
    return new Float32Array([
      2 * x * x - 1, 2 * x * y, 2 * x * z,
      2 * x * y, 2 * y * y - 1, 2 * y * z,
      2 * x * z, 2 * y * z, 2 * z * z - 1,
    ]);
  }

  const [x, y, z] = axis.map((v) => v / s);
  const k = 1 - dot;
  return new Float32Array([
    dot + x * x * k, y * x * k + z * s, z * x * k - y * s,
    x * y * k - z * s, dot + y * y * k, z * y * k + x * s,
    x * z * k + y * s, y * z * k - x * s, dot + z * z * k,
  ]);
}

/** Apply a column-major 3x3 matrix to a 3-vector. */
export function rotate(m, v) {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

/** LSD radix sort of indices by float key — used when workers are unavailable. */
export function radixSort(depths) {
  const n = depths.length;
  const keys = new Uint32Array(n);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < n; i++) {
    f32[0] = depths[i];
    const bits = u32[0];
    keys[i] = (bits & 0x80000000) ? (~bits >>> 0) : ((bits | 0x80000000) >>> 0);
  }
  let order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  let tmp = new Uint32Array(n);
  const count = new Uint32Array(256);
  for (let shift = 0; shift < 32; shift += 8) {
    count.fill(0);
    for (let i = 0; i < n; i++) count[(keys[order[i]] >>> shift) & 255]++;
    let sum = 0;
    for (let i = 0; i < 256; i++) { const c = count[i]; count[i] = sum; sum += c; }
    for (let i = 0; i < n; i++) {
      const idx = order[i];
      tmp[count[(keys[idx] >>> shift) & 255]++] = idx;
    }
    const swap = order; order = tmp; tmp = swap;
  }
  return order;
}

export { SH_C0, BLOCK };
