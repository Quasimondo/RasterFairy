/*
 * RasterFairy (JavaScript port)
 * ---------------------------------------------------------------------------
 * Transforms a 2D point cloud into a regular grid while trying to preserve the
 * neighbourhood relations of the original cloud, using the same recursive
 * slicing idea as the Python reference implementation.
 *
 * This is an independent JavaScript port of the core algorithm — it produces a
 * valid, bijective assignment (every point lands on its own grid cell) and is
 * neighbourhood-preserving, but it is NOT bit-identical to the Python version
 * (the Python code uses an exact Hungarian assignment for small leaf slices;
 * this port uses a scanline assignment). See README.md.
 *
 * Original algorithm: "Raster Fairy by Mario Klingemann"
 *   https://github.com/Quasimondo/RasterFairy
 *
 * No dependencies. Works in the browser and in Node (ES module).
 */

/* ------------------------------------------------------------------------- */
/* Core transform                                                            */
/* ------------------------------------------------------------------------- */

/**
 * Map a point cloud onto the free cells of a mask.
 *
 * @param {Array<[number,number]>} points  Cloud as an array of [x, y] pairs.
 * @param {{w:number,h:number,free:Uint8Array}} [mask]  Target grid. `free` is a
 *        row-major w*h array where 1 marks a usable cell. The number of free
 *        cells must equal `points.length`. Defaults to `makeRect(points.length)`.
 * @returns {Array<[number,number]>} One [gridX, gridY] per input point, in the
 *        same order as `points`.
 */
export function transformPointCloud2D(points, mask) {
  if (!mask) mask = makeRect(points.length, 'auto');
  const n = points.length, W = mask.w, H = mask.h;

  // normalize points into grid coordinate space (scale independence)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
  }
  const sx = (maxX - minX) || 1, sy = (maxY - minY) || 1;
  const px = new Float64Array(n), py = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    px[i] = (points[i][0] - minX) / sx * (W - 1);
    py[i] = (points[i][1] - minY) / sy * (H - 1);
  }
  const free = mask.free.slice(); // 1 = usable cell
  const out = new Float64Array(2 * n);

  const freeAt = (x, y) => free[y * W + x];

  function cropGrid(g) { // trim empty edge rows/cols, in place
    let changed = true;
    while (changed && g[2] > 0 && g[3] > 0) {
      changed = false;
      let s;
      s = 0; for (let y = g[1]; y < g[1] + g[3]; y++) s += freeAt(g[0], y);
      if (!s) { g[0]++; g[2]--; changed = true; continue; }
      s = 0; for (let y = g[1]; y < g[1] + g[3]; y++) s += freeAt(g[0] + g[2] - 1, y);
      if (!s) { g[2]--; changed = true; continue; }
      s = 0; for (let x = g[0]; x < g[0] + g[2]; x++) s += freeAt(x, g[1]);
      if (!s) { g[1]++; g[3]--; changed = true; continue; }
      s = 0; for (let x = g[0]; x < g[0] + g[2]; x++) s += freeAt(x, g[1] + g[3] - 1);
      if (!s) { g[3]--; changed = true; }
    }
  }

  function assignScanline(idx, g) { // fallback + leaf assignment
    const cells = [];
    for (let y = g[1]; y < g[1] + g[3] && cells.length < idx.length; y++)
      for (let x = g[0]; x < g[0] + g[2] && cells.length < idx.length; x++)
        if (freeAt(x, y)) cells.push([x, y]);
    idx.sort((a, b) => py[a] - py[b] || px[a] - px[b]);
    for (let i = 0; i < idx.length; i++) {
      const c = cells[i] || cells[cells.length - 1];
      out[2 * idx[i]] = c[0]; out[2 * idx[i] + 1] = c[1];
      free[c[1] * W + c[0]] = 0;
    }
  }

  const stack = [{ idx: Array.from({ length: n }, (_, i) => i), g: [0, 0, W, H] }];
  while (stack.length) {
    const q = stack.pop();
    cropGrid(q.g);
    if (q.idx.length === 1) { assignScanline(q.idx, q.g); continue; }
    const g = q.g, half = q.idx.length >> 1;

    // free-cell counts per column / row of the subgrid
    let countX = 0, splitCol = 0;
    for (let x = g[0]; x < g[0] + g[2] && countX < half; x++) {
      let c = 0; for (let y = g[1]; y < g[1] + g[3]; y++) c += freeAt(x, y);
      countX += c; splitCol = x - g[0] + 1;
    }
    let countY = 0, splitRow = 0;
    for (let y = g[1]; y < g[1] + g[3] && countY < half; y++) {
      let c = 0; for (let x = g[0]; x < g[0] + g[2]; x++) c += freeAt(x, y);
      countY += c; splitRow = y - g[1] + 1;
    }

    const mk = (idx, g) => ({ idx, g });
    let sX = null;
    if (countX > 0 && countX < q.idx.length && splitCol < g[2]) {
      const o = q.idx.slice().sort((a, b) => px[a] - px[b] || py[a] - py[b]);
      sX = [mk(o.slice(0, countX), [g[0], g[1], splitCol, g[3]]),
            mk(o.slice(countX), [g[0] + splitCol, g[1], g[2] - splitCol, g[3]])];
    }
    let sY = null;
    if (countY > 0 && countY < q.idx.length && splitRow < g[3]) {
      const o = q.idx.slice().sort((a, b) => py[a] - py[b] || px[a] - px[b]);
      sY = [mk(o.slice(0, countY), [g[0], g[1], g[2], splitRow]),
            mk(o.slice(countY), [g[0], g[1] + splitRow, g[2], g[3] - splitRow])];
    }
    if (!sX && !sY) { assignScanline(q.idx, q.g); continue; }

    let pick;
    if (sX && sY) {
      const dev = s => Math.max(...s.map(o => {
        const w = Math.max(o.g[2], 1), h = Math.max(o.g[3], 1);
        return Math.abs(1 - Math.min(w, h) / Math.max(w, h));
      }));
      pick = dev(sX) <= dev(sY) ? sX : sY;
    } else pick = sX || sY;
    stack.push(pick[0], pick[1]);
  }

  // pack flat Float64Array back into [x, y] pairs, input order
  const grid = new Array(n);
  for (let i = 0; i < n; i++) grid[i] = [out[2 * i], out[2 * i + 1]];
  return grid;
}

/* ------------------------------------------------------------------------- */
/* Target masks                                                              */
/* ------------------------------------------------------------------------- */

/**
 * Divisor pair of `n` closest to square, as [rows, cols] with rows <= cols.
 * @returns {[number,number]|null} null for n < 1.
 */
export function rectArrangement(n) {
  let best = null;
  for (let d = 1; d * d <= n; d++) if (n % d === 0) best = [d, n / d];
  return best;
}

/**
 * A w*h rectangle keeping the first `keep` cells (extras dropped from the end).
 * @returns {{w:number,h:number,free:Uint8Array,hex:boolean}}
 */
export function fullMask(w, h, keep) {
  const free = new Uint8Array(w * h).fill(1);
  let drop = w * h - keep;
  for (let i = w * h - 1; i >= 0 && drop > 0; i--) { free[i] = 0; drop--; }
  return { w, h, free, hex: false };
}

/**
 * Rectangular target for `n` points.
 * @param {number} n
 * @param {'auto'|'square'} [mode]  'auto' prefers a near-square exact divisor
 *        rectangle; falls back to a padded ceil(sqrt) rectangle when none is
 *        close enough to square (e.g. prime n). 'square' always pads.
 * @returns {{w:number,h:number,free:Uint8Array,hex:boolean}}
 */
export function makeRect(n, mode = 'auto') {
  if (mode === 'auto') {
    const r = rectArrangement(n);
    if (r && r[0] / r[1] >= 0.4)
      return { w: r[1], h: r[0], free: new Uint8Array(r[0] * r[1]).fill(1), hex: false };
  }
  const w = Math.ceil(Math.sqrt(n)), h = Math.ceil(n / w);
  return fullMask(w, h, n);
}

/**
 * Circular (or hexagonally-packed) target holding exactly `n` cells.
 * @param {number} n
 * @param {boolean} [hex]  Offset alternate rows for hex packing.
 * @returns {{w:number,h:number,free:Uint8Array,hex:boolean}}
 */
export function makeCircle(n, hex = false) {
  const f = hex ? Math.sqrt(3) / 2 : 1;
  let r = Math.sqrt(n / Math.PI) / Math.sqrt(f);
  for (; ; r += 0.15) {
    const w = Math.ceil(2 * r) + 2, h = Math.ceil(2 * r / f) + 2, cx = w / 2 - 0.5, cy = (h / 2 - 0.5);
    const cells = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const hx = hex ? x + (y % 2 ? 0.5 : 0) : x, hy = y * f;
      const d = (hx - cx) ** 2 + (hy - cy * f) ** 2;
      if (d <= r * r) cells.push([d, y * w + x]);
    }
    if (cells.length >= n) {
      cells.sort((a, b) => a[0] - b[0]);
      const free = new Uint8Array(w * h);
      for (let i = 0; i < n; i++) free[cells[i][1]] = 1;
      return { w, h, free, hex };
    }
  }
}

/**
 * Count the usable cells of a mask (sanity helper).
 * @param {{free:Uint8Array}} mask
 * @returns {number}
 */
export function countFree(mask) {
  let s = 0; for (let i = 0; i < mask.free.length; i++) s += mask.free[i];
  return s;
}
