/*
 * Validity tests for the RasterFairy JS port.
 * Run with:  node rasterfairy_js/test.mjs
 *
 * These do NOT check bit-parity with the Python reference (the leaf assignment
 * differs). They check the guarantees that matter: every point is placed on a
 * distinct free cell of the mask (a bijection), for a range of sizes, target
 * shapes, and awkward input scales.
 */
import { transformPointCloud2D, makeRect, makeCircle, countFree } from './rasterfairy.js';

let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

function cloud(n, scale) {
  const p = [];
  for (let i = 0; i < n; i++) p.push([rnd() * scale, rnd() * scale]);
  return p;
}

function assert(cond, msg) { if (!cond) { console.error('FAIL:', msg); process.exitCode = 1; } }

function checkBijection(points, mask, label) {
  const grid = transformPointCloud2D(points, mask);
  assert(grid.length === points.length, `${label}: length`);
  const seen = new Set();
  let inMask = true, integral = true;
  for (const [x, y] of grid) {
    if (!Number.isInteger(x) || !Number.isInteger(y)) integral = false;
    const idx = y * mask.w + x;
    if (x < 0 || y < 0 || x >= mask.w || y >= mask.h || mask.free[idx] !== 1) inMask = false;
    seen.add(idx);
  }
  assert(integral, `${label}: grid coords are integers`);
  assert(inMask, `${label}: every point lands on a free mask cell`);
  assert(seen.size === points.length, `${label}: bijection (${seen.size}/${points.length} unique cells)`);
  console.log(`  ok  ${label.padEnd(34)} n=${points.length} grid=${mask.w}x${mask.h} free=${countFree(mask)}`);
}

console.log('RasterFairy JS — validity tests\n');

// rectangles across sizes; include the [0,1] small-range case that used to collapse
for (const n of [16, 63, 256, 1000, 1024, 2048]) {
  checkBijection(cloud(n, 1), makeRect(n, 'auto'), `rect auto, scale [0,1]`);
  checkBijection(cloud(n, 1000), makeRect(n, 'square'), `rect square, scale [0,1000]`);
}

// circular and hex targets
for (const n of [200, 777, 1500]) {
  checkBijection(cloud(n, 1), makeCircle(n, false), `circle`);
  checkBijection(cloud(n, 1), makeCircle(n, true), `hex circle`);
}

// default mask (mask omitted -> makeRect auto)
{
  const pts = cloud(512, 1);
  const grid = transformPointCloud2D(pts);
  assert(grid.length === 512, 'default-mask: length');
  const seen = new Set(grid.map(([x, y]) => x + ',' + y));
  assert(seen.size === 512, 'default-mask: bijection');
  console.log(`  ok  ${'default mask (omitted)'.padEnd(34)} n=512`);
}

console.log(process.exitCode ? '\nSOME TESTS FAILED' : '\nAll tests passed.');
