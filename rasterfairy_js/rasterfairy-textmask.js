/*
 * RasterFairy — text mask (browser only)
 * ---------------------------------------------------------------------------
 * Builds a target mask shaped like a piece of text, for use as the `mask`
 * argument of transformPointCloud2D. Requires a DOM (uses <canvas>), so it is
 * kept out of the core module — import it only in the browser.
 */

/**
 * Build a mask holding exactly `n` cells in the shape of `text`.
 * @param {number} n
 * @param {string} [text]
 * @param {string} [fontFamily]  CSS font stack for the glyphs.
 * @returns {{w:number,h:number,free:Uint8Array,hex:boolean}}
 */
export function makeTextMask(n, text = 'RF', fontFamily = 'sans-serif') {
  text = String(text).trim() || 'RF';
  for (let gw = Math.ceil(Math.sqrt(n)) * 2; ; gw = Math.round(gw * 1.25)) {
    const c = document.createElement('canvas');
    const meas = c.getContext('2d');
    meas.font = '900 100px ' + fontFamily;
    const m = meas.measureText(text);
    const ar = (m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) / m.width;
    const gh = Math.max(5, Math.round(gw * ar));
    c.width = gw; c.height = gh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    let fs = gh * 1.2;
    do { ctx.font = `900 ${fs}px ${fontFamily}`; fs *= 0.94; }
    while (ctx.measureText(text).width > gw * 0.96 && fs > 4);
    ctx.fillText(text, gw / 2, gh / 2);
    const a = ctx.getImageData(0, 0, gw, gh).data;
    const cells = [];
    for (let i = 0; i < gw * gh; i++) if (a[4 * i + 3] > 96) cells.push(i);
    if (cells.length >= n) {
      const free = new Uint8Array(gw * gh);
      cells.forEach(i => free[i] = 1);
      // trim excess from thinnest spots: fewest filled neighbours first
      let excess = cells.length - n;
      while (excess > 0) {
        let worst = -1, worstN = 9;
        for (const i of cells) {
          if (!free[i]) continue;
          const x = i % gw, y = (i / gw) | 0; let nn = 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx >= 0 && yy >= 0 && xx < gw && yy < gh && free[yy * gw + xx]) nn++;
          }
          if (nn < worstN) { worstN = nn; worst = i; if (nn <= 3) break; }
        }
        free[worst] = 0; excess--;
      }
      return { w: gw, h: gh, free, hex: false };
    }
    if (gw > 600) throw new Error('makeTextMask: could not fit ' + n + ' cells into text "' + text + '"');
  }
}
