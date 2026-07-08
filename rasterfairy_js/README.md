# RasterFairy (JavaScript)

A dependency-free JavaScript port of the RasterFairy core: it maps a 2D point
cloud onto a regular grid while preserving the neighbourhood relations of the
cloud. Runs in the browser and in Node.

> **This is an independent port, not the Python package.** It uses the same
> recursive-slicing idea and is guaranteed to produce a **valid bijection**
> (every point lands on its own grid cell) that preserves neighbourhoods well.
> It is **not bit-identical** to the Python reference: the Python code solves
> small leaf slices with an exact Hungarian assignment, whereas this port uses a
> scanline assignment. For the canonical implementation and for benchmarks, use
> the [Python package](../README.md).

## Files

| File | Purpose |
|------|---------|
| `rasterfairy.js` | Core module (ESM). Transform + rectangle/circle mask builders. No DOM. |
| `rasterfairy-textmask.js` | Optional `makeTextMask` (browser only — uses `<canvas>`). |
| `example.html` | Minimal browser example (serve over http, see below). |
| `test.mjs` | Validity tests (`node rasterfairy_js/test.mjs`). |

## Quick start — browser (via CDN, no build step)

jsDelivr serves the files straight from this repository, so no npm install is
needed:

```html
<script type="module">
  import {
    transformPointCloud2D, makeCircle
  } from 'https://cdn.jsdelivr.net/gh/Quasimondo/RasterFairy@master/rasterfairy_js/rasterfairy.js';

  const points = [/* [x, y], [x, y], ... your t-SNE / UMAP / any 2D layout */];
  const mask   = makeCircle(points.length);        // target shape
  const grid   = transformPointCloud2D(points, mask);
  // grid[i] === [gridX, gridY] for points[i]
</script>
```

Pin `@master` to a specific tag or commit (e.g. `@v1.2.0`) for a stable,
cache-friendly URL once a release is tagged.

## Quick start — Node (ES module)

Copy `rasterfairy.js` into your project (or import it from the CDN URL above)
and import it from an ES module (`.mjs`, or a `package.json` with
`"type": "module"`):

```js
import { transformPointCloud2D, makeRect } from './rasterfairy.js';

const points = [[0.1, 0.9], [0.4, 0.2], /* ... */];
const grid = transformPointCloud2D(points, makeRect(points.length));
console.log(grid); // [[x, y], ...] aligned to a grid, same order as input
```

## Running the example locally

ES-module imports do **not** work from `file://` — serve the folder over http:

```bash
cd /path/to/RasterFairy
python3 -m http.server
# then open http://localhost:8000/rasterfairy_js/example.html
```

## API

### `transformPointCloud2D(points, mask?) → Array<[x, y]>`

Maps `points` onto the free cells of `mask`.

- `points` — `Array<[number, number]>`, the cloud. Any coordinate scale works;
  input is normalised internally.
- `mask` — `{ w, h, free }` (see below). The number of free cells **must equal**
  `points.length`. Defaults to `makeRect(points.length, 'auto')` if omitted.
- **Returns** one `[gridX, gridY]` per input point, in the same order as
  `points`. Coordinates are integers in `[0, w) × [0, h)`.

### Mask format

A mask is a plain object:

```js
{
  w: Number,           // grid width  (columns)
  h: Number,           // grid height (rows)
  free: Uint8Array,    // length w*h, row-major; 1 = usable cell, 0 = blocked
  hex: Boolean         // informational: whether rows are hex-offset for display
}
```

`free` lets you target arbitrary shapes — anything with exactly `points.length`
ones. The builders below produce ready-made masks.

### Mask builders (in `rasterfairy.js`)

| Function | Returns |
|----------|---------|
| `makeRect(n, mode?)` | Rectangle for `n` points. `mode='auto'` (default) prefers a near-square exact divisor rectangle, falling back to a padded `ceil(sqrt)` rectangle (e.g. for prime `n`); `mode='square'` always pads. |
| `makeCircle(n, hex?)` | A disc (or hex-packed disc) holding exactly `n` cells. |
| `rectArrangement(n)` | `[rows, cols]` divisor pair closest to square (`rows ≤ cols`), or `null`. |
| `countFree(mask)` | Number of usable cells in a mask (sanity check). |

### Text mask (in `rasterfairy-textmask.js`, browser only)

```js
import { makeTextMask } from './rasterfairy-textmask.js';
const mask = makeTextMask(points.length, 'RF');   // grid shaped like the text
```

## Tests

```bash
node rasterfairy_js/test.mjs
```

Checks that every point is placed on a distinct free mask cell (a bijection)
across a range of sizes, target shapes, and input scales — including the small
`[0, 1]` range that older versions mishandled. These are validity tests, not a
bit-parity comparison against the Python implementation.

## Citation

The algorithm is *Raster Fairy by Mario Klingemann* —
<https://github.com/Quasimondo/RasterFairy>. If you port or reuse it, please
keep the attribution (see the porting note in the main README).
