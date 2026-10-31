// Depth sort for the splat renderer, kept off the main thread.
//
// Splats must be composited back-to-front, so every camera move invalidates the
// ordering.  A comparison sort of 200k+ elements costs tens of milliseconds and
// would stutter the drag; an LSD radix sort over 32-bit keys is linear and runs
// in a few milliseconds, and the worker keeps even that off the render loop.
//
// Float keys are converted to a monotonic uint32 ordering first (flip the sign
// bit for positives, invert everything else), so a plain unsigned radix sort
// yields the correct float order.

function toMonotonicKeys(depths) {
  const keys = new Uint32Array(depths.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < depths.length; i++) {
    f32[0] = depths[i];
    const bits = u32[0];
    keys[i] = (bits & 0x80000000) ? (~bits >>> 0) : ((bits | 0x80000000) >>> 0);
  }
  return keys;
}

function radixSort(depths) {
  const n = depths.length;
  const keys = toMonotonicKeys(depths);

  let order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  let tmp = new Uint32Array(n);
  const count = new Uint32Array(256);

  for (let shift = 0; shift < 32; shift += 8) {
    count.fill(0);
    for (let i = 0; i < n; i++) count[(keys[order[i]] >>> shift) & 255]++;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
      const c = count[i];
      count[i] = sum;
      sum += c;
    }
    for (let i = 0; i < n; i++) {
      const idx = order[i];
      tmp[count[(keys[idx] >>> shift) & 255]++] = idx;
    }
    const swap = order;
    order = tmp;
    tmp = swap;
  }
  return order;
}

self.onmessage = function (event) {
  const { depths, generation } = event.data;
  const order = radixSort(depths);
  self.postMessage({ order, generation }, [order.buffer]);
};
