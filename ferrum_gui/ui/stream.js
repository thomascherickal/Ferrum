// Pure helpers for streamed generation, shared by the browser UI (app.js) and
// the Node test (stream.test.js). No DOM / Tauri dependencies so it is unit
// testable. Loaded as a plain <script> in the browser and `require`d in Node.
"use strict";
(function (global) {
  // Reconcile the streamed output against the authoritative full string the
  // backend returns (G1). `generate_slm` returns `seed + continuation`, while
  // the UI accumulates only the streamed continuation fragments. The final
  // `gen-fragment` IPC event can arrive after the command promise resolves (or
  // be dropped), so on completion we recompute the continuation directly from
  // the returned string instead of trusting the stream tail.
  function streamContinuation(full, seed) {
    full = full || "";
    seed = seed || "";
    return full.startsWith(seed) ? full.slice(seed.length) : full;
  }

  const api = { streamContinuation };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.streamContinuation = streamContinuation;
  }
})(typeof window !== "undefined" ? window : globalThis);
