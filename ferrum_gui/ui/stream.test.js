// Node test for the streamed-generation reconciliation (G1). Run with:
//   node ui/stream.test.js
"use strict";
const assert = require("node:assert");
const { streamContinuation } = require("./stream.js");

// The backend returns `seed + continuation`; the UI shows only the continuation.
assert.strictEqual(streamContinuation("the quick brown", "the q"), "uick brown");

// Recovers the full continuation even if the streamed tail was dropped: the
// return value is authoritative regardless of what fragments arrived.
assert.strictEqual(streamContinuation("hello world!", "hello"), " world!");

// Empty seed → the whole string is continuation.
assert.strictEqual(streamContinuation("abc", ""), "abc");

// Defensive: if the return value does not start with the seed, show it verbatim
// rather than slicing off characters that are not the seed.
assert.strictEqual(streamContinuation("xyz", "seed"), "xyz");

// Null/undefined inputs degrade to empty strings.
assert.strictEqual(streamContinuation(undefined, undefined), "");
assert.strictEqual(streamContinuation("abc", undefined), "abc");

// Multi-byte seed prefix slices correctly (prefix match, not char counting).
assert.strictEqual(streamContinuation("café latte", "café "), "latte");

console.log("stream.test.js: all assertions passed");
