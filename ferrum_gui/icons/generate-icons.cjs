// Generates placeholder PNG icons (no external deps) for the Ferrum GUI.
// Replace with real artwork via `cargo tauri icon <source.png>` for release.
const fs = require('fs');
const zlib = require('zlib');
const path = require('path');

const CRC = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c;
  }
  return (buf) => {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) c = t[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  };
})();

function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const t = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4); crc.writeUInt32BE(CRC(Buffer.concat([t, data])), 0);
  return Buffer.concat([len, t, data, crc]);
}

function png(size) {
  const w = size, h = size;
  const px = Buffer.alloc(w * h * 4);
  const set = (x, y, r, g, b, a) => {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    const o = (y * w + x) * 4; px[o]=r; px[o+1]=g; px[o+2]=b; px[o+3]=a;
  };
  // Background slate.
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) set(x, y, 0x1e, 0x29, 0x3b, 0xff);
  // Rust-orange rounded-ish square.
  const m = Math.round(size * 0.16);
  for (let y = m; y < h - m; y++) for (let x = m; x < w - m; x++) set(x, y, 0xea, 0x58, 0x0c, 0xff);
  // White "F".
  const fx = Math.round(size*0.34), fy = Math.round(size*0.30);
  const bw = Math.round(size*0.085), bh = Math.round(size*0.40);
  const armW = Math.round(size*0.26);
  for (let y = fy; y < fy+bh; y++) for (let x = fx; x < fx+bw; x++) set(x,y,255,255,255,255); // stem
  for (let x = fx; x < fx+armW; x++) for (let y = fy; y < fy+bw; y++) set(x,y,255,255,255,255); // top arm
  const my = fy + Math.round(bh*0.42);
  for (let x = fx; x < fx+Math.round(armW*0.8); x++) for (let y = my; y < my+bw; y++) set(x,y,255,255,255,255); // mid arm

  // Build raw scanlines with filter byte 0.
  const raw = Buffer.alloc(h * (w * 4 + 1));
  for (let y = 0; y < h; y++) {
    raw[y * (w*4+1)] = 0;
    px.copy(raw, y*(w*4+1)+1, y*w*4, y*w*4 + w*4);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8]=8; ihdr[9]=6; ihdr[10]=0; ihdr[11]=0; ihdr[12]=0;
  const sig = Buffer.from([0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]);
  return Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

const dir = __dirname;
const out = { '32x32.png':32, '128x128.png':128, '128x128@2x.png':256, 'icon.png':512, 'Square150x150Logo.png':150, 'StoreLogo.png':50 };
for (const [name, size] of Object.entries(out)) {
  fs.writeFileSync(path.join(dir, name), png(size));
  console.log('wrote', name, size + 'x' + size);
}
