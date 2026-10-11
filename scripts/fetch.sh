#!/usr/bin/env bash
# Download <url> to <out> and verify its sha256; fails (and leaves nothing
# behind) on a mismatch. Usage: scripts/fetch.sh <url> <sha256> <out>
set -euo pipefail
url=${1:?url}; want=${2:?sha256}; out=${3:?out}
mkdir -p "$(dirname "$out")"
tmp="$out.part"
trap 'rm -f "$tmp"' EXIT
curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"
got=$(sha256sum "$tmp" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then
  echo "sha256 mismatch for $url: got $got, want $want" >&2
  exit 1
fi
mv "$tmp" "$out"
