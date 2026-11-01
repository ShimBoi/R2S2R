#!/bin/bash
# Isaac Sim 6.2's DLSS Ray Reconstruction (310.7.128) can't be created on driver 610. The 310.1.0
# build from Isaac Sim 5.1 works; this pulls just that file (~34 MB) out of the 5.1 aarch64 wheel.
# enter_container.sh mounts it over the bundled one on 610 nodes.
set -euo pipefail

W=${R2S2R_WORKSPACE:-$SCRATCH/r2s2r}
OUT=$W/.cache/dlssd/libnvidia-ngx-dlssd.so.310.1.0
URL=https://pypi.nvidia.com/isaacsim-extscache-kit/isaacsim_extscache_kit-5.1.0.0-cp311-none-manylinux_2_35_aarch64.whl

[ -f "$OUT" ] && { echo "[fetch_dlssd] already present: $OUT"; exit 0; }
mkdir -p "$(dirname "$OUT")"
python3 - "$URL" "$OUT" <<'PY'
import io, sys, urllib.request, zipfile

class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size or not len(buf):
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)

url, out = sys.argv[1:]
wheel = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 20))
member = next(m for m in wheel.namelist() if m.endswith("/libnvidia-ngx-dlssd.so.310.1.0"))
with open(out, "wb") as f:
    f.write(wheel.read(member))
PY
echo "[fetch_dlssd] done: $OUT"
