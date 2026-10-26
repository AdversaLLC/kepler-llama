#!/bin/bash
# Run a command in a bubblewrap sandbox: only ./work (rw), the Kepler toolchain (ro),
# system /usr (ro) and the NVIDIA device nodes are visible. No home, no network unless NET=1.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tool="$HOME/AdversaVault/code/src/cuda118"
net=(--unshare-net)
[ "${NET:-0}" = "1" ] && net=(--share-net)
devs=()
for d in /dev/nvidia*; do [ -e "$d" ] && devs+=(--dev-bind "$d" "$d"); done
exec bwrap --unshare-pid --unshare-ipc --unshare-uts "${net[@]}" --die-with-parent \
  --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib /lib64 \
  --ro-bind /etc/ssl /etc/ssl --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --proc /proc --dev /dev --tmpfs /tmp "${devs[@]}" \
  --ro-bind "$tool" /opt/cuda118 --bind "$here/work" /work \
  --setenv HOME /work --setenv PATH /usr/bin --chdir /work "$@"
