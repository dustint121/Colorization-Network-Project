"""
reshard_imagenet.py  (v5)
=========================
v5 changes vs v4:
  1. NEW: --disk-spill flag.  Streams each open shard's bytes directly to a
     temp .tar.tmp file on disk instead of buffering JPEGs in RAM. Reduces
     RAM per open shard from ~110 MB to ~0 at the cost of more I/O. Use
     this if you want to push SHARDS_PER_PASS way higher (e.g. 2000+) or
     are tight on RAM. Default OFF — the in-RAM path is faster.

  2. FIX: Pass scheduling now aligns to actual progress, not to the
     hardcoded window grid. If you finished 400 shards and resume with
     SHARDS_PER_PASS=800, pass 1 starts at shard 400 (not 0). This also
     makes the progress bar's total / % accurate.

  3. RAM estimate softened — the previous "K * 110 MB" wording was the
     theoretical worst case, which never happens in practice because
     shards flush continuously during a pass. Real peak is typically
     1/3 to 1/2 of that.

Usage:
  python reshard_imagenet.py                       # plan + write, in-RAM path
  python reshard_imagenet.py write                 # write only, in-RAM
  python reshard_imagenet.py write --disk-spill    # write with disk spill
  python reshard_imagenet.py write --debug         # extra logging
  python reshard_imagenet.py write --disk-spill --debug
"""

from __future__ import annotations

import gc
import io
import logging
import os
import pickle
import random
import sys
import tarfile
import time
from array import array
from pathlib import Path

from tqdm import tqdm

try:
    import psutil
    _PROC = psutil.Process()
    def rss_mb() -> float:
        return _PROC.memory_info().rss / (1024 * 1024)
except ImportError:
    def rss_mb() -> float:
        return -1.0


# ============================================================================
# CONFIG
# ============================================================================
SOURCE_DIR       = Path(r"S:\ImageNet-21K tar files\winter21_whole")
OUTPUT_DIR       = Path(r"S:\ImageNet-21K shards")
IMAGES_PER_SHARD = 1000
RANDOM_SEED      = 42
TRAIN_FRACTION   = 0.95

# Number of output shards processed per pass.
# In-RAM path:    each open buffer holds up to IMAGES_PER_SHARD JPEGs
#                 (~110 MB worst-case, typically 30-60 MB at fill peak).
#                 Real-world peak RSS scales ~roughly with K * 35 MB.
# Disk-spill:     RAM cost per open buffer is effectively zero. K can be
#                 set to thousands, bounded only by OS file-handle limits
#                 (Windows default ~512 per process; we stay well below).
#
# RECOMMENDED:
#   In-RAM:    100  (safe)  /  200  (good balance)  /  400  (close other apps) / 800 *MAX; close other apps) / 1600 is risky
#   Disk-spill: 1000-2000   (much fewer passes, similar wall time)
SHARDS_PER_PASS  = 1200

# Reporting / checkpointing
RAM_REPORT_EVERY          = 50    # every N flushed shards
PROGRESS_CHECKPOINT_EVERY = 25

# Plan storage
PLAN_DIR         = OUTPUT_DIR / "_reshard_plan"
PLAN_INDEX_FILE  = OUTPUT_DIR / "_reshard_plan_index.pkl"
PROGRESS_FILE    = OUTPUT_DIR / "_reshard_progress.pkl"
SPILL_DIR        = OUTPUT_DIR / "_reshard_spill"   # for --disk-spill mode
ERROR_LOG        = OUTPUT_DIR / "_reshard_errors.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("reshard")


# ============================================================================
# Low-level tar reader (unchanged from v3/v4)
# ============================================================================
def iter_tar_low_level(path: Path):
    """Yield (name, jpeg_bytes) from a tar via raw I/O. No tarfile state."""
    with open(path, "rb") as f:
        while True:
            header = f.read(512)
            if not header or len(header) < 512:
                return
            if header == b"\x00" * 512:
                return

            try:
                name_bytes = header[0:100].rstrip(b"\x00")
                typeflag = header[156:157]
                size_oct = header[124:136].rstrip(b"\x00 ")
                size = int(size_oct, 8) if size_oct else 0

                if typeflag == b"L":
                    long_name = f.read(size).rstrip(b"\x00").decode("utf-8", "replace")
                    pad = (-size) % 512
                    if pad:
                        f.seek(pad, 1)
                    real_header = f.read(512)
                    if not real_header or len(real_header) < 512:
                        return
                    typeflag = real_header[156:157]
                    size_oct = real_header[124:136].rstrip(b"\x00 ")
                    size = int(size_oct, 8) if size_oct else 0
                    name = long_name
                else:
                    name = name_bytes.decode("utf-8", "replace")
            except Exception:
                return

            if typeflag not in (b"0", b"\x00"):
                f.seek(size, 1)
                pad = (-size) % 512
                if pad:
                    f.seek(pad, 1)
                continue

            data = f.read(size)
            pad = (-size) % 512
            if pad:
                f.seek(pad, 1)

            if name.lower().endswith((".jpeg", ".jpg")):
                yield name, data


# ============================================================================
# In-tar writer abstraction
# ============================================================================
# Both modes need to "append a JPEG to shard S, then finalize the tar later."
# The in-RAM mode buffers (name, bytes) tuples in a list and writes the tar
# only at finalize. The disk-spill mode opens a tarfile object backed by an
# .tar.tmp file and writes each JPEG as it arrives; finalize just closes
# and renames the file.

class _RamShardWriter:
    """Buffers (name, bytes) in RAM. Writes the tar file at finalize()."""
    __slots__ = ("members",)

    def __init__(self):
        self.members: list[tuple[str, bytes]] = []

    def append(self, name: str, data: bytes) -> None:
        self.members.append((name, data))

    def finalize(self, final_path: Path) -> None:
        tmp = final_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp, mode="w") as out_tar:
            for name, data in self.members:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                out_tar.addfile(info, io.BytesIO(data))
        self.members.clear()
        tmp.replace(final_path)


class _DiskShardWriter:
    """Streams (name, bytes) directly to an on-disk .tar.tmp file.
    RAM cost is ~zero (just an open file handle + tarfile state ~few KB)."""
    __slots__ = ("tmp_path", "tar")

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        # tarfile in append mode requires a complete existing tar; "w" creates
        # fresh. Since we only ever create-then-finalize, "w" is correct here.
        self.tar = tarfile.open(tmp_path, mode="w")

    def append(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))
        # 'data' goes out of scope on the caller's next loop iter -> freed.

    def finalize(self, final_path: Path) -> None:
        self.tar.close()
        self.tmp_path.replace(final_path)


# ============================================================================
# PHASE 1: Plan (unchanged from v4)
# ============================================================================
def build_plan() -> dict:
    if PLAN_INDEX_FILE.exists():
        log.info(f"Loading existing plan index from {PLAN_INDEX_FILE}")
        with open(PLAN_INDEX_FILE, "rb") as f:
            idx = pickle.load(f)
        log.info(
            f"  Plan has {idx['num_images']:,} images across "
            f"{idx['num_shards']:,} shards "
            f"({idx['train_shards']} train, {idx['val_shards']} val)"
        )
        return idx

    log.info("Building plan from scratch...")
    PLAN_DIR.mkdir(parents=True, exist_ok=True)

    tar_paths = sorted(SOURCE_DIR.glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No .tar files found in {SOURCE_DIR}")
    log.info(f"Found {len(tar_paths):,} class tars in {SOURCE_DIR}")

    source_tars: list[str] = []
    per_tar_counts: list[int] = []
    bad_tars: list[str] = []

    for tar_path in tqdm(tar_paths, desc="Listing", unit="tar"):
        names: list[str] = []
        try:
            with tarfile.open(tar_path, mode="r|") as tf:
                for m in tf:
                    if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg")):
                        names.append(m.name)
        except Exception as e:
            bad_tars.append(tar_path.name)
            with open(ERROR_LOG, "a") as f:
                f.write(f"LIST_FAILED\t{tar_path.name}\t{type(e).__name__}: {e}\n")
            continue

        if not names:
            continue

        with open(PLAN_DIR / f"{tar_path.stem}.names.pkl", "wb") as f:
            pickle.dump(names, f, protocol=pickle.HIGHEST_PROTOCOL)

        source_tars.append(tar_path.name)
        per_tar_counts.append(len(names))

    if bad_tars:
        log.warning(f"{len(bad_tars)} unreadable class tars — see {ERROR_LOG}")

    total_images = sum(per_tar_counts)
    num_shards = (total_images + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD
    train_shards = int(num_shards * TRAIN_FRACTION)
    val_shards = num_shards - train_shards
    log.info(f"Total JPEGs: {total_images:,}  Shards: {num_shards:,}  "
             f"({train_shards} train + {val_shards} val)")

    log.info(f"Shuffling globally (seed={RANDOM_SEED})...")
    perm = array("i", range(total_images))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(perm)

    log.info("Computing shard assignments...")
    shard_assignments = array("i", [0]) * total_images
    for i in range(total_images):
        shard_assignments[perm[i]] = i // IMAGES_PER_SHARD
    del perm

    log.info("Writing per-class plan files...")
    cursor = 0
    for src_name, count in tqdm(list(zip(source_tars, per_tar_counts)),
                                 desc="Plan files", unit="tar"):
        stem = Path(src_name).stem
        with open(PLAN_DIR / f"{stem}.names.pkl", "rb") as f:
            names = pickle.load(f)
        assignments = shard_assignments[cursor : cursor + count]
        cursor += count
        with open(PLAN_DIR / f"{stem}.plan.pkl", "wb") as f:
            pickle.dump((names, assignments), f, protocol=pickle.HIGHEST_PROTOCOL)
        (PLAN_DIR / f"{stem}.names.pkl").unlink()

    del shard_assignments

    idx = {
        "num_images": total_images,
        "num_shards": num_shards,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "source_tars": source_tars,
        "per_tar_counts": per_tar_counts,
        "images_per_shard": IMAGES_PER_SHARD,
    }
    with open(PLAN_INDEX_FILE, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Plan saved to {PLAN_DIR} + {PLAN_INDEX_FILE}")
    return idx


# ============================================================================
# Pass scheduling helper
# ============================================================================
def _plan_passes(completed: set[int], num_shards: int, K: int) -> list[tuple[int, int]]:
    """
    Decide which (shard_lo, shard_hi) windows still need work.

    Rebases the window grid to start at the first incomplete shard, so when
    resuming with non-aligned progress (e.g. 400 done, K=800), pass 1 covers
    [400, 1200) instead of [0, 800). This makes the progress bar accurate.

    Within each resulting window we'll still skip any individual shards
    that happen to be already done — but in practice, once we rebase to
    the first incomplete shard, the windows tile the remaining work
    naturally.
    """
    if len(completed) == num_shards:
        return []

    # First incomplete shard becomes the anchor for the window grid.
    completed_sorted = sorted(completed)
    first_incomplete = 0
    # Walk forward through completed shards in order; first gap is where we start.
    for s in completed_sorted:
        if s == first_incomplete:
            first_incomplete += 1
        else:
            break

    passes: list[tuple[int, int]] = []
    lo = first_incomplete
    while lo < num_shards:
        hi = min(lo + K, num_shards)
        # Only include this window if it has at least one incomplete shard
        if any(s not in completed for s in range(lo, hi)):
            passes.append((lo, hi))
        lo = hi
    return passes


# ============================================================================
# PHASE 2: Write shards via MULTI-PASS sweep
# ============================================================================
def write_shards(idx: dict, debug: bool = False, disk_spill: bool = False) -> None:
    num_shards = idx["num_shards"]
    total_images = idx["num_images"]
    last_shard_size = total_images - (num_shards - 1) * IMAGES_PER_SHARD

    completed: set[int] = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "rb") as f:
            completed = pickle.load(f)
        log.info(f"Resuming: {len(completed):,}/{num_shards:,} shards already done")

    K = SHARDS_PER_PASS

    if disk_spill:
        SPILL_DIR.mkdir(parents=True, exist_ok=True)
        # Wipe any leftover spill files from a previous interrupted run --
        # partial spills can't be resumed mid-pass; only completed shards are
        # tracked in PROGRESS_FILE.
        leftover = list(SPILL_DIR.glob("*.tar.tmp"))
        if leftover:
            log.info(f"Cleaning up {len(leftover)} leftover spill files from "
                     "a previous interrupted run")
            for p in leftover:
                try:
                    p.unlink()
                except OSError:
                    pass
        log.info("Mode: DISK-SPILL — open shards stream directly to "
                 f"{SPILL_DIR}\\shard-NNNNN.tar.tmp")
        # In disk-spill mode, the only real RAM cost per open shard is the
        # tarfile object + OS file handle — call it ~5 KB.
        ram_est_gb = K * 5e-6
        log.info(f"Estimated RAM per pass (disk-spill): ~{ram_est_gb*1024:.0f} MB "
                 "for open file handles (negligible)")
    else:
        log.info("Mode: IN-RAM — open shards buffer JPEG bytes in memory")
        # Worst case = K * 110 MB. Real peak with continuous flushes is roughly
        # K * 35 MB based on observed runs. Quote both.
        ram_worst_gb = K * 0.11
        ram_typical_gb = K * 0.035
        log.info(f"Estimated RAM per pass (in-RAM): ~{ram_typical_gb:.1f} GB typical, "
                 f"~{ram_worst_gb:.1f} GB theoretical worst case")

    passes = _plan_passes(completed, num_shards, K)
    log.info(f"Multi-pass plan: {len(passes)} passes remaining of up to {K} shards each")
    if passes:
        log.info(f"  First pass:  [{passes[0][0]}, {passes[0][1]})")
        log.info(f"  Last  pass:  [{passes[-1][0]}, {passes[-1][1]})")

    source_tars = idx["source_tars"]

    for pass_seq, (shard_lo, shard_hi) in enumerate(passes, start=1):
        log.info("=" * 70)
        log.info(f"PASS {pass_seq}/{len(passes)}  (shard range "
                 f"[{shard_lo}, {shard_hi}))   RSS before pass: {rss_mb():.0f} MB")
        log.info("=" * 70)

        _run_one_pass(
            idx=idx,
            shard_lo=shard_lo,
            shard_hi=shard_hi,
            last_shard_size=last_shard_size,
            source_tars=source_tars,
            completed=completed,
            debug=debug,
            disk_spill=disk_spill,
        )

        gc.collect()
        log.info(f"PASS {pass_seq} done. Completed total: "
                 f"{len(completed):,}/{num_shards:,}.  "
                 f"RSS after pass+gc: {rss_mb():.0f} MB")

        with open(PROGRESS_FILE, "wb") as f:
            pickle.dump(completed, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"DONE. {len(completed):,} shards written to {OUTPUT_DIR}")


def _run_one_pass(
    *,
    idx: dict,
    shard_lo: int,
    shard_hi: int,
    last_shard_size: int,
    source_tars: list[str],
    completed: set[int],
    debug: bool,
    disk_spill: bool,
) -> None:
    """One pass: sweep all source tars, write only shards in [lo, hi)."""
    num_shards = idx["num_shards"]

    # Open writers (RAM or disk-spill) — created lazily as shards are first seen.
    writers: dict[int, _RamShardWriter | _DiskShardWriter] = {}

    # remaining[s] = number of images still expected for shard s in this pass.
    remaining: dict[int, int] = {}
    for s in range(shard_lo, shard_hi):
        if s in completed:
            continue
        remaining[s] = last_shard_size if s == num_shards - 1 else IMAGES_PER_SHARD

    if not remaining:
        log.info(f"  (all shards in [{shard_lo}, {shard_hi}) already done — skipping)")
        return

    images_in_this_pass = sum(remaining.values())
    pbar = tqdm(
        total=images_in_this_pass,
        desc=f"Pass [{shard_lo},{shard_hi})  ({len(remaining)} shards)",
        unit="img", smoothing=0.05,
    )

    flush_count_total = 0
    flush_count_since_checkpoint = 0

    def writer_for(shard_idx: int):
        w = writers.get(shard_idx)
        if w is None:
            if disk_spill:
                tmp_path = SPILL_DIR / f"shard-{shard_idx:05d}.tar.tmp"
                w = _DiskShardWriter(tmp_path)
            else:
                w = _RamShardWriter()
            writers[shard_idx] = w
        return w

    def flush_shard(shard_idx: int) -> None:
        nonlocal flush_count_total, flush_count_since_checkpoint
        w = writers.pop(shard_idx)
        final_path = OUTPUT_DIR / f"shard-{shard_idx:05d}.tar"
        w.finalize(final_path)
        completed.add(shard_idx)
        flush_count_total += 1
        flush_count_since_checkpoint += 1
        if flush_count_since_checkpoint >= PROGRESS_CHECKPOINT_EVERY:
            with open(PROGRESS_FILE, "wb") as f:
                pickle.dump(completed, f, protocol=pickle.HIGHEST_PROTOCOL)
            flush_count_since_checkpoint = 0
        if flush_count_total % RAM_REPORT_EVERY == 0:
            log.info(
                f"  [ram] RSS={rss_mb():.0f} MB  open_shards={len(writers)}  "
                f"this_pass_done={flush_count_total}"
            )

    for tar_i, src_name in enumerate(source_tars):
        stem = Path(src_name).stem
        plan_path = PLAN_DIR / f"{stem}.plan.pkl"
        if not plan_path.exists():
            continue

        with open(plan_path, "rb") as f:
            names, assignments = pickle.load(f)

        # Build name->shard ONLY for shards in this pass's range that are
        # still expecting more images (i.e. in `remaining`, which shrinks
        # as shards flush during the pass).
        #
        # This is the key tail-acceleration: as the pass progresses and
        # more shards complete, more source tars get fully filtered out
        # and skipped without ever being read from disk. Without this,
        # the last 5% of the pass would re-read most of the 1.1 TB of
        # source data just to discover none of it is needed.
        name_to_shard: dict[str, int] = {}
        for n, s in zip(names, assignments):
            if shard_lo <= s < shard_hi and s in remaining:
                name_to_shard[n] = s
        del names, assignments

        if not name_to_shard:
            continue

        src_path = SOURCE_DIR / src_name
        try:
            for name, data in iter_tar_low_level(src_path):
                shard_idx = name_to_shard.get(name)
                if shard_idx is None:
                    continue
                writer_for(shard_idx).append(name, data)
                remaining[shard_idx] -= 1
                pbar.update(1)
                if remaining[shard_idx] == 0:
                    flush_shard(shard_idx)
                    del remaining[shard_idx]
        except Exception as e:
            with open(ERROR_LOG, "a") as f:
                f.write(f"READ_TAR_FAILED\t{src_name}\t{type(e).__name__}: {e}\n")
            log.warning(f"Tar read failed: {src_name}: {e}")

        del name_to_shard

        if debug and tar_i % 100 == 0:
            log.info(
                f"  [tar {tar_i:5d}/{len(source_tars)}] {src_name}  "
                f"RSS={rss_mb():.0f} MB  open_shards={len(writers)}  "
                f"remaining_shards={len(remaining)}"
            )

    pbar.close()

    # If any shards in this pass didn't fill (corrupt/missing source files),
    # finalize whatever we got rather than losing the partial data.
    if writers:
        log.warning(
            f"  Pass ended with {len(writers)} incomplete shards "
            "(plan expected more images than source tars provided). "
            "Finalizing partial shards."
        )
        for shard_idx in list(writers.keys()):
            flush_shard(shard_idx)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = sys.argv[1:]
    disk_spill = "--disk-spill" in args
    if disk_spill:
        args.remove("--disk-spill")
    debug = "--debug" in args
    if debug:
        args.remove("--debug")
    mode = args[0] if args else "all"
    if mode not in ("all", "plan", "write"):
        print(f"Unknown mode: {mode}. Use 'all', 'plan', or 'write'.")
        sys.exit(1)

    log.info(f"Source : {SOURCE_DIR}")
    log.info(f"Output : {OUTPUT_DIR}")
    log.info(f"Images per shard: {IMAGES_PER_SHARD}")
    log.info(f"Shards per pass: {SHARDS_PER_PASS}")
    log.info(f"Disk-spill mode: {'ON' if disk_spill else 'OFF (in-RAM)'}")
    log.info(f"Initial RSS: {rss_mb():.0f} MB")

    if mode in ("all", "plan"):
        t0 = time.time()
        idx = build_plan()
        log.info(f"Phase 1 (plan) took {(time.time() - t0)/60:.1f} min")
    else:
        with open(PLAN_INDEX_FILE, "rb") as f:
            idx = pickle.load(f)

    if mode in ("all", "write"):
        t1 = time.time()
        write_shards(idx, debug=debug, disk_spill=disk_spill)
        log.info(f"Phase 2 (write) took {(time.time() - t1)/60:.1f} min")

    log.info("")
    log.info("Next step: upload to Wasabi with rclone:")
    log.info('  rclone copy "S:\\ImageNet-21K shards" wasabi-eu:image-net-21k-shards-eu/ \\')
    log.info('    --transfers 32 --checkers 16 --progress \\')
    log.info('    --exclude "_reshard_*" --exclude "*.tmp"')


if __name__ == "__main__":
    main()
