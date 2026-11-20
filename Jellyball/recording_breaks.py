"""Commercial breaks in Jellyfin recordings of Jellyball channels.

Streaming sites replace the broadcaster's ads with their own still "Commercial
break" card, and Jellyball fills source dropouts with its No Signal slate.
Comskip finds neither. With RECORDINGS_DIR set to the folder Jellyfin records
into, every .ts recording there is scanned while it is being written: one small
grayscale thumbnail per keyframe, compared with the known card and slate images
(assets/break-*.png, plus any PNGs in <data dir>/break-cards). Runs that match
are written to "<recording>.edl" beside it; Jellyfin's Comskip Commercial
Segments plugin imports that file as the recording's commercials instead of
running Comskip, as long as the file is there when the recording ends.

Nothing is written for a recording with no matches (an over-the-air recording
keeps Comskip), and an .edl that Jellyball did not write is never touched.
"""

import asyncio
import json
import operator
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cpu_pool
import ffmpeg_proc
from config import _log_failure, _resource_path, DATA_DIR, LOGGER
from network_safety import bounded_float

RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "").strip()
POLL_SECONDS = bounded_float(os.getenv("RECORDING_BREAKS_POLL_SECONDS", "60"), 60.0, 15.0, 3600.0)
# Recordings last written longer ago than this are left alone.
MAX_AGE_HOURS = bounded_float(os.getenv("RECORDING_BREAKS_MAX_AGE_HOURS", "48"), 48.0, 1.0, 24.0 * 365)
# A recording that has not grown for this long is finished.
FINISHED_AFTER_SECONDS = 180.0
THUMB_W, THUMB_H = 64, 36
THUMB_BYTES = THUMB_W * THUMB_H
# Mean absolute difference (0-255) below which a thumbnail matches a reference.
# Measured: card and slate frames 0-11 from their reference, game footage 25+.
MATCH_DISTANCE = 18.0
# Matches closer together than this belong to one break.
MERGE_GAP_SECONDS = 6.0
MIN_BREAK_SECONDS = 8.0
# Each pass re-reads this much before where the last one stopped, so a keyframe
# that was only half written last time is decoded whole.
REWIND_BYTES = 188 * 44_000
PASS_TIMEOUT_SECONDS = 3600.0
STATE_FILE = DATA_DIR / "recording_breaks.json"
EXTRA_CARDS_DIR = DATA_DIR / "break-cards"
_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")


@dataclass
class _Scan:
    offset: int = 0
    start: Optional[float] = None
    frames: Dict[float, bool] = field(default_factory=dict)
    written: List[Tuple[float, float]] = field(default_factory=list)


def _ffmpeg_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": ffmpeg_proc._ffmpeg_creationflags()}
    return {"preexec_fn": lambda: os.nice(10)}


async def _run_ffmpeg(args: List[str], timeout: float) -> Tuple[bytes, str]:
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_proc.FFMPEG_PATH, "-hide_banner", "-nostats", *args,
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        **_ffmpeg_kwargs(),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except BaseException:
        proc.kill()
        await proc.wait()
        raise
    return out, err.decode("utf-8", "replace")


def _thumb_filter(extra: str = "") -> str:
    return f"scale={THUMB_W}:{THUMB_H}:flags=area,format=gray{extra}"


async def _load_references() -> List[bytes]:
    paths = sorted(_resource_path("assets").glob("break-*.png"))
    if EXTRA_CARDS_DIR.is_dir():
        paths += sorted(EXTRA_CARDS_DIR.glob("*.png"))
    refs = []
    for path in paths:
        out, _err = await _run_ffmpeg(
            ["-loglevel", "error", "-i", str(path), "-frames:v", "1", "-vf", _thumb_filter(), "-f", "rawvideo", "pipe:1"],
            30.0,
        )
        if len(out) == THUMB_BYTES:
            refs.append(out)
        else:
            LOGGER.warning("Recording breaks: could not read reference image %s", path)
    return refs


def _distance(a: bytes, b: bytes) -> float:
    return sum(map(abs, map(operator.sub, a, b))) / THUMB_BYTES


def _classify(thumbs: List[bytes], refs: List[bytes]) -> List[bool]:
    return [any(_distance(t, r) < MATCH_DISTANCE for r in refs) for t in thumbs]


async def _scan_pass(path: Path, scan: _Scan, size: int, refs: List[bytes]) -> None:
    """Thumbnail every keyframe from just before scan.offset to the end of the file."""
    skip = max(0, scan.offset - REWIND_BYTES)
    skip -= skip % 188
    out, err = await _run_ffmpeg(
        [
            "-loglevel", "info", "-skip_frame", "nokey", "-copyts",
            "-skip_initial_bytes", str(skip), "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", _thumb_filter(",showinfo"), "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
        ],
        PASS_TIMEOUT_SECONDS,
    )
    times = [float(m.group(1)) for line in err.splitlines() if "showinfo" in line for m in [_PTS_RE.search(line)] if m]
    thumbs = [out[i:i + THUMB_BYTES] for i in range(0, len(out) - THUMB_BYTES + 1, THUMB_BYTES)]
    count = min(len(times), len(thumbs))
    if not count:
        return
    times, thumbs = times[:count], thumbs[:count]
    if skip == 0:
        scan.start = times[0]
    first = times[0]
    scan.frames = {t: hit for t, hit in scan.frames.items() if t < first}
    scan.frames.update(zip(times, await cpu_pool.run_cpu(_classify, thumbs, refs)))
    scan.offset = size


def _breaks(scan: _Scan) -> List[Tuple[float, float]]:
    if scan.start is None:
        return []
    hits = [t - scan.start for t, hit in sorted(scan.frames.items()) if hit]
    groups: List[List[float]] = []
    for t in hits:
        if groups and t - groups[-1][1] <= MERGE_GAP_SECONDS:
            groups[-1][1] = t
        else:
            groups.append([t, t])
    return [(round(max(0.0, a), 2), round(b, 2)) for a, b in groups if b - a >= MIN_BREAK_SECONDS]


def _write_edl(edl: Path, breaks: List[Tuple[float, float]]) -> None:
    tmp = edl.with_name(edl.name + ".jellyball-tmp")
    tmp.write_text("".join(f"{a:.2f}\t{b:.2f}\t0\n" for a, b in breaks), encoding="utf-8")
    os.replace(tmp, edl)


def _load_state() -> Dict[str, dict]:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: Dict[str, dict]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def _recordings(root: Path, now: float) -> List[Tuple[Path, os.stat_result]]:
    found = []
    oldest = now - MAX_AGE_HOURS * 3600
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".ts"):
                path = Path(folder) / name
                try:
                    st = path.stat()
                except OSError:
                    continue
                if st.st_mtime >= oldest:
                    found.append((path, st))
    return found


def _prune(state: Dict[str, dict], scans: Dict[str, _Scan]) -> bool:
    """Forget deleted recordings, and delete the .edl Jellyball wrote for them."""
    changed = False
    for key in list(state):
        if Path(key).exists():
            continue
        if state[key].get("edl"):
            try:
                Path(key).with_suffix(".edl").unlink()
            except OSError:
                pass
        del state[key]
        scans.pop(key, None)
        changed = True
    return changed


async def _tick(root: Path, refs: List[bytes], state: Dict[str, dict], scans: Dict[str, _Scan], warned: set) -> None:
    now = time.time()
    changed = _prune(state, scans)
    for path, st in _recordings(root, now):
        key = str(path)
        record = state.get(key, {})
        edl = path.with_suffix(".edl")
        if edl.exists() and not record.get("edl"):
            if key not in warned:
                warned.add(key)
                LOGGER.info("Recording breaks: %s already has an .edl from another program; leaving it", path.name)
            continue
        if record.get("done") and record.get("size") == st.st_size:
            continue
        scan = scans.setdefault(key, _Scan())
        finished = now - st.st_mtime >= FINISHED_AFTER_SECONDS
        if st.st_size > scan.offset:
            await _scan_pass(path, scan, st.st_size, refs)
        elif not finished:
            continue
        breaks = _breaks(scan)
        if breaks and breaks != scan.written:
            _write_edl(edl, breaks)
            scan.written = breaks
            record["edl"] = True
            LOGGER.info(
                "Recording breaks: %d break(s), %.1f min, in %s",
                len(breaks), sum(b - a for a, b in breaks) / 60, path.name,
            )
        record["size"] = st.st_size
        record["done"] = finished
        state[key] = record
        changed = True
        if finished:
            scans.pop(key, None)
    if changed:
        _save_state(state)


async def recording_breaks_loop() -> None:
    if not RECORDINGS_DIR:
        return
    root = Path(RECORDINGS_DIR)
    if not root.is_dir():
        LOGGER.warning("Recording breaks: RECORDINGS_DIR %s is not a folder; not scanning recordings", root)
        return
    refs = await _load_references()
    if not refs:
        LOGGER.warning("Recording breaks: no reference images; not scanning recordings")
        return
    LOGGER.info("Recording breaks: scanning %s every %.0f s (%d reference images)", root, POLL_SECONDS, len(refs))
    state = _load_state()
    scans: Dict[str, _Scan] = {}
    warned: set = set()
    while True:
        try:
            await _tick(root, refs, state, scans, warned)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log_failure("scan recordings for breaks", exc)
        await asyncio.sleep(POLL_SECONDS)
