"""On-demand mode: channels sleep until someone tunes in or a recording is due.

With ON_DEMAND_MODE=1 a channel does no stream discovery, health probing or
standby checking unless it is awake. A channel wakes when Jellyfin requests its
playlist (a viewer or a DVR recording tuning in), when a Jellyfin recording on it
is about to start, or on a manual rescrape from the dashboard. It goes back to
sleep ON_DEMAND_SLEEP_SECONDS after the last request. Chromium is launched only
when a scrape needs it and closed once every channel has been asleep for
ON_DEMAND_BROWSER_IDLE_SECONDS.

Schedule lookups (ESPN), the guide and the M3U keep working while asleep, so
Jellyfin always sees every channel.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from config import LOGGER, _log_failure, _m3u_title
from network_safety import bounded_float
from state import stream_state

ENABLED = os.getenv("ON_DEMAND_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
# Stay awake this long after the last playlist request, so flipping back to a
# channel is instant.
SLEEP_SECONDS = bounded_float(os.getenv("ON_DEMAND_SLEEP_SECONDS", "600"), 600.0, 60.0, 86400.0)
# Wake a channel this long before a Jellyfin recording on it starts (on top of
# the recording's own pre-padding), so it records the real stream from the start.
PREWAKE_SECONDS = bounded_float(os.getenv("ON_DEMAND_PREWAKE_SECONDS", "600"), 600.0, 0.0, 7200.0)
# How often to read Jellyfin's scheduled recordings.
TIMER_POLL_SECONDS = bounded_float(os.getenv("ON_DEMAND_TIMER_POLL_SECONDS", "300"), 300.0, 30.0, 3600.0)
# Close Chromium after every channel has been asleep this long.
BROWSER_IDLE_SECONDS = bounded_float(os.getenv("ON_DEMAND_BROWSER_IDLE_SECONDS", "300"), 300.0, 30.0, 86400.0)
# A sleeping channel's scrape loop only refreshes its schedule this often.
IDLE_REFRESH_SECONDS = bounded_float(os.getenv("ON_DEMAND_IDLE_REFRESH_SECONDS", "3600"), 3600.0, 300.0, 86400.0)
# On a tune-in to a channel with no streams yet, wait this long for the search to
# find one before falling back to "No Signal". Opening on No Signal and switching
# later can leave Jellyfin without audio when the stream's audio format differs.
TUNE_IN_WAIT_SECONDS = bounded_float(os.getenv("ON_DEMAND_TUNE_IN_WAIT_SECONDS", "40"), 40.0, 0.0, 170.0)
LOOP_INTERVAL = 30.0

_AWAKE_UNTIL: Dict[str, float] = {}
_LAST_AWAKE_AT = 0.0
_LAST_TIMER_POLL = 0.0


def _watched(team_id: str) -> bool:
    from sessions import SESSIONS
    return SESSIONS.is_watched(team_id)


def is_awake(team_id: str) -> bool:
    """Whether this channel should discover, probe and refresh streams now."""
    if not ENABLED:
        return True
    if time.monotonic() < _AWAKE_UNTIL.get(team_id, 0.0):
        return True
    return _watched(team_id)


def any_awake() -> bool:
    if not ENABLED:
        return True
    now = time.monotonic()
    if any(until > now for until in _AWAKE_UNTIL.values()):
        return True
    return any(_watched(team_id) for team_id in list(stream_state))


def wake(team_id: str, reason: str, seconds: Optional[float] = None) -> None:
    """Keep a channel awake for `seconds` (default SLEEP_SECONDS) from now. A
    channel that was asleep starts searching for streams immediately."""
    if not ENABLED or team_id not in stream_state:
        return
    now = time.monotonic()
    was_awake = is_awake(team_id)
    _AWAKE_UNTIL[team_id] = max(_AWAKE_UNTIL.get(team_id, 0.0), now + (seconds or SLEEP_SECONDS))
    if not was_awake:
        LOGGER.info("On-demand: waking channel=%s (%s)", team_id, reason)
        from failover import kick_scrape_loop
        kick_scrape_loop(team_id)


def touch(team_id: str) -> None:
    """A playlist request: wake the channel or keep it awake."""
    wake(team_id, "tune-in")


async def wait_for_streams(team_id: str) -> None:
    """Hold a tune-in until the channel has a stream to play, up to
    TUNE_IN_WAIT_SECONDS, so Jellyfin opens on the real stream."""
    if not ENABLED or TUNE_IN_WAIT_SECONDS <= 0:
        return
    data = stream_state.get(team_id)
    if data is None or data.get("type") == "multiview" or data.get("candidates"):
        return
    started = time.monotonic()
    deadline = started + TUNE_IN_WAIT_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        data = stream_state.get(team_id)
        if data is None:
            return
        if data.get("candidates"):
            LOGGER.info("On-demand: channel=%s found a stream %.1fs after tune-in", team_id, time.monotonic() - started)
            return
        if time.monotonic() - started > 2.0:
            # The search finished without finding anything: stop waiting.
            from failover import _SCRAPE_IN_FLIGHT
            if team_id not in _SCRAPE_IN_FLIGHT:
                return
    LOGGER.info("On-demand: channel=%s has no stream after %.0fs; showing No Signal meanwhile", team_id, TUNE_IN_WAIT_SECONDS)


def _channels_by_title() -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for team_id, data in list(stream_state.items()):
        title = _m3u_title(str(data.get("name") or team_id)).casefold()
        titles.setdefault(title, team_id)
    return titles


def _parse_jellyfin_date(value: object) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    # Jellyfin sends 7 fractional digits ("2026-11-21T17:00:00.0000000Z").
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    head, sep, rest = text.partition(".")
    if sep:
        digits = "".join(ch for ch in rest if ch.isdigit())
        zone = rest[len(digits):]
        text = f"{head}.{digits[:6]}{zone}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def wake_for_timers(timers: list, now: Optional[datetime] = None) -> None:
    """Wake every channel with a Jellyfin recording starting within
    PREWAKE_SECONDS (or already running), until the recording ends."""
    current = now or datetime.now(timezone.utc)
    titles = _channels_by_title()
    for timer in timers:
        if not isinstance(timer, dict):
            continue
        start = _parse_jellyfin_date(timer.get("StartDate"))
        end = _parse_jellyfin_date(timer.get("EndDate"))
        if start is None or end is None:
            continue
        try:
            pre = float(timer.get("PrePaddingSeconds") or 0)
            post = float(timer.get("PostPaddingSeconds") or 0)
        except (TypeError, ValueError):
            pre = post = 0.0
        wake_from = start.timestamp() - pre - PREWAKE_SECONDS
        wake_to = end.timestamp() + post
        if not (wake_from <= current.timestamp() < wake_to):
            continue
        name = str(timer.get("ChannelName") or "")
        team_id = titles.get(_m3u_title(name).casefold())
        if team_id is None:
            LOGGER.warning("On-demand: recording on channel %r matches no Jellyball channel", name)
            continue
        wake(team_id, f"recording {timer.get('Name') or ''}".strip(), seconds=wake_to - current.timestamp() + SLEEP_SECONDS)


async def _poll_timers() -> None:
    from alerts import get_jellyfin_config
    from jellyfin_client import fetch_timers
    timers = await fetch_timers(await get_jellyfin_config())
    if timers is not None:
        wake_for_timers(timers)


async def on_demand_loop() -> None:
    """Wake channels for upcoming recordings; close Chromium once all are asleep."""
    global _LAST_AWAKE_AT, _LAST_TIMER_POLL
    if not ENABLED:
        return
    LOGGER.info(
        "On-demand mode: channels sleep %.0fs after the last viewer; recordings wake them %.0fs early",
        SLEEP_SECONDS, PREWAKE_SECONDS,
    )
    _LAST_AWAKE_AT = time.monotonic()
    while True:
        try:
            now = time.monotonic()
            if now - _LAST_TIMER_POLL >= TIMER_POLL_SECONDS:
                _LAST_TIMER_POLL = now
                await _poll_timers()
            for team_id, until in list(_AWAKE_UNTIL.items()):
                if until <= now and not _watched(team_id):
                    del _AWAKE_UNTIL[team_id]
                    LOGGER.info("On-demand: channel=%s asleep", team_id)
            if any_awake():
                _LAST_AWAKE_AT = now
            elif now - _LAST_AWAKE_AT >= BROWSER_IDLE_SECONDS:
                import scrapers
                await scrapers.close_idle_browser()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            _log_failure("on-demand loop", exc)
        await asyncio.sleep(LOOP_INTERVAL)
