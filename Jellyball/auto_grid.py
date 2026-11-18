"""Auto Multi-View: one grid channel built from the channels you just watched.

With MULTIVIEW_AUTO_GRID=1 Jellyball lists a "Multi-View" channel, driven
entirely from the Jellyfin app. On each fresh tune-in it looks at the channels
opened since the grid was last watched:

- two or more: a new grid of those (the four most recent, in the order opened);
- one: added to the previous grid (the oldest pane drops when it is full);
- none: the previous grid again.

Two channels show side by side, three or four in a 2x2 grid. With fewer than
two channels it fills in with games that are live now (your team channels
first, then networks carrying a game).
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Tuple, cast

from config import LOGGER
from state import _spawn_background_task, stream_state

ENABLED = os.getenv("MULTIVIEW_AUTO_GRID", "0").strip().lower() in ("1", "true", "yes", "on")
CHANNEL_ID = "multiview_auto"
NAME = "Multi-View"
TVG_ID = "jellyball.multiview.auto"
# A playlist request this long after the grid's previous one is a new tune-in
# (players refresh a live playlist every few seconds while watching).
RETUNE_GAP_SECONDS = 15.0
# How long a tune-in waits for the members' streams before starting the grid.
MEMBER_WAIT_SECONDS = 25.0
_EMPTY_PREFIX = "__empty_"


def ensure_channel() -> None:
    """Create the grid's DB row once, so startup loads it like any Multi-View."""
    if not ENABLED:
        return
    from db import load_multiview_channels, save_multiview_channel
    if any(row.channel_id == CHANNEL_ID for row in load_multiview_channels()):
        return
    save_multiview_channel(
        CHANNEL_ID, NAME, "side_by_side_2", [f"{_EMPTY_PREFIX}1", f"{_EMPTY_PREFIX}2"],
        f"{_EMPTY_PREFIX}1", tvg_id=TVG_ID, group_title="Multi-View",
    )
    LOGGER.info("Created the auto Multi-View channel")


def is_auto_grid(channel_id: str) -> bool:
    return ENABLED and channel_id == CHANNEL_ID


def _live_games() -> List[str]:
    """Channels showing a game right now: team channels first, then networks."""
    import network_games
    from catalog import _channel_is_always_live, is_stream_window_active, parse_team_schedule
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    states = cast(Dict[str, Dict[str, Any]], stream_state)
    teams: List[str] = []
    networks: List[str] = []
    for team_id, data in list(states.items()):
        if data.get("type") == "multiview":
            continue
        if _channel_is_always_live(data):
            if network_games.network_key(data) is not None and network_games.current_game(data, states):
                networks.append(team_id)
            continue
        start, stop = parse_team_schedule(data)
        if start and stop and start <= now <= stop and is_stream_window_active(data, now):
            teams.append(team_id)
    return teams + networks


def choose_members(previous: List[str], since: float) -> Tuple[List[str], str]:
    """(member ids, layout) for a fresh tune-in, given the previous grid's
    members and when the grid was last watched (monotonic time)."""
    import on_demand
    opened = on_demand.tuned_since(since, 4)
    kept = [m for m in previous if not m.startswith(_EMPTY_PREFIX) and m in stream_state]
    if len(opened) >= 2:
        members = opened
    else:
        members = [m for m in kept if m not in opened] + opened
        members = members[-4:]
    if len(members) < 2:
        for team_id in _live_games():
            if team_id not in members:
                members.append(team_id)
            if len(members) == 4:
                break
    target = 2 if len(members) <= 2 else 4
    members = members[:4]
    while len(members) < target:
        members.append(f"{_EMPTY_PREFIX}{len(members) + 1}")
    return members, ("side_by_side_2" if target == 2 else "grid_2x2")


async def prepare_tune_in(channel_id: str) -> None:
    """On a fresh tune-in, pick the members again and restart the grid if they
    changed, then wake them and give them a moment to find streams."""
    import multiview
    import on_demand
    from db import save_multiview_channel_async

    data = stream_state.get(channel_id)
    if data is None:
        return
    last = multiview._MULTIVIEW_LAST_VIEWER.get(channel_id, 0.0)
    now = time.monotonic()
    if now - last < RETUNE_GAP_SECONDS:
        return  # a playlist refresh from someone already watching
    # Never watched since startup: count channels opened within the wake window.
    since = last if last > 0 else now - on_demand.SLEEP_SECONDS
    members, layout = choose_members(list(data.get("member_team_ids") or []), since)
    real = [m for m in members if not m.startswith(_EMPTY_PREFIX)]
    if members != data.get("member_team_ids") or layout != data.get("layout"):
        LOGGER.info("Auto Multi-View: showing %s", ", ".join(real) or "nothing (no recent channels or live games)")
        await multiview._cancel_multiview_start(channel_id)
        await multiview._stop_multiview_process(channel_id)
        # Start the viewer's session over too: carrying on from the old grid
        # (a discontinuity) can leave Jellyfin silent when the new first
        # member's audio format differs.
        from sessions import SESSIONS
        for session_id in multiview._multiview_view_session_ids(channel_id):
            await SESSIONS.close(session_id)
        for mapping in (multiview._MULTIVIEW_FAILURES, multiview._MULTIVIEW_HOLDS, multiview._MULTIVIEW_REFUSALS):
            mapping.pop(channel_id, None)
        data["member_team_ids"] = members
        data["layout"] = layout
        data["active_audio_team_id"] = members[0]
        _spawn_background_task(
            save_multiview_channel_async(
                channel_id, NAME, layout, members, members[0], tvg_id=TVG_ID, group_title="Multi-View",
            ),
            "save auto Multi-View members",
        )
    for team_id in real:
        on_demand.wake(team_id, "Multi-View")
    if real:
        await _wait_members(real)


async def _wait_members(team_ids: List[str]) -> None:
    import on_demand
    try:
        await asyncio.wait_for(
            asyncio.gather(*(on_demand.wait_for_streams(t) for t in team_ids), return_exceptions=True),
            timeout=MEMBER_WAIT_SECONDS,
        )
    except asyncio.TimeoutError:
        LOGGER.info("Auto Multi-View: starting before every member has a stream")
