"""Load the raw BAR-ATH tracking and Smart Tagging files, plus small signal helpers."""
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'starter'))

import games  # noqa: E402
import load_smart_tagging  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

GAME = games.GAME
FPS = games.GAMES[GAME]['fps']                        # 25
N_FRAMES = games.GAMES[GAME]['n_video_frames']        # 199,319
PITCH_LENGTH, PITCH_WIDTH = games.GAMES[GAME]['pitch']  # 105.3 x 68.0 m
HALVES = games.half_video_blocks(GAME)                # [(half, first_frame, last_frame), ...]
SECOND_HALF_KICKOFF = 113742                          # FAQ.md: frames before it in the 2nd half are dead

BARCA, ATHLETIC, DEAD, UNLABELLED = 0, 1, 2, -1      # same codes as evaluation/possession_metrics.py
LABEL_NAME = {BARCA: 'H', ATHLETIC: 'A', DEAD: 'D', UNLABELLED: '-'}
TEAM_NAME = {BARCA: 'Barça', ATHLETIC: 'Athletic'}
GOALKEEPER_SLOT = {BARCA: 24 - 1, ATHLETIC: 46 - 1}   # slots 24 and 46, explore_tracking.ipynb section 2


def load_tracking():
    """Read the raw ATD file. Returns positions in metres (NaN = not tracked).

    players_xy    (frames, 46 slots, 2)
    players_speed (frames, 46 slots)   m/s, the file's own speed field
    ball_xy       (frames, 2)
    slot_team     (46,)                 BARCA or ATHLETIC
    """
    meta_path, raw_path = games.require(['atd_meta', 'atd_raw'])
    with open(raw_path) as handle:
        # Line format "frame:x,y,s;x,y,s;...:x,y,s" -> one comma-separated row per frame.
        text = handle.read().replace(':', ',').replace(';', ',')
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64).to_numpy()
    assert len(table) == N_FRAMES and (table[:, 0] == np.arange(1, N_FRAMES + 1)).all()

    n_slots = (table.shape[1] - 4) // 3
    players = table[:, 1:1 + 3 * n_slots].reshape(N_FRAMES, n_slots, 3)
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH])
    return {
        'players_xy': (players[:, :, :2] * scale).astype(np.float32),
        'players_speed': players[:, :, 2].astype(np.float32),
        'ball_xy': table[:, -3:-1] * scale,
        'slot_team': team_of_each_track(meta_path),
    }


def load_events():
    """Smart Tagging intervals: dicts with code, start_frame, end_frame, tags."""
    return load_smart_tagging.load()


def graded_frames():
    """True on the frames that are scored."""
    graded = np.zeros(N_FRAMES, bool)
    for _, first, last in HALVES:
        graded[first:last + 1] = True
    return graded


def half_of_each_frame():
    return np.where(np.arange(N_FRAMES) < HALVES[1][1], 1, 2)


def runs_of(mask):
    """Stretches of consecutive True values as (start, end_exclusive).

    runs_of([F, T, T, F, T]) -> [(1, 3), (4, 5)]
    """
    edges = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def rolling_mean(values, frames):
    """Centred moving average over `frames` frames."""
    return pd.Series(values).rolling(frames, center=True, min_periods=1).mean().to_numpy()


def ball_speed(ball_xy):
    """m/s from the distance the ball moves between frame t-2 and t+2 (file's ball speed is unusable)."""
    speed = np.full(len(ball_xy), np.nan)
    speed[2:-2] = np.linalg.norm(ball_xy[4:] - ball_xy[:-4], axis=1) * FPS / 4
    return speed


def ball_distance_to_line(ball_xy):
    """Distance to the nearest touchline or goal line: positive inside the pitch, negative outside."""
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    return np.minimum.reduce([x, PITCH_LENGTH - x, y, PITCH_WIDTH - y])


def attacking_direction(players_xy):
    """{(team, half): +1 if the team attacks towards x = 105, -1 if towards x = 0}.

    A team attacks away from its own goalkeeper, so the keeper's median x decides it.
    """
    direction = {}
    for half, first, last in HALVES:
        start = SECOND_HALF_KICKOFF if half == 2 else first
        for team, slot in GOALKEEPER_SLOT.items():
            keeper_x = np.nanmedian(players_xy[start:last + 1, slot, 0])
            direction[(team, half)] = 1 if keeper_x < PITCH_LENGTH / 2 else -1
    return direction


def distance_from_own_goal(players_xy, slot_team, direction):
    """(frames, slots): each player's x measured from their own goal line (0 = own goal, 105 = opponent's)."""
    half = half_of_each_frame()
    out = np.empty(players_xy.shape[:2], np.float32)
    for team in (BARCA, ATHLETIC):
        slots = slot_team == team
        x = players_xy[:, slots, 0]
        attacks_up = np.where(half == 1, direction[(team, 1)], direction[(team, 2)]) > 0
        out[:, slots] = np.where(attacks_up[:, None], x, PITCH_LENGTH - x)
    return out
