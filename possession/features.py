"""Per-frame features from the players only (no ball), plus optional Smart Tagging features.

Names: barca_<x> / athletic_<x> per team, diff_<x> = Barça minus Athletic, <x>_1s / <x>_5s = the
same value averaged over 1 s / 5 s.
"""
import warnings

import numpy as np
import pandas as pd

from .data import ATHLETIC, BARCA, FPS, PITCH_LENGTH, half_of_each_frame, rolling_mean

RUNNING_SPEED = 4.0      # m/s
VELOCITY_STEP = 5        # frames each side for player velocity
MAX_PLAYER_SPEED = 12.0  # m/s: faster = a slot jumping to another player, not running


def forward_velocity(players_xy, slot_team, direction):
    """(frames, slots) m/s towards the opponent's goal (negative = running back)."""
    s = VELOCITY_STEP
    velocity = np.full(players_xy.shape, np.nan, np.float32)
    velocity[s:-s] = (players_xy[2 * s:] - players_xy[:-2 * s]) * FPS / (2 * s)
    velocity[np.linalg.norm(velocity, axis=2) > MAX_PLAYER_SPEED] = np.nan
    half = half_of_each_frame()
    forward = np.empty(players_xy.shape[:2], np.float32)
    for team in (BARCA, ATHLETIC):
        slots = slot_team == team
        sign = np.where(half == 1, direction[(team, 1)], direction[(team, 2)])
        forward[:, slots] = velocity[:, slots, 0] * sign[:, None]
    return forward


def marking_distance(players_xy, slot_team, team, chunk=20000):
    """Per frame: mean distance from each of the team's players to the nearest opponent."""
    mine, theirs = players_xy[:, slot_team == team], players_xy[:, slot_team != team]
    out = np.full(len(players_xy), np.nan, np.float32)
    for s in range(0, len(out), chunk):
        dist = np.linalg.norm(mine[s:s + chunk, :, None] - theirs[s:s + chunk, None], axis=3)
        nearest = np.min(np.where(np.isnan(dist), np.inf, dist), axis=2)
        out[s:s + chunk] = np.nanmean(np.where(np.isinf(nearest), np.nan, nearest), axis=1)
    return out


def team_features(players_xy, players_speed, slot_team, from_own_goal, forward):
    """One row per frame, the team-shape columns for both teams."""
    with warnings.catch_warnings():  # frames with no player tracked give NaN, which is fine
        warnings.simplefilter('ignore', RuntimeWarning)
        return _team_features(players_xy, players_speed, slot_team, from_own_goal, forward)


def _team_features(players_xy, players_speed, slot_team, from_own_goal, forward):
    cols = {}
    for team, name in ((BARCA, 'barca'), (ATHLETIC, 'athletic')):
        slots = slot_team == team
        depth = from_own_goal[:, slots]          # m from own goal line
        across = players_xy[:, slots, 1]
        speed = players_speed[:, slots]
        cols[f'{name}_players_tracked'] = np.sum(~np.isnan(depth), axis=1)
        cols[f'{name}_centre_from_own_goal'] = np.nanmean(depth, axis=1)
        cols[f'{name}_centre_across'] = np.nanmean(across, axis=1)
        cols[f'{name}_spread_length'] = np.nanstd(depth, axis=1)
        cols[f'{name}_spread_width'] = np.nanstd(across, axis=1)
        cols[f'{name}_defensive_line'] = np.sort(depth, axis=1)[:, 1]  # 2nd deepest: the deepest is the keeper
        cols[f'{name}_front_player'] = np.nanmax(depth, axis=1)
        cols[f'{name}_in_opponent_half'] = np.sum(depth > PITCH_LENGTH / 2, axis=1)
        cols[f'{name}_mean_speed'] = np.nanmean(speed, axis=1)
        cols[f'{name}_running'] = np.sum(speed > RUNNING_SPEED, axis=1)
        cols[f'{name}_forward_speed'] = np.nanmean(forward[:, slots], axis=1)
        cols[f'{name}_marking_distance'] = marking_distance(players_xy, slot_team, team)
    df = pd.DataFrame(cols)
    for x in ['centre_from_own_goal', 'spread_length', 'spread_width', 'defensive_line', 'front_player',
              'in_opponent_half', 'mean_speed', 'forward_speed', 'marking_distance']:
        df[f'diff_{x}'] = df[f'barca_{x}'] - df[f'athletic_{x}']
    return df.astype(np.float32)


def add_time_context(df):
    """Add 1 s and 5 s averages of the main columns, and how far each team moved up in 2 s."""
    out = {}
    for col in ['barca_centre_from_own_goal', 'athletic_centre_from_own_goal', 'barca_forward_speed',
                'athletic_forward_speed', 'diff_forward_speed', 'diff_marking_distance',
                'barca_spread_length', 'athletic_spread_length', 'diff_in_opponent_half',
                'barca_mean_speed', 'athletic_mean_speed', 'share_walking']:
        for seconds in (1, 5):
            out[f'{col}_{seconds}s'] = rolling_mean(df[col].to_numpy(), seconds * FPS)
    for name in ('barca', 'athletic'):
        centre = df[f'{name}_centre_from_own_goal'].to_numpy()
        out[f'{name}_moved_up_2s'] = centre - np.r_[np.full(2 * FPS, np.nan), centre[:-2 * FPS]]
    return pd.concat([df, pd.DataFrame(out).astype(np.float32)], axis=1)


def tag_features(events, n_frames):
    """Per Smart Tagging code: +1 while a Barça interval is on, -1 for Athletic, 0 otherwise.
    Plus tag_set_piece_on and tag_neutral_on (an interval with team N/A is on)."""
    side = {'Barcelona': 1, 'Atletic Club': -1}
    cols = {}
    for ev in events:
        code = 'tag_' + ev['code'].strip().lower().replace(' ', '_')
        col = cols.setdefault(code, np.zeros(n_frames, np.float32))
        col[ev['start_frame']:ev['end_frame'] + 1] = side.get(ev['tags'].get('Team'), 0)
    for name, pick in [('tag_set_piece_on', lambda ev: ev['code'] == 'SET PIECES'),
                       ('tag_neutral_on', lambda ev: ev['tags'].get('Team') == 'N/A')]:
        col = cols.setdefault(name, np.zeros(n_frames, np.float32))
        for ev in filter(pick, events):
            col[ev['start_frame']:ev['end_frame'] + 1] = 1
    return pd.DataFrame(cols)


def player_features(tracking):
    """All player features of 2_lightgbm.ipynb (sections 2-4) in one call."""
    from .data import attacking_direction, distance_from_own_goal
    from .rules import share_walking
    players_xy, players_speed, slot_team = tracking['players_xy'], tracking['players_speed'], tracking['slot_team']
    direction = attacking_direction(players_xy)
    from_own_goal = distance_from_own_goal(players_xy, slot_team, direction)
    forward = forward_velocity(players_xy, slot_team, direction)
    X = team_features(players_xy, players_speed, slot_team, from_own_goal, forward)
    X['share_walking'] = share_walking(players_speed)
    return add_time_context(X)
