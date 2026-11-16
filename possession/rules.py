"""Rules that label a frame H (Barça), A (Athletic) or D (dead) when the tracking makes it clear.

All functions take plain arrays (one value per frame), so they work on the full match and on toy
examples alike.
"""
import numpy as np

from .data import (ATHLETIC, BARCA, DEAD, FPS, HALVES, PITCH_LENGTH, PITCH_WIDTH, SECOND_HALF_KICKOFF,
                   UNLABELLED, runs_of, rolling_mean)

# Clear control (rules_analysis.ipynb section 2)
CONTROL_DISTANCE = 1.5   # m: a player this close to the ball ...
OPPONENT_MARGIN = 1.5    # m: ... and the nearest opponent at least this much further away
MAX_BALL_SPEED = 6.0     # m/s: ball slower than this (not a pass in flight)
MIN_CONTROL_FRAMES = 5   # all of the above for this many frames in a row

# Dead-ball rules R1-R4 and the restart (rules_analysis.ipynb section 3)
OUT_MARGIN = 0.5         # m past a line = ball out
LOST_NEAR_LINE = 2.0     # m: ball disappears this close to a line
STILL_SPEED = 0.5        # m/s: ball not moving ...
STILL_FRAMES = FPS       # ... for 1 s
TAG_LEAD = 8 * FPS       # dead time before a set-piece tag starts
RESTART_SPEED = 3.0      # m/s: play restarts when the ball moves inside the pitch ...
RESTART_FRAMES = 10      # ... for this many frames
MAX_STOPPAGE = 120 * FPS

# New rules (checked in 1_initial_labels.ipynb)
KEEPER_HOLD_DISTANCE = 1.5     # m: a still ball this close to a keeper is held, i.e. live
WALK_SPEED = 2.0               # m/s
WALK_SHARE = 0.9               # R5: at least this share of players walking ...
WALK_FRAMES = 10 * FPS         # ... for 10 s, with the ball missing
FORMATION_SHARE = 0.9          # R6: this share of each team in its own half
FORMATION_FRAMES = 2 * FPS
FORMATION_LOOKBACK = 150 * FPS
MERGE_GAP = 2 * FPS            # stoppages closer than this are one stoppage
CARRY_MAX_MISSING = FPS        # carry rule: longest ball-missing stretch it may cross ...
CARRY_MAX_MISSING_SAME_TEAM = 10 * FPS  # ... or this, when the same team has the ball on both sides
UNCERTAIN_DEAD_RULES = {'R2 lost at a line', 'R5 players walking'}  # not used as D for training


# -- clear control ------------------------------------------------------------------------------

def nearest_player_distance(players_xy, slot_team, ball_xy, chunk=20000):
    """Distance from the ball to the nearest Barça player and to the nearest Athletic player."""
    n = len(ball_xy)
    nearest = {BARCA: np.full(n, np.inf), ATHLETIC: np.full(n, np.inf)}
    for s in range(0, n, chunk):
        dist = np.linalg.norm(players_xy[s:s + chunk] - ball_xy[s:s + chunk, None, :], axis=2)
        dist = np.where(np.isnan(dist), np.inf, dist)
        for team in nearest:
            nearest[team][s:s + chunk] = dist[:, slot_team == team].min(axis=1)
    return nearest[BARCA], nearest[ATHLETIC]


def clear_control(dist_barca, dist_athletic, ball_speed, ball_to_line, graded):
    """BARCA / ATHLETIC where one team clearly has the ball at a player's feet, else UNLABELLED."""
    closest = np.fmin(dist_barca, dist_athletic)
    with np.errstate(invalid='ignore'):  # inf - inf when nobody is tracked
        margin = np.abs(dist_barca - dist_athletic)
    ok = (graded & (ball_to_line > 0) & (ball_speed < MAX_BALL_SPEED)
          & (closest < CONTROL_DISTANCE) & (margin > OPPONENT_MARGIN))
    closer_team = np.where(dist_barca < dist_athletic, BARCA, ATHLETIC)
    labels = np.full(len(ok), UNLABELLED, np.int8)
    for team in (BARCA, ATHLETIC):
        for a, b in runs_of(ok & (closer_team == team)):
            if b - a >= MIN_CONTROL_FRAMES:
                labels[a:b] = team
    return labels


# -- dead ball: each rule returns stretches (start, end) -----------------------------------------

def restart_finder(ball_speed, ball_to_line):
    """Function frame -> first frame from there on where play restarts.

    Restart = ball inside the pitch and moving for RESTART_FRAMES frames. If that never happens
    within MAX_STOPPAGE, the next frame the ball is seen inside; else MAX_STOPPAGE later.
    """
    inside = ball_to_line > 0
    moving = inside & (ball_speed > RESTART_SPEED)

    def restart(frame):
        end = min(frame + MAX_STOPPAGE, len(inside))
        for a, b in runs_of(moving[frame:end]):
            if b - a >= RESTART_FRAMES:
                return frame + a
        seen = np.flatnonzero(inside[frame:end])
        return frame + seen[0] if len(seen) else end
    return restart


def ball_out(ball_to_line, graded, restart):
    """R1: the ball is seen more than OUT_MARGIN outside the pitch."""
    return [(a, max(restart(b), b)) for a, b in runs_of(graded & (ball_to_line < -OUT_MARGIN))]


def ball_lost_at_line(ball_to_line, graded, restart):
    """R2: the ball disappears within LOST_NEAR_LINE of a line (or outside)."""
    visible = ~np.isnan(ball_to_line)
    spans = []
    for a, b in runs_of(~visible & graded):
        if a > 0 and visible[a - 1] and ball_to_line[a - 1] < LOST_NEAR_LINE:
            spans.append((a - 1, max(restart(b), b)))
    return spans


def keeper_holding(players_xy, ball_xy, keeper_slots):
    """True where a goalkeeper is next to the ball outside the 6-yard box (holding it, not a goal kick)."""
    near_keeper = np.zeros(len(ball_xy), bool)
    for slot in keeper_slots:
        near_keeper |= np.linalg.norm(players_xy[:, slot] - ball_xy, axis=1) < KEEPER_HOLD_DISTANCE
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    in_six_yard_box = ((x < 5.5) | (x > PITCH_LENGTH - 5.5)) & (np.abs(y - PITCH_WIDTH / 2) < 9.16)
    return near_keeper & ~in_six_yard_box


def ball_still(ball_speed, held_by_keeper, graded, restart):
    """R3: the ball stands still for STILL_FRAMES, unless a keeper holds it for most of that time."""
    return [(a, max(restart(b), b)) for a, b in runs_of(graded & (ball_speed < STILL_SPEED))
            if b - a >= STILL_FRAMES and held_by_keeper[a:b].mean() <= 0.5]


def set_piece_tag(events, graded, restart):
    """R4: from TAG_LEAD before each SET PIECES tag until the restart."""
    return [(max(ev['start_frame'] - TAG_LEAD, 0), max(restart(ev['start_frame']), ev['start_frame']))
            for ev in events if ev['code'] == 'SET PIECES' and graded[ev['start_frame']]]


def share_walking(players_speed):
    """Per frame: share of tracked players slower than WALK_SPEED, averaged over 1 s."""
    tracked = ~np.isnan(players_speed)
    share = np.sum(players_speed < WALK_SPEED, axis=1) / np.maximum(tracked.sum(axis=1), 1)
    return rolling_mean(share, FPS)


def players_walking(walking, ball_visible, graded):
    """R5: at least WALK_SHARE of players walking, ball missing, for WALK_FRAMES."""
    hold = graded & (walking >= WALK_SHARE) & ~ball_visible
    return [(a, b) for a, b in runs_of(hold) if b - a >= WALK_FRAMES]


def share_in_own_half(from_own_goal, slot_team, team):
    """Per frame: share of the team's tracked players in their own half."""
    d = from_own_goal[:, slot_team == team]
    return np.sum(d < PITCH_LENGTH / 2 + 1, axis=1) / np.maximum(np.sum(~np.isnan(d), axis=1), 1)


def kickoff_after_goal(events, in_formation, restart):
    """R6: before a 'Kick Off (After Goal)' tag, dead from the first kickoff formation in the
    FORMATION_LOOKBACK before it (both teams in their own half, players walking) until the restart."""
    spans = []
    for ev in events:
        if ev['tags'].get('Type') != 'Kick Off (After Goal)':
            continue
        kickoff = ev['start_frame']
        look = max(kickoff - FORMATION_LOOKBACK, 0)
        runs = [(a, b) for a, b in runs_of(in_formation[look:kickoff + 5 * FPS]) if b - a >= FORMATION_FRAMES]
        if runs:
            spans.append((look + runs[0][0], max(restart(kickoff), kickoff)))
    return spans


def second_half_pre_kickoff():
    """The 2nd half is graded from its first frame but kicks off later (FAQ.md): dead until then."""
    return [(HALVES[1][1], SECOND_HALF_KICKOFF)]


def merge_stoppages(spans_by_rule, graded):
    """Join all rule stretches; stretches closer than MERGE_GAP become one stoppage.

    Returns ([(start, end, {rules})], dead mask on graded frames).
    """
    spans = sorted((a, b, rule) for rule, spans in spans_by_rule.items() for a, b in spans)
    stoppages = []
    for a, b, rule in spans:
        if stoppages and a <= stoppages[-1][1] + MERGE_GAP:
            stoppages[-1][1] = max(stoppages[-1][1], b)
            stoppages[-1][2].add(rule)
        else:
            stoppages.append([a, b, {rule}])
    dead = np.zeros(len(graded), bool)
    for a, b, _ in stoppages:
        dead[a:b] = True
    return stoppages, dead & graded


# -- putting it together ------------------------------------------------------------------------

def carry(control, dead, ball_visible, graded):
    """Final labels: D on dead frames, clear control elsewhere, and between two clear-control
    frames the earlier team keeps the ball, unless there is a D in between or the ball is missing
    for more than CARRY_MAX_MISSING frames (CARRY_MAX_MISSING_SAME_TEAM if both sides are the same team).

    control H H - - H H -> H H H H H H        control H H - - A A -> H H H H A A
    """
    control = np.where(dead, UNLABELLED, control).astype(np.int8)
    labels = control.copy()
    labels[dead] = DEAD
    known = np.flatnonzero(control != UNLABELLED)
    for i, j in zip(known[:-1], known[1:]):
        gap = slice(i + 1, j)
        longest_missing = max((b - a for a, b in runs_of(~ball_visible[gap])), default=0)
        limit = CARRY_MAX_MISSING_SAME_TEAM if control[i] == control[j] else CARRY_MAX_MISSING
        if j - i > 1 and not dead[gap].any() and graded[gap].all() and longest_missing <= limit:
            labels[gap] = control[i]
    labels[~graded] = UNLABELLED
    return labels


def initial_labels(tracking, events):
    """The whole chain of 1_initial_labels.ipynb in one call.

    Returns (labels, trusted_dead): one label per frame, and True on dead frames that a rule other
    than UNCERTAIN_DEAD_RULES finds.
    """
    from .data import (GOALKEEPER_SLOT, attacking_direction, ball_distance_to_line, ball_speed,
                       distance_from_own_goal, graded_frames)
    players_xy, ball_xy, slot_team = tracking['players_xy'], tracking['ball_xy'], tracking['slot_team']
    graded = graded_frames()
    visible = ~np.isnan(ball_xy[:, 0])
    speed, to_line = ball_speed(ball_xy), ball_distance_to_line(ball_xy)
    restart = restart_finder(speed, to_line)
    from_own_goal = distance_from_own_goal(players_xy, slot_team, attacking_direction(players_xy))
    walking = share_walking(tracking['players_speed'])
    in_formation = graded & (walking > 0.5) & np.all(
        [share_in_own_half(from_own_goal, slot_team, t) >= FORMATION_SHARE for t in (BARCA, ATHLETIC)], axis=0)

    spans = {
        'R1 ball out': ball_out(to_line, graded, restart),
        'R2 lost at a line': ball_lost_at_line(to_line, graded, restart),
        'R3 ball still': ball_still(speed, keeper_holding(players_xy, ball_xy, GOALKEEPER_SLOT.values()),
                                    graded, restart),
        'R4 set-piece tag': set_piece_tag(events, graded, restart),
        'R5 players walking': players_walking(walking, visible, graded),
        'R6 kickoff after goal': kickoff_after_goal(events, in_formation, restart),
        '2nd half before kickoff': second_half_pre_kickoff(),
    }
    _, dead = merge_stoppages(spans, graded)
    _, trusted_dead = merge_stoppages({k: v for k, v in spans.items() if k not in UNCERTAIN_DEAD_RULES}, graded)
    control = clear_control(*nearest_player_distance(players_xy, slot_team, ball_xy), speed, to_line, graded)
    return carry(control, dead, visible, graded), trusted_dead
