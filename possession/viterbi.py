"""Viterbi smoothing over H / A / D: the single most likely label sequence given per-frame
evidence and a cost for every change of label.

D is a chain of MIN_DEAD states that must be walked through in order, so a stoppage lasts at
least MIN_DEAD frames (no 3-frame D used as a cheap way from H to A).
"""
import numpy as np

from .data import ATHLETIC, BARCA, DEAD, FPS, UNLABELLED

RULE_CONFIDENCE = 0.95   # evidence for a rule label (the other two share the rest)
MODEL_DEAD = 0.001       # evidence for D on frames without a rule label: D comes only from rules
MIN_DEAD = FPS           # a stoppage lasts at least 1 s


def evidence(labels, p_barca):
    """(frames, 3) probabilities for H, A, D.

    Rule-labelled frames: RULE_CONFIDENCE for their label. Unlabelled frames: the model's P(H)
    and P(A) rescaled to sum to 1 - MODEL_DEAD, and MODEL_DEAD for D.
    """
    ev = np.full((len(labels), 3), (1 - RULE_CONFIDENCE) / 2)
    for label in (BARCA, ATHLETIC, DEAD):
        ev[labels == label, label] = RULE_CONFIDENCE
    free = labels == UNLABELLED
    ev[free, BARCA] = p_barca[free] * (1 - MODEL_DEAD)
    ev[free, ATHLETIC] = (1 - p_barca[free]) * (1 - MODEL_DEAD)
    ev[free, DEAD] = MODEL_DEAD
    return ev


def change_costs(switch_cost, dead_cost, min_dead=MIN_DEAD):
    """(states, states) cost of going from one state to the next. States: H, A, D1 ... Dmin_dead.

    H <-> A costs switch_cost; H/A -> D1 and Dmin_dead -> H/A cost dead_cost; D1 -> D2 -> ... is
    free and forced; Dmin_dead may stay. Everything else is impossible (inf).
    """
    n = 2 + min_dead
    cost = np.full((n, n), np.inf)
    cost[BARCA, BARCA] = cost[ATHLETIC, ATHLETIC] = 0
    cost[BARCA, ATHLETIC] = cost[ATHLETIC, BARCA] = switch_cost
    first_dead, last_dead = 2, n - 1
    cost[[BARCA, ATHLETIC], first_dead] = dead_cost
    cost[last_dead, [BARCA, ATHLETIC]] = dead_cost
    for d in range(first_dead, last_dead):
        cost[d, d + 1] = 0
    cost[last_dead, last_dead] = 0
    return cost


def viterbi(ev, cost):
    """Most likely sequence of labels (H, A or D per frame) for evidence `ev` (frames, 3)."""
    n_frames, n_states = len(ev), len(cost)
    state_label = np.r_[BARCA, ATHLETIC, np.full(n_states - 2, DEAD)]
    log_ev = np.log(ev)[:, state_label]            # every D state shares the D evidence
    best = log_ev[0].copy()
    best[2:-1] = -np.inf                           # may start inside a stoppage, but not mid-chain
    back = np.zeros((n_frames, n_states), np.int16)
    for t in range(1, n_frames):
        options = best[:, None] - cost             # options[from, to]
        back[t] = options.argmax(axis=0)
        best = log_ev[t] + options.max(axis=0)
    states = np.empty(n_frames, np.int16)
    states[-1] = best.argmax()
    for t in range(n_frames - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return state_label[states].astype(np.int8)
