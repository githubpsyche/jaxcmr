"""Analysis utilities for VRT selective interference experiments.

Provides helpers for deriving cue attributions, computing
interference-specific recall metrics, and preparing data for
selective interference analyses.

"""

import numpy as np


def derive_cue_clips(recall_items, recall_types, recalls):
    """Derive cue attributions from the interleaved event sequence.

    Parameters
    ----------
    recall_items : ndarray, shape [n_trials, max_events]
        Clip/scene numbers from the interleaved event stream.
    recall_types : ndarray, shape [n_trials, max_events]
        Event type codes: 0=padding, 1=cue, 2=recall, 3=foil.
    recalls : ndarray, shape [n_trials, max_recalls]
        Unique-first recall sequence (within-list positions).

    Returns
    -------
    ndarray, shape [n_trials, max_recalls]
        Cue clip number preceding each recall entry:
        0 = no cue, 1..11 = clip number of the preceding cue.

    """
    n_trials = recall_items.shape[0]
    cue_clips = np.zeros_like(recalls)

    for t in range(n_trials):
        pending_cue = 0
        matched = set()

        for e in range(recall_items.shape[1]):
            etype = int(recall_types[t, e])
            item = int(recall_items[t, e])

            if etype == 0:
                break
            elif etype == 1:
                pending_cue = item
            elif etype == 3:
                pending_cue = 0
            elif etype == 2:
                for r in range(recalls.shape[1]):
                    if int(recalls[t, r]) == 0:
                        break
                    if int(recalls[t, r]) == item and r not in matched:
                        cue_clips[t, r] = pending_cue
                        matched.add(r)
                        break
                pending_cue = 0

    return cue_clips


def build_transition_masks(recalls, cue_clips):
    """Construct boolean masks for transition-type filtering.

    Parameters
    ----------
    recalls : ndarray, shape [n_trials, max_recalls]
        Unique-first recall sequence.
    cue_clips : ndarray, shape [n_trials, max_recalls]
        Cue attributions from ``derive_cue_clips``.

    Returns
    -------
    dict
        Boolean masks shaped [n_trials, max_recalls]:
        ``uncued``, ``cue_matched``, ``doubly_uncued``, ``from_cued``.

    """
    valid = recalls > 0
    cue_matched = (cue_clips == recalls) & (cue_clips > 0) & valid
    uncued = (cue_clips == 0) & valid

    prev_matched = np.zeros_like(cue_matched)
    prev_matched[:, 1:] = cue_matched[:, :-1]

    prev_not_matched = np.zeros_like(cue_matched)
    prev_not_matched[:, 1:] = ~cue_matched[:, :-1] & valid[:, :-1]

    doubly_uncued = (~cue_matched & valid) & prev_not_matched
    from_cued = prev_matched & valid

    return {
        "uncued": uncued,
        "cue_matched": cue_matched,
        "doubly_uncued": doubly_uncued,
        "from_cued": from_cued,
    }
