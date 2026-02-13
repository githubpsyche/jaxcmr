"""Analysis utilities for VRT selective interference experiments.

Provides helpers for deriving cue attributions, computing
interference-specific recall metrics, and preparing data for
selective interference analyses.

"""

import numpy as np


def derive_cue_clips(recall_items, recall_types, recalls=None):
    """Derive cue attributions from the interleaved event sequence.

    Parameters
    ----------
    recall_items : ndarray, shape [n_trials, max_events]
        Clip/scene numbers from the interleaved event stream.
    recall_types : ndarray, shape [n_trials, max_events]
        Event type codes: 0=padding, 1=cue, 2=recall, 3=foil,
        4=unclassified.
    recalls : ndarray, shape [n_trials, max_recalls], optional
        Unique-first recall sequences.  When provided, the output is
        projected from event-aligned to recall-aligned so that each
        position corresponds to the matching entry in *recalls*.

    Returns
    -------
    ndarray
        If *recalls* is ``None``, shape ``[n_trials, max_events]``
        (event-aligned).  If *recalls* is given, shape
        ``[n_trials, max_recalls]`` (recall-aligned).

    """
    n_trials, max_events = recall_items.shape
    cue_clips = np.zeros_like(recall_items)

    for t in range(n_trials):
        pending_cue = 0
        for e in range(max_events):
            etype = int(recall_types[t, e])
            item = int(recall_items[t, e])

            if etype == 0:
                break
            elif etype == 1:
                pending_cue = item
            elif etype == 3:
                pending_cue = 0
            elif etype == 2:
                cue_clips[t, e] = pending_cue
                pending_cue = 0
            elif etype == 4:
                pending_cue = 0

    if recalls is None:
        return cue_clips

    # Project event-aligned → recall-aligned (unique-first)
    max_recalls = recalls.shape[1]
    recall_cue_clips = np.zeros_like(recalls)
    for t in range(n_trials):
        recall_idx = 0
        seen = set()
        for e in range(max_events):
            etype = int(recall_types[t, e])
            if etype == 0:
                break
            if etype == 2:
                item = int(recall_items[t, e])
                if item > 0 and item not in seen:
                    seen.add(item)
                    if recall_idx < max_recalls:
                        recall_cue_clips[t, recall_idx] = cue_clips[t, e]
                        recall_idx += 1
    return recall_cue_clips


def build_transition_masks(recalls, cue_clips):
    """Construct boolean masks for transition-type filtering.

    Parameters
    ----------
    recalls : ndarray, shape [n_trials, max_recalls]
        Unique-first recall sequences (recall-aligned).
    cue_clips : ndarray, shape [n_trials, max_recalls]
        Recall-aligned cue attributions from ``derive_cue_clips``.

    Returns
    -------
    dict
        Boolean masks shaped ``[n_trials, max_recalls]``:
        ``uncued``, ``cue_matched``, ``doubly_uncued``,
        ``from_cued``.

    """
    is_valid = recalls > 0
    cue_matched = (cue_clips == recalls) & (cue_clips > 0) & is_valid
    uncued = (cue_clips == 0) & is_valid

    n_trials, max_recalls = recalls.shape
    prev_matched = np.zeros_like(cue_matched)
    prev_not_matched = np.zeros_like(cue_matched)

    for t in range(n_trials):
        last_recall_matched = None
        for r in range(max_recalls):
            if recalls[t, r] == 0:
                break
            if last_recall_matched is not None:
                if last_recall_matched:
                    prev_matched[t, r] = True
                else:
                    prev_not_matched[t, r] = True
            last_recall_matched = bool(cue_matched[t, r])

    doubly_uncued = (~cue_matched & is_valid) & prev_not_matched
    from_cued = prev_matched & is_valid

    return {
        "uncued": uncued,
        "cue_matched": cue_matched,
        "doubly_uncued": doubly_uncued,
        "from_cued": from_cued,
    }
