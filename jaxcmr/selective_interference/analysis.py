"""Analysis utilities for VRT selective interference experiments.

Provides helpers for deriving cue attributions, computing
interference-specific recall metrics, and preparing data for
selective interference analyses.

"""

import numpy as np


def derive_cue_clips(recall_items, recall_types):
    """Derive cue attributions from the interleaved event sequence.

    Parameters
    ----------
    recall_items : ndarray, shape [n_trials, max_events]
        Clip/scene numbers from the interleaved event stream.
    recall_types : ndarray, shape [n_trials, max_events]
        Event type codes: 0=padding, 1=cue, 2=recall, 3=foil,
        4=unclassified.

    Returns
    -------
    ndarray, shape [n_trials, max_events]
        Pending cue at each event position: nonzero only at recall
        events (type 2) where a cue was active.  Zero elsewhere.
        Unclassified utterances (type 4) consume the pending cue
        without producing a match.

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

    return cue_clips


def build_transition_masks(recall_items, recall_types, cue_clips):
    """Construct boolean masks for transition-type filtering.

    Parameters
    ----------
    recall_items : ndarray, shape [n_trials, max_events]
        Clip/scene numbers from the interleaved event stream.
    recall_types : ndarray, shape [n_trials, max_events]
        Event type codes: 0=padding, 1=cue, 2=recall, 3=foil,
        4=unclassified.
    cue_clips : ndarray, shape [n_trials, max_events]
        Cue attributions from ``derive_cue_clips``.

    Returns
    -------
    dict
        Boolean masks shaped [n_trials, max_events], nonzero only at
        recall events: ``uncued``, ``cue_matched``, ``doubly_uncued``,
        ``from_cued``.

    """
    is_recall = recall_types == 2
    cue_matched = (cue_clips == recall_items) & (cue_clips > 0) & is_recall
    uncued = (cue_clips == 0) & is_recall

    # Build "previous recall" masks by finding recall-to-recall adjacency.
    # For each recall event, look back to the most recent prior recall event.
    n_trials, max_events = recall_items.shape
    prev_matched = np.zeros_like(cue_matched)
    prev_not_matched = np.zeros_like(cue_matched)

    for t in range(n_trials):
        last_recall_matched = None
        for e in range(max_events):
            if int(recall_types[t, e]) == 0:
                break
            if int(recall_types[t, e]) == 2:
                if last_recall_matched is not None:
                    if last_recall_matched:
                        prev_matched[t, e] = True
                    else:
                        prev_not_matched[t, e] = True
                last_recall_matched = bool(cue_matched[t, e])

    doubly_uncued = (~cue_matched & is_recall) & prev_not_matched
    from_cued = prev_matched & is_recall

    return {
        "uncued": uncued,
        "cue_matched": cue_matched,
        "doubly_uncued": doubly_uncued,
        "from_cued": from_cued,
    }
