import numpy as np
from tqdm import tqdm
from jaxcmr.helpers import generate_trial_mask
# from numba import njit
from jaxcmr.typing import Integer, Array, Int_
from jax import lax, numpy as jnp


# @njit
def shuffle_matrix(rng, matrix, experiment_count):
    # Initialize an empty list to store each shuffled matrix
    shufflings = np.zeros(
        (int(experiment_count * matrix.shape[0]), matrix.shape[1]), dtype=matrix.dtype
    )

    for _ in range(experiment_count):
        # Copy and shuffle the matrix
        shuffled_matrix = matrix.copy()
        rng.shuffle(shuffled_matrix)
        shufflings[_ * matrix.shape[0] : (_ + 1) * matrix.shape[0]] = shuffled_matrix

    return shufflings


def control_dataset(
    data: dict,
    mixed_trial_query: str,
    ctrl_trial_query: str,
    control_experiment_count: int = 1000,
):
    trial_mask = generate_trial_mask(data, mixed_trial_query)
    control_trial_mask = generate_trial_mask(data, ctrl_trial_query)
    trials = data["recalls"]
    list_length = np.max(data["listLength"][trial_mask])
    presentations = data["pres_itemnos"][:, :list_length]
    subjects = data["subject"].flatten()

    result_trials = []
    result_presentations = []
    result_subjects = []
    rng = np.random.default_rng(1)
    for subject_index, subject in tqdm(enumerate(np.unique(subjects))):
        subject_specific_trial_mask = np.logical_and(subjects == subject, trial_mask)
        ctrl_subject_specific_trial_mask = np.logical_and(
            subjects == subject, control_trial_mask
        )

        trial_count = np.sum(subject_specific_trial_mask)
        if trial_count == 0:
            continue

        shuffled_control_trials = shuffle_matrix(
            rng, trials[ctrl_subject_specific_trial_mask], control_experiment_count
        )
        repeated_mixed_presentations = np.tile(
            presentations[subject_specific_trial_mask],
            (control_experiment_count, 1),
        )

        result_trials.append(shuffled_control_trials)
        result_presentations.append(repeated_mixed_presentations)
        result_subjects += [subject] * trial_count * control_experiment_count

    result_trials = np.concatenate(result_trials)
    result_presentations = np.concatenate(result_presentations)

    return {
        "subject": np.expand_dims(result_subjects, axis=1),
        "recalls": result_trials,
        "pres_itemnos": result_presentations,
        "listLength": np.full((result_trials.shape[0], 1), list_length),
    }

# @njit
def njit_item_to_study_positions(
        item: int,
        presentation: np.ndarray):
    """Return the one-indexed study positions of an item in a 1D presentation sequence.

    Args:
        item: the item index.
        presentation: the 1D presentation sequence.
        size: number of non-zero entries to return.
    """
    return (
        np.empty(0, dtype=np.int64)
        if item == 0
        else np.nonzero(presentation == item)[0] + 1
    )
    

def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
):
    """Return the one-indexed study positions of an item in a 1D presentation sequence.

    Args:
        item: the item index.
        presentation: the 1D presentation sequence.
        size: number of non-zero entries to return.
    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )


def all_study_positions(
    study_position: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
):
    """Return the one-indexed study positions associated with a given study position.

    Args:
        study_position: the study position.
        presentation: the 1D presentation sequence.
        size: number of non-zero entries to return.
    """
    item = lax.cond(
        study_position > 0,
        lambda: presentation[study_position - 1],
        lambda: 0,
    )
    return item_to_study_positions(item, presentation, size)