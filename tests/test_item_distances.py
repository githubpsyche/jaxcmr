"""Distance coverage concerns selected study alternatives, not unused vocabulary."""
import numpy as np
import pytest
from jaxcmr.analyses.distance import validate_item_distances
from jaxcmr.analyses.distrank import subject_distrank
from jaxcmr.helpers import make_dataset


def test_selected_pairs_padding_and_unused_vocabulary_are_preserved():
    data = {'pres_itemids': np.array([[1,2,0],[1,4,0]])}
    distances = np.full((4,4), np.nan)
    distances[0,1] = distances[1,0] = .3
    before = distances.copy()
    validate_item_distances(data, distances, np.array([True,False]))
    np.testing.assert_array_equal(distances, before)
    np.testing.assert_array_equal(data['pres_itemids'], [[1,2,0],[1,4,0]])
    with pytest.raises(ValueError, match='missing distances'):
        validate_item_distances(data, distances)


def test_repeated_item_at_distinct_positions_requires_diagonal_distance():
    data={'pres_itemids':np.array([[1,1]])}
    with pytest.raises(ValueError, match='missing distances'):
        validate_item_distances(data, [[np.nan]])
    validate_item_distances(data, [[0.]])
    validate_item_distances({'pres_itemids':np.array([[1,0]])}, [[np.nan]])


def test_subject_analysis_rejects_out_of_range_item_ids():
    data = make_dataset([1,2])
    data['pres_itemids'] = np.array([[1, 99]])
    with pytest.raises(ValueError, match="item coverage"):
        subject_distrank(data, np.array([True]), np.array([[0., 1.], [1., 0.]]))


def test_unselected_out_of_range_ids_do_not_change_selected_validation():
    validate_item_distances({'pres_itemids':np.array([[1,2],[7,8]])}, [[0,1],[1,0]], [True,False])
    validate_item_distances({'pres_itemids':np.array([[0,0]])}, np.zeros((0,0)))
    with pytest.raises(ValueError, match='pres_itemids'):
        validate_item_distances({}, np.zeros((2,2)))
    # The explicit validator accepts ArrayLike, including nonsquare matrices.
    with pytest.raises(ValueError, match='square'):
        validate_item_distances({'pres_itemids': np.array([[1, 2]])}, [[0, 1]])
