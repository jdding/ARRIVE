"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np
import cue_donors as m


def test_same_bin_and_partition_isolation():
    args=([1,2,1,3],[15]*4,[2]*4,[0,0,1,1],42,[],[],[])
    d,s,t=m.select_donors(*args)
    assert np.array_equal(d,[2,1,3,1])
    assert np.array_equal(t,[0]*4)
    assert np.array_equal(d,m.select_donors(*args)[0])


def test_duplicate_true_cues_are_all_excluded():
    d,s,t=m.select_donors([1,1,1,2],[15]*4,[2]*4,[0]*4,42,[],[],[])
    assert np.array_equal(d,[2,2,2,1])


def test_nearest_lower_bin_then_catalog_preserves_availability():
    d,s,t=m.select_donors([1,2,3,8],[15,15,15,0],[2,1,3,4],[0,0,0,1],42,[9,10],[0,15],[4,4])
    assert d[0]==2 and t[0]==1
    assert d[3]==9 and t[3]==2 and s[3]==-1


def test_insufficient_group_retained_and_flagged():
    d,s,t=m.select_donors([1],[15],[2],[0],42,[1,2],[15,0],[2,2])
    assert (d[0],s[0],t[0])==(-1,-1,3)


def test_four_matching_fields_exact_against_existing_producer():
    from scipy import sparse
    from candidate_features import row_feature
    rng=np.random.default_rng(7);n=12
    audio=rng.normal(size=(n,4)).astype(np.float32)
    genre=sparse.csr_matrix(rng.normal(size=(n,3)).astype(np.float32))
    tags=sparse.csr_matrix(rng.normal(size=(n,5)).astype(np.float32))
    artists=np.array([-1,1,2,1,4,5,6,7,8,9,10,11]);ids=np.array([1,3,7,2],np.int32)
    items=rng.normal(size=(n,3)).astype(np.float32);state=rng.normal(size=3).astype(np.float32)
    old=row_feature(ids,state,items,[{}, {}, {}, {}],30,0,np.ones(n),3,[1,2],artists,tags,audio,genre)
    assert np.array_equal(m.cue_match_values(ids,3,artists,tags,audio,genre),old[:,[12,13,14,16]])


def test_availability_bits_equal_existing_fields():
    from scipy import sparse
    artists=np.array([-1,7,8,9]);audio=np.array([[0,0],[1,0],[0,0],[1,1]],np.float32)
    tags=sparse.csr_matrix(np.array([[0],[1],[1],[0]],np.float32))
    genre=sparse.csr_matrix(np.array([[0],[1],[0],[1]],np.float32))
    assert np.array_equal(m.availability_codes(artists,tags,audio,genre),[0,15,9,7])


def test_wrong_matching_keeps_27_fields_and_uses_absolute_query_indices():
    from scipy import sparse
    rng=np.random.default_rng(19);n=10
    artists=np.arange(n);audio=rng.normal(size=(n,4)).astype(np.float32)
    tags=sparse.csr_matrix(rng.normal(size=(n,3)).astype(np.float32));genre=tags.copy()
    ids=np.array([3,4,5,3,6,7,4]);queries=np.array([8,8,8,2,2,8,8])
    donors=np.full(12,6);donors[8]=4
    block=rng.normal(size=(len(ids),31)).astype(np.float32);before=block.copy()
    m.apply_wrong_matching(block,ids,queries,donors,artists,tags,audio,genre)
    keep=np.setdiff1d(np.arange(31),[12,13,14,16])
    assert np.array_equal(block[:,keep],before[:,keep])
    for lo,hi in ((0,3),(3,5),(5,7)):
        assert np.array_equal(block[lo:hi][:,[12,13,14,16]],m.cue_match_values(ids[lo:hi],donors[queries[lo]],artists,tags,audio,genre))
    assert np.array_equal(block[:,[19,21,24,27]],before[:,[19,21,24,27]])
