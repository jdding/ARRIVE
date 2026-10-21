"""Selected native v2 coverage functions; raw-input reconstruction is omitted."""
import numpy as np


def recover_sources(ids,features,complete=None):
    """Invert frozen rank encoding, with an exact FP32 round-trip check."""
    ids=np.asarray(ids);values=np.asarray(features[:,2:6])
    if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<3)
        or len(np.unique(ids))!=len(ids) or values.shape!=(len(ids),4)
        or not np.isfinite(values).all() or np.any(values<0) or np.any(values>1)):
        raise ValueError('source_rank_encoding')
    sources=[]
    if complete is not None and len(complete)!=4:raise ValueError('complete_four_sources_required')
    for k,column in enumerate(values.T):
        present=column>0
        ranks=np.rint(np.exp2(1.0/column[present].astype(np.float64))-1).astype(np.int64)
        if np.any(ranks<1) or np.any(ranks>200):raise ValueError('source_rank_range')
        r=ranks.astype(np.float32)
        encoded=1.0/np.log2(r+1.0)
        if not np.array_equal(encoded,column[present]):raise ValueError('source_rank_roundtrip')
        order=np.argsort(ranks,kind='stable')
        if complete is None:
            if not np.array_equal(ranks[order],np.arange(1,len(ranks)+1)):
                raise ValueError('source_ranks_incomplete_or_duplicate')
            sources.append(ids[present][order].astype(np.int32))
        else:
            src=np.asarray(complete[k])
            if src.ndim!=1 or not np.issubdtype(src.dtype,np.integer) or np.any(src<3) or len(src)>200 or len(np.unique(src))!=len(src):raise ValueError('complete_source_identity')
            rankmap={int(item):at+1 for at,item in enumerate(src)}
            expected=np.asarray([rankmap.get(int(item),0) for item in ids],np.float32)
            encoded=np.where(expected>0,1.0/np.log2(np.maximum(expected,1.0)+1.0),0.0)
            if not np.array_equal(encoded,column):raise ValueError('complete_source_observed_ranks_differ')
            sources.append(src.astype(np.int32))
    if len(sources[0])!=200:raise ValueError('backbone200_required_for_equal_budget')
    return sources


def replay_row(ids,features,rrf_indices,truth,complete=None):
    from run_arrive_attribution_controls import rrf_pool
    truth=np.asarray(truth);rrf_indices=np.asarray(rrf_indices)
    if (truth.ndim!=1 or not len(truth) or not np.issubdtype(truth.dtype,np.integer)
        # The existing evaluator preserves unseen target identities as negative
        # IDs. They stay in the denominator and cannot match catalog candidates.
        or np.any(np.isin(truth,[-1,0,1,2])) or len(np.unique(truth))!=len(truth)
        or rrf_indices.shape!=(200,) or not np.issubdtype(rrf_indices.dtype,np.integer)
        or np.any(rrf_indices<0) or np.any(rrf_indices>=len(ids)) or len(np.unique(rrf_indices))!=200):
        raise ValueError('truth_or_RRF_identity')
    sources=recover_sources(ids,features,complete)
    full=rrf_pool(sources)[0]
    if not np.array_equal(full,np.asarray(ids)[rrf_indices]):
        raise ValueError('stored_full_RRF200_replay_difference')
    other=[sources[0],sources[2],sources[3]]
    reduced=rrf_pool(other)[0]
    if not np.array_equal(full[:10],reduced[:10]):raise ValueError('protected_Top10_changed')
    unique=set(map(int,sources[1]))-set().union(*map(set,other))
    targets=set(map(int,truth))
    full_hits=set(map(int,full))&targets;reduced_hits=set(map(int,reduced))&targets
    return {'full_coverage':len(full_hits)/len(targets),
        'removed_coverage':len(reduced_hits)/len(targets),
        'unique_relevant':len(unique&targets),
        'lost_relevant':len(full_hits-reduced_hits),'gained_relevant':len(reduced_hits-full_hits),
        'changed':not np.array_equal(full,reduced),
        'full_union':len(set().union(*map(set,sources))),
        'removed_union':len(set().union(*map(set,other))),
        'backbone_supplement':0}
