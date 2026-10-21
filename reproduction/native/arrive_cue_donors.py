"""Label-free controlled wrong-cue selection; no candidate or target inputs."""
import numpy as np

def select_donors(cues, availability, popularity_bin, partition, seed,
                  catalog_cues, catalog_availability, catalog_popularity_bin):
    """Draw another cue within partition and availability, then fixed fallback.

    Popularity bins are floor(log2(1+Train_count)); callers supply Train-only
    counts. Prefer same-bin request donors, then nearest bin (lower on ties),
    then a Train-known catalog cue. Catalog metadata never contains targets.
    Unresolved rows stay -1 and must be disclosed as insufficient controls.
    """
    cues, availability, popularity_bin, partition = [np.asarray(v) for v in
        (cues, availability, popularity_bin, partition)]
    cat, cat_av, cat_bin = [np.asarray(v) for v in
        (catalog_cues, catalog_availability, catalog_popularity_bin)]
    if cues.ndim != 1 or cat.ndim != 1 or any(v.ndim != 1 or len(v) != len(cues) for v in
           (availability, popularity_bin, partition)):
        raise ValueError('request_alignment')
    if any(v.ndim != 1 or len(v) != len(cat) for v in (cat_av, cat_bin)):
        raise ValueError('catalog_alignment')
    if len(np.unique(cat)) != len(cat) or np.any(cat < 0) or np.any(cues < 0):
        raise ValueError('invalid_cue_identity')
    if np.any(availability < 0) or np.any(availability > 15) or np.any(cat_av < 0) or np.any(cat_av > 15):
        raise ValueError('four_bit_availability_required')
    if any(np.any(~np.isfinite(v)) or np.any(v != np.floor(v)) or np.any(v < 0)
           for v in (cues,availability,popularity_bin,partition,cat,cat_av,cat_bin)):
        raise ValueError('finite_nonnegative_integer_metadata_required')
    rng=np.random.default_rng(seed)
    donor=np.full(len(cues),-1,np.int64);source=np.full(len(cues),-1,np.int64)
    tier=np.full(len(cues),3,np.uint8)
    def groups(values,av,bins,part=None):
        result={}
        for i in range(len(values)):
            key=(int(part[i]),int(av[i]),int(bins[i])) if part is not None else (int(av[i]),int(bins[i]))
            result.setdefault(key,[]).append(i)
        ordered={k:np.asarray(ix,np.int64)[np.argsort(values[ix],kind='stable')] for k,ix in result.items()}
        return {k:(ix,values[ix]) for k,ix in ordered.items()}
    request=groups(cues,availability,popularity_bin,partition)
    catalog=groups(cat,cat_av,cat_bin)
    def choose(group,actual):
        pool,ordered=group
        lo=np.searchsorted(ordered,actual,'left');hi=np.searchsorted(ordered,actual,'right')
        eligible=len(pool)-(hi-lo)
        if not eligible:return None
        pos=int(rng.integers(eligible))
        if pos>=lo:pos+=hi-lo
        return int(pool[pos])
    for i in range(len(cues)):
        p,av,b=int(partition[i]),int(availability[i]),int(popularity_bin[i])
        bins=sorted((k[2] for k in request if k[:2]==(p,av)),key=lambda q:(abs(q-b),q))
        for other in bins:
            chosen=choose(request[p,av,other],cues[i])
            if chosen is not None:
                donor[i]=cues[chosen];source[i]=chosen;tier[i]=0 if other==b else 1
                break
        if donor[i]<0:
            bins=sorted((k[1] for k in catalog if k[0]==av),key=lambda q:(abs(q-b),q))
            for other in bins:
                chosen=choose(catalog[av,other],cues[i])
                if chosen is not None:
                    donor[i]=cat[chosen];tier[i]=2
                    break
    known=donor>=0
    if np.any(donor[known]==cues[known]):raise AssertionError('self_cue_selected')
    from_request=source>=0
    if np.any(partition[source[from_request]]!=partition[from_request]) or np.any(availability[source[from_request]]!=availability[from_request]):
        raise AssertionError('partition_or_availability_crossed')
    return donor,source,tier

def cue_match_values(ids, cue, artists, tags, audio, genre):
    """Reuse the four row_feature matching expressions, not a new encoder."""
    ids=np.asarray(ids,dtype=np.int32)
    if cue < 0 or cue >= len(artists) or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(artists)):
        raise ValueError('matching_item_range')
    result=np.column_stack((
        ((artists[ids]==artists[cue]) & (artists[cue]>=0)).astype(np.float32),
        tags[ids].dot(tags.getrow(cue).T).toarray().ravel(),
        np.asarray(audio[ids]) @ np.asarray(audio[cue]),
        genre[ids].dot(genre.getrow(cue).T).toarray().ravel(),
    )).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('matching_nonfinite')
    return result

def availability_codes(artists, tags, audio, genre):
    """Exactly the existing four cue-availability fields, encoded as bits."""
    n=len(artists)
    if tags.shape[0]!=n or audio.shape[0]!=n or genre.shape[0]!=n:
        raise ValueError('content_catalog_alignment')
    audio_known=np.empty(n,bool)
    for lo in range(0,n,65536):
        audio_known[lo:lo+65536]=np.linalg.norm(audio[lo:lo+65536],axis=1)>0
    fields=np.column_stack((np.asarray(artists)>=0,
        audio_known,genre.getnnz(axis=1)>0,tags.getnnz(axis=1)>0))
    return fields.astype(np.uint8) @ np.array([1,2,4,8],np.uint8)

def apply_wrong_matching(block, ids, query_ids, donors, artists, tags, audio, genre, eligible=None):
    """Overwrite four fields in a new bounded block; never allocate a full export.

    Query ids refer to the original metadata rows, not the fit subset's group
    ids. Each query uses one donor for all candidates, also across chunk cuts.
    """
    ids=np.asarray(ids);queries=np.asarray(query_ids);donors=np.asarray(donors)
    if (block.ndim!=2 or block.shape[1]!=31 or block.dtype!=np.float32 or not block.flags.writeable
        or ids.ndim!=1 or queries.ndim!=1 or len(ids)!=len(block) or len(queries)!=len(block)
        or donors.ndim!=1 or np.any(queries<0) or np.any(queries>=len(donors))
        or np.any(queries!=np.floor(queries)) or np.any(donors<0)
        or np.any(~np.isfinite(donors)) or np.any(donors!=np.floor(donors))):
        raise ValueError('wrong_matching_block_or_unresolved_control')
    eligible=np.ones(len(donors),bool) if eligible is None else np.asarray(eligible)
    if eligible.dtype!=bool or eligible.shape!=donors.shape:
        raise ValueError('wrong_matching_eligibility_alignment')
    # Contiguous runs keep the cost linear; repeated runs still use the same donor.
    starts=np.r_[0,np.flatnonzero(queries[1:]!=queries[:-1])+1]
    ends=np.r_[starts[1:],len(queries)]
    for lo,hi in zip(starts,ends):
        if lo==hi:continue
        if not eligible[int(queries[lo])]:
            if np.any(block[lo:hi][:,[12,13,14,16,19,21,24,27]]!=0):
                raise ValueError('unavailable_cue_has_nonzero_matching_or_availability')
            continue
        block[lo:hi,[12,13,14,16]]=cue_match_values(ids[lo:hi],int(donors[int(queries[lo])]),artists,tags,audio,genre)
    return block


class CueTreatment:
    """Read-only metadata plus an O(requests) mapping; no feature copy."""
    def __init__(self, donors, artists, tags, audio, genre, eligible=None):
        self.donors=donors
        self.eligible=np.ones(len(donors),bool) if eligible is None else np.asarray(eligible)
        self.artists,self.tags,self.audio,self.genre=artists,tags,audio,genre
        self.count=0
        self.sums=np.zeros((2,4),np.float64)
        self.squares=np.zeros((2,4),np.float64)

    def apply(self, block, ids, query_ids):
        before=block[:,[12,13,14,16]].astype(np.float64)
        result=apply_wrong_matching(block,ids,query_ids,self.donors,
                                   self.artists,self.tags,self.audio,self.genre,self.eligible)
        after=block[:,[12,13,14,16]].astype(np.float64)
        self.count+=len(block)
        for at,v in enumerate((before,after)):
            self.sums[at]+=v.sum(axis=0);self.squares[at]+=(v*v).sum(axis=0)
        return result

    def distribution(self):
        mean=self.sums/max(self.count,1)
        std=np.sqrt(np.maximum(0,self.squares/max(self.count,1)-mean*mean))
        return {'processed_candidate_occurrences':self.count,'columns':[12,13,14,16],
                'original_mean':mean[0].tolist(),'wrong_mean':mean[1].tolist(),
                'original_std':std[0].tolist(),'wrong_std':std[1].tolist(),
                'scope':'processed blocks; Train includes fit and repeated calibration, not unique candidates'}


def controlled_mapping(cues,av,bins,part,seed,catalog,catalog_av,catalog_bins):
    """Keep zero-availability requests; randomize usable cues label-free."""
    cues=np.asarray(cues);av=np.asarray(av);bins=np.asarray(bins);part=np.asarray(part)
    if cues.ndim!=1 or any(v.shape!=cues.shape for v in (av,bins,part)):
        raise ValueError('controlled_mapping_alignment')
    if any(np.any(~np.isfinite(v)) or np.any(v!=np.floor(v)) or np.any(v<0)
           for v in (cues,av,bins,part)) or np.any(av>15):
        raise ValueError('controlled_mapping_finite_integer_metadata')
    if np.any((cues<3)&(cues!=1)) or np.any((cues==1)&(av!=0)):
        raise ValueError('controlled_mapping_invalid_special_cue')
    eligible=(cues>=3)&(av!=0);selected=np.flatnonzero(eligible)
    donor=cues.astype(np.int64,copy=True);source=np.full(len(cues),-1,np.int64)
    tier=np.full(len(cues),4,np.uint8)  # 4: unchanged, no usable cue content.
    if len(selected):
        d,s,t=select_donors(cues[selected],av[selected],bins[selected],part[selected],seed,
                          catalog,catalog_av,catalog_bins)
        if np.any(d<0):raise ValueError('wrong_cue_control_insufficient_for_usable_cue')
        donor[selected]=d;tier[selected]=t
        from_request=s>=0;source[selected[from_request]]=selected[s[from_request]]
    return donor,source,tier,eligible

def verify_unchanged_rows(x,offsets,eligible):
    """Bounded row checks retain full candidate arrays in place."""
    for row in np.flatnonzero(~eligible):
        lo,hi=offsets[row:row+2]
        if np.any(np.asarray(x[lo:hi])[:,[12,13,14,16,19,21,24,27]]!=0):
            raise ValueError('unavailable_cue_has_nonzero_matching_or_availability')

def build_treatments(project,dataset,seed,train_meta,val_meta,train_x,val_x,
                     train_manifest,val_manifest,context_root=None):
    """Construct isolated, label-free donors against hash-bound A inputs.

    Popularity bins use prepared Train-event counts for BOTH surfaces. These
    counts are fixed before C, not derived from any evaluation target. The
    existing OOF feature/index counts are unchanged and are never replaced.
    """
    import hashlib,json,time
    from pathlib import Path
    from scipy import sparse
    from arrive_framework_datasets import bundle,metadata_args,file_hash
    from run_music4all_artist_component_b import load_metadata
    from run_music4all_candidate_support_a1 import user_partition
    began=time.monotonic();ds=bundle(project,dataset);prepared=Path(ds['data_dir'])
    receipts={}
    if dataset=='music4all_onion':
        if context_root is None:raise ValueError('wrong_cue_context_root_required')
        receipts={s:json.loads((Path(context_root)/f's{seed}'/s/'REQUEST_CONTEXT.json').read_text())
                  for s in ('train_cv','validation')}
    inputs={}
    for name in ('events.npy','sessions.npy','track_identity.tsv','train_frequencies.npy'):
        actual=file_hash(prepared/name)
        for surface,m in (('train_cv',train_manifest),('validation',val_manifest)):
            expected=(receipts[surface] if receipts else m)['prepared_sha256'][name]
            if actual!=expected:raise ValueError('wrong_cue_prepared_hash:'+name)
        inputs[name]=actual
    if receipts:
        for surface in receipts:
            for name,expected in receipts[surface]['content_sha256'].items():
                p=Path(ds['content_dir'])/name
                if file_hash(p)!=expected:raise ValueError('wrong_cue_content_hash:'+name)
                inputs[name]=expected
            for key,field in (('tags','tag_sha256'),('information','artist_sha256')):
                p=Path(ds[key]);actual=file_hash(p)
                if actual!=receipts[surface][field]:raise ValueError('wrong_cue_metadata_hash:'+key)
                inputs[key]=actual
    else:
        for key,field in (('content_audio','content_audio'),('content_genre','content_genre'),
                          ('artist_mapping','artist_mapping')):
            actual=file_hash(ds[key])
            if actual!=val_manifest['serving_inputs'][field]:raise ValueError('wrong_cue_content_hash:'+key)
            inputs[key]=actual
    audio=np.load(ds['content_audio'],mmap_mode='r',allow_pickle=False)
    genre=sparse.load_npz(ds['content_genre']).tocsr()
    tags,artists,_=load_metadata(metadata_args(ds,prepared),len(audio))
    # Same conversion as framework_export: Yambda metadata returns dense audio
    # as its legacy tag representation; Onion already returns CSR.
    tags=sparse.csr_matrix(tags)
    av=availability_codes(artists,tags,audio,genre)
    freq=np.load(prepared/'train_frequencies.npy',mmap_mode='r',allow_pickle=False)
    if freq.shape!=(len(av),) or not np.isfinite(freq).all() or np.any(freq<0):
        raise ValueError('wrong_cue_Train_frequency_shape')
    bins=np.floor(np.log2(1+freq)).astype(np.int64)
    catalog=np.flatnonzero((freq>0)&(np.arange(len(freq))>=3))
    sessions=np.load(prepared/'sessions.npy',mmap_mode='r',allow_pickle=False)
    events=np.load(prepared/'events.npy',mmap_mode='r',allow_pickle=False)
    objects=[];reports={};mappings={}
    for surface,meta,x,split in (('train_cv',train_meta,train_x,0),('validation',val_meta,val_x,1)):
        rows=meta['row_ids'];users=meta['users']
        if not np.array_equal(sessions[rows,0],users) or not np.all(sessions[rows,5]==split):
            raise ValueError('wrong_cue_row_user_split')
        cues=np.asarray(events[sessions[rows,2]],dtype=np.int64)
        if np.any((cues<3)&(cues!=1)) or np.any(cues>=len(av)):raise ValueError('wrong_cue_actual_identity')
        raw=np.asarray(x[meta['offsets'][:-1]][:,[19,21,24,27]])
        if not np.isin(raw,(0,1)).all() or not np.array_equal(raw.astype(np.uint8)@np.array([1,2,4,8],np.uint8),av[cues]):
            raise ValueError('wrong_cue_actual_availability')
        part=(~user_partition(users)).astype(np.uint8) if split==0 else np.zeros(len(rows),np.uint8)
        donor,source,tier,eligible=controlled_mapping(cues,av[cues],bins[cues],part,seed,catalog,av[catalog],bins[catalog])
        verify_unchanged_rows(x,meta['offsets'],eligible)
        if np.any(av[donor]!=av[cues]):raise ValueError('wrong_cue_donor_availability')
        unique,counts=np.unique(donor[eligible],return_counts=True)
        reports[surface]={'rows':len(rows),'eligible_rows':int(eligible.sum()),
            'unchanged_rows':int((~eligible).sum()),'unk_rows':int((cues==1).sum()),
            'known_zero_availability_rows':int(((cues>=3)&(~eligible)).sum()),
            'unchanged_fraction':float(np.mean(~eligible)),
            'unresolved':0,'self_cue_rows':int(((donor==cues)&eligible).sum()),
            'tier_counts':{str(k):int((tier==k).sum()) for k in range(5)},
            'partition_counts':{str(k):int((part==k).sum()) for k in np.unique(part)},
            'distinct_donor_cues':len(unique),'max_donor_reuse':int(counts.max()) if len(counts) else 0,
            'donor_quality_scope':'eligible_rows_only',
            'same_popularity_bin_fraction':float(np.mean(bins[donor[eligible]]==bins[cues[eligible]])) if eligible.any() else None,
            'mean_popularity_bin_distance':float(np.mean(np.abs(bins[donor[eligible]]-bins[cues[eligible]]))) if eligible.any() else None,
            'donor_mapping_sha256':hashlib.sha256(donor.astype('<i8').tobytes()).hexdigest(),
            'source_rows_sha256':hashlib.sha256(source.astype('<i8').tobytes()).hexdigest()}
        objects.append(CueTreatment(donor,artists,tags,audio,genre,eligible))
        mappings[surface+'_row_ids']=rows;mappings[surface+'_donors']=donor
        mappings[surface+'_source_rows']=source;mappings[surface+'_tiers']=tier
        mappings[surface+'_eligible']=eligible;mappings[surface+'_actual_cues']=cues
    report={'schema':'arrive.controlled_wrong_cue.v2','status':'pass','dataset':dataset,'seed':seed,
        'frequency_basis':'prepared Train-event train_frequencies.npy; existing OOF features unchanged',
        'rule':'usable cues: same availability, same bin then nearest bin (lower on ties), Train-known catalog fallback; zero-availability cues retain original zero matching; all requests retained',
        'unknown_cue_policy':'retain_zero_unavailable_v1',
        'inputs_sha256':inputs,'reports':reports,'selector_reads_targets':False,'test_accessed':False,
        'helper_sha256':file_hash(Path(__file__)),'preparation_seconds':time.monotonic()-began}
    return objects[0],objects[1],report,mappings
