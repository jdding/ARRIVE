#!/usr/bin/env python3
"""Shared bounded Train/Validation controls: ordinary fusion and refitted minus-STC."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import lightgbm as lgb
from scipy import sparse

from lfm_timeconcat_control import LFMTimeConcatData, exact_catalog_topk
from run_music4all_candidate_support_a1 import atomic_json, atomic_npz, digest, stable_topk, load_standard_base, target_lookup
from run_music4all_candidate_support_a0 import ranked_sparse_row, ranked_history, recent_anchors
from run_music4all_artist_component_b import load_artist_features, load_metadata, artist_members
from run_music4all_component_c2_validation import frozen_predecessor
from analyze_music4all_component_d_content_conversion import build_features, source_contexts, split_calibration
from analyze_music4all_component_d_content_validation import metric_table
from analyze_music4all_component_d_selective_exchange import row_metrics, user_macro
from run_music4all_component_d_joint_content_validation import fit_ranker, choose_alpha, blended_top10
from run_arrive_independent_mig import batched_values

ROOT = Path('runs/arrive_repair_20260912/arrive_independent')
GRID = [(220, 31), (220, 63), (440, 31), (440, 63)]
ALPHAS = (0., .05, .1, .2, .5, 1., 2., 'tree_only')
FEATURE_NAMES = ['encoder_score_z','rrf_score','encoder_rank','first_track_transition_rank',
                 'history_transition_rank','artist_rank','log_popularity','log_gap',
                 'gap_30_90','gap_90_180','gap_180_365','gap_gt365',
                 'same_artist','tag_cosine_first','audio_cosine_first','audio_cosine_history',
                 'genre_cosine_first','genre_cosine_history','artist_known','first_artist_known',
                 'audio_available','first_audio_available','history_audio_available',
                 'genre_available','first_genre_available','history_genre_available',
                 'tag_available','first_tag_available']


def rrf_pool(sources, budget=200, k=60):
    """Pure inference inputs, deterministic ties, protects exact encoder Top-10."""
    maps = [{int(item): at + 1 for at, item in enumerate(source) if int(item) >= 3} for source in sources]
    universe = sorted(set().union(*maps))
    score = {i: sum(1. / (k + m[i]) for m in maps if i in m) for i in universe}
    protected = list(map(int, sources[0][:10]))
    ranked = sorted((i for i in universe if i not in protected), key=lambda i: (-score[i], i))
    pool = protected + ranked[:budget-len(protected)]
    if len(pool) != budget or len(set(pool)) != budget:
        raise ValueError('rrf_budget_or_identity_invalid')
    ranks = np.array([[m.get(i, 0) for m in maps] for i in pool], np.float32)
    return np.array(pool, np.int32), np.array([score[i] for i in pool], np.float32), ranks


def actual_truth(data, data_dir, surface, rows, counts):
    lookup = None
    if surface != 'train_cv':
        lookup = target_lookup(data_dir)
    result = np.full((len(rows), int(counts.max())), -1, np.int64)
    for at, row in enumerate(rows):
        values = list(dict.fromkeys(data.known_suffix(int(row)))) if surface == 'train_cv' else list(lookup[int(row)])
        if len(values) != counts[at]:
            raise ValueError('full_target_denominator_mismatch')
        result[at, :len(values)] = values
    return result


def load_raw(args, surface):
    raw, manifest = load_artist_features(args.features_root/f'{surface}_features.npz', args.features_root/f'{surface}_features_manifest.json', surface)
    sessions = np.load(args.data_dir/'sessions.npy', mmap_mode='r')
    split_code = {'train_cv': 0, 'validation': 1, 'test': 2}.get(surface)
    if split_code is None:
        raise ValueError('unknown_surface')
    if not np.all(sessions[raw['row_ids'],5] == split_code):
        raise ValueError('split_mismatch')
    return raw, manifest


@torch.no_grad()
def construct(args, surface, audio, genre, metadata):
    raw, manifest = load_raw(args, surface)
    data = LFMTimeConcatData(args.data_dir, max_context=129, horizon=64)
    pred = {k: raw[k] for k in ('row_ids','users','gaps','bucket_ids','target_counts')}
    bindings = {'raw_features': manifest['feature_sha256']}
    if args.arm == 'without_stc':
        candidates, scores, top, _, _, bindings = frozen_predecessor(args, raw, surface)
        pred.update(candidates=candidates, scores=scores)
        pred['targets'] = actual_truth(data, args.data_dir, surface, pred['row_ids'], pred['target_counts'])
        pools, matrices, labels = build_features(pred, {'method_top10':top}, raw, audio, genre,
                                                 source_contexts(pred,args.data_dir,audio,genre),np.arange(len(top)),50)
    else:
        tags, artists = metadata
        cross = ROOT/'carrier_crossfit'/f's{args.seed}'
        full = ROOT.parent/'sasrec/formal'/f's{args.seed}'
        directory = cross if surface == 'train_cv' else full
        checkpoint = directory/('final_model.pt' if surface == 'train_cv' else 'best_model.pt')
        model, bm = load_standard_base(args,data,checkpoint_path=checkpoint,manifest_path=directory/'training_manifest.json')
        if bm['binding']['configuration']['seed'] != args.seed or digest(checkpoint) != manifest['base_checkpoint_sha256']:
            raise ValueError('encoder_identity_mismatch')
        bindings.update(encoder_sha256=digest(checkpoint), encoder_manifest_sha256=digest(directory/'training_manifest.json'))
        suffix = 'fit' if surface == 'train_cv' else 'full'
        index = Path('data/music4all_onion/a0_candidate_index_v1')
        im = json.loads((index/'index_manifest.json').read_text())
        for name in (f'x_{suffix}.npz',f'h_{suffix}.npz',f'{suffix}_target_frequency.npy'):
            if digest(index/name) != im['artifacts'][name]['sha256']:
                raise ValueError('index_hash_mismatch')
        bindings['index_manifest_sha256'] = digest(index/'index_manifest.json')
        frequency = np.load(index/f'{suffix}_target_frequency.npy')
        x = sparse.load_npz(index/f'x_{suffix}.npz').tocsr()
        h = sparse.load_npz(index/f'h_{suffix}.npz').tocsr()
        members = artist_members(artists,frequency)
        ao,ah,go,gh = source_contexts(pred,args.data_dir,audio,genre)
        pools,matrices,scores_all = [],[],[]
        xcache = {}
        for start in range(0,len(raw['row_ids']),128):
            ids = raw['row_ids'][start:start+128]
            batch = data.serving_batch([data.context(int(r)) for r in ids],[data.gap_days(int(r)) for r in ids],torch.device(args.device))
            hidden = model.encode_all(batch['sequence'],batch['gap_days'])
            states = model.gather_states(hidden,batch['state_positions'])
            base = exact_catalog_topk(model,states,k=200).cpu().numpy()
            local = []
            for off,rid in enumerate(ids):
                row = start+off
                session = data.sessions[int(rid)]
                first = int(data.events[int(session[2])])
                if first not in xcache:
                    xcache[first] = ranked_sparse_row(x,first,frequency,0.,200)
                hist = ranked_history(h,recent_anchors(data.events,session,5),frequency,0.,200)
                art = [i for i in members.get(int(artists[first]),[]) if i != first][:20]
                pool,rrf,ranks = rrf_pool([base[off],xcache[first],hist,art])
                local.append(pool)
                rr = np.where(ranks>0,1./np.log2(np.maximum(ranks,1)+1),0.)
                audio_c = np.asarray(audio[pool])
                gc = genre[pool]
                tc = tags[pool]
                features = np.column_stack([np.zeros(200),rrf,rr,
                    np.log1p(data.frequencies[pool]),np.full(200,np.log1p(raw['gaps'][row])/8),
                    np.tile(np.eye(4)[raw['bucket_ids'][row]],(200,1)),
                    (artists[pool]==artists[first])&(artists[first]>=0),
                    tc.dot(tags[first].T).toarray().ravel(),audio_c@ao[row],audio_c@ah[row],
                    gc.dot(go[row].T).toarray().ravel(),gc.dot(gh[row].T).toarray().ravel(),
                    artists[pool]>=0,np.full(200,artists[first]>=0),
                    np.linalg.norm(audio_c,axis=1)>0,np.full(200,np.linalg.norm(ao[row])>0),
                    np.full(200,np.linalg.norm(ah[row])>0),gc.getnnz(axis=1)>0,
                    np.full(200,go[row].nnz>0),np.full(200,gh[row].nnz>0),
                    tc.getnnz(axis=1)>0,np.full(200,tags[first].nnz>0)]).astype(np.float32)
                matrices.append(features)
            cand = torch.as_tensor(np.array(local),device=args.device,dtype=torch.long)
            values = torch.einsum('bd,bkd->bk',states,model.item_embedding(cand)).cpu().numpy()
            for off,score in enumerate(values):
                matrices[start+off][:,0] = (score-score.mean())/max(float(score.std()),1e-6)
            pools.extend(local); scores_all.extend(values)
            if start == 0 or (start//128)%40==0:
                print(json.dumps({'event':'fusion_inputs','surface':surface,'rows':min(start+128,len(raw['row_ids']))}),flush=True)
        pred.update(candidates=np.asarray(pools),scores=np.asarray(scores_all))
        top = stable_topk(pred['scores'],pred['candidates'])
        pred['targets'] = actual_truth(data,args.data_dir,surface,pred['row_ids'],pred['target_counts'])
        labels = [np.isin(p,pred['targets'][r,:pred['target_counts'][r]]).astype(np.uint8) for r,p in enumerate(pools)]
        del model
        if args.device == 'cuda': torch.cuda.empty_cache()
    if any(not np.isfinite(m).all() for m in matrices):
        raise ValueError('nonfinite_input')
    print(json.dumps({'event':'inputs_complete','arm':args.arm,'surface':surface,'rows':len(pools),'features':matrices[0].shape[1]}),flush=True)
    return pools,matrices,labels,pred,top,bindings


def generic_fit(matrices,labels,selected,seed,trees,leaves):
    rows = np.flatnonzero(selected)
    model = lgb.LGBMRanker(objective='lambdarank',metric='ndcg',eval_at=(10,),label_gain=(0,1),
        n_estimators=trees,num_leaves=leaves,learning_rate=.035,min_child_samples=80,
        subsample=.8,subsample_freq=1,colsample_bytree=.9,reg_lambda=1.,random_state=seed,n_jobs=8,verbosity=-1)
    model.fit(np.concatenate([matrices[r] for r in rows]),np.concatenate([labels[r] for r in rows]),
              group=[len(labels[r]) for r in rows],callbacks=[lambda env: print(json.dumps({'event':'tree','iteration':env.iteration+1,'trees':trees,'leaves':leaves}),flush=True) if (env.iteration+1)%100==0 else None])
    return model


def selected_output(pools,matrices,values,alpha,top):
    if alpha == 'tree_only':
        return np.array([p[np.lexsort((p,-v))[:10]] for p,v in zip(pools,values)],np.int32)
    return blended_top10(pools,matrices,values,np.ones(len(top),bool),float(alpha),top)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',required=True,choices=('without_stc','ordinary_fusion'))
    ap.add_argument('--seed',type=int,choices=(2024,2025,2026),required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    ap.add_argument('--data-dir',type=Path,default=Path('data/music4all_onion/prepared_v1'))
    ap.add_argument('--content-dir',type=Path,default=Path('data/music4all_onion/component_b2_attainability_v1'))
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError('control_output_exists')
    torch.set_num_threads(4)
    args.rcce_dir=ROOT/'component1'/f's{args.seed}'
    args.features_root=args.rcce_dir/'artist_features'
    args.eval_batch_size=256
    started=time.monotonic()
    audio=np.load(args.content_dir/'musicnn_normalized.npy',mmap_mode='r')
    genre=sparse.load_npz(args.content_dir/'genre_normalized.npz').tocsr()
    metadata=None
    if args.arm=='ordinary_fusion':
        meta_args=SimpleNamespace(data_dir=args.data_dir,tags=Path('data/music4all_onion/raw/id_tags.csv'),information=Path('data/music4all_onion/raw/id_information.csv'))
        tags,artists,_=load_metadata(meta_args,len(audio)); metadata=(tags,artists)
    pools,matrices,labels,pred,top,bindings=construct(args,'train_cv',audio,genre,metadata)
    calibration=split_calibration(pred['users'],99)
    if args.arm=='without_stc':
        model=fit_ranker(matrices,labels,~calibration,slice(None),40001+args.seed)
        chosen,grid=choose_alpha(model,pools,matrices,slice(None),{'method_top10':top},pred,calibration)
        model=fit_ranker(matrices,labels,np.ones(len(top),bool),slice(None),41001+args.seed)
    else:
        grid=[]
        for trees,leaves in GRID:
            model=generic_fit(matrices,labels,~calibration,40001+args.seed,trees,leaves)
            # Predict only selection queries; never evaluate Validation during fitting.
            indices=np.flatnonzero(calibration)
            pp=[pools[i] for i in indices]; mm=[matrices[i] for i in indices]
            values=batched_values(model,mm)
            for alpha in ALPHAS:
                output=selected_output(pp,mm,values,alpha,top[indices])
                r,n=row_metrics(output,pred['targets'][indices],pred['target_counts'][indices])
                u=pred['users'][indices]; long=pred['bucket_ids'][indices]==3
                objective=sum(user_macro(v,u,mask) for v in (r,n) for mask in (np.ones(len(u),bool),long))
                grid.append(dict(trees=trees,leaves=leaves,alpha=alpha,objective=objective))
            print(json.dumps({'event':'selection_fit_complete','trees':trees,'leaves':leaves}),flush=True)
        chosen=max(grid,key=lambda r:(r['objective'],-r['trees'],-r['leaves']))
        model=generic_fit(matrices,labels,np.ones(len(top),bool),41001+args.seed,chosen['trees'],chosen['leaves'])
    args.output.mkdir(parents=True)
    model.booster_.save_model(str(args.output/'model.txt'))
    restored=SimpleNamespace(booster_=lgb.Booster(model_file=str(args.output/'model.txt')))
    probe=np.concatenate(matrices[:8])
    np.testing.assert_array_equal(model.booster_.predict(probe,num_threads=8),restored.booster_.predict(probe,num_threads=8))
    train_seconds=time.monotonic()-started
    manifest={'schema':'arrive.attribution_control.v1','status':'frozen_before_validation','arm':args.arm,'seed':args.seed,
              'selection':chosen,'selection_grid':grid,'model_sha256':digest(args.output/'model.txt'),
              'train_bindings':bindings,'test_accessed':False,'fit_scope':'train_cv_only','selection_scope':'hash_user_train_internal',
              'feature_names':FEATURE_NAMES if args.arm=='ordinary_fusion' else 'unchanged_MIG_17_features',
              'candidate_budget':200,'ranking_depth':200 if args.arm=='ordinary_fusion' else 50,
              'producer_sha256':digest(Path(__file__)),'training_seconds':train_seconds,
              'audio_sha256':digest(args.content_dir/'musicnn_normalized.npy'),'genre_sha256':digest(args.content_dir/'genre_normalized.npz'),
              'shared_code_sha256':{p:digest(Path(__file__).with_name(p)) for p in ('run_arrive_independent_mig.py','run_music4all_component_d_joint_content_validation.py','analyze_music4all_component_d_content_conversion.py','run_music4all_component_c2_validation.py')}}
    atomic_json(args.output/'MODEL_MANIFEST.json',manifest)
    print(json.dumps({'event':'model_frozen','arm':args.arm,'selection':chosen}),flush=True)
    del matrices,labels,pools,model
    pools,matrices,labels,pred,top,bindings=construct(args,'validation',audio,genre,metadata)
    output=selected_output(pools,matrices,batched_values(restored,matrices),chosen['alpha'],top)
    metrics,arrays=metric_table(output,top,pred,60000+args.seed)
    path=args.output/'validation_predictions.npz'
    atomic_npz(path,row_ids=pred['row_ids'],users=pred['users'],bucket_ids=pred['bucket_ids'],
               top10=output,base_top10=top,candidates=pred['candidates'],targets=pred['targets'],target_counts=pred['target_counts'],**arrays)
    result={'status':'pass','arm':args.arm,'seed':args.seed,'metrics':metrics,'selection':chosen,
            'predictions_sha256':digest(path),'model_manifest_sha256':digest(args.output/'MODEL_MANIFEST.json'),
            'validation_bindings':bindings,'seconds':time.monotonic()-started,'test_accessed':False,'inference_uses_saved_model':True}
    atomic_json(args.output/'VALIDATION_RESULT.json',result)
    print(json.dumps({'event':'control_complete','arm':args.arm,'seed':args.seed,'seconds':result['seconds']}),flush=True)


if __name__=='__main__': main()
