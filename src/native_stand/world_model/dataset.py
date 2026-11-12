"""Episode/parameter-tuple isolation and strict transition provenance."""
from dataclasses import fields
from pathlib import Path
import json
import numpy as np
import torch
from . import STATE_VERSION
from .state import State


UNIFORM_MECHANICS='uniform_episode_v1'
CONTACT_MECHANICS='contact_event25_v1'


def contact_event_starts(contact,window):
    """TRAIN label sampling only; never a feature given to WM/Actor.

    Either wheel <=2N for >=5 consecutive1ms labels, expanded20ms each side.
    Starts before100ms are excluded from this added quota; the uniform75% still
    includes initialization. Each start must have a full consecutive window.
    """
    contact=np.asarray(contact)
    if contact.ndim!=2 or contact.shape[1]!=2 or not np.isfinite(contact).all() or (contact<0).any():
        raise ValueError('finite nonnegative two-wheel1ms force labels required')
    if type(window) is not int or window<1:raise ValueError('positive complete window required')
    stop=len(contact)-window+1
    if stop<=0:return np.empty(0,dtype=np.int64)
    lost=(contact<=2).any(-1);marked=np.zeros(len(contact),dtype=bool)
    edges=np.flatnonzero(np.diff(np.r_[False,lost,False].astype(np.int8)))
    for start,end in edges.reshape(-1,2):
        if end-start>=5:marked[max(0,start-20):min(len(marked),end+20)]=True
    marked[:100]=False
    return np.flatnonzero(marked[:stop])


def mechanics_indices(data,n,rng,*,episode_pool,window,sampling=UNIFORM_MECHANICS):
    """Preserve bootstrap multiplicities and the historical uniform draw order."""
    if sampling not in (UNIFORM_MECHANICS,CONTACT_MECHANICS):raise ValueError('unknown mechanics sampler version')
    if type(n) is not int or n<1:raise ValueError('positive batch required')
    pool=list(episode_pool)
    if not pool or any(type(e) not in (int,np.int64,np.int32) or e<0 or e>=len(data) for e in pool):
        raise ValueError('explicit member episode pool required')
    if any(len(data[e]['micro_q'])<window for e in pool):raise ValueError('bootstrap pool lacks complete window')
    event_count=n//4 if sampling==CONTACT_MECHANICS else 0
    if sampling==CONTACT_MECHANICS and not event_count:raise ValueError('contact25 batch requires at least4 samples')
    uniform_n=n-event_count
    selected=[int(rng.choice(pool)) for _ in range(uniform_n)]
    ids=[int(rng.integers(len(data[e]['micro_q'])-window+1)) for e in selected]
    if event_count:
        events={e:contact_event_starts(data[e]['micro_contact'],window) for e in set(pool)}
        event_pool=[e for e in pool if len(events[e])]
        if not event_pool:raise ValueError('member has no train contact events; cannot silently replace declared quota')
        extra=[int(rng.choice(event_pool)) for _ in range(event_count)]
        selected+=extra;ids += [int(rng.choice(events[e])) for e in extra]
    return selected,ids,event_count


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False)


def write_episode(folder, case, rows, micro, blobs, geometry, budget=None):
    from .real import sha
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=False)
    if not rows or len(micro) != len(rows)*10:
        raise ValueError('complete1ms substeps for every10ms policy transition required')
    arrays = {}
    for side in ('before','after'):
        for f in fields(State):
            arrays[f'{side}_{f.name}'] = torch.cat([getattr(r[side],f.name) for r in rows]).numpy()
    for key in ('action','observation','reward','terminated','truncated','reward_raw','reward_density','reward_dt'):
        arrays[key] = torch.cat([r[key] for r in rows]).numpy()
    for key in micro[0]:
        arrays['micro_'+key] = np.stack([r[key] for r in micro])
    arrays['integration_state'] = np.stack(blobs)
    if budget is not None:budget.storage_guard(sum(v.nbytes for v in arrays.values())+1024**2)
    with (folder/'transitions.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    record = dict(version=STATE_VERSION,episode_id=case['id'],case=case,split=case['split'],
        tuple_id=case['tuple_id'],transitions=len(rows),physics_steps=len(micro),
        terminal=bool(arrays['terminated'][-1]),truncated=bool(arrays['truncated'][-1]),
        pre_action=True,next_state='post-physics/pre-reset',reset_labels=False,
        reward_fields=list(rows[0]['reward_fields']),geometry=geometry,
        trace_sha256=sha(folder/'transitions.npz'))
    write_json(folder/'episode.json',record)
    return record


def build_manifest(episodes, output, identities):
    from .real import sha
    rows, ids, tuples, physical_tuples, geometry = [], set(), {}, {}, None
    plants=set()
    for folder in map(Path,episodes):
        p=folder/'episode.json'; r=json.loads(p.read_text())
        if r['version'] != STATE_VERSION or not r['pre_action'] or r['reset_labels']:
            raise ValueError('complete causal Markov episode schema required; legacy31D traces cannot be upgraded by guessing')
        if r['split'] not in ('train','validation','test') or r['episode_id'] in ids:
            raise ValueError('episode split absent or duplicated')
        if sha(folder/'transitions.npz') != r['trace_sha256']:
            raise ValueError('episode data SHA differs')
        previous = tuples.setdefault(r['tuple_id'],r['split'])
        if previous != r['split']:
            raise ValueError('execution/initialization tuple leaks across episode splits')
        case=r['case']
        if case.get('plant_identity') is not None:
            plants.add(case['plant_identity'])
            if len(plants)>1:raise ValueError('different solver/plant identities cannot be mixed as the same mechanics')
            numeric={k:v for k,v in case['uncertainty'].items() if k not in ('name','family')}
            key=json.dumps(dict(uncertainty=numeric,yaw=case['yaw']),sort_keys=True)
            if physical_tuples.setdefault(key,r['split'])!=r['split']:
                raise ValueError('physical execution/initialization tuple leaks despite different tuple labels')
        if geometry is None: geometry=r['geometry']
        elif geometry!=r['geometry']: raise ValueError('different assets/kinematic contexts need explicit versioned mixture')
        with np.load(folder/'transitions.npz',allow_pickle=False) as z:
            t=r['transitions']
            if z['before_q'].shape[0] != t or z['micro_q'].shape[0]!=10*t:
                raise ValueError('incomplete transition labels')
            if z['terminated'][:-1].any() or z['truncated'][:-1].any():
                raise ValueError('reset/terminal transition spliced into normal dynamics')
            if not bool(z['terminated'][-1] or z['truncated'][-1]):
                raise ValueError('complete episode terminal/timeout label absent')
            if not np.array_equal(z['after_tick'],z['before_tick']+10):raise ValueError('physics/policy clock labels differ')
            if t>1 and not np.array_equal(z['after_q'][:-1],z['before_q'][1:]):
                raise ValueError('non-contiguous physical labels')
            for k in z.files:
                if not np.isfinite(z[k]).all(): raise ValueError('nonfinite dataset field '+k)
        ids.add(r['episode_id'])
        rows.append(dict(path=str(folder.resolve()),manifest_sha256=sha(p),**{k:r[k] for k in
            ('episode_id','split','tuple_id','transitions','trace_sha256','terminal','physics_steps')}))
    if not rows: raise ValueError('no complete Markov episodes; collect new1ms state instead of inferring missing queues')
    result=dict(version=STATE_VERSION,episodes=rows,geometry=geometry,identities=identities,
        split_unit='whole episode AND execution/initialization tuple',normalization='train episodes only',
        tasks=['Motion'],plant_identities=sorted(plants),Recovery='NOT_COVERED',Jump='NOT_COVERED_NO_FLIGHT_OR_LANDING_DATA')
    write_json(output,result)
    return result


class Replay:
    def __init__(self, manifest, split='train', *, permit_test=False):
        from .real import sha
        if split=='test' and not permit_test:
            raise ValueError('final test episodes cannot select/tune models')
        self.manifest=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest
        self.split=split; self.rows=[r for r in self.manifest['episodes'] if r['split']==split]
        if not self.rows: raise ValueError('requested episode split has no data')
        self.data=[];self.cases=[]
        for r in self.rows:
            p=Path(r['path']); ep=json.loads((p/'episode.json').read_text())
            if sha(p/'episode.json')!=r['manifest_sha256'] or sha(p/'transitions.npz')!=r['trace_sha256']:
                raise ValueError('replay evidence changed')
            if ep['version']!=STATE_VERSION: raise ValueError('state schema differs')
            self.cases.append(ep['case'])
            with np.load(p/'transitions.npz',allow_pickle=False) as z:
                self.data.append({k:z[k] for k in z.files})

    def state(self, episode, indices, side='before', device='cpu'):
        z=self.data[episode]
        return State(**{f.name:torch.as_tensor(z[f'{side}_{f.name}'][indices],device=device) for f in fields(State)})

    def batch(self, n, rng, *, device='cpu', horizon=1, actor_only=False):
        # Equal episode draws preserve rare failures; no neighboring-frame split.
        choices=[]
        for ep,z in enumerate(self.data):
            stop=len(z['reward'])-horizon+1
            if stop<=0: continue
            valid=np.arange(stop)
            if actor_only: valid=valid[z['before_owner'][:stop]==1]
            if len(valid): choices.append((ep,valid))
        if not choices: raise ValueError('no contiguous valid starts for requested horizon')
        selected=[]
        for _ in range(n):
            ep,valid=choices[int(rng.integers(len(choices)))]; selected.append((ep,int(rng.choice(valid))))
        before=State.cat([self.state(ep,slice(i,i+1),device=device) for ep,i in selected])
        after=State.cat([self.state(ep,slice(i,i+1),'after',device) for ep,i in selected])
        tensors={k:torch.as_tensor(np.stack([self.data[ep][k][i] for ep,i in selected]),device=device)
                 for k in ('reward','terminated','truncated','action')}
        return before,after,tensors,selected

    def for_controllers(self,names,checkpoint_sha=None):
        import copy
        selected=[i for i,c in enumerate(self.cases) if c.get('controller') in names and
                  (checkpoint_sha is None or c.get('controller_sha256')==checkpoint_sha)]
        if not selected:raise ValueError('real current-policy Critic episodes absent')
        result=copy.copy(self)
        result.rows=[self.rows[i] for i in selected];result.data=[self.data[i] for i in selected]
        result.cases=[self.cases[i] for i in selected]
        return result

    def mechanics(self,n,rng,*,device='cpu',episode_pool=None,window=1,sampling=UNIFORM_MECHANICS):
        if sampling!=UNIFORM_MECHANICS and self.split!='train':
            raise ValueError('contact-focused sampling is train-only; validation/test distribution is fixed')
        pool=list(range(len(self.data))) if episode_pool is None else episode_pool
        selected,ids,_=mechanics_indices(self.data,n,rng,episode_pool=pool,window=window,sampling=sampling)
        keys=('q','v','warm','applied','next_q','next_v','next_warm','contact','body_contact')
        result={k:torch.as_tensor(np.stack([self.data[e]['micro_'+k][i] if window==1 else
                     self.data[e]['micro_'+k][i:i+window] for e,i in zip(selected,ids)]),
                                  dtype=torch.float32,device=device) for k in keys}
        result['context']=torch.as_tensor(np.stack([self.data[e]['before_context'][0] for e in selected]),device=device)
        return result
