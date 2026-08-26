"""Core metrics for a run, computed identically across runs.

The ``NOTES.md`` core-metrics table is produced from this module, so two runs'
tables are always comparable. Add a metric here, never in a one-off script.

    python -m evollm.analysis.core_metrics runs/<run> [runs/<comparison>]
"""
import json, glob, os, numpy as np
from collections import defaultdict, Counter
from evollm.analysis import (build_phenotypes, Pedigree, load_fingerprints,
                             parent_offspring_concordance)
from evollm.analysis.suite import effective_number

ROOMS = {'gpu0','gpu1','gpu2','gpu3'}
PH = {'room_id','agent_id','sender_id','your_id','text','room','agent','sender','you'}

def regress(x, y):
    x, y = np.asarray(x,float), np.asarray(y,float)
    m = np.isfinite(x)&np.isfinite(y); x,y = x[m],y[m]
    if len(x) < 40 or x.std() == 0: return None
    b,_ = np.polyfit(x,y,1); r = np.corrcoef(x,y)[0,1]
    se = np.sqrt((1-r**2)/(len(x)-2))*(y.std()/x.std())
    return float(b), float(se), int(len(x))

def core(run):
    R = f"runs/{run}"
    out = {}
    last={}; b=de=0; mg=0; occ=defaultdict(list); deaths=[]; origins=Counter()
    turns=defaultdict(list); ph=Counter(); mv=Counter()
    for f in glob.glob(f"{R}/events/*.jsonl"):
        room = os.path.basename(f).split('.')[0]
        for line in open(f):
            e=json.loads(line); s=e.get("step",0); t=e["type"]
            last[room]=max(last.get(room,0),s)
            if t=="birth": b+=1; mg=max(mg,e.get("generation",0)); origins[e.get("origin","?")]+=1
            elif t=="death": de+=1; deaths.append(e)
            elif t=="occupancy": occ[e["room"]].append(e)
            elif t=="move": mv[e['agent']]+=1
            elif t=="move_failed":
                a=e['agent']; mv[a]+=1
                if e.get('to')!=room and e.get('to') not in ROOMS: ph[a]+=1
            elif t=="turn": turns[e["agent"]].append((s, e.get("form")))
    rs=sum(last.values())
    out["rooms"]=len(occ); out["room_steps"]=rs; out["per_room_final"]=dict(sorted(last.items()))
    out["births"]=b; out["deaths"]=de; out["maxgen"]=mg; out["origins"]=dict(origins)
    out["births_per_1k_rs"]=b/max(rs,1)*1000
    ag=[];cx=[];agf=[];cxf=[]
    for r,v in occ.items():
        v=sorted(v,key=lambda x:x["step"])
        ag += [x["agents"] for x in v]; cx += [x["mean_context"] for x in v]
        agf.append(v[-1]["agents"]); cxf.append(v[-1]["mean_context"])
    out["agents_mean"]=float(np.mean(ag)); out["agents_final"]=float(np.mean(agf))
    out["ctx_mean"]=float(np.mean(cx));   out["ctx_final"]=float(np.mean(cxf))
    # context at death vs room mean
    rel=[]
    for e in deaths:
        v=occ.get(e["room"])
        if not v: continue
        st=np.array([x["step"] for x in v]); mc=np.array([x["mean_context"] for x in v])
        m=float(mc[np.argmin(np.abs(st-e["step"]))])
        if m>0: rel.append(e["tokens"]/m)
    out["ctx_at_death_ratio"]=float(np.median(rel)) if rel else None
    # phenotypes
    ped=Pedigree.from_run(R); t=build_phenotypes(R,pedigree=ped)
    k=t['children'].astype(float)
    out["n_pheno"]=len(t); out["Vk"]=float(k.var()); out["k_mean"]=float(k.mean())
    out["k_max"]=int(k.max()); out["k_zero_pct"]=float((k==0).mean()*100)
    N=out["agents_mean"]; Ne=(4*N-2)/(k.var()+2)
    out["Ne"]=float(Ne); out["drift_threshold"]=float(1/(2*Ne))
    out["canonical"]=float(np.nanmean(t['canonical_rate'].astype(float))*100)
    out["eff_lineages"]=float(effective_number(t['lineage']))
    fam=ped.families(); out["largest_family_pct"]=float(
        max(Counter(fam.get(a,a) for a in t.index).values())/len(t)*100)
    # heritability on canonical rate
    rate={a:v for a,v in zip(t.index,t['canonical_rate'].astype(float))}
    X=[];Y=[];SX=[]
    for a in t.index:
        par=ped.parents.get(a)
        if not par or len(par)!=2: continue
        c,p1,p2=rate.get(a),rate.get(par[0]),rate.get(par[1])
        if None in (c,p1,p2): continue
        X.append((p1+p2)/2); Y.append(c); SX.append(p1)
    mp,sp=regress(X,Y),regress(SX,Y)
    out["h2"]=mp[0] if mp else None; out["h2_se"]=mp[1] if mp else None
    out["h2_n"]=mp[2] if mp else None
    out["h2_ratio"]=(mp[0]/sp[0]) if (mp and sp and abs(sp[0])>1e-9) else None
    # strategy concordance
    mate=dict(zip(t.index,t['mate_share'].astype(float)))
    move=dict(zip(t.index,t['move_share'].astype(float)))
    strat={a:(1 if mate[a]>move[a] else 0) for a in t.index
           if np.isfinite(mate[a]) and np.isfinite(move[a])}
    room=dict(zip(t.index,t['room_born'])); step=dict(zip(t.index,t['birth_step'].astype(float)))
    con=parent_offspring_concordance(strat,ped,room,step)
    out["concord_excess"]=con.get("excess"); out["concord_z"]=con.get("z")
    # genome drift and diversity
    rows,sites=load_fingerprints(R)
    if rows:
        gen={a:ped.generation.get(a,0) for a in rows if a in ped.generation}
        ags=[a for a in rows if a in gen]
        G=np.array([[rows[a][s] for s in sites] for a in ags]); g=np.array([gen[a] for a in ags])
        tot=np.sqrt((G**2).sum(1)); base=tot[g==0].mean()
        late=g>=np.percentile(g,80)
        rng=np.random.default_rng(0); sub=G[late]
        idx=rng.choice(len(sub),min(300,len(sub)),replace=False); S=sub[idx]
        d=np.sqrt(((S[:,None,:]-S[None,:,:])**2).sum(-1))
        out["drift_vs_gen0"]=float(tot[late].mean()/base)
        out["diversity"]=float(d[np.triu_indices(len(S),1)].mean()/tot[late].mean())
        out["n_genomes"]=len(ags)
    # placeholder share of move attempts
    tot_mv=sum(mv.values()); out["placeholder_pct"]=float(sum(ph.values())/max(tot_mv,1)*100)
    # within-lifetime change
    long=[a for a,v in turns.items() if len(v)>=20]
    if long:
        diffs=[]
        for a in long:
            v=sorted(turns[a]); n=len(v); kk=max(1,n//5)
            diffs.append(np.mean([f=="canonical" for _,f in v[-kk:]])
                         - np.mean([f=="canonical" for _,f in v[:kk]]))
        diffs=np.array(diffs)
        out["within_life_pp"]=float(diffs.mean()*100)
        out["within_life_ci"]=float(1.96*diffs.std(ddof=1)/np.sqrt(len(diffs))*100)
        out["within_life_n"]=len(diffs)
    return out


def _fmt(v, dp=2):
    return "—" if v is None else f"{v:,.{dp}f}"


def main(argv=None):
    import sys
    args = (argv if argv is not None else sys.argv[1:])
    if not args:
        print(__doc__)
        return 1
    runs = [a.rstrip("/").split("runs/")[-1] for a in args]
    res = {r: core(r) for r in runs}
    keys = [k for k in next(iter(res.values())) if isinstance(
        next(iter(res.values()))[k], (int, float, type(None)))]
    w = max(len(k) for k in keys)
    print(f"{'metric':{w}s} " + " ".join(f"{r[-18:]:>18s}" for r in runs))
    for k in keys:
        print(f"{k:{w}s} " + " ".join(
            f"{_fmt(res[r].get(k)):>18s}" for r in runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
