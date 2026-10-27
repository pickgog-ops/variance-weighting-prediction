import sys, os, pickle, time
from multiprocessing import Pool
import lib
R=200
def job(args):
    cfg,a=args
    rows=[]
    for r in range(R):
        rows+=lib.one_rep(cfg,r,a_override=a)
    return cfg,a,rows
if __name__=='__main__':
    which=sys.argv[1]
    cfgs=lib.configs()
    if which=='main': tasks=[(c,None) for c in cfgs]
    else: tasks=[(dict(c,cid=c['cid']+1000),0.5) for c in cfgs if c['p']==30 and c['mis']==1]   # sensitivity a=0.5, distinct seeds
    t=time.time(); res=[]
    with Pool(2) as pool:
        for i,out in enumerate(pool.imap_unordered(job,tasks)):
            res.append(out)
            if i%8==0: print(i,len(tasks),'%.0fs'%(time.time()-t),flush=True)
    pickle.dump(res,open(f'results_{which}.pkl','wb'))
    print('done',time.time()-t)
