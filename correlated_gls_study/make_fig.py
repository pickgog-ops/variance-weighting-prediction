import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':0.6,'axes.edgecolor':'#b8c0c8',
  'xtick.color':'#5b6670','ytick.color':'#5b6670','axes.labelcolor':'#1f2933','legend.frameon':False})
S=pd.read_csv('summary_main.csv')
ST={'Oracle':('#555555','--','o'),'Diag':('#0072B2','-','s'),'LW-linear':('#CC79A7','-','^'),
    'LW-NL':('#D55E00','-','v'),'QIS':('#E69F00',':','D'),'CV-linear':('#56B4E9','-','<'),'CV-power':('#009E73','-','>')}
fig,axs=plt.subplots(2,3,figsize=(7.0,3.9),sharex=True)
for j,p in enumerate((10,30,60)):
    for i,(mis,metric,lab) in enumerate([(0,'se_b0','coefficient MSE vs OLS (%)'),(1,'test_mse','test MSE vs OLS (%)')]):
        ax=axs[i,j]; sub=S[(S.p==p)&(S.mis==mis)&(S.v==20)]
        piv=sub.pivot_table(index=['T','rho'],columns='method',values=metric)
        rel=(piv.div(piv['OLS'],axis=0)-1)*100
        rel=rel.groupby(level=0).mean()                 # average over rho (v = 20)
        ax.axhline(0,color='#9aa1a8',lw=0.6)
        for m,(c,ls,mk) in ST.items():
            ax.plot(rel.index,rel[m],color=c,ls=ls,marker=mk,ms=3.2,lw=1.1,label=m)
        ax.set_xscale('log'); ax.set_xticks([20,40,100,400]); ax.set_xticklabels(['20','40','100','400']); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis='y',color='#e6e9ed',lw=0.5)
        for s in ('top','right'): ax.spines[s].set_visible(False)
        if j==0: ax.set_ylabel(lab)
        if i==1: ax.set_xlabel('training periods $T$')
        ax.set_title('abcdef'[3*i+j]+f'   p = {p}, '+('correct mean' if mis==0 else 'omitted curvature'),fontsize=7,loc='left')
h,l=axs[0,0].get_legend_handles_labels()
fig.legend(h,l,ncol=7,loc='upper center',bbox_to_anchor=(0.5,1.0),fontsize=6.5,handlelength=2.2,columnspacing=1.0)
fig.tight_layout(rect=(0,0,1,0.94)); fig.savefig('FigS9_correlated_gls.pdf'); fig.savefig('FigS9_correlated_gls.png',dpi=160)
