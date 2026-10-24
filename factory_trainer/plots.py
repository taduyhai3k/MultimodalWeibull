import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .numerics import curves


def weibull_figure(k,rate,w,horizon,case_id,time,censor):
    grid=np.linspace(max(horizon*1e-5,1e-7),horizon,500)
    values=curves(k,rate,w,grid)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=plt.cm.tab10(np.linspace(0,1,len(w)))
    for i,color in enumerate(colors):
        label=f'{i+1}: k={k[i]:.3g}, lambda={rate[i]:.3g}, pi={w[i]:.3f}'
        axes[0,0].plot(grid,values['pdf'][i],color=color,label=label)
        axes[0,1].plot(grid,values['weighted_pdf'][i],color=color,label=f'pi[{i+1}] * f[{i+1}]')
        axes[1,0].plot(grid,values['survival'][i],color=color,alpha=.7,label=f'S[{i+1}]')
    axes[0,0].plot(grid,values['mixture_pdf'],'k',lw=2.5,label='Mixture PDF')
    axes[0,1].plot(grid,values['mixture_pdf'],'k',lw=2.5,label='Sum = mixture PDF')
    axes[1,0].plot(grid,values['mixture_survival'],'k',lw=2.5,label='Mixture survival')
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.axvline(time,color='gray',ls='--',label='Observed/censor time')
        ax.set_xlabel('Time (original dataset units)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0,0].set_title('Individual component PDFs + mixture')
    axes[0,1].set_title('Weighted component contributions')
    axes[1,0].set_title('Survival functions');axes[1,0].set_ylim(0,1.03)
    axes[1,1].bar(np.arange(1,len(w)+1),w,color=colors)
    axes[1,1].set_title('Weibull mixture probabilities');axes[1,1].set_ylim(0,1)
    axes[1,1].set_xlabel('Component');axes[1,1].set_ylabel('pi')
    fig.suptitle(f'{case_id} | t={time:.3g}, censor={int(censor)} | grid truncated at {horizon:.3g}')
    return fig


def history_figure(records):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    epochs=[r['epoch'] for r in records]
    for key in sorted(records[-1]['train_loss']):
        axes[0,0].plot(epochs,[r['train_loss'].get(key,np.nan) for r in records],label=key)
    axes[0,0].set_title('Train objective components')
    aux=records[-1].get('val',{}).get('aux_label_conditioned',{})
    for key in sorted(aux):
        axes[0,1].plot(epochs,[r.get('val',{}).get('aux_label_conditioned',{}).get(key,np.nan) for r in records],label=key)
    axes[0,1].set_title('Val objective diagnostic (label-conditioned)')
    for split in ('train_eval','val'):
        axes[1,0].plot(epochs,[r.get(split,{}).get('nll',np.nan) for r in records],label=split+' NLL')
        values=[r.get(split,{}).get('c_index') for r in records]
        axes[1,1].plot(epochs,[v if v is not None else np.nan for v in values],label=split+' C-index')
    axes[1,0].set_title('Label-free predictive NLL');axes[1,1].set_title('Label-free C-index')
    for ax in axes[0]:ax.set_yscale('symlog',linthresh=1.)
    for ax in axes.flat:
        ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    return fig


def poe_figure(records,names):
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    epochs=[r['epoch'] for r in records]
    for i,name in enumerate(names):
        axes[0].plot(epochs,[r['poe_phi'][i] for r in records],label=name)
        axes[1].plot(epochs,[r['poe_alpha'][i] for r in records],label=name)
    axes[0].set_title('PoE raw parameters phi')
    axes[1].set_title('PoE alpha = softmax(phi)');axes[1].set_ylim(0,1)
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    return fig
