import copy, csv, hashlib, importlib, inspect, json, logging, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .numerics import c_index, nll, risk_rmst
from .plots import weibull_figure, history_figure, poe_figure
import matplotlib.pyplot as plt


def resolve(name):
    module,attribute=name.split(':');return getattr(importlib.import_module(module),attribute)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def worker_seed(_):
    value=torch.initial_seed()%2**32;random.seed(value);np.random.seed(value)


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)


def checkpoint(path,state):
    temp=Path(str(path)+'.tmp');torch.save(state,temp);temp.replace(path)


class WithIndex(Dataset):
    """Only attaches index to the original __getitem__; no data processing."""
    def __init__(self,original):self.original=original
    def __len__(self):return len(self.original)
    def __getitem__(self,i):return self.original[i],i


def collate_original(batch):
    records=[]
    for sample,index in batch:
        # pathways: patch, list[omics], y, time, censor, clinical, mask
        if isinstance(sample[1],(list,tuple)):
            x,groups,y,t,c,clinical,mask=sample
        else:
            # groups: patch, *omics_groups, y, time, censor, clinical, mask
            x=sample[0];groups=list(sample[1:-5]);y,t,c,clinical,mask=sample[-5:]
        if not groups:raise ValueError('Trainer requires pathways/groups format, not gene-only')
        records.append((x,groups,y,t,c,mask,index))
    b=len(records);n=max(r[0].size(0) for r in records);dim=records[0][0].size(1)
    x=torch.zeros(b,n,dim);mask=torch.ones(b,n,dtype=torch.bool)
    for i,r in enumerate(records):
        size=r[0].size(0);x[i,:size]=r[0];mask[i,:size]=r[5].reshape(-1).bool()
    if mask.all(1).any():raise ValueError('Dataset returned all-padding patient; check WSI files')
    groups=[torch.stack([r[1][g] for r in records]) for g in range(len(records[0][1]))]
    scalar=lambda pos,dtype:torch.tensor([torch.as_tensor(r[pos]).item() for r in records],dtype=dtype)
    return dict(x=x,groups=groups,y=scalar(2,torch.long),time=scalar(3,torch.float32),
                censor=scalar(4,torch.float32),mask=mask,index=[r[6] for r in records])


def to_device(batch,device):
    return {k:([v.to(device) for v in value] if k=='groups' else value.to(device) if isinstance(value,torch.Tensor) else value) for k,value in batch.items()}


def objective_output(model,b):
    output=model(b['x'],b['groups'],b['y'],b['time'],b['censor'],b['mask'],return_aux=True)
    if not isinstance(output,dict):raise TypeError('Model must return auxiliary dictionary with return_aux=True, as in your working notebook')
    return output


def actual_parameters(output,eps):
    x=output['x'].float()
    shape=torch.nn.functional.softplus(torch.einsum('bd,bmd->bm',x,output['shapes'].float()))+eps
    rate=torch.nn.functional.softplus(torch.einsum('bd,bmd->bm',x,output['scales'].float()))+eps
    weight=output['weights'].float().clamp_min(eps);weight=weight/weight.sum(-1,keepdim=True)
    return shape,rate,weight


@torch.no_grad()
def predict_parameters(model,b):
    # Each call removes padding; never passes y, time or censor into prediction.
    ks=[];rates=[];weights=[]
    for i in range(len(b['index'])):
        valid=~b['mask'][i]
        prediction=model.predict(b['x'][i:i+1,valid],[g[i:i+1] for g in b['groups']])
        if not isinstance(prediction,dict):raise TypeError('model.predict must return dict shapes/scales/weights [B,M]')
        for key,dest in [('shapes',ks),('scales',rates),('weights',weights)]:
            value=prediction[key].float()
            if value.ndim!=2 or value.size(0)!=1 or not torch.isfinite(value).all():raise ValueError(f'Invalid predicted {key}: expected actual parameters [1,M]')
            if key!='weights' and (value<=0).any():raise ValueError('Weibull k and lambda must be positive')
            if key=='weights' and ((value<0).any() or value.sum()<=0):raise ValueError('Invalid mixture probabilities')
            dest.append(value)
    return [torch.cat(v).cpu().double().numpy() for v in (ks,rates,weights)]


class Trainer:
    def __init__(self,model,criterion,train_dataset,val_dataset,outdir,config,preprocessing=None):
        self.cfg=copy.deepcopy(config);self.out=Path(outdir);self.out.mkdir(parents=True,exist_ok=True)
        self.device=torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
        self.model=model.to(self.device);self.criterion=criterion.to(self.device)
        self.train_dataset=train_dataset;self.val_dataset=val_dataset;self.preprocessing=preprocessing
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=config.get('lr',1e-4),weight_decay=config.get('weight_decay',1e-4))
        self.club_opt=torch.optim.Adam(self.criterion.club_parameters(),lr=config.get('club_lr',1e-4))
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,mode='min',factor=.5,patience=config.get('lr_patience',5))
        self.seed=config.get('seed',42);self.generator=torch.Generator().manual_seed(self.seed)
        self.history=[];self.step=0;self.start=0;self.best=float('inf');self.bad=0
        self.fingerprint=self._fingerprint()
        self._validate_contract()
        if config.get('resume',False):self._restore()
        elif (self.out/'last.pt').exists():raise FileExistsError('Output contains checkpoint: enable resume or choose new output')
        self.writer=SummaryWriter(str(self.out/'tensorboard'),purge_step=self.step+1 if self.start else None,flush_secs=10)
        self.horizon=float(config.get('plot_horizon') or max(np.quantile(train_dataset.metadata[train_dataset.label_col],.9),1.))
        self.risk_horizon=float(config.get('risk_horizon') or self.horizon)
        write_json(self.out/'trainer_config.json',self.cfg)
        write_json(self.out/'metric_definitions.json',dict(censor='0:event,1:right_censored',time_scale=1,
            risk='negative_RMST',risk_horizon=self.risk_horizon,plot_horizon=self.horizon,
            train_loss='label-conditioned objective in train mode',val_aux='label-conditioned diagnostic in eval mode',
            val='label-free prediction; labels used only for scoring',train_eval_sampling='unchanged dataset.sample'))
        self._save_sources()

    def _validate_contract(self):
        if not hasattr(self.model,'predict') or not callable(self.model.predict):
            raise TypeError('Your imported model has no predict(). Use your debugged model version with label-free predict; trainer will not rewrite model.py')
        if 'return_aux' not in inspect.signature(self.model.forward).parameters:
            raise TypeError('Your imported model is the old version: forward must accept return_aux=True, as in notebook')
        if not hasattr(self.model,'PoE') or not hasattr(self.model.PoE,'phi'):raise TypeError('Expected model.PoE.phi')
        if self.model.PoE.phi.numel()!=len(self.cfg.get('poe_names',['histology','pathway'])):raise ValueError('poe_names must follow phi order')
        if not hasattr(self.criterion,'club_learning_loss'):raise TypeError('Expected criterion.club_learning_loss(output)')
        if self.cfg.get('batch_size',4)<2:raise ValueError('Training batch_size >= 2 required for CLUB')
        for split,ds in [('train',self.train_dataset),('val',self.val_dataset)]:
            if len(ds)==0:raise ValueError(f'Empty {split} dataset')
            self._targets(ds) # validate outcomes without changing the dataset

    def _targets(self,ds):
        times=ds.metadata[ds.label_col].to_numpy(float);c=ds.metadata[ds.censorship_var].to_numpy(float)
        if not np.isfinite(times).all() or (times<0).any() or not np.isin(c,[0,1]).all():raise ValueError('Invalid survival target')
        return times,c

    def _fingerprint(self):
        cfg={k:v for k,v in self.cfg.items() if k not in ('epochs','resume')}
        meta=[ds.metadata[['case_id',ds.label_col,ds.censorship_var]].to_json() for ds in (self.train_dataset,self.val_dataset)]
        code=[]
        for obj in (self.model.__class__,self.criterion.__class__,self.train_dataset.__class__):
            path=Path(inspect.getfile(obj));code.append(hashlib.sha256(path.read_bytes()).hexdigest())
        return hashlib.sha256(json.dumps([cfg,meta,code],sort_keys=True).encode()).hexdigest()

    def _save_sources(self):
        folder=self.out/'source_snapshot';folder.mkdir(exist_ok=True)
        import shutil
        for prefix,obj in [('model',self.model.__class__),('criterion',self.criterion.__class__),('dataset',self.train_dataset.__class__)]:
            path=Path(inspect.getfile(obj));shutil.copy2(path,folder/(prefix+'_'+path.name))

    def _loader(self,ds,training=False):
        generator=self.generator if training else torch.Generator().manual_seed(self.seed+100)
        return DataLoader(WithIndex(ds),batch_size=self.cfg.get('batch_size',4) if training else self.cfg.get('eval_batch_size',4),
            shuffle=training,num_workers=self.cfg.get('workers',0),collate_fn=collate_original,
            worker_init_fn=worker_seed,generator=generator,pin_memory=self.device.type=='cuda',drop_last=False)

    def _state(self,epoch):
        return dict(epoch=epoch,step=self.step,best=self.best,bad=self.bad,history=self.history,fingerprint=self.fingerprint,
            model=self.model.state_dict(),criterion=self.criterion.state_dict(),optimizer=self.opt.state_dict(),club_optimizer=self.club_opt.state_dict(),scheduler=self.scheduler.state_dict(),
            preprocessing=self.preprocessing,config=self.cfg,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),loader_rng=self.generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None)

    def _restore(self):
        path=self.out/'last.pt'
        if not path.exists():raise FileNotFoundError(f'Resume checkpoint absent: {path}')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['fingerprint']!=self.fingerprint:raise ValueError('Resume config, source or patient IDs changed')
        self.model.load_state_dict(state['model']);self.criterion.load_state_dict(state['criterion'])
        self.opt.load_state_dict(state['optimizer']);self.club_opt.load_state_dict(state['club_optimizer']);self.scheduler.load_state_dict(state['scheduler'])
        self.start=state['epoch']+1;self.step=state['step'];self.best=state['best'];self.bad=state['bad'];self.history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);self.generator.set_state(state['loader_rng'])
        if self.device.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        # Discard history lines from epochs not committed to last.pt after a crash.
        (self.out/'epochs.jsonl').write_text(''.join(json.dumps(json_safe(r))+'\n' for r in self.history))

    def train_epoch(self):
        self.model.train();self.criterion.train();sums={};count=0
        for batch in tqdm(self._loader(self.train_dataset,True),desc='train',leave=False):
            b=to_device(batch,self.device);n=len(b['index'])
            if n==1:logging.warning('Singleton train batch: CLUB MI equals zero for this batch')
            self.opt.zero_grad(set_to_none=True);self.club_opt.zero_grad(set_to_none=True)
            output=objective_output(self.model,b)
            fit=self.criterion.club_learning_loss(output)
            if not torch.isfinite(fit):raise FloatingPointError('Nonfinite CLUB fit')
            fit.backward();torch.nn.utils.clip_grad_norm_(self.criterion.club_parameters(),self.cfg.get('clip',5.),error_if_nonfinite=True)
            self.club_opt.step();self.club_opt.zero_grad(set_to_none=True)
            # Explicitly freeze estimators even if criterion implements its own freeze.
            params=list(self.criterion.club_parameters());flags=[p.requires_grad for p in params]
            try:
                for p in params:p.requires_grad_(False)
                losses=self.criterion(output,b['y'],b['time'],b['censor'])
                if 'total' not in losses:raise KeyError('criterion must return dict with total')
                bad_losses = {
                    name: value.detach().cpu().tolist()
                    for name, value in losses.items()
                    if not torch.isfinite(value).all().item()
                }

                if bad_losses:
                    raise FloatingPointError(
                        f"Nonfinite training loss: {bad_losses}; "
                        f"batch_indices={b['index']}"
                    )
                #if not all(torch.isfinite(v).all() for v in losses.values()):raise FloatingPointError('Nonfinite training loss')
                losses['total'].backward()
            finally:
                for p,flag in zip(params,flags):p.requires_grad_(flag)
            grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.get('clip',5.),error_if_nonfinite=True)
            self.opt.step();self.step+=1;count+=n
            for key,value in {**losses,'club_fit':fit}.items():
                v=float(value.detach());sums[key]=sums.get(key,0.)+v*n;self.writer.add_scalar('batch/'+key,v,self.step)
            self.writer.add_scalar('batch/grad_norm',float(grad),self.step)
        return {k:v/count for k,v in sums.items()}

    @torch.no_grad()
    def evaluate(self,ds,split,epoch,auxiliary=False,plot=False):
        self.model.eval();self.criterion.eval();rows=[];all_k=[];all_r=[];all_w=[];sums={};count=0
        # Isolate evaluation RNG so plotting/validation do not change training RNG.
        py_state=random.getstate();np_state=np.random.get_state()
        devices=[self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                seed_all(self.seed+100)
                for batch in tqdm(self._loader(ds),desc=split,leave=False):
                    b=to_device(batch,self.device);n=len(b['index']);count+=n
                    k,r,w=predict_parameters(self.model,b);all_k.extend(k);all_r.extend(r);all_w.extend(w)
                    for i,index in enumerate(b['index']):
                        rows.append(dict(index=index,case_id=str(ds.metadata['case_id'].iloc[index]),time=float(b['time'][i]),censor=int(b['censor'][i])))
                    if auxiliary:
                        output=objective_output(self.model,b)
                        values=self.criterion(output,b['y'],b['time'],b['censor'])
                        for key,value in values.items():
                            v=float(value.detach())
                            if not np.isfinite(v):raise FloatingPointError(f'Nonfinite {split} diagnostic {key}')
                            sums[key]=sums.get(key,0)+v*n
        finally:random.setstate(py_state);np.random.set_state(np_state)
        df=pd.DataFrame(rows);k=np.asarray(all_k);r=np.asarray(all_r);w=np.asarray(all_w)
        w=np.maximum(w,1e-7);w=w/w.sum(1,keepdims=True)
        losses=nll(k,r,w,df.time.to_numpy(),df.censor.to_numpy());risk=risk_rmst(k,r,w,self.risk_horizon)
        if not np.isfinite(losses).all():raise FloatingPointError('Nonfinite predictive NLL')
        result=dict(nll=float(losses.mean()),**c_index(df.time,df.censor,risk),n_patients=len(df),events=int((df.censor==0).sum()))
        result['aux_label_conditioned']={key:value/count for key,value in sums.items()}
        df['risk']=risk;df['nll']=losses
        for j in range(k.shape[1]):df[f'k_{j}']=k[:,j];df[f'lambda_{j}']=r[:,j];df[f'pi_{j}']=w[:,j]
        pred_dir=self.out/'predictions';pred_dir.mkdir(exist_ok=True)
        df.to_csv(pred_dir/f'{split}_latest.csv',index=False)
        if plot:
            folder=self.out/'figures'/f'epoch_{epoch:04d}';folder.mkdir(parents=True,exist_ok=True)
            for i in range(min(self.cfg.get('plot_cases',3),len(df))):
                row=df.iloc[i];fig=weibull_figure(k[i],r[i],w[i],self.horizon,row.case_id,row.time,row.censor)
                fig.savefig(folder/f'{split}_case_{i:02d}.png',dpi=130)
                self.writer.add_figure(f'weibull/{split}/case_{i:02d}',fig,global_step=epoch,close=False);plt.close(fig)
            df.head(self.cfg.get('plot_cases',3)).to_csv(folder/f'{split}_plot_parameters.csv',index=False)
        return result

    def _record(self,record):
        epoch=record['epoch'];self.history.append(record)
        with (self.out/'epochs.jsonl').open('a') as f:f.write(json.dumps(json_safe(record),allow_nan=False)+'\n')
        flat={}
        def flatten(prefix,value):
            if isinstance(value,dict):
                for k,v in value.items():flatten(prefix+'/'+k if prefix else k,v)
            elif isinstance(value,(int,float)) and np.isfinite(value):flat[prefix]=value
        for key in ('train_loss','train_eval','val'):flatten(key,record.get(key,{}))
        for key,value in flat.items():self.writer.add_scalar(key,value,epoch)
        self.writer.add_scalar('optimizer/lr',record['lr'],epoch)
        names=self.cfg.get('poe_names',['histology','pathway'])
        for i,name in enumerate(names):
            self.writer.add_scalar('poe/phi_'+name,record['poe_phi'][i],epoch)
            self.writer.add_scalar('poe/alpha_'+name,record['poe_alpha'][i],epoch)
        flat_rows=[]
        for r in self.history:
            row={'epoch':r['epoch'],'seconds':r['seconds'],'lr':r['lr']}
            for key in ('train_loss','train_eval','val'):
                for sub,value in r.get(key,{}).items():
                    if isinstance(value,dict):
                        for a,b in value.items():row[f'{key}/{sub}/{a}']=b
                    else:row[f'{key}/{sub}']=value
            for i,name in enumerate(names):row['poe/phi_'+name]=r['poe_phi'][i];row['poe/alpha_'+name]=r['poe_alpha'][i]
            flat_rows.append(row)
        pd.DataFrame(flat_rows).to_csv(self.out/'history.csv',index=False)
        figs=self.out/'figures';figs.mkdir(exist_ok=True)
        for name,fig in [('history',history_figure(self.history)),('poe',poe_figure(self.history,names))]:
            fig.savefig(figs/(name+'.png'),dpi=130);self.writer.add_figure(name,fig,epoch,close=False);plt.close(fig)
        self.writer.flush()

    def fit(self,test_dataset=None):
        try:
            for epoch in range(self.start,self.cfg.get('epochs',100)):
                if self.bad>=self.cfg.get('patience',15):break
                begin=time.time();train_loss=self.train_epoch()
                number=epoch+1;plot=number==1 or number%self.cfg.get('plot_every',5)==0
                val=self.evaluate(self.val_dataset,'val',number,auxiliary=self.cfg.get('val_auxiliary',True),plot=plot)
                train_eval={}
                if number%self.cfg.get('train_eval_every',1)==0:
                    train_eval=self.evaluate(self.train_dataset,'train_eval',number,auxiliary=False)
                score=val['nll'];improved=score<self.best-self.cfg.get('min_delta',0.)
                if improved:self.best=score;self.bad=0
                else:self.bad+=1
                self.scheduler.step(score)
                phi=self.model.PoE.phi.detach().float().reshape(-1)
                record=dict(epoch=number,step=self.step,seconds=time.time()-begin,lr=self.opt.param_groups[0]['lr'],
                    train_loss=train_loss,train_eval=train_eval,val=val,poe_phi=phi.cpu().tolist(),poe_alpha=phi.softmax(0).cpu().tolist())
                if self.device.type=='cuda':
                    self.writer.add_scalar('hardware/peak_gpu_gb',torch.cuda.max_memory_allocated()/2**30,number);torch.cuda.reset_peak_memory_stats()
                self._record(record);state=self._state(epoch)
                checkpoint(self.out/'last.pt',state)
                if improved:checkpoint(self.out/'best.pt',state)
                logging.info('epoch=%d train_total=%.5g val_nll=%.5g val_cindex=%s PoE=%s',number,train_loss['total'],score,val['c_index'],record['poe_alpha'])
            if not (self.out/'best.pt').exists():raise RuntimeError('No completed epoch/checkpoint')
            best=torch.load(self.out/'best.pt',map_location='cpu',weights_only=False)
            self.model.load_state_dict(best['model']);self.criterion.load_state_dict(best['criterion'])
            metrics=self.evaluate(self.val_dataset,'val_best',best['epoch']+1,auxiliary=self.cfg.get('val_auxiliary',True),plot=True)
            metrics['best_epoch']=best['epoch']+1;write_json(self.out/'val_best_metrics.json',metrics)
            if test_dataset is not None:
                test=self.evaluate(test_dataset,'test_best',best['epoch']+1,auxiliary=False,plot=True)
                test['best_epoch']=best['epoch']+1;write_json(self.out/'test_best_metrics.json',test)
            return metrics
        finally:self.writer.close()
