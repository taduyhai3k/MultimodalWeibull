"""Run from project root: python -m factory_trainer.run --config ..."""
import argparse, contextlib, copy, hashlib, inspect, json, logging, pickle, subprocess, sys
from pathlib import Path
import pandas as pd
import torch
from .trainer import Trainer, resolve, seed_all, write_json, to_device, objective_output


def absolute(root,value):
    path=Path(value);return str(path if path.is_absolute() else root/path)


def create_datasets(config,root,cohort,spec,fold,out):
    Factory=resolve(config.get('factory_class','dataset.DatasetFactory:SurvivalDatasetFactory'))
    kwargs=copy.deepcopy(config.get('factory_kwargs',{}));kwargs.update(spec['factory_kwargs'])
    for key in ('label_file','omics_file','clinical_file','signature_file','wsi_dir'):
        kwargs[key]=absolute(root,kwargs[key])
    kwargs.setdefault('study_name',cohort)
    factory=Factory(**kwargs) # Your original class, unchanged.
    split_path=absolute(root,spec['split_template'].format(fold=fold))
    table=pd.read_csv(split_path)
    train,scaler=factory._get_split_from_df(split_path=split_path,split_key='train',fold=fold,scaler=None,valid_cols=None)
    val,_=factory._get_split_from_df(split_path=split_path,split_key='val',fold=fold,scaler=scaler,valid_cols=None)
    test=None
    if config.get('evaluate_test',True) and 'test' in table and table['test'].notna().any():
        test,_=factory._get_split_from_df(split_path=split_path,split_key='test',fold=fold,scaler=scaler,valid_cols=None)
    if train is None or val is None:raise ValueError('Your Factory returned empty train or val')
    # Report actual factory outputs; do not alter IDs, labels, normalization or slides.
    rows=[]
    for split,ds in [('train',train),('val',val),('test',test)]:
        if ds is None:continue
        present=set(ds.metadata.case_id.astype(str));requested=table[split].dropna().astype(str).tolist()
        for pid in requested:rows.append(dict(split=split,case_id=pid,in_factory_dataset=pid in present))
        ds.metadata.to_csv(out/f'{split}_metadata.csv',index=False)
        ds.omic_data_dict[['temp_index']].to_csv(out/f'{split}_omics_order.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'factory_membership.csv',index=False)
    # Overlap check only, no repair or patient dropping.
    actual={name:set(ds.metadata.case_id.astype(str)) for name,ds in [('train',train),('val',val),('test',test)] if ds is not None}
    for a in actual:
        for b in actual:
            if a!=b and actual[a]&actual[b]:raise ValueError(f'Factory output {a}/{b} overlap')
    prep=dict(factory_class=config.get('factory_class','dataset.DatasetFactory:SurvivalDatasetFactory'),factory_kwargs=kwargs,
        omic_names=factory.omic_names,omic_sizes=factory.omic_sizes,label_discrete_dict=factory.label_discrete_dict,
        scaler=scaler,split_path=split_path,split_sha256=hashlib.sha256(Path(split_path).read_bytes()).hexdigest())
    with (out/'factory_preprocessing.pkl').open('wb') as f:pickle.dump(prep,f)
    write_json(out/'imports.json',dict(factory=inspect.getfile(Factory),dataset=inspect.getfile(type(train))))
    return factory,train,val,test,prep


def summarize(output):
    rows=[]
    for path in sorted(Path(output).glob('*/fold_*/*_best_metrics.json')):
        v=json.loads(path.read_text());v.pop('aux_label_conditioned',None)
        rows.append(dict(cohort=path.parent.parent.name,fold=int(path.parent.name.split('_')[-1]),split=path.name.split('_')[0],**v))
    if rows:
        df=pd.DataFrame(rows);df.to_csv(Path(output)/'fold_metrics.csv',index=False)
        df.groupby(['cohort','split'])[['nll','c_index']].agg(['count','mean','std']).to_csv(Path(output)/'summary.csv')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',default='.')
    p.add_argument('--output',default=None);p.add_argument('--cohorts',nargs='+');p.add_argument('--folds',type=int,nargs='+')
    p.add_argument('--resume',action='store_true');p.add_argument('--preflight',action='store_true')
    args=p.parse_args();config=json.loads(Path(args.config).read_text());root=Path(args.root).resolve()
    output=Path(args.output or config.get('output','runs/factory_v1')).resolve();output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(output/'training.log'),logging.StreamHandler()])
    write_json(output/'config.json',config)
    (output/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write_json(output/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version))
    for cohort in args.cohorts or list(config['cohorts']):
        for fold in args.folds or config.get('folds',[0,1,2,3,4]):
            out=output/cohort/f'fold_{fold}';out.mkdir(parents=True,exist_ok=True)
            cfg=copy.deepcopy(config['training']);cfg['seed']=cfg.get('seed',42)+fold;cfg['resume']=args.resume
            seed_all(cfg['seed'])
            logging.info('Preparing %s fold %s using YOUR DatasetFactory',cohort,fold)
            with (out/'factory_stdout.log').open('w') as capture,contextlib.redirect_stdout(capture):
                factory,train,val,test,prep=create_datasets(config,root,cohort,config['cohorts'][cohort],fold,out)
            Model=resolve(config.get('model_class','model:MultiModelWeibull'));Criterion=resolve(config.get('criterion_class','block.loss:PIBDWeibullLoss'))
            kwargs=copy.deepcopy(config.get('model_kwargs',{}));kwargs['omic_sizes']=factory.omic_sizes
            model=Model(**kwargs);criterion=Criterion(**config.get('criterion_kwargs',{}))
            logging.info('Imported model: %s; criterion: %s',inspect.getfile(Model),inspect.getfile(Criterion))
            trainer=Trainer(model,criterion,train,val,out,cfg,preprocessing=prep)
            if args.preflight:
                try:
                    batch=to_device(next(iter(trainer._loader(train,True))),trainer.device)
                    result=objective_output(trainer.model,batch)
                    loss=trainer.criterion(result,batch['y'],batch['time'],batch['censor'])
                    if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite preflight loss')
                    loss['total'].backward()
                    if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in trainer.model.parameters()):raise FloatingPointError('Nonfinite preflight gradients')
                    trainer.model.zero_grad(set_to_none=True);trainer.criterion.zero_grad(set_to_none=True)
                    # Full validation checks model.predict and creates requested plots.
                    trainer.evaluate(val,'val_preflight',0,auxiliary=False,plot=True)
                    logging.info('Preflight passed, train total=%s',float(loss['total'].detach()))
                finally:trainer.writer.close()
            else:
                trainer.fit(test_dataset=test);summarize(output)

if __name__=='__main__':main()
