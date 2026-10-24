"""Create config copy only; never modify user's dataset/model or original config."""
import argparse,json,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cohorts',nargs='+',required=True);p.add_argument('--folds',type=int,nargs='+',required=True)
    p.add_argument('--require-cuda',type=int,default=1);a=p.parse_args()
    root=Path(a.root).resolve();config=json.loads(Path(a.config).read_text())
    for cohort in a.cohorts:
        if cohort not in config['cohorts']:raise ValueError(f'Unknown cohort {cohort}')
        spec=config['cohorts'][cohort];values=spec['factory_kwargs']
        current=Path(values['wsi_dir']);current=current if current.is_absolute() else root/current
        if not (current/'pt_files').is_dir():
            candidates=sorted({f.parent.parent for f in (root/'WSIs').glob(f'{cohort}-*/**/pt_files/*.pt')})
            if len(candidates)!=1:raise ValueError(f'{cohort}: found {len(candidates)} feature directories; set wsi_dir explicitly in config')
            current=candidates[0]
        values['wsi_dir']=str(current)
        for key in ('label_file','omics_file','clinical_file','signature_file'):
            path=Path(values[key]);path=path if path.is_absolute() else root/path
            if not path.is_file():raise FileNotFoundError(str(path))
        for fold in a.folds:
            path=root/spec['split_template'].format(fold=fold)
            if not path.is_file():raise FileNotFoundError(str(path))
    config['pipeline_source_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'model.py', *sorted((root/'block').glob('*.py')), *sorted((root/'dataset').glob('*.py'))] if f.is_file()}
    config['folds']=a.folds
    if a.require_cuda:config['training']['device']='cuda'
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(config,sort_keys=True,indent=2)
    # Prevent mixed run results when rerunning with a changed configuration.
    if out.exists() and json.loads(out.read_text())!=config:
        raise ValueError('Runtime config changed. Choose new EXPERIMENT/OUTPUT_DIR or restore original settings.')
    out.write_text(payload)
    digest=hashlib.sha256(payload.encode()).hexdigest()
    print(f'Runtime config: {out}\nConfiguration SHA256: {digest}')

if __name__=='__main__':main()
