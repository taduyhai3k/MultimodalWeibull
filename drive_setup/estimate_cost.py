"""Use epoch wall times from train.py plus offer prices; no invented benchmark."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--folds',type=int,default=5);p.add_argument('--gpu-hour',type=float,required=True)
    p.add_argument('--disk-gb',type=float,default=0);p.add_argument('--storage-gb-hour',type=float,default=0)
    p.add_argument('--download-gb',type=float,default=0);p.add_argument('--upload-gb',type=float,default=0)
    p.add_argument('--in-tb-price',type=float,default=0);p.add_argument('--out-tb-price',type=float,default=0)
    p.add_argument('--setup-hours',type=float,default=0);p.add_argument('--evaluation-hours',type=float,default=0)
    args=p.parse_args(); samples={}
    for path in Path(args.runs).glob('*/fold_*/epochs.jsonl'):
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Exclude first epoch warmup when possible; take last completed epochs.
        seconds=[r['seconds'] for r in records[1:] or records]
        if seconds: samples.setdefault(path.parent.parent.name,[]).extend(seconds[-5:])
    if not samples: raise ValueError('No completed epochs.jsonl found')
    cohort_hours={c:float(np.median(s))*args.epochs*args.folds/3600 for c,s in samples.items()}
    hours=sum(cohort_hours.values())+args.setup_hours+args.evaluation_hours
    costs=dict(gpu=hours*args.gpu_hour,storage=hours*args.disk_gb*args.storage_gb_hour,
               ingress=args.download_gb/1000*args.in_tb_price,egress=args.upload_gb/1000*args.out_tb_price)
    result=dict(measured_cohorts=list(samples),projected_hours_by_cohort=cohort_hours,
        hours=hours,usd=costs,total_usd=sum(costs.values()),
        note='Only measured cohorts included. Epoch seconds include train+validation. Final evaluation and setup entered separately. Storage assumes running price over these hours; add stopped-instance retention separately. No early stopping discount assumed.')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
