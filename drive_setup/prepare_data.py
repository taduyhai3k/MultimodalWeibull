"""Download Google Drive ZIPs and install into train.py's data layout."""
import argparse, hashlib, json, logging, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def safe_path(root, relative):
    relative=PurePosixPath(str(relative).replace('\\','/'))
    if relative.is_absolute() or '..' in relative.parts or any(':' in p for p in relative.parts):
        raise ValueError(f'Unsafe path: {relative}')
    root=Path(root).resolve(); path=(root/str(relative)).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes project root')
    return path


def plan_members(archive,item,root):
    entries=[]; seen=set(); kind=item['kind']
    target=safe_path(root,item['destination'])
    if kind not in ('wsi','tree'): raise ValueError('kind must be wsi or tree')
    prefix=item.get('source_subdir','').strip('/')
    files=[m for m in archive.infolist() if not m.is_dir()]
    for m in files:
        path=PurePosixPath(m.filename.replace('\\','/'))
        safe_path(root,str(path)) # reject unsafe source members too
        if ((m.external_attr>>16)&0o170000)==0o120000: raise ValueError('Archive symlinks unsupported')
        if prefix:
            try: path=path.relative_to(prefix)
            except ValueError: continue
        if kind=='wsi':
            if path.suffix.lower()!='.pt': continue
            # train.py uses recursive pt_files discovery, not the original archive nesting.
            relative=PurePosixPath('pt_files')/path.name
        else:
            # Auto remove parent wrappers before datasets_csv/5foldcv, if present.
            anchor=target.name
            if anchor in path.parts:
                relative=PurePosixPath(*path.parts[path.parts.index(anchor)+1:])
            else: relative=path
        destination=safe_path(target,str(relative))
        if destination in seen: raise ValueError(f'Archive maps two members to {destination}')
        seen.add(destination); entries.append((m,destination))
    if not entries: raise ValueError(f'No matching files in {item["name"]}')
    return entries


def extract_archive(path,item,root,reserve_gb=2):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        entries=plan_members(archive,item,root)
        missing=[(m,d) for m,d in entries if not d.exists()]
        need=sum(m.file_size for m,d in missing)+int(reserve_gb*1024**3)
        free=shutil.disk_usage(root).free
        if free<need: raise OSError(f'Insufficient disk: need {need/2**30:.1f} GiB, free {free/2**30:.1f} GiB')
        for i,(member,destination) in enumerate(entries,1):
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                # Compare existing bytes, do not silently replace user data.
                if destination.stat().st_size!=member.file_size:
                    raise FileExistsError(f'Conflicting existing file: {destination}')
                h=hashlib.sha256()
                with archive.open(member) as f:
                    for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
                if h.hexdigest()!=digest(destination): raise FileExistsError(f'Conflicting existing file: {destination}')
                continue
            temp=destination.with_name(destination.name+'.extracting')
            with archive.open(member) as source,temp.open('wb') as out:
                shutil.copyfileobj(source,out,8*1024*1024)
            if temp.stat().st_size!=member.file_size: raise IOError('Extracted size mismatch')
            temp.replace(destination)  # ZipFile checks CRC while reading each member.
            if i%100==0 or i==len(entries): logging.info('%s: extracted %d/%d',item['name'],i,len(entries))
        return [dict(path=str(d.relative_to(root)),bytes=m.file_size) for m,d in entries]


def download(item,cache,retries):
    name=item['name']
    if not name.endswith('.zip') or Path(name).name!=name: raise ValueError('name must be a ZIP basename')
    target=cache/name; partial=cache/(name+'.partial')
    if target.exists():
        if not zipfile.is_zipfile(target): raise ValueError(f'Invalid cached ZIP: {target}; rename it then retry')
        return target
    source=item.get('url','')
    if not source or 'PASTE_' in source: raise ValueError(f'Fill Google Drive URL for {name}')
    # CLI flags below are supported by pinned gdown 5.2.0.
    for attempt in range(retries):
        logging.info('Downloading %s attempt %d/%d',name,attempt+1,retries)
        code=subprocess.call([sys.executable,'-m','gdown',source,'--fuzzy','--continue','-O',str(partial)])
        if code==0 and partial.exists() and zipfile.is_zipfile(partial):
            partial.replace(target); return target
        if code==0: raise ValueError(f'{name} was not a ZIP (Drive permission/quota or wrong link)')
        if attempt+1<retries: time.sleep(min(5*2**attempt,30))
    raise RuntimeError(f'Download failed: {name}. Check sharing and Drive quota; partial retained.')


def verify_layout(root,cohorts,family,folds):
    root=Path(root); missing=[]; summary={}
    for cohort in cohorts:
        required=[root/'datasets_csv/metadata'/f'tcga_{cohort}.csv',
                  root/'datasets_csv/raw_rna_data'/family/cohort/'rna_clean.csv']
        required.extend(root/'5foldcv'/f'tcga_{cohort}'/f'splits_{fold}.csv' for fold in folds)
        missing.extend(str(p) for p in required if not p.is_file())
        bags=list((root/'WSIs').glob(f'{cohort}-*/**/pt_files/*.pt'))
        summary[cohort]=dict(feature_files=len(bags),feature_bytes=sum(p.stat().st_size for p in bags))
        if not bags: missing.append(f'WSIs/{cohort}-*/**/pt_files/*.pt')
    signatures=root/'datasets_csv/metadata'/f'{family}_signatures.csv'
    if not signatures.is_file(): missing.append(str(signatures))
    report=dict(cohorts=summary,missing=missing)
    (root/'download_report.json').write_text(json.dumps(report,indent=2))
    if missing: raise FileNotFoundError('Missing training inputs:\n'+'\n'.join(missing))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='drive_manifest.json');p.add_argument('--root',required=True)
    p.add_argument('--cache',default=None);p.add_argument('--cohorts',nargs='+',default=['blca','brca','coadread','hnsc','stad'])
    p.add_argument('--family',default='combine',choices=['combine','hallmarks','xena'])
    p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--retries',type=int,default=3);p.add_argument('--verify-only',action='store_true')
    args=p.parse_args(); root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(root/'download.log')])
    if not args.verify_only:
        config=json.loads(Path(args.manifest).read_text()); items=config['files']
        cache=Path(args.cache).resolve() if args.cache else root/'download_cache';cache.mkdir(parents=True,exist_ok=True)
        selected=[v for v in items if v.get('enabled',True) and (not v.get('cohort') or v['cohort'] in args.cohorts)]
        if len({i['name'] for i in selected})!=len(selected): raise ValueError('Duplicate archive names')
        for item in selected:
            path=download(item,cache,args.retries)
            actual=digest(path)
            if item.get('sha256') and actual.lower()!=item['sha256'].lower(): raise ValueError(f'Checksum mismatch: {path.name}')
            stamp=cache/(path.name+'.installed.json')
            marker=json.loads(stamp.read_text()) if stamp.exists() else {}
            valid=(marker.get('sha256')==actual and marker.get('root')==str(root) and marker.get('item')==item
                   and all(safe_path(root,f['path']).is_file() and safe_path(root,f['path']).stat().st_size==f['bytes'] for f in marker.get('files',[]))
                   and bool(marker.get('files')))
            if valid: logging.info('Already installed %s',path.name);continue
            files=extract_archive(path,item,root)
            temp=stamp.with_suffix('.tmp');temp.write_text(json.dumps(dict(sha256=actual,root=str(root),item=item,files=files),indent=2));temp.replace(stamp)
            logging.info('Installed %s (%d files)',path.name,len(files))
    report=verify_layout(root,args.cohorts,args.family,args.folds)
    logging.info('Training inputs ready: %s',json.dumps(report['cohorts']))

if __name__=='__main__': main()
