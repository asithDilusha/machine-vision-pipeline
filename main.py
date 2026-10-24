"""Command-line entry point. Pillow is used only for image I/O and mode decoding."""
import argparse
import json
from pathlib import Path
import sys
import time
from PIL import Image, UnidentifiedImageError
from vision import grayscale, median3, gaussian3, segment, trace_boundary, resample, chain_code, cluster


def save_gray(path, pixels):
    image=Image.new('L',(len(pixels[0]),len(pixels)))
    image.putdata([max(0,min(255,round(v))) for row in pixels for v in row])
    image.save(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Native 32-digit chain-code image grouping')
    parser.add_argument('input',type=Path,help='Folder of PNG/JPG/JPEG images (nonrecursive)')
    parser.add_argument('--output',type=Path,default=Path('results/groups.txt'))
    parser.add_argument('--diagnostics',type=Path,default=Path('results/diagnostics'))
    parser.add_argument('--threshold',type=float,default=.06,help='Complete-link distance cutoff in [0,1] (default .06)')
    parser.add_argument('--max-size',type=int,default=320,help='Longest processing dimension (default 320)')
    args=parser.parse_args(argv)
    if not args.input.is_dir(): parser.error('Input folder does not exist')
    if not 0<=args.threshold<=1: parser.error('Threshold must be in [0,1]')
    if args.max_size<32: parser.error('Maximum size must be at least 32')
    paths=sorted((p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}),key=lambda p:p.name.casefold())
    if not paths: parser.error('Input folder contains no PNG/JPG/JPEG images')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.diagnostics.mkdir(parents=True,exist_ok=True)
    records=[]; errors=[]; started=time.perf_counter()
    for index,path in enumerate(paths):
        try:
            with Image.open(path) as source:
                source.load(); rgb=source.convert('RGB'); w,h=rgb.size
                pixels = rgb.get_flattened_data() if hasattr(rgb, 'get_flattened_data') else rgb.getdata()
                gray=grayscale(list(pixels),w,h,args.max_size)
            filtered=gaussian3(median3(gray))
            mask,threshold=segment(filtered)
            boundary=trace_boundary(mask); sampled=resample(boundary); code=chain_code(sampled)
            prefix=f'{index+1:03d}_{path.stem}'
            save_gray(args.diagnostics/f'{prefix}_gray.png',gray)
            save_gray(args.diagnostics/f'{prefix}_filtered.png',filtered)
            save_gray(args.diagnostics/f'{prefix}_mask.png',[[255*int(v) for v in row] for row in mask])
            overlay=[list(row) for row in [[255*int(v) for v in row] for row in mask]]
            for x,y in sampled: overlay[round(y)][round(x)]=128
            save_gray(args.diagnostics/f'{prefix}_boundary.png',overlay)
            records.append(dict(filename=path.name,source_size=[w,h],processing_size=[len(gray[0]),len(gray)],threshold=round(threshold,3),area=sum(map(sum,mask)),boundary_pixels=len(boundary),chain_code=code,sampled_boundary=sampled))
            print(f'{path.name}: {code}',flush=True)
        except (OSError,UnidentifiedImageError,ValueError) as exc:
            errors.append(dict(filename=path.name,error=str(exc)))
            print(f'ERROR {path.name}: {exc}',file=sys.stderr)
    groups,distances=cluster([r['chain_code'] for r in records],args.threshold)
    lines=['Machine vision image groups',f'Distance threshold: {args.threshold}',f'Processed: {len(records)}; failed: {len(errors)}','']
    for k,group in enumerate(groups,1):
        lines.append(f'Group {k} ({len(group)} images)')
        lines.extend('  '+records[i]['filename'] for i in group); lines.append('')
    if errors:
        lines.append('FAILED INPUTS'); lines.extend(f"  {r['filename']}: {r['error']}" for r in errors)
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metadata=dict(parameters=dict(threshold=args.threshold,max_size=args.max_size),elapsed_seconds=round(time.perf_counter()-started,3),images=records,groups=[[records[i]['filename'] for i in g] for g in groups],distances=[dict(a=records[i]['filename'],b=records[j]['filename'],distance=round(d,6)) for (i,j),d in distances.items()],errors=errors)
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Wrote {args.output}; {len(groups)} groups; {len(errors)} errors')
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
