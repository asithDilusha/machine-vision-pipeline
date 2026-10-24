"""Build the assignment archive from explicit deliverables, never .git or a venv."""
import argparse
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True)
    args=parser.parse_args()
    if not args.index.isalnum(): parser.error('Index must contain only letters and digits')
    root=Path(__file__).resolve().parents[1]
    files=['main.py','vision.py','requirements.txt','README.md','report.docx',
           'tests/test_pipeline.py','tools/package_submission.py',
           'results/groups.txt','results/groups.json']
    for name in files:
        if not (root/name).is_file(): raise SystemExit(f'Missing deliverable: {name}')
    out=root/'submission';out.mkdir(exist_ok=True)
    target=out/(args.index+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files: archive.write(root/name,name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None: raise SystemExit('Archive integrity check failed')
    print(target)


if __name__=='__main__': main()
