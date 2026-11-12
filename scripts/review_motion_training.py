"""One-shot report of existing Motion metrics; never launch physics or a watcher."""
import argparse,json
from pathlib import Path
from native_stand.students.motion_evaluation_review import report,markdown

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--window',type=int,default=100);p.add_argument('--output-dir',type=Path)
 a=p.parse_args()
 if not 1<=a.window<=1000:p.error('--window must be1..1000')
 result=report(a.run_dir,a.window)
 if a.output_dir:
  a.output_dir.mkdir(parents=True,exist_ok=False)
  (a.output_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
  (a.output_dir/'REPORT.md').write_text(markdown(result))
 print(markdown(result))
if __name__=='__main__':main()
