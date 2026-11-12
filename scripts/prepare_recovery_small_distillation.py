"""Recovery-specific data adapter for the shared small-student BC trainer; no physics."""
import argparse,json
from pathlib import Path
from native_stand.students.recovery_distillation import prepare,export
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');p.add_argument('--output-dir',type=Path);p.add_argument('--export-student',type=Path);p.add_argument('--dataset',type=Path);a=p.parse_args()
out=a.output_dir or ROOT/'isolated'/new_experiment_name('recovery1500_small31_dataset')
if a.export_student:
 if not a.dataset or not a.output_dir:p.error('--export-student requires --dataset and --output-dir')
 if a.write:print(json.dumps(export(a.export_student,a.dataset,a.output_dir),indent=2))
 else:print(json.dumps({'status':'PLAN_ONLY','student':str(a.export_student),'dataset':str(a.dataset),'output':str(a.output_dir),'simulation':0,'optimizer_steps':0},indent=2))
elif not a.write:print(json.dumps({'status':'PLAN_ONLY','output':str(out),'student':[31,128,64,32,6],'trainer':'native_stand.students.distill.train','optimizer_steps':0,'simulation':0},indent=2))
else:print(json.dumps(prepare(ROOT/'runs/2026_10_17-17_59__recovery31_standing_contact_quality/formal/model_update_1500.pt',ROOT/'configs/experiments/2026_10_17-17_59__recovery31_standing_contact_quality/spec.json',ROOT/'isolated/recovery31_contact_quality_u1500_frozen_20261018/manifest.json',out),indent=2))
