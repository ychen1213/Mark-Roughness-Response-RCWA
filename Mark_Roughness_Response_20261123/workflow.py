"""Finish fitting and reveal the frozen holdout after the native queues supply its inputs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from design import blind_jobs, kernel_jobs, pair_jobs, reference_jobs
from response_geometry import ROOT
from response_model import OUTPUT

MODEL=OUTPUT/"response_model.npz"


def completed(job):
    return (job.folder/"done.json").is_file()


def stages():
    modes=kernel_jobs(3,10)
    interactions=pair_jobs(3,10)
    core=[j for j in modes if j.group=="mode_reference" or
          (j.group=="mode_bank" and j.mark.info["terms"][0][2]<=16)]
    cross=[j for j in interactions if max(t[2] for t in j.mark.info["terms"])<=16]
    training=reference_jobs(3)
    nominal=[j for j in training if not j.mark.vertically_varying]
    return {"single_modes":core,"mixed_modes":cross,"joint_training":training,
            "eligible_for_nominal_calibration":nominal,"blind_reference":blind_jobs(3)}


def output_status(checks,finished=False):
    report={"local_time":datetime.now().astimezone().isoformat(),"frozen_model_ready":MODEL.is_file(),
            "primary_score_complete":finished}
    for group,jobs in checks.items():
        missing=[j.mark.name for j in jobs if not completed(j)]
        report[group]={"complete":len(jobs)-len(missing),"planned":len(jobs),"examples_waiting":missing[:6]}
    path=OUTPUT/"status.json"
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)
    return report


def call(script,*args):
    subprocess.run([sys.executable,"-u","-B",str(ROOT/script),*map(str,args)],
                   cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Write a progress snapshot and exit")
    parser.add_argument("--parameter-sweeps",action="store_true")
    options=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    checks=stages()
    while True:
        state=output_status(checks)
        if options.check:
            print(json.dumps(state,ensure_ascii=False,indent=2))
            return
        required=("single_modes","mixed_modes","eligible_for_nominal_calibration")
        if not MODEL.is_file() and all(completed(j) for name in required for j in checks[name]):
            call("response_model.py","fit","--calibrate","--out",MODEL)
        if MODEL.is_file() and all(completed(j) for j in checks["blind_reference"]):
            break
        time.sleep(30)
    call("report_analysis.py","--model",MODEL,"--blind")
    output_status(checks,finished=True)
    if options.parameter_sweeps:
        call("decisions.py","--model",MODEL)
    print("Primary held-out analysis and artifact generation complete.",flush=True)


if __name__=="__main__":
    main()

