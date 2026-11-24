"""Replay cases that were deferred by RAM pressure when there is actual headroom."""

import argparse
import json
from pathlib import Path
import re
import time

from design import named_lanes, unique
from rcwa_runner import Job, Setting, estimate_memory, intervals, native, run_jobs
from response_geometry import ROOT, load_mark


def limited():
    missing={}
    for logfile in sorted((ROOT/"queue").glob("*.jsonl")):
        for line in logfile.read_text(encoding="utf-8").splitlines():
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                continue
            folder=ROOT/row["folder"]
            deferred=row.get("result")=="resource_limit"
            interrupted=(row.get("result")=="failed" and "Failed to evaluate code" in row.get("error","")
                         and (folder/"model.fsp").is_file())
            if not (deferred or interrupted):
                continue
            if not (folder/"done.json").is_file():
                missing[folder]=row
    return missing


def settings(folder):
    pattern=r"k(\d+)x(\d+)_d([0-9.]+)x([0-9.]+)_a(\d+)_h(\d+)(_li)?"
    found=re.fullmatch(pattern,folder.name)
    if found is None:
        raise ValueError(f"Unknown setting directory {folder.name}")
    ku,kv,du,dv,n,h,li=found.groups()
    return Setting(int(ku),int(kv),float(du),float(dv),int(n),int(h),bool(li))


def materialize(missing):
    legacy=None
    for folder,row in missing.items():
        file=ROOT/"geometry"/f"{row['case']}.npz"
        if file.is_file():
            yield Job(load_mark(file),settings(folder),row["group"])
        else:
            if legacy is None:
                legacy={j.folder:j for joblist in named_lanes(3,10).values() for j in joblist}
            if folder in legacy:
                yield legacy[folder]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait",action="store_true")
    p.add_argument("--threads",type=int,default=5)
    args=p.parse_args()
    attempted=set()
    while True:
        pending=[j for j in unique(materialize(limited())) if j.folder not in attempted]
        if not pending:
            break
        # Close each native session before looking for the next memory-heavy case.
        possible=[j for j in pending
                  if estimate_memory(j.setting,len(intervals(j.mark,j.setting))-1)+2<native.free_ram_gib()]
        if possible:
            selected=possible[0]
            print("Retrying deferred native case",selected.mark.name,selected.setting.key,flush=True)
            run_jobs([selected],"ram_retry",args.threads)
            attempted.add(selected.folder)
        elif args.wait:
            time.sleep(40)
        else:
            break
    tail=limited()
    print(f"{len(tail)} cases still lack native results after the RAM deferral.",flush=True)


if __name__=="__main__":
    main()

