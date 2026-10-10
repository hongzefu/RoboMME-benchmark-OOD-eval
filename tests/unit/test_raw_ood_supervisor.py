"""真实JSON往返及GPU守卫终止夹具，无真实Slurm操作。"""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

HERE=Path(__file__).resolve().parents[2]/'dev-scripts/gl'
sys.path.insert(0,str(HERE))
import raw_ood_supervisor as S


def test_heartbeat_missing_fields_and_staleness(tmp_path):
    base=dict(run='r',run_name='r',exec_commit='a'*40,config_sha256='b'*64,controller_job_id='100')
    p=tmp_path/'beat.json'; S.atomic(p,dict(base,time=time.time(),pending=0,moved=0,last_progress=time.time()))
    assert S.check_heartbeat(p,base,now=time.time())['pending']==0
    broken=dict(base,time=time.time()-121); S.atomic(p,broken)
    with pytest.raises(RuntimeError): S.check_heartbeat(p,base,now=time.time())
    del broken['run']; S.atomic(p,broken)
    with pytest.raises(KeyError): S.check_heartbeat(p,base,now=time.time())


def test_gpu_guard_term_with_complete_never_stop_or_cancel(tmp_path):
    control=tmp_path/'control'; control.mkdir()
    base=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,config_sha256='b'*64)
    S.atomic(control/'launch.ready',base); S.atomic(control/'gpu_work_complete.json',base)
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    scancel=bindir/'scancel'; scancel.write_text('#!/bin/bash\necho "$*" >> '+str(cancelled)+'\n'); scancel.chmod(0o700)
    marker=tmp_path/'python-started'
    interpreter=bindir/'managed-python'
    interpreter.write_text('#!/bin/bash\ntouch '+str(marker)+'\nsleep .15\nexec '+sys.executable+' "$@"\n'); interpreter.chmod(0o700)
    env=dict(os.environ,BENCH_PY=str(interpreter),SLURM_JOB_ID='101',PATH=str(bindir)+':'+os.environ['PATH'])
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    # 在正常标记存在时立即TERM，覆盖尚未轮询到标记的清理路径。
    deadline=time.monotonic()+3
    while not marker.exists() and time.monotonic()<deadline: time.sleep(.01)
    assert marker.exists()
    if p.poll() is None: p.terminate()
    out=p.communicate(timeout=5)[0]
    assert p.returncode==0,out
    assert not (control/'STOP').exists() and not cancelled.exists()


def test_gpu_guard_cpu_missing_before_ready_stops_registered_only(tmp_path):
    control=tmp_path/'control'; control.mkdir(); (control/'gpu-job-ids.txt').write_text('101\n102\n103\n104\n')
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    for name,body in [('squeue','exit 0'),('scancel','echo "$*" >> '+str(cancelled))]:
        p=bindir/name; p.write_text('#!/bin/bash\n'+body+'\n'); p.chmod(0o700)
    env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',PATH=str(bindir)+':'+os.environ['PATH'])
    p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,capture_output=True,text=True,timeout=5)
    assert p.returncode!=0 and (control/'STOP').exists()
    assert cancelled.read_text().splitlines()==['102','103','104','101']
