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


def test_gpu_guard_hanging_query_is_bounded(tmp_path):
    control=tmp_path/'control'; control.mkdir(); (control/'gpu-job-ids.txt').write_text('101\n102\n103\n104\n')
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    for name,body in [('squeue','exec sleep 60'),('scancel','echo "$*" >> '+str(cancelled))]:
        p=bindir/name; p.write_text('#!/bin/bash\n'+body+'\n'); p.chmod(0o700)
    env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',PATH=str(bindir)+':'+os.environ['PATH'])
    start=time.monotonic()
    p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+120))],env=env,capture_output=True,text=True,timeout=36)
    elapsed=time.monotonic()-start
    assert p.returncode==7 and elapsed<35,p.stdout+p.stderr
    assert (control/'STOP').exists() and cancelled.read_text().splitlines()==['102','103','104','101']


def test_supervisor_guard_launch_and_exit_contract(tmp_path,monkeypatch):
    import types
    (tmp_path/'control').mkdir(); c=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,
         config_sha256='b'*64,python=sys.executable,cpu_end_time=1e12,seats=[{'job_id':str(i)} for i in range(101,105)])
    seen=[]
    def popen(cmd,**kw):
        seen.append((cmd,kw)); return types.SimpleNamespace(pid=1000+len(seen),poll=lambda:None)
    monkeypatch.setattr(S.subprocess,'Popen',popen); monkeypatch.setenv('SLURM_JOB_ID','100'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(S.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout='RUNNING\n',stderr=''))
    guardians=S.start_guardians(c)
    assert len(seen)==4
    for cmd,kw in seen:
        assert '--gres=none' in cmd and '--job-name=raw-ood-lifecycle-guard' in cmd
        assert 'SLURM_JOB_ID' not in kw['env'] and 'CUDA_VISIBLE_DEVICES' not in kw['env']
    S.check_guardians(guardians,c)
    guardians[0]['process'].poll=lambda:0
    with pytest.raises(RuntimeError): S.check_guardians(guardians,c)
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    S.atomic(tmp_path/'control/gpu_work_complete.json',base)
    guardians[0]['process'].poll=lambda:143
    S.check_guardians(guardians,c)
    for g in guardians: g['log'].close()


def test_guardians_wait_pending_without_srun_or_duplicate(tmp_path,monkeypatch):
    import types
    (tmp_path/'control').mkdir(); c=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,
         config_sha256='b'*64,python=sys.executable,cpu_end_time=1e12,seats=[{'job_id':str(i)} for i in range(101,105)])
    state={str(i):'RUNNING' for i in range(101,104)}; state['104']='PENDING'; launched=[]
    def query(cmd,**kw): return types.SimpleNamespace(returncode=0,stdout=state[cmd[cmd.index('-j')+1]],stderr='')
    def popen(cmd,**kw):
        launched.append(cmd); return types.SimpleNamespace(pid=1000+len(launched),poll=lambda:None)
    monkeypatch.setattr(S.subprocess,'run',query); monkeypatch.setattr(S.subprocess,'Popen',popen)
    guards=S.start_guardians(c); assert len(guards)==3 and not any('--jobid=104' in x for x in launched)
    S.start_guardians(c,guards); assert len(launched)==3
    state['104']='RUNNING'; S.start_guardians(c,guards); assert len(guards)==4 and len(launched)==4
    for g in guards: g['log'].close()


def test_real_guard_child_death_interrupts_supervisor_within_five_seconds(tmp_path,monkeypatch):
    import json, signal, types
    from test_raw_ood_controller import config
    c=config(tmp_path); del c['config_sha256']; cfg=tmp_path/'run-config.json'; cfg.write_text(json.dumps(c))
    c=S.load_config(cfg); control=tmp_path/'control'; control.mkdir()
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    S.atomic(control/'launch.ready',dict(base,root=str(tmp_path)))
    S.atomic(control/'jobs.json',dict(run=c['run'],controller_job_id='100',gpu_job_ids=['101','102','103','104']))
    S.atomic(control/'mover-heartbeat.json',dict(base,time=time.time(),pending=0,moved=0,last_progress=time.time()))
    original_popen=subprocess.Popen; children=[]; cancelled=[]
    def popen(cmd,**kw):
        script='import time;time.sleep(.2);raise SystemExit(3)' if cmd[0]=='srun' else 'import time;time.sleep(30)'
        p=original_popen([sys.executable,'-c',script],**kw); children.append(p); return p
    def query(cmd,**kw):
        if cmd[0]=='scancel': cancelled.append(cmd[1]); output=''
        elif '-o' in cmd and cmd[cmd.index('-o')+1]=='%T': output='RUNNING\n'
        else: output=''
        return types.SimpleNamespace(returncode=0,stdout=output,stderr='')
    monkeypatch.setattr(S.subprocess,'Popen',popen); monkeypatch.setattr(S.subprocess,'run',query)
    monkeypatch.setenv('SLURM_JOB_ID','100')
    old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    start=time.monotonic()
    try: assert S.run(cfg)==1
    finally:
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)
        for p in children:
            if p.poll() is None: p.kill()
            p.wait(timeout=5)
    assert time.monotonic()-start<5
    assert cancelled==['101','102','103','104']
    assert (control/'STOP').exists() and '独立守卫' in S.read(control/'failure.json')['message']


def test_guard_retries_cleanup_lock_after_first_owner_disappears(tmp_path):
    import fcntl, threading
    control=tmp_path/'control'; control.mkdir(); (control/'gpu-job-ids.txt').write_text('101\n102\n103\n104\n')
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    for name,body in [('squeue','exit 0'),('scancel','echo "$*" >> '+str(cancelled))]:
        p=bindir/name; p.write_text('#!/bin/bash\n'+body+'\n'); p.chmod(0o700)
    lock=(control/'cleanup.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX)
    thread=threading.Thread(target=lambda:(time.sleep(2.5),fcntl.flock(lock,fcntl.LOCK_UN))); thread.start()
    env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',PATH=str(bindir)+':'+os.environ['PATH'])
    try:
        p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,capture_output=True,text=True,timeout=6)
    finally: thread.join(); lock.close()
    assert p.returncode!=0 and cancelled.read_text().splitlines()==['102','103','104','101']
