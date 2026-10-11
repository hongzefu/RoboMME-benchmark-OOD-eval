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


def guard_complete(control, base):
    """序列化严格完整终态，旧弱标记不再作为成功夹具。"""
    import hashlib
    identities=[f'smvla:ood:fixture-{i}' for i in range(4000)]
    config={**base,'root':str(control.parent),'expected_identities':identities,'expected_publications':[hashlib.sha256(k.encode()).hexdigest()+'.json' for k in identities]}
    config.pop('config_sha256',None)
    cfg=control/'terminal-config.json'; S.atomic(cfg,config)
    current={**base,'root':str(control.parent),'config_path':str(cfg),'config_sha256':S.sha(cfg)}
    ready=control/'launch.ready'
    if (control/'resume-ready.json').exists(): ready=control/'resume-ready.json'
    S.atomic(ready,current)
    terminal={k:current[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    terminal.update(schema='raw-ood-terminal-v1',status='COMPLETE',expected=4000,settled=4000,published=4000,infra=0,unresolved=0,unattempted=0,
                    published_identities=identities,failure_documents={},unresolved_identities=[],unattempted_identities=[])
    S.atomic(control/'gpu_work_complete.json',terminal)
    return terminal


def delivery_terminal_fixture(tmp_path, monkeypatch, *, infra):
    """真实控制器统计生产器写出四千身份终态；只有预算与小报告，无GPU或媒体。"""
    import json
    import raw_ood_controller as C
    from test_raw_ood_controller import watchdog_fixture
    ctl,_,_,_,_,_=watchdog_fixture(tmp_path,monkeypatch)
    cfg=ctl.control/'delivery-config.json'
    original=dict(ctl.c); original.pop('config_sha256'); C.atomic(cfg,original)
    ctl.c['config_sha256']=C.sha(cfg); ctl.base['config_sha256']=ctl.c['config_sha256']
    for path in tmp_path.rglob('*.json'):
        doc=C.read(path)
        if isinstance(doc,dict) and 'config_sha256' in doc:
            doc['config_sha256']=ctl.c['config_sha256']; C.atomic(path,doc)
    ctl.resume_existing_dispatches('c'*40)
    failed=next(iter(ctl.failures))
    if not infra:
        ctl.failures.clear()
        rows=[json.loads(line) for line in Path(ctl.c['budget_ledger']).read_text().splitlines()]
        for row in rows:
            if row.get('kind')=='commit' and row.get('infra') is True: row.update(infra=False,status='fail')
        Path(ctl.c['budget_ledger']).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    rows=[json.loads(line) for line in Path(ctl.c['budget_ledger']).read_text().splitlines()]
    reserve=next(row for row in rows if row['kind']=='reserve')
    commit=next(row for row in rows if row['kind']=='commit')
    normal=set(ctl.c['expected_identities'])-({failed} if infra else set())
    for index,key in enumerate(sorted(normal)):
        model,_,episode=key.split(':',2)
        # 不重复运行真实媒体编码；终态生产器只消费这些明确标注的CPU小报告。
        directory=ctl.gl/'cpu-terminal-fixture'/str(index)
        C.atomic(directory/'result.json',dict(task='VideoUnmask',tier='xhard1',task_success=False))
        C.atomic(ctl.gl/'published'/C.publication_name(key),dict(ctl.base,model=model,identity_key=key,relative_dir=str(directory.relative_to(ctl.gl))))
        route=ctl.c['models'][model]['policy']+('/'+C.ROUTES[model][2] if C.ROUTES[model][2] else '')+'/seed7/new'
        if not any(r.get('kind')=='reserve' and r.get('route')==route and r.get('key')==episode for r in rows):
            rid=f'cpu-delivery-{index}'; token=route+'|'+episode+'|a1'
            rows.extend([dict(reserve,rid=rid,key=episode,route=route,token=token),
                         dict(kind='reset_claim',rid=rid,what='build'),dict(kind='reset_claim',rid=rid,what='reset'),
                         dict(commit,rid=rid,infra=False,status='fail')])
    Path(ctl.c['budget_ledger']).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    ctl.published=normal; ctl.unresolved.clear()
    for active in ctl.active.values(): active['log'].close()
    ctl.active.clear()
    assert ctl.report()==(2 if infra else 0)
    terminal=C.read(ctl.gl/'gpu_work_complete.json')
    ready=dict(ctl.base,root=str(tmp_path),config_path=str(cfg)); C.atomic(ctl.control/'launch.ready',ready)
    return ctl,terminal,cfg


@pytest.mark.parametrize('infra',[False,True])
def test_real_terminal_producer_delivery_guard_and_attached_consume(tmp_path,monkeypatch,infra):
    import types
    from eval_video_mover import raw_validate_delivery
    ctl,terminal,cfg=delivery_terminal_fixture(tmp_path,monkeypatch,infra=infra)
    report=dict(ctl.base,schema='raw-ood-delivery-v1',status=terminal['status'],expected=4000,moved=4000-int(infra),infra=int(infra),
                unresolved=0,unattempted=0,raw_missing=int(infra),missing=int(infra),decode_fail=0,sha_mismatch=0,pending=0,
                moved_identities=terminal['published_identities'],terminal_sha256=S.sha(ctl.gl/'gpu_work_complete.json'),time=time.time())
    S.atomic(ctl.gl/'mover/completed.json',report)
    assert raw_validate_delivery(S.read(ctl.gl/'mover/completed.json'),terminal,ctl.c)==report
    proof=S.controller_terminal_proof(ctl.c,ctl.base); assert proof['terminal']['infra']==int(infra)
    monkeypatch.setattr(S,'process_identity',lambda pid:None)
    attached=S.AttachedController(ctl.c,123,{}); assert attached.poll()==0
    receipt=S.read(ctl.control/'controller-attach-terminal.json'); assert receipt['terminal']['status']==terminal['status']
    S.atomic(ctl.control/'jobs.json',dict(run='fixture',controller_job_id='100',gpu_job_ids=['101','102','103','104']))
    S.atomic(ctl.control/'mover-heartbeat.json',dict(ctl.base,time=time.time(),pending=0,moved=report['moved'],last_progress=time.time()))
    monkeypatch.setenv('SLURM_JOB_ID','100')
    monkeypatch.setattr(S,'start_guardians',lambda c,guardians=None:[])
    monkeypatch.setattr(S.subprocess,'Popen',lambda *a,**kw:types.SimpleNamespace(pid=123,poll=lambda:2 if infra else 0))
    assert S.run(cfg)==(2 if infra else 0)
    final=S.read(ctl.control/'final-report.json')
    assert final['cpu_hold'] is True and final['delivery']['raw_missing']==int(infra)
    assert S.read(ctl.control/'intentional-supervisor-stop.json')['gpu_action']=='none'
    for zero in ('infra','unresolved','unattempted'):
        broken=dict(terminal); broken.pop(zero)
        from raw_ood_context import validate_terminal_marker
        with pytest.raises(ValueError): validate_terminal_marker(broken,ctl.c)
    for mutation in ('unknown','duplicate'):
        broken=dict(terminal); broken['published_identities']=list(terminal['published_identities'])
        if mutation=='unknown': broken['published_identities'][0]='smvla:ood:unknown'
        else: broken['published_identities'].append(broken['published_identities'][0])
        with pytest.raises(ValueError): validate_terminal_marker(broken,ctl.c)
    # 真shellguard通过同一配置原字节与共同helper，不查询/取消任何真实作业。
    monkeypatch.undo()
    proc=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'fixture','100',str(int(time.time()+60))],
                        env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',SLURM_STEP_ID='0'),capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stdout+proc.stderr
    assert 'GUARD_TERMINAL status='+terminal['status'] in proc.stdout
    assert not (ctl.control/'STOP').exists()


def test_heartbeat_missing_fields_and_staleness(tmp_path):
    base=dict(run='r',run_name='r',exec_commit='a'*40,config_sha256='b'*64,controller_job_id='100')
    p=tmp_path/'beat.json'; S.atomic(p,dict(base,time=time.time(),pending=0,moved=0,last_progress=time.time()))
    assert S.check_heartbeat(p,base,now=time.time())['pending']==0
    broken=dict(base,time=time.time()-121); S.atomic(p,broken)
    with pytest.raises(RuntimeError): S.check_heartbeat(p,base,now=time.time())
    del broken['run']; S.atomic(p,broken)
    with pytest.raises(KeyError): S.check_heartbeat(p,base,now=time.time())


def test_guard_historical_original_remains_but_new_error_is_rejected(tmp_path):
    import types
    control=tmp_path/'control'; control.mkdir()
    base=dict(root=str(tmp_path),run='r',run_name='r',exec_commit='a'*40,config_sha256='b'*64,controller_job_id='100')
    original=control/'guard-failure-101.json'
    S.atomic(original,dict(base,gpu_job_id='101',time=time.time(),message='已认可历史故障'))
    saved=original.read_bytes()
    c=dict(base,seats=[dict(job_id='101')],historical_errors=[dict(path=str(original),sha256=S.sha(original))])
    guardians=[dict(job_id='101',process=types.SimpleNamespace(poll=lambda:None))]
    S.check_guardians(guardians,c)
    current=control/'guard-failure-101.current-fixture.json'; S.atomic(current,dict(base,gpu_job_id='101',time=time.time(),message='本实例新错误'))
    with pytest.raises(RuntimeError,match='本实例新错误'): S.check_guardians(guardians,c)
    assert original.read_bytes()==saved
    original.write_text('{"error":"同路径改变"}')
    with pytest.raises(ValueError,match='SHA'): S.check_guardians(guardians,c)


def test_gpu_guard_term_with_complete_never_stop_or_cancel(tmp_path):
    control=tmp_path/'control'; control.mkdir()
    base=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,config_sha256='b'*64)
    S.atomic(control/'launch.ready',base); guard_complete(control,base)
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    scancel=bindir/'scancel'; scancel.write_text('#!/bin/bash\necho "$*" >> '+str(cancelled)+'\n'); scancel.chmod(0o700)
    marker=tmp_path/'python-started'
    interpreter=bindir/'managed-python'
    interpreter.write_text('#!/bin/bash\ntouch '+str(marker)+'\nsleep .15\nexec '+sys.executable+' "$@"\n'); interpreter.chmod(0o700)
    env=dict(os.environ,BENCH_PY=str(interpreter),SLURM_JOB_ID='101',SLURM_STEP_ID='0',PATH=str(bindir)+':'+os.environ['PATH'])
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    # 在正常标记存在时立即TERM，覆盖尚未轮询到标记的清理路径。
    deadline=time.monotonic()+3
    while not marker.exists() and time.monotonic()<deadline: time.sleep(.01)
    assert marker.exists()
    if p.poll() is None: p.terminate()
    out=p.communicate(timeout=5)[0]
    assert p.returncode==0,out
    assert not (control/'STOP').exists() and not cancelled.exists()


def test_gpu_guard_cpu_missing_before_ready_exits_without_cancel(tmp_path):
    control=tmp_path/'control'; control.mkdir(); (control/'gpu-job-ids.txt').write_text('101\n102\n103\n104\n')
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    for name,body in [('squeue','exit 0'),('scancel','echo "$*" >> '+str(cancelled))]:
        p=bindir/name; p.write_text('#!/bin/bash\n'+body+'\n'); p.chmod(0o700)
    env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',SLURM_STEP_ID='0',PATH=str(bindir)+':'+os.environ['PATH'])
    p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,capture_output=True,text=True,timeout=5)
    assert p.returncode!=0 and not (control/'STOP').exists() and not cancelled.exists()


def test_gpu_guard_hanging_query_is_bounded(tmp_path):
    control=tmp_path/'control'; control.mkdir(); (control/'gpu-job-ids.txt').write_text('101\n102\n103\n104\n')
    base=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,config_sha256='b'*64)
    S.atomic(control/'launch.ready',base); S.atomic(control/'cpu-heartbeat.json',dict(base,t=time.time()-121,loop_t=time.time()-121))
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    for name,body in [('squeue','exec sleep 60'),('scancel','echo "$*" >> '+str(cancelled))]:
        p=bindir/name; p.write_text('#!/bin/bash\n'+body+'\n'); p.chmod(0o700)
    env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',SLURM_STEP_ID='0',PATH=str(bindir)+':'+os.environ['PATH'])
    start=time.monotonic()
    p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+120))],env=env,capture_output=True,text=True,timeout=36)
    elapsed=time.monotonic()-start
    assert p.returncode==8 and elapsed<12,p.stdout+p.stderr
    assert not (control/'STOP').exists() and not cancelled.exists()


@pytest.mark.parametrize('explicit',[False,True])
def test_supervisor_guard_launch_and_exit_contract(tmp_path,monkeypatch,explicit):
    import types
    (tmp_path/'control').mkdir(); c=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,
         config_sha256='b'*64,python=sys.executable,cpu_end_time=1e12,seats=[{'job_id':str(i)} for i in range(101,105)])
    seen=[]
    def popen(cmd,**kw):
        seen.append((cmd,kw)); return types.SimpleNamespace(pid=1000+len(seen),poll=lambda:None)
    monkeypatch.setattr(S.subprocess,'Popen',popen); monkeypatch.setenv('SLURM_JOB_ID','100'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(S.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout='RUNNING\n',stderr=''))
    if explicit: c['ready_path']=str(tmp_path/'control/resume-ready.json')
    ready=Path(c.get('ready_path',tmp_path/'control/launch.ready')); S.atomic(ready,c)
    guardians=S.start_guardians(c)
    assert len(seen)==4
    for cmd,kw in seen:
        assert cmd[-1]==str(ready)
        assert '--gres=none' in cmd and '--job-name=raw-ood-lifecycle-guard' in cmd
        assert 'SLURM_JOB_ID' not in kw['env'] and 'CUDA_VISIBLE_DEVICES' not in kw['env']
    S.check_guardians(guardians,c)
    guardians[0]['process'].poll=lambda:0
    with pytest.raises(RuntimeError): S.check_guardians(guardians,c)
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    guard_complete(tmp_path/'control',base)
    c.update(S.read(tmp_path/'control/terminal-config.json')); c['config_sha256']=S.sha(tmp_path/'control/terminal-config.json')
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
    S.atomic(tmp_path/'control/launch.ready',c)
    guards=S.start_guardians(c); assert len(guards)==3 and not any('--jobid=104' in x for x in launched)
    S.start_guardians(c,guards); assert len(launched)==3
    state['104']='RUNNING'
    cache=S.GUARD_QUERY_FAILURES[(c['root'],c['config_sha256'])]
    monkeypatch.setattr(S.time,'monotonic',lambda:cache['104']['next_query'])
    S.start_guardians(c,guards); assert len(guards)==4 and len(launched)==4
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
    try:
        assert S.run(cfg)==1
        assert children[-1].poll() is None
    finally:
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)
        for p in children:
            if p.poll() is None: p.kill()
            p.wait(timeout=5)
    assert time.monotonic()-start<5
    assert cancelled==[]
    assert not (control/'STOP').exists() and '独立守卫' in S.read(control/'supervisor-failure.json')['message']
    assert S.read(control/'intentional-supervisor-stop.json')['notification_established'] is False


def test_guard_exit_never_touches_other_cleanup_lock_or_jobs(tmp_path):
    import fcntl, threading
    control=tmp_path/'control'; control.mkdir(); (control/'gpu-job-ids.txt').write_text('101\n102\n103\n104\n')
    bindir=tmp_path/'bin'; bindir.mkdir(); cancelled=tmp_path/'cancelled'
    for name,body in [('squeue','exit 0'),('scancel','echo "$*" >> '+str(cancelled))]:
        p=bindir/name; p.write_text('#!/bin/bash\n'+body+'\n'); p.chmod(0o700)
    lock=(control/'cleanup.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX)
    thread=threading.Thread(target=lambda:(time.sleep(2.5),fcntl.flock(lock,fcntl.LOCK_UN))); thread.start()
    env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',SLURM_STEP_ID='0',PATH=str(bindir)+':'+os.environ['PATH'])
    try:
        p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,capture_output=True,text=True,timeout=6)
    finally: thread.join(); lock.close()
    assert p.returncode!=0 and not cancelled.exists() and not (control/'STOP').exists()


def guard_fixture(root):
    control=root/'control'; control.mkdir(parents=True)
    base=dict(root=str(root),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,config_sha256='b'*64,expected=4000,published=4000)
    S.atomic(control/'launch.ready',base); S.atomic(control/'cpu-heartbeat.json',dict(base,t=time.time(),loop_t=time.time()))
    (control/'gpu-job-ids.txt').write_text('101\n102\n103\n104\n')
    bindir=root/'bin'; bindir.mkdir(); cancelled=root/'cancelled'
    for name,body in [('squeue','echo RUNNING'),('scancel','echo "$*" >> '+str(cancelled))]:
        p=bindir/name; p.write_text('#!/bin/bash\n'+body+'\n'); p.chmod(0o700)
    env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='101',SLURM_STEP_ID='0',PATH=str(bindir)+':'+os.environ['PATH'])
    return control,base,env,cancelled


def test_continuous_query_failure_does_not_override_fresh_heartbeat(tmp_path):
    import threading
    control,base,env,cancelled=guard_fixture(tmp_path)
    query=tmp_path/'bin/squeue'; query.write_text('#!/bin/bash\nexit 1\n')
    stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            S.atomic(control/'cpu-heartbeat.json',dict(base,t=time.time(),loop_t=time.time()))
            stop.wait(.2)
    writer=threading.Thread(target=heartbeat); writer.start()
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+120))],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        deadline=time.monotonic()+3
        while not (control/'guardian-heartbeat-101.json').exists() and p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        assert (control/'guardian-heartbeat-101.json').exists()
        time.sleep(31)
        assert p.poll() is None
        stop.set(); writer.join(timeout=2)
        S.atomic(control/'cpu-heartbeat.json',dict(base,t=time.time()-121,loop_t=time.time()-121))
        out=p.communicate(timeout=15)[0]
        assert p.returncode==8 and 'state=UNKNOWN' in out
        assert S.read(control/'guard-failure-101.json')['phase']=='heartbeat_validate'
        assert not cancelled.exists() and not (control/'STOP').exists()
    finally:
        stop.set(); writer.join(timeout=2)
        if p.poll() is None: p.kill(); p.wait(timeout=3)


def test_unknown_query_without_verified_ready_has_initial_120_second_bound(tmp_path):
    control,base,env,cancelled=guard_fixture(tmp_path); (control/'launch.ready').unlink()
    (tmp_path/'bin/squeue').write_text('#!/bin/bash\nexit 1\n')
    # 注入只用于夹具的wall clock，真正shell走120秒边界，避免实等两分钟。
    counter=tmp_path/'clock-count'; clock=tmp_path/'bin/date'; initial=int(time.time())
    clock.write_text('#!/bin/bash\nn=0\n[[ ! -f '+str(counter)+' ]] || n=$(cat '+str(counter)+')\necho $((n+1)) > '+str(counter)+'\nif (( n==0 )); then echo '+str(initial)+'; else echo '+str(initial+121)+'; fi\n'); clock.chmod(0o700)
    p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(initial+600)],env=env,capture_output=True,text=True,timeout=5)
    assert p.returncode==7 and 'state=UNKNOWN' in p.stdout
    assert not cancelled.exists() and not (control/'guardian-ready-101.json').exists()


@pytest.mark.parametrize('kind',['new','symlink','outside'])
def test_explicit_guard_ready_binds_new_context_without_cancel(tmp_path,kind):
    control,base,env,cancelled=guard_fixture(tmp_path)
    current=dict(base,exec_commit='d'*40,config_sha256='e'*64)
    ready=control/'resume-ready.json'; S.atomic(ready,current)
    S.atomic(control/'cpu-heartbeat.json',dict(current,t=time.time(),loop_t=time.time()))
    if kind=='symlink':
        linked=control/'linked.json'; linked.symlink_to(ready); ready=linked
    elif kind=='outside': ready=tmp_path/'outside.json'; S.atomic(ready,current)
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60)),str(ready)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if kind=='new':
        marker=control/'guardian-ready-101.json'; deadline=time.monotonic()+5
        while not marker.exists() and p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        assert marker.exists()
        row=S.read(marker); assert row['config_sha256']=='e'*64 and row['exec_commit']=='d'*40
        guard_complete(control,current); p.terminate()
    out=p.communicate(timeout=20)[0]
    assert p.returncode==(0 if kind=='new' else 8),out
    assert not cancelled.exists() and not (control/'STOP').exists()


@pytest.mark.parametrize('kind',['healthy_old_receipt','current_intent','wrong_intent'])
def test_guard_exit_uses_current_payload_state_not_old_receipt(tmp_path,kind):
    control,base,env,cancelled=guard_fixture(tmp_path)
    receipt=control/'guardian-intent-accepted-101.json'
    stale=dict(base,config_sha256='c'*64,exec_commit='d'*40)
    S.atomic(receipt,stale); before=receipt.read_bytes()
    if kind!='healthy_old_receipt':
        S.atomic(control/'intentional-supervisor-stop.json',base if kind=='current_intent' else stale)
    if kind=='current_intent': S.atomic(control/'controller-heartbeat.json',dict(base,t=time.time(),loop_t=time.time(),pid=123,start_ticks=10))
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        if kind in ('healthy_old_receipt','current_intent'):
            marker=control/'guardian-ready-101.json'; deadline=time.monotonic()+3
            while not marker.exists() and p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
            assert marker.exists()
            time.sleep(.2)
            assert p.poll() is None
            if kind=='healthy_old_receipt': assert receipt.read_bytes()==before
            else: assert S.read(control/'guardian-heartbeat-101.json')['monitor']=='controller'
            # 正常完成标记仅用于结束本夹具，不触及其它作业。
            guard_complete(control,base); p.terminate()
        out=p.communicate(timeout=15)[0]
        assert p.returncode==(8 if kind=='wrong_intent' else 0),out
        if kind=='current_intent': assert S.read(receipt)==base and 'phase=intentional_supervisor_stop' in out
        elif kind=='wrong_intent': assert receipt.read_bytes()==before and S.read(control/'guard-failure-101.json')['phase']=='intentional_supervisor_stop'
        assert not cancelled.exists() and not (control/'STOP').exists()
    finally:
        if p.poll() is None: p.kill(); p.wait(timeout=5)


@pytest.mark.parametrize('rc',[0,1])
def test_supervisor_hold_wrapper_preserves_controller_after_supervisor_exit(tmp_path,rc):
    import json,signal
    control=tmp_path/'control'; control.mkdir()
    pidfile=control/'controller.pid'
    code="import subprocess,pathlib,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); sys.exit(int(sys.argv[2]))"
    parent=subprocess.Popen(['bash',str(HERE/'raw_ood_supervisor_hold.sh'),str(tmp_path),sys.executable,'-c',code,str(pidfile),str(rc)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    child=None
    try:
        record=control/f'supervisor-hold-{parent.pid}.json'; deadline=time.monotonic()+4
        while not record.exists() and parent.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        assert record.exists() and parent.poll() is None
        child=int(pidfile.read_text()); os.kill(child,0)
        report=json.loads(record.read_text()); assert report['supervisor_rc']==rc and report['hold'] is True
        assert report['restarts']==report['child_signals']==report['gpu_cancellations']==0
        parent.terminate(); parent.wait(timeout=3)
        # wrapper的TERM只结束自身，没有向仍存活的控制器发送信号。
        os.kill(child,0)
        assert not (control/'STOP').exists()
    finally:
        # 只清理此夹具独立进程组，绝不匹配其它任务。
        os.killpg(parent.pid,signal.SIGKILL)
        parent.wait(timeout=3)


def test_old_five_second_startup_can_stop_fresh_heartbeat_new_isolated_startup_passes(tmp_path):
    legacy_source=subprocess.run(['git','show','4f3cc20b9513294674e17ecb5f9f31b7d3dd23c1:dev-scripts/gl/raw_ood_hold_guard.sh'],
                                 cwd=HERE,capture_output=True,text=True,check=True).stdout
    legacy=tmp_path/'legacy.sh'; legacy.write_text(legacy_source)
    for name,script in [('old',legacy),('new',HERE/'raw_ood_hold_guard.sh')]:
        root=tmp_path/name; control,base,env,cancelled=guard_fixture(root)
        marker=root/'python-once'; py=root/'bin/delayed-python'
        py.write_text('#!/bin/bash\nif [[ ! -f '+str(marker)+' ]]; then touch '+str(marker)+'; sleep 6; fi\nexec '+sys.executable+' "$@"\n'); py.chmod(0o700)
        env['BENCH_PY']=str(py)
        p=subprocess.Popen(['bash',str(script),str(root),'r','100',str(int(time.time()+120))],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if name=='old':
            out=p.communicate(timeout=9)[0]
            assert p.returncode==8 and not (control/'guardian-ready-101.json').exists(),out
            assert time.time()-S.read(control/'cpu-heartbeat.json')['t']<120
            assert cancelled.exists()
        else:
            deadline=time.monotonic()+10
            while not (control/'guardian-ready-101.json').exists() and time.monotonic()<deadline: time.sleep(.02)
            assert (control/'guardian-ready-101.json').exists()
            guard_complete(control,base)
            out=p.communicate(timeout=15)[0]
            assert p.returncode==0 and not (control/'STOP').exists() and not cancelled.exists(),out
            assert 'phase=python_entered' in out and 'phase=validated' in out and 'payload_rc=0' in out


def test_actual_read_deadline_and_first_failure_phase_are_preserved(tmp_path):
    control,base,env,cancelled=guard_fixture(tmp_path)
    heartbeat=control/'cpu-heartbeat.json'; heartbeat.unlink(); os.mkfifo(heartbeat)
    p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+120))],
                     env=env,capture_output=True,text=True,timeout=10)
    d=S.read(control/'guard-failure-101.json')
    assert p.returncode==8 and d['phase']=='heartbeat_read' and d['error_kind']=='TimeoutError' and d['payload_rc']==1
    assert not cancelled.exists() and not (control/'STOP').exists() and 'GUARD_CHECK_END gpu=101 payload_rc=1' in p.stdout


@pytest.mark.parametrize('kind',['write_once','rename_once','expired_cache','no_cache'])
def test_filesystem_timeout_retries_only_with_unexpired_verified_heartbeat(tmp_path,kind):
    control,base,env,cancelled=guard_fixture(tmp_path)
    if kind=='expired_cache': S.atomic(control/'cpu-heartbeat.json',dict(base,t=time.time()-119,loop_t=time.time()-119))
    marker=tmp_path/'slow-injected'; shim=tmp_path/'isolated-python'
    shim.write_text('#!'+sys.executable+'\n'+
        "import os,pathlib,sys,time\nargs=sys.argv[1:]; i=args.index('-'); sys.argv=['-',*args[i+1:]]\n"+
        'marker=pathlib.Path('+repr(str(marker))+'); mode='+repr(kind)+'\n'+
        "original_write=pathlib.Path.write_text; original_read=pathlib.Path.read_text; original_replace=os.replace\n"+
        "def slow_once():\n if not marker.exists(): marker.touch(); time.sleep(6)\n"+
        "def write(self,*a,**k):\n if self.name.startswith('guardian-ready-') and mode in ('write_once','expired_cache'): slow_once()\n return original_write(self,*a,**k)\n"+
        "def read(self,*a,**k):\n if self.name=='cpu-heartbeat.json' and mode=='no_cache': slow_once()\n return original_read(self,*a,**k)\n"+
        "def replace(src,dest):\n if pathlib.Path(dest).name.startswith('guardian-ready-') and mode=='rename_once': slow_once()\n return original_replace(src,dest)\n"+
        "pathlib.Path.write_text=write; pathlib.Path.read_text=read; os.replace=replace\nexec(compile(sys.stdin.read(),'<真实guard payload>','exec'))\n")
    shim.chmod(0o700); env['BENCH_PY']=str(shim)
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+120))],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        deadline=time.monotonic()+3
        while not marker.exists() and p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        assert marker.exists()
        time.sleep(5.2)
        if kind in ('write_once','rename_once'):
            assert p.poll() is None
            assert not (control/'guardian-ready-101.json').exists() and not (control/'guardian-heartbeat-101.json').exists()
            deadline=time.monotonic()+14
            while not (control/'guardian-heartbeat-101.json').exists() and p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
            assert (control/'guardian-heartbeat-101.json').exists() and p.poll() is None
            assert S.read(control/'guardian-heartbeat-101.json')['verified_t']==S.read(control/'cpu-heartbeat.json')['t']
            guard_complete(control,base); p.terminate(); out=p.communicate(timeout=15)[0]
            assert p.returncode==0 and 'GUARD_NFS_RETRY' in out
        else:
            out=p.communicate(timeout=3)[0]
            assert p.returncode==8 and 'GUARD_NFS_RETRY' not in out
            assert S.read(control/'guard-failure-101.json')['error_kind']=='TimeoutError'
        assert not cancelled.exists() and not (control/'STOP').exists()
    finally:
        if p.poll() is None: p.kill(); p.wait(timeout=3)


def test_stale_heartbeat_threshold_unchanged_and_primary_error_precedes_secondary_pipe(tmp_path):
    import types
    control,base,env,cancelled=guard_fixture(tmp_path); env['SLURM_JOB_ID']='102'
    S.atomic(control/'cpu-heartbeat.json',dict(base,t=time.time()-121))
    p=subprocess.run(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+120))],
                     env=env,capture_output=True,text=True,timeout=5)
    d=S.read(control/'guard-failure-102.json')
    assert p.returncode==8 and d['phase']=='heartbeat_validate' and d['error_kind']=='AssertionError'
    c=dict(base,seats=[dict(job_id=str(i)) for i in range(101,105)])
    guardians=[dict(job_id='101',process=types.SimpleNamespace(poll=lambda:141)),dict(job_id='102',process=types.SimpleNamespace(poll=lambda:8))]
    with pytest.raises(RuntimeError,match='原始失败') as error: S.check_guardians(guardians,c)
    assert '102' in str(error.value) and 'heartbeat_validate' in str(error.value)


def test_final_cleanup_fresh_query_preserves_initial_completing_evidence(tmp_path,monkeypatch):
    import types
    (tmp_path/'control').mkdir(); base=dict(run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,config_sha256='b'*64)
    c=dict(base,root=str(tmp_path),seats=[dict(job_id=str(i)) for i in range(101,105)])
    initial=dict(records=[dict(job_id=str(i),rc=0) for i in range(101,105)],active='101 COMPLETING',query_rc=0,cleanup_incomplete=True)
    path=tmp_path/'control/gpu-cleanup.json'; S.atomic(path,initial); original=path.read_bytes()
    monkeypatch.setattr(S.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout='',stderr=''))
    final=S.final_cleanup_evidence(c,base)
    assert not final['cleanup_incomplete'] and final['initial']['cleanup_incomplete'] and path.read_bytes()==original
    assert S.read(tmp_path/'control/gpu-cleanup-final.json')['final_query']['active']==''
    monkeypatch.setattr(S.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout='101 COMPLETING',stderr=''))
    assert S.final_cleanup_evidence(c,base)['cleanup_incomplete']
    monkeypatch.setattr(S.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=1,stdout='',stderr='query failed'))
    assert S.final_cleanup_evidence(c,base)['cleanup_incomplete']


@pytest.mark.parametrize('child_code',[0,1])
def test_cpu_bootstrap_keeps_parent_alive_after_child_exit(tmp_path,child_code):
    import json
    control=tmp_path/'control'; control.mkdir(); cfg=tmp_path/'config.json'; cfg.write_text('{}')
    child=tmp_path/'fake-supervisor.py'; child.write_text('import sys\nraise SystemExit('+str(child_code)+')\n')
    base=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,config_sha256=S.sha(cfg),
              config_path=str(cfg),python=sys.executable,supervisor=str(child))
    S.atomic(control/'launch.ready',base); S.atomic(control/'jobs.json',dict(run='r',controller_job_id='100',gpu_job_ids=['101','102','103','104']))
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_cpu_bootstrap.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],
                        env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='100'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        until=time.monotonic()+5
        while not (control/'cpu-bootstrap-outcome.json').exists() and time.monotonic()<until: time.sleep(.02)
        outcome=S.read(control/'cpu-bootstrap-outcome.json')
        assert outcome['supervisor_exit_code']==child_code and outcome['cpu_hold'] and not outcome['notification_established']
        assert p.poll() is None and not (control/'STOP').exists()
        assert S.read(control/'intentional-supervisor-stop.json')['config_sha256']==base['config_sha256']
    finally: p.terminate(); p.communicate(timeout=5)


def test_prepare_timeout_reports_once_and_keeps_cpu_hold(tmp_path):
    control=tmp_path/'control'; control.mkdir()
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_cpu_bootstrap.sh'),str(tmp_path),'r','100',str(int(time.time()-1))],
                        env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='100'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        until=time.monotonic()+3
        while not (control/'cpu-bootstrap-outcome.json').exists() and time.monotonic()<until: time.sleep(.02)
        d=S.read(control/'cpu-bootstrap-outcome.json')
        assert d['supervisor_exit_code']==5 and d['phase']=='prepare' and d['cpu_hold']
        assert p.poll() is None and not (control/'STOP').exists()
    finally: p.terminate(); p.communicate(timeout=5)


def test_guard_legitimate_supervisor_stop_keeps_fresh_controller_guard_alive(tmp_path):
    control,base,env,cancelled=guard_fixture(tmp_path)
    S.atomic(control/'cpu-heartbeat.json',dict(base,t=time.time()-121))
    S.atomic(control/'intentional-supervisor-stop.json',dict(base,reason='exception_stop_only_supervisor',cpu_hold=True))
    S.atomic(control/'controller-heartbeat.json',dict(base,t=time.time(),loop_t=time.time(),pid=123,start_ticks=10))
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        marker=control/'guardian-heartbeat-101.json'; deadline=time.monotonic()+3
        while not marker.exists() and p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        assert marker.exists() and p.poll() is None and S.read(marker)['monitor']=='controller'
        guard_complete(control,base); p.terminate(); out=p.communicate(timeout=15)[0]
        assert p.returncode==0 and not cancelled.exists() and not (control/'STOP').exists()
        assert (control/'guardian-intent-accepted-101.json').exists() and 'phase=intentional_supervisor_stop' in out
    finally:
        if p.poll() is None: p.kill(); p.wait(timeout=3)


def test_supervisor_process_exits_but_controller_guard_and_batch_pids_live(tmp_path):
    import json, signal
    from test_raw_ood_controller import config
    c=config(tmp_path); del c['config_sha256']; cfg=tmp_path/'config.json'; cfg.write_text(json.dumps(c)); c=S.load_config(cfg)
    control=tmp_path/'control'; control.mkdir(); base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    S.atomic(control/'launch.ready',dict(base,root=str(tmp_path)))
    S.atomic(control/'jobs.json',dict(run=c['run'],controller_job_id='100',gpu_job_ids=['101','102','103','104']))
    S.atomic(control/'mover-heartbeat.json',dict(base,time=time.time(),pending=0,moved=0,last_progress=time.time()))
    S.atomic(control/'mover-error.json',dict(base,error='fixture mover error'))
    code='''import json,os,pathlib,subprocess,sys,types
sys.path.insert(0,sys.argv[1]); import raw_ood_supervisor as S
root=pathlib.Path(sys.argv[2]); original=subprocess.Popen
def popen(cmd,**kw):
 p=original([sys.executable,"-c","import time;time.sleep(30)"],**kw)
 with (root/"control/fixture-children.jsonl").open("a") as f: f.write(json.dumps({"pid":p.pid,"role":"guard" if cmd[0]=="srun" else "controller"})+"\\n")
 return p
def query(cmd,**kw):
 if cmd[0]=="scancel": (root/"unexpected-scancel").write_text("called")
 return types.SimpleNamespace(returncode=0,stdout="RUNNING\\n" if "-o" in cmd and cmd[cmd.index("-o")+1]=="%T" else "",stderr="")
S.subprocess.Popen=popen; S.subprocess.run=query
raise SystemExit(S.run(pathlib.Path(sys.argv[3])))
'''
    batch=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    parent=subprocess.Popen([sys.executable,'-c',code,str(HERE),str(tmp_path),str(cfg)],env=dict(os.environ,SLURM_JOB_ID='100'),
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    child_pids=[]
    try:
        out=parent.communicate(timeout=5)[0]; assert parent.returncode==1,out
        rows=[json.loads(x) for x in (control/'fixture-children.jsonl').read_text().splitlines()]
        child_pids=[r['pid'] for r in rows]; assert len(child_pids)==5
        for pid in child_pids: os.kill(pid,0)
        assert batch.poll() is None and not (tmp_path/'unexpected-scancel').exists() and not (control/'STOP').exists()
        d=S.read(control/'supervisor-failure.json'); assert d['gpu_action']=='none' and d['controller_action']=='none' and not d['notification_established']
    finally:
        for pid in child_pids:
            try: os.kill(pid,signal.SIGTERM)
            except ProcessLookupError: pass
        if parent.poll() is None: parent.kill(); parent.wait(timeout=5)
        batch.terminate(); batch.wait(timeout=5)


def test_guard_batch_misuse_preserves_allocation_hold(tmp_path):
    control,base,env,cancelled=guard_fixture(tmp_path); env['SLURM_STEP_ID']='batch'
    p=subprocess.Popen(['bash',str(HERE/'raw_ood_hold_guard.sh'),str(tmp_path),'r','100',str(int(time.time()+60))],
                        env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        time.sleep(.1)
        assert p.poll() is None and not (control/'STOP').exists() and not cancelled.exists()
    finally: p.terminate(); out=p.communicate(timeout=5)[0]
    assert 'GUARD_BATCH_MISUSE' in out


def mail_config(tmp_path):
    c=dict(root=str(tmp_path),run='r',run_name='r',controller_job_id='100',exec_commit='a'*40,config_sha256='b'*64,
           notification=dict(enabled=True,to=S.MAIL_TO,command='/usr/sbin/sendmail',timeout=30))
    (tmp_path/'control').mkdir()
    return c


@pytest.mark.parametrize('mode',['existing','paused','both','neither'])
def test_attach_binds_exact_original_recovery_mode_without_new_controller(tmp_path,monkeypatch,mode):
    c=mail_config(tmp_path); c['python']=sys.executable; control=tmp_path/'control'
    cfg=control/'config.json'; S.atomic(cfg,c); source=tmp_path/'raw_ood_controller.py'; source.write_text('可信原源码夹具')
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}; now=time.time()
    origin=dict(base,orchestration_commit='c'*40,resume_existing_dispatches=mode in ('existing','both'),resume_paused=mode in ('paused','both'),resume_completed_smokes=False)
    S.atomic(control/'orchestration-started.json',origin); S.atomic(control/'supervisor-started.json',dict(base,controller_pid=777,time=now))
    S.atomic(control/'controller-heartbeat.json',dict(base,pid=777,t=now,loop_t=now,start_ticks=12))
    manifest=control/'original-lock.json'; S.atomic(manifest,dict(orchestration_commit='c'*40,worker_exec_commit=c['exec_commit'],config_sha256=c['config_sha256'],files={'raw_ood_controller.py':dict(bytes=source.stat().st_size,sha256=S.sha(source))}))
    flag='--resume-paused' if mode=='paused' else '--resume-existing-dispatches'
    proc=dict(state='S',command=[sys.executable,str(source),'run','--config',str(cfg),flag,'--orchestration-commit','c'*40],start_ticks=12,start_time=now,cgroup='0::/slurm/job_100/step_12/task0')
    monkeypatch.setattr(S,'process_identity',lambda pid:proc)
    monkeypatch.setattr(S.subprocess,'Popen',lambda *a,**kw:pytest.fail('附着禁止新控制器'))
    before={p:p.read_bytes() for p in control.iterdir()}
    if mode in ('both','neither'):
        with pytest.raises(ValueError): S.attach_controller(c,cfg,777,source,manifest,S.sha(manifest))
    else:
        process,proof=S.attach_controller(c,cfg,777,source,manifest,S.sha(manifest))
        assert process.poll() is None and proof['command']==proc['command'] and proof['budget_action']=='none'
    assert all(p.read_bytes()==raw for p,raw in before.items())


@pytest.mark.parametrize('kind',['valid','missing','context','channel'])
def test_legacy_notification_reuses_only_same_channel_received_proof(tmp_path,kind):
    import copy
    c=mail_config(tmp_path); c['models']={}; old=copy.deepcopy(c); old.pop('config_sha256')
    path=tmp_path/'control/old-config.json'; S.atomic(path,old); old=S.load_config(path)
    current=copy.deepcopy(old); current.update(exec_commit='d'*40,config_sha256='e'*64,
        legacy_source=dict(config=str(path),sha256=S.sha(path),exec_commit=old['exec_commit']))
    base={k:old[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    S.atomic(tmp_path/'control/notification-probe.json',dict(base,event='probe',state='submitted',rc=0,to=S.MAIL_TO,subject='原探针',time=time.time()))
    proof=dict(base,received=True,source='user_confirmation',to=S.MAIL_TO,subject='原探针')
    if kind=='context': proof['controller_job_id']='999'
    if kind=='channel': current['notification']['command']='/usr/bin/mailx'
    if kind!='missing': S.atomic(tmp_path/'control/mail-delivery-confirmed.json',proof)
    before={p:p.read_bytes() for p in (tmp_path/'control').iterdir()}
    if kind=='valid':
        report=S.require_probe(current); assert report['reused'] and report['delivery_verified']
        assert report['legacy_config_sha256']==old['config_sha256']
    else:
        with pytest.raises((ValueError,FileNotFoundError)): S.require_probe(current)
    assert all(p.read_bytes()==data for p,data in before.items())


def test_notification_submit_is_bound_idempotent_and_not_delivery(tmp_path,monkeypatch):
    import types
    c=mail_config(tmp_path); calls=[]
    def run(cmd,**kw):
        calls.append((cmd,kw)); return types.SimpleNamespace(returncode=0,stdout=b'queued fixture',stderr=b'')
    monkeypatch.setattr(S.subprocess,'run',run)
    report=S.notify(c,'probe',{'test':'只用CPU邮件替身'})
    assert report['state']=='submitted' and report['rc']==0 and not report['delivery_verified'] and not report['notification_established']
    assert calls[0][0]==['/usr/sbin/sendmail','-i','-t'] and b'hongzefu@umich.edu' in calls[0][1]['input']
    S.require_probe(c); assert S.notify(c,'probe',{})==report and len(calls)==1
    bad=dict(c,config_sha256='c'*64)
    with pytest.raises(ValueError): S.require_probe(bad)


def test_each_supervisor_instance_notifies_once_and_preserves_legacy_receipt(tmp_path,monkeypatch):
    import types
    c=mail_config(tmp_path); calls=[]; legacy=tmp_path/'control/notification-failure.json'
    S.atomic(legacy,dict(config_sha256='old',state='failed')); original=legacy.read_bytes()
    monkeypatch.setattr(S.subprocess,'run',lambda cmd,**kw:calls.append(cmd) or types.SimpleNamespace(returncode=0,stdout=b'queued',stderr=b''))
    for instance in ('instance-1','instance-2'):
        monkeypatch.setattr(S,'NOTIFICATION_INSTANCE',instance)
        record=S.notify(c,'failure',dict(error='本实例错误'))
        assert record['supervisor_instance']==instance and record['state']=='submitted'
        assert S.notify(c,'failure',{})==record
        assert S.read(Path(record['receipt_path']))==record
    assert len(calls)==2 and legacy.read_bytes()==original
    latest=S.read(tmp_path/'control/notification-latest-failure.json'); assert latest['supervisor_instance']=='instance-2'
    assert (tmp_path/'control/notifications/instance-1/failure.json').exists()


def test_notification_failure_does_not_stop_other_pids(tmp_path,monkeypatch):
    import types
    c=mail_config(tmp_path); child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    monkeypatch.setattr(S.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=75,stdout=b'',stderr=b'fixture MTA refused'))
    try:
        assert S.fail(c,RuntimeError('fixture failure'),process=child)==1 and child.poll() is None
        r=S.read(tmp_path/'control/notification-failure.json')
        assert r['state']=='failed' and r['rc']==75 and r['stderr']=='fixture MTA refused'
        assert not (tmp_path/'control/STOP').exists()
    finally: child.terminate(); child.wait(timeout=5)


def test_notification_timeout_and_recipient_guard(tmp_path,monkeypatch):
    c=mail_config(tmp_path)
    def hung(*a,**kw): raise subprocess.TimeoutExpired(a[0],30)
    monkeypatch.setattr(S.subprocess,'run',hung)
    r=S.notify(c,'completed',{})
    assert r['state']=='failed' and r['error_kind']=='TimeoutExpired' and r['rc'] is None
    c['notification']['to']='other@example.com'
    with pytest.raises(ValueError): S.notification_config(c)


@pytest.mark.parametrize('resume',[False,True])
def test_supervisor_resume_flag_is_only_explicit_cli_passthrough(monkeypatch,resume):
    calls=[]
    monkeypatch.setattr(S,'run',lambda path,**kwargs:calls.append((path,kwargs)) or 0)
    argv=['supervisor','run','--config','fixture.json']
    if resume: argv+=['--resume-completed-smokes','--orchestration-commit','c'*40]
    monkeypatch.setattr(sys,'argv',argv)
    assert S.main()==0
    assert calls[0][1]['resume_completed_smokes'] is resume
    assert calls[0][1]['orchestration_commit']==('c'*40 if resume else None)


def test_supervisor_existing_flag_only_explicit_passthrough(monkeypatch):
    calls=[]; monkeypatch.setattr(S,'run',lambda path,**kw:calls.append(kw) or 0)
    monkeypatch.setattr(sys,'argv',['supervisor','run','--config','fixture.json','--resume-existing-dispatches','--orchestration-commit','c'*40])
    assert S.main()==0 and calls[0]['resume_existing_dispatches'] is True and calls[0]['resume_completed_smokes'] is False


def test_supervisor_unknown_query_cooldown_keeps_guardians_unlaunched(tmp_path,monkeypatch):
    import types
    c=mail_config(tmp_path); c.update(python=sys.executable,cpu_end_time=1e12,seats=[{'job_id':'104'}]); S.atomic(tmp_path/'control/launch.ready',c)
    clock=[0.]; calls=[]; monkeypatch.setattr(S.time,'monotonic',lambda:clock[0])
    def query(cmd,**kw):
        calls.append(cmd); raise subprocess.TimeoutExpired(cmd,5)
    monkeypatch.setattr(S.subprocess,'run',query); monkeypatch.setattr(S.subprocess,'Popen',lambda *a,**kw:pytest.fail('未知席不能启动守卫'))
    guards=S.start_guardians(c); assert not guards and len(calls)==1
    clock[0]=4; S.start_guardians(c,guards); assert len(calls)==1
    clock[0]=30
    with pytest.raises(RuntimeError,match='SLURM_QUERY_WINDOW'): S.start_guardians(c,guards)
