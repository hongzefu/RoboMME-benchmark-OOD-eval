"""控制器CPU契约：身份互斥、配置缺项和精确清理。"""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

HERE=Path(__file__).resolve().parents[2]/'dev-scripts/gl'
sys.path.insert(0,str(HERE))
import raw_ood_controller as C


def identities():
    rows=[]
    for task,counts in C.TASK_COUNTS.items():
        ep=0
        for n,count in enumerate(counts,1):
            tier='xhard4' if task in ('MoveCube','InsertPeg') else f'xhard{n}'
            for seed in range(count):
                rows.append(dict(task=task,tier=tier,dataset='ood',builder_episode=ep,key=f'{task}_{tier}_{seed}',candidate=seed,spec_sha256='a'*64))
                ep+=1
    return rows


def config(tmp_path):
    p=tmp_path/'ids.jsonl'; p.write_text(''.join(json.dumps(r)+'\n' for r in identities()))
    return dict(run='fixture',run_name='fixture',root=str(tmp_path),gl_root=str(tmp_path/'gl'),controller_job_id='100',
                exec_commit='a'*40,config_sha256='b'*64,identities=str(p),identities_sha256=C.sha(p),models={m:dict(policy=C.ROUTES[m][0],label=C.ROUTES[m][1],
                   args=(['--groundsg-variant',C.ROUTES[m][2]] if C.ROUTES[m][2] else [])+
                        (['--qwenvl-groundsg-adapter','/locked/adapter'] if m=='qwenvl' else [])) for m in C.MODELS},
                seats=[dict(job_id=str(i),seat=f's{i}',port_base=22000+i,end_time=1e12) for i in range(101,105)],
                python=sys.executable,runner=str(HERE/'run_eval_gl.sh'),budget_ledger=str(tmp_path/'budget'),cpu_end_time=1e12,
                expected_identities=[C.identity(m,r) for m in C.MODELS for r in identities()],
                expected_publications=[C.publication_name(C.identity(m,r)) for m in C.MODELS for r in identities()])


def test_shards_and_smoke_have_no_extra_identity(tmp_path):
    c=config(tmp_path); rows=C.validate_config(c); shards=C.make_shards(rows)
    assert len(shards)==40
    for model in C.MODELS:
        flattened=[r for s in shards.values() if s['model']==model for r in s['rows']]
        assert len(flattened)==800 and len({r['key'] for r in flattened})==800
        assert shards[model+'-00']['rows'][0]['task']=='VideoUnmask'


def test_config_missing_identity_and_duplicate_job_fail(tmp_path):
    c=config(tmp_path); c['seats'][1]['job_id']=c['seats'][0]['job_id']
    with pytest.raises(ValueError): C.validate_config(c)
    c=config(tmp_path); p=Path(c['identities']); p.write_text(p.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError): C.validate_config(c)


def test_config_file_digest_does_not_self_reference(tmp_path):
    p=tmp_path/'config.json'; p.write_text('{"run":"fixture"}')
    assert C.load_config(p)['config_sha256']==C.sha(p)


def test_cancel_only_registered_gpu(tmp_path,monkeypatch):
    c=config(tmp_path); calls=[]
    def run(args,**kw):
        calls.append(args); return types.SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(C.subprocess,'run',run)
    report=C.cancel_gpu(c)
    assert [a for a in calls if a[0]=='scancel']==[['scancel',str(i)] for i in range(101,105)]
    assert report['cleanup_incomplete'] is False


def test_dispatch_clears_cpu_slurm_env_and_expands_ownership(tmp_path,monkeypatch):
    c=config(tmp_path); c['models']['qwenvl']['args']+=['--work-dir','{work_dir}','--cfg','server_dir={work_dir}/server']
    c['models']['qwenvl']['env']={'BENCH_PY':'/client/python'}
    controller=C.Controller(c); seen={}
    monkeypatch.setenv('SLURM_JOB_ID','100'); monkeypatch.setenv('SBATCH_GRES','gpu:1'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    def popen(cmd,**kw):
        seen.update(cmd=cmd,**kw); return types.SimpleNamespace(pid=999)
    monkeypatch.setattr(C.subprocess,'Popen',popen)
    controller.start(c['seats'][0],'qwenvl-00',smoke=True)
    assert 'SLURM_JOB_ID' not in seen['env'] and 'SBATCH_GRES' not in seen['env']
    assert seen['env']['BENCH_PY']=='/client/python'
    assert 'CUDA_VISIBLE_DEVICES' not in seen['env']
    assert seen['cmd'][seen['cmd'].index('--gpus')+1]==''
    assert '{work_dir}' not in ' '.join(seen['cmd']) and '--jobid=101' in seen['cmd']
    controller.active['101']['log'].close()


def test_missing_report_key_and_infra_block_publication(tmp_path):
    c=config(tmp_path); controller=C.Controller(c)
    a=dict(model='qwenvl',rows=controller.shards['qwenvl-00']['rows'],shard='qwenvl-00',smoke=True)
    with pytest.raises(ValueError): controller.publish(a,{'infra':False})
    row=dict(key=a['rows'][0]['key'],infra=True,final=True,accepted=True,result='none',exec_steps=1,
             task_success=False,budget_exhausted=False,run_blocked=False)
    with pytest.raises(RuntimeError): controller.publish(a,row)


def test_normal_task_failure_is_published_with_explicit_zero(tmp_path):
    c=config(tmp_path); controller=C.Controller(c); r=controller.shards['qwenvl-00']['rows'][0]
    directory=Path(c['gl_root'])/'rollouts/qwenvl/ood/seed7/raw/ep0'; directory.mkdir(parents=True)
    for name in ('front.mkv','wrist.mkv','arrays.npz','trace.jsonl','meta.json','events.jsonl','frames-front.jsonl','frames-wrist.jsonl'):
        (directory/name).write_text('fixture')
    result=directory/'result.json'; result.write_text(json.dumps(dict(key=r['key'],dataset='ood',infra=False,task_success=False)))
    a=dict(model='qwenvl',rows=[r],shard='qwenvl-00',smoke=True)
    row=dict(key=r['key'],infra=False,final=True,accepted=True,result=str(result),exec_steps=1800,
             task_success=False,budget_exhausted=False,run_blocked=False)
    controller.publish(a,row)
    key=C.identity('qwenvl',r); publication=C.read(Path(c['gl_root'])/'published'/C.publication_name(key))
    assert publication['identity_key']==key and publication['delete_files']==['front.mkv','wrist.mkv','arrays.npz']
    event=json.loads((tmp_path/'control/events.jsonl').read_text().splitlines()[-1])
    assert event['task_success'] is False


def test_real_format_report_nullable_policy_timing(tmp_path,monkeypatch):
    import budget_ledger
    c=config(tmp_path); controller=C.Controller(c)
    caps=dict(kind='config',schema='sgeval-budget/3',trajectory_cap=4000,reset_cap=8000,planned_first_tries=4000,shared_infra_cap=0,expired_cap=0)
    Path(c['budget_ledger']).write_text(json.dumps(caps)+'\n')
    pubs={}; results={}
    for m in C.MODELS:
        for r in identities():
            key=C.identity(m,r); controller.published.add(key)
            directory=f"{m}/{r['key']}"
            pubs[C.publication_name(key)]=dict(model=m,relative_dir=directory)
            results[directory]=dict(task=r['task'],tier=r['tier'],task_success=False,
                timing={'episode_wall_s':1.0,'recorder':None,'policy':None})
    def read(path):
        if path.name=='result.json': return results[path.parent.relative_to(Path(c['gl_root'])).as_posix()]
        return pubs[path.name]
    monkeypatch.setattr(C,'read',read)
    monkeypatch.setattr(budget_ledger,'BudgetState',lambda _:types.SimpleNamespace(bad_rows=0,trajectories=4000,resets=8000,
         retries=[],recovery_used=0,astra=0,claimed={'x':8000},orphan_claims=0,config=caps))
    controller.report()
    report=json.loads((tmp_path/'control/results-report.json').read_text())
    assert report['infra']==0 and report['missing']==0
    for m in C.MODELS:
        stat=report['models'][m]
        assert stat['accepted']==800 and stat['failed']==800 and stat['policy_timing_missing']==800
        assert stat['steady_decision_wall_ms_mean'] is None and stat['recorder_finalize_measured']==0


def test_controller_heartbeat_has_single_thread_writer(tmp_path,monkeypatch):
    import threading
    c=config(tmp_path); controller=C.Controller(c); writers=[]; original=C.atomic
    def atomic(path,obj):
        if path.name=='controller-heartbeat.json': writers.append(threading.get_ident())
        original(path,obj)
    monkeypatch.setattr(C,'atomic',atomic)
    monkeypatch.setattr(controller,'scan',lambda:(tmp_path/'control/STOP').write_text('fixture'))
    monkeypatch.setattr(controller,'dispatch',lambda:None)
    with pytest.raises(RuntimeError,match='STOP'): controller.run()
    assert writers and len(set(writers))==1 and writers[0]!=threading.get_ident()


@pytest.mark.parametrize('mutation',['astra','memer','budget','cfg_gpu','cfg_variant_alias','option_variant_alias'])
def test_route_or_budget_override_blocked_before_process(tmp_path,mutation):
    c=config(tmp_path)
    if mutation=='astra': c['models']['qwenvl']['policy']='astra'
    elif mutation=='memer': c['models']['qwenvl']['args'][1]='ground-sg-memer'
    elif mutation=='budget': c['models']['pp']['args']+=['--trajectory-cap','9999']
    elif mutation=='cfg_gpu': c['models']['pp']['args']+=['--cfg','gpus=0']
    elif mutation=='cfg_variant_alias': c['models']['qwenvl']['args']+=['--cfg','groundsg-variant=ground-sg-memer']
    else: c['models']['qwenvl']['args']+=['--groundsg_variant','ground-sg-memer']
    with pytest.raises(ValueError): C.validate_config(c)


def test_identity_same_size_mutation_and_manifest_mismatch_blocked(tmp_path):
    c=config(tmp_path); p=Path(c['identities']); old=p.read_bytes()
    p.write_bytes(old.replace(b'aaaaaaaa',b'bbbbbbbb',1)); assert p.stat().st_size==len(old)
    with pytest.raises(ValueError,match='SHA'): C.validate_config(c)
    c=config(tmp_path); c['expected_identities'][0]='wrong'
    with pytest.raises(ValueError,match='manifest'): C.validate_config(c)


def test_guardian_step_does_not_hide_worker_first_step(tmp_path,monkeypatch):
    import time
    c=config(tmp_path); controller=C.Controller(c)
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**kw:types.SimpleNamespace(pid=999))
    controller.start(c['seats'][0],'qwenvl-00',smoke=False); a=controller.active['101']
    def run(*args,**kwargs):
        return types.SimpleNamespace(returncode=0,stdout=f"101.1 raw-ood-lifecycle-guard\n101.2 {a['worker_name']}\n",stderr='')
    monkeypatch.setattr(C.subprocess,'run',run)
    row=a['rows'][1]
    C.atomic(a['progress'],dict(label=c['models']['qwenvl']['label'],phase='episode',key=row['key'],pid=250,t=time.time(),slurm_job_id='101',slurm_step_id='2'))
    ep=Path(c['gl_root'])/'rollouts'/c['models']['qwenvl']['label']/'ood/seed7/progress.json'
    C.atomic(ep,dict(key=row['key'],pid=250,step=1))
    controller.check_worker(a); controller.poll_ack(a)
    ack=C.read(tmp_path/'control/first_dispatch.json')
    assert ack['step_id']=='101.2' and ack['first_step']==1 and ack['identity_key']==C.identity('qwenvl',row)
    a['log'].close()


def test_controller_exception_keeps_existing_step_process_alive(tmp_path,monkeypatch):
    controller=C.Controller(config(tmp_path))
    child=C.subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    controller.active={'fixture':{'process':child}}
    def broken(): raise RuntimeError('fixture controller failure')
    monkeypatch.setattr(controller,'scan',broken)
    try:
        with pytest.raises(RuntimeError,match='fixture'): controller.run()
        assert child.poll() is None and not (tmp_path/'control/STOP').exists()
    finally: child.terminate(); child.wait(timeout=5)


def prior_config(tmp_path):
    import budget_ledger
    c=config(tmp_path); c['run']=c['run_name']='ood-five-raw-seed7-20261010-02'
    control=tmp_path/'control'; control.mkdir()
    old=control/'prior-budget-ledger.jsonl'
    old.write_bytes((HERE.parents[1]/'docs/validation/ood-five-raw-seed7-20261010-01/records/stopped-budget-ledger.jsonl').read_bytes())
    c['prior_work']=dict(ledger=str(old),sha256=C.PRIOR_SHA256,expected_trajectories=1,expected_reset_claims=2,
                        combined_trajectory_cap=4001,combined_reset_cap=8002)
    budget_ledger.BudgetLedger(c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000).check()
    return c


def test_cross_round_budget_start_and_final_are_read_only(tmp_path):
    c=prior_config(tmp_path); old=Path(c['prior_work']['ledger']); original=old.read_bytes()
    assert len(C.validate_config(c))==800
    assert C.work_budget(c)['combined']['trajectories']==1
    final=types.SimpleNamespace(trajectories=4000,claimed={'all':8000},orphan_claims=0,resets=8000,retries=[],bad_rows=0,astra=0,recovery_used=0)
    report=C.work_budget(c,current=final)
    assert report['combined']['trajectories']==4001 and report['combined']['reset_claims']==8002
    assert old.read_bytes()==original and not old.with_suffix(old.suffix+'.lock').exists()


@pytest.mark.parametrize('kind',['tamper','missing_prior','wrong_cap','wrong_expectation','missing_current','over_current'])
def test_cross_round_budget_rejects_tamper_omission_and_overage(tmp_path,kind):
    c=prior_config(tmp_path)
    if kind=='tamper':
        p=Path(c['prior_work']['ledger']); p.write_bytes(p.read_bytes().replace(b'4000',b'4001',1))
    elif kind=='missing_prior': del c['prior_work']
    elif kind=='wrong_cap': c['prior_work']['combined_reset_cap']=8000
    elif kind=='wrong_expectation': c['prior_work']['expected_trajectories']=0
    elif kind=='missing_current': Path(c['budget_ledger']).unlink()
    else:
        state=types.SimpleNamespace(trajectories=4001,claimed={'all':8001},orphan_claims=0,resets=8001,retries=[],bad_rows=0,astra=0,recovery_used=0)
        with pytest.raises(ValueError): C.work_budget(c,current=state)
        return
    with pytest.raises((ValueError,KeyError)): C.validate_config(c)


def scan_fixture(tmp_path,*,rc=0):
    import time
    c=config(tmp_path); ctl=C.Controller(c); ident=ctl.shards['pp-00']['rows'][0]
    inv=tmp_path/'control/invocations/pp-00-smoke'; inv.mkdir(parents=True)
    progress=inv/'progress.json'; results=Path(c['gl_root'])/'seats/pp/seed7/s3-pp-00-smoke/seat-results.jsonl'
    results.parent.mkdir(parents=True)
    C.atomic(progress,dict(label='pp',phase='finished',key=None,pid=1000,t=time.time()-300,slurm_job_id='103',slurm_step_id='1'))
    directory=Path(c['gl_root'])/'rollouts/pp/ood/seed7/raw/VideoUnmask_ep0_xhard1'; directory.mkdir(parents=True)
    for name in ('front.mkv','wrist.mkv','arrays.npz','trace.jsonl','meta.json','events.jsonl','frames-front.jsonl','frames-wrist.jsonl'):
        (directory/name).write_text('fixture')
    result=directory/'result.json'; C.atomic(result,dict(key=ident['key'],dataset='ood',infra=False,task_success=1))
    row=dict(key=ident['key'],infra=False,final=True,accepted=True,result=str(result),exec_steps=276,
             task_success=1,budget_exhausted=False,run_blocked=False)
    a=dict(process=types.SimpleNamespace(poll=lambda:rc),log=(inv/'worker.log').open('a'),seat=c['seats'][2],shard='pp-00',
           smoke=True,model='pp',progress=progress,results=results,rows=[ident],offset=0,started=time.time()-300,
           acked=False,phase_key=None,phase_start=time.time()-300,step_id='103.1',worker_name='raw-ood-worker-pp-00-smoke')
    ctl.active['103']=a
    return ctl,a,row


def test_terminal_single_lf_report_is_read_once_despite_stale_heartbeat(tmp_path):
    ctl,a,row=scan_fixture(tmp_path); a['results'].write_text(json.dumps(row)+'\n')
    ctl.scan()
    assert 'pp' in ctl.smoked and not ctl.active and a['offset']==1
    proof=C.read(a['progress'].parent/'final-confirmation.json')
    assert proof['state']=='confirmed' and proof['rc']==0 and proof['consumed_lines']==1 and proof['missing_identities']==[]


def test_terminal_file_appears_late_without_relaunch_or_republish(tmp_path,monkeypatch):
    ctl,a,row=scan_fixture(tmp_path); clock=[10.]
    monkeypatch.setattr(C.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**k:pytest.fail('不得重新启动worker'))
    monkeypatch.setattr(C.subprocess,'run',lambda *a,**k:pytest.fail('已退出worker不应查询或取消作业'))
    ctl.scan(); assert ctl.active and not ctl.smoked and a['offset']==0
    clock[0]=30; a['results'].write_text(json.dumps(row)+'\n'); ctl.scan()
    assert not ctl.active and ctl.smoked=={'pp'} and a['offset']==1 and len(ctl.published)==1


def test_terminal_partial_tail_becomes_complete_jsonl(tmp_path):
    ctl,a,row=scan_fixture(tmp_path); text=json.dumps(row)
    a['results'].write_text(text[:len(text)//2]); ctl.scan()
    assert ctl.active and a['offset']==0
    a['results'].write_text(text+'\n'); ctl.scan()
    assert not ctl.active and a['offset']==1


def test_terminal_permanent_missing_fails_at_bounded_deadline_with_evidence(tmp_path,monkeypatch):
    ctl,a,row=scan_fixture(tmp_path); clock=[0.]
    monkeypatch.setattr(C.time,'monotonic',lambda:clock[0])
    ctl.scan(); clock[0]=119; ctl.scan(); assert ctl.active
    clock[0]=120
    with pytest.raises(ValueError,match='WORKER_REPORT_CONFIRMATION=FAIL'): ctl.scan()
    proof=C.read(a['progress'].parent/'final-confirmation.json')
    assert proof['state']=='failed' and proof['elapsed_s']==120 and proof['expected_path']==str(a['results'])
    assert proof['path_exists'] is False and proof['stat_size'] is None and proof['consumed_lines']==0
    assert len(proof['missing_identities'])==1 and proof['rc']==0
    a['log'].close()


@pytest.mark.parametrize('kind',['bad_json','duplicate','cross_identity','wrong_slurm'])
def test_terminal_confirmation_never_accepts_invalid_reports_or_identity(tmp_path,kind):
    ctl,a,row=scan_fixture(tmp_path)
    if kind=='bad_json': text='{bad}\n'
    elif kind=='duplicate': text=(json.dumps(row)+'\n')*2
    elif kind=='cross_identity': row['key']='foreign'; text=json.dumps(row)+'\n'
    else:
        p=C.read(a['progress']); p['slurm_job_id']='104'; C.atomic(a['progress'],p); text=json.dumps(row)+'\n'
    a['results'].write_text(text)
    with pytest.raises((ValueError,json.JSONDecodeError)): ctl.scan()
    assert ctl.active and not ctl.smoked
    a['log'].close()


def test_live_worker_heartbeat_gate_is_not_relaxed(tmp_path):
    ctl,a,row=scan_fixture(tmp_path,rc=None)
    with pytest.raises(RuntimeError,match='心跳失效'): ctl.scan()
    a['log'].close()


def resume_fixture(tmp_path,monkeypatch):
    import budget_ledger, time
    c=config(tmp_path); rows=identities()
    for row in rows: row['seed']=int(row['key'].rsplit('_',1)[1])
    p=Path(c['identities']); p.write_text(''.join(json.dumps(r)+'\n' for r in rows)); c['identities_sha256']=C.sha(p)
    led=budget_ledger.BudgetLedger(c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    led.check(); creator=C.Controller(c)
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**kw:types.SimpleNamespace(pid=999))
    for index,model in enumerate(('qwenvl','smvla','pp')):
        seat=c['seats'][index]; creator.start(seat,model+'-00',smoke=True); a=creator.active[str(seat['job_id'])]; ident=a['rows'][0]
        seat_name=f"{seat['seat']}-{model}-00-smoke"; label=c['models'][model]['label']
        C.atomic(a['progress'],dict(label=label,phase='finished',rc=0,episodes_done=1,pid=1000,seat=seat_name,policy_seed=7,
                                   slurm_job_id=seat['job_id'],slurm_step_id='1',key=None,t=time.time()))
        route=c['models'][model]['policy']+('/'+C.ROUTES[model][2] if C.ROUTES[model][2] else '')+'/seed7/new'; token=route+'|'+ident['key']+'|a1'
        rid=led.reserve(resets=2,route=route,key=ident['key'],token=token,kind_of_try='first',attempt_no=1,slurm_job_id=seat['job_id'])
        led.claim_reset(rid,'build'); led.claim_reset(rid,'reset'); led.commit(rid,status='success',infra=False,slurm_job_id=seat['job_id'])
        directory=Path(c['gl_root'])/'rollouts'/label/'ood/seed7/raw/VideoUnmask_ep0_xhard1'; directory.mkdir(parents=True)
        for name in ('front.mkv','wrist.mkv','arrays.npz','events.jsonl','frames-front.jsonl','frames-wrist.jsonl'): (directory/name).write_text('fixture')
        C.atomic(directory/'meta.json',dict(identity=ident,pid=1000))
        C.atomic(directory/'summary.json',dict(RECORDER_VERIFY='PASS',errors=[],decode_mismatch=0,summary={'exec_steps':1},
            streams={s:dict(error=None,mismatch=0,timestamps_ok=True,encoded=1,decoded=1) for s in ('front','wrist')}))
        (directory/'trace.jsonl').write_text(json.dumps(dict(kind='header',identity=ident))+'\n'+json.dumps(dict(kind='end',exec_steps=1))+'\n')
        result=dict(ident,infra=False,policy_label=label,policy_seed=7,recorder_verify='PASS',exec_steps=1,task_success=1,status='success')
        C.atomic(directory/'result.json',result)
        report=dict(key=ident['key'],dataset='ood',seat=seat_name,attempt=1,attempt_id='fixture-'+model,policy_seed=7,policy=c['models'][model]['policy'],exec_steps=1,budget_rid=rid,budget_token=token,
                    status='success',infra=False,final=True,accepted=True,result=str(directory/'result.json'),task_success=1,budget_exhausted=False,run_blocked=False)
        a['results'].parent.mkdir(parents=True,exist_ok=True); a['results'].write_text(json.dumps(report)+'\n'); a['log'].close()
        C.atomic(Path(c['gl_root'])/'queue'/label/'seed7/accepted'/('ood__'+ident['key']+'.json'),
                 dict(key=ident['key'],dataset='ood',attempt=1,seat=seat_name,result=str(directory/'result.json'),attempt_id=report['attempt_id']))
    ctl=C.Controller(c)
    def query(cmd,**kw):
        if cmd[0]=='squeue': return types.SimpleNamespace(returncode=0,stdout='',stderr='')
        step=cmd[cmd.index('-j')+1]; model=('qwenvl','smvla','pp')[int(step.split('.')[0])-101]
        return types.SimpleNamespace(returncode=0,stdout=step+'|raw-ood-worker-'+model+'-00-smoke|COMPLETED|0:0\n',stderr='')
    monkeypatch.setattr(C.subprocess,'run',query)
    return ctl,query


def test_manual_resume_validates_all_closed_smokes_then_skips_existing(tmp_path,monkeypatch):
    ctl,query=resume_fixture(tmp_path,monkeypatch); ledger=Path(ctl.c['budget_ledger']); before=ledger.read_bytes()
    ctl.resume_completed_smokes('c'*40)
    assert ctl.smoked=={'qwenvl','smvla','pp'} and len(ctl.published)==3 and ledger.read_bytes()==before
    assert C.read(ctl.control/'resume-completed-smokes.json')['orchestration_commit']=='c'*40
    launches=[]
    def run(cmd,**kw): return types.SimpleNamespace(returncode=0,stdout='RUNNING\n',stderr='')
    monkeypatch.setattr(C.subprocess,'run',run)
    for seat in ctl.c['seats']:
        C.atomic(ctl.control/f"guardian-ready-{seat['job_id']}.json",dict(ctl.base,gpu_job_id=seat['job_id'],pid=2000,time=C.time.time(),step_id='2'))
    def launch(seat,shard,smoke=False):
        launches.append(shard); ctl.active[str(seat['job_id'])]={'model':shard.rsplit('-',1)[0]}
    monkeypatch.setattr(ctl,'start',launch)
    ctl.dispatch()
    assert launches==['oracle-00','framesamp-00']


@pytest.mark.parametrize('kind',['running','unknown_exit','wrong_job','half_line','duplicate','uncommitted','missing_field','media_identity','missing_accepted'])
def test_resume_rejects_before_any_publication_and_never_changes_budget(tmp_path,monkeypatch,kind):
    ctl,query=resume_fixture(tmp_path,monkeypatch); ledger=Path(ctl.c['budget_ledger'])
    inv=ctl.control/'invocations/pp-00-smoke'; path=Path(ctl.c['gl_root'])/'seats/pp/seed7/s103-pp-00-smoke/seat-results.jsonl'
    if kind=='running': monkeypatch.setattr(C.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='103.1 raw-ood-worker-pp-00-smoke RUNNING\n',stderr=''))
    elif kind=='unknown_exit': monkeypatch.setattr(C.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='',stderr=''))
    elif kind=='wrong_job':
        p=C.read(inv/'progress.json'); p['slurm_job_id']='104'; C.atomic(inv/'progress.json',p)
    elif kind=='half_line':
        path.write_text(path.read_text()[:-1]); clock=[0]
        monkeypatch.setattr(C.time,'monotonic',lambda:clock[0]); monkeypatch.setattr(C.time,'sleep',lambda x:clock.__setitem__(0,clock[0]+121))
    elif kind=='duplicate': path.write_text(path.read_text()*2)
    elif kind=='uncommitted':
        rows=[json.loads(x) for x in ledger.read_text().splitlines()]; rows=[r for r in rows if not (r['kind']=='commit' and r['slurm_job_id']=='103')]
        # 原fixture commit默认无job；按PP报告rid准确删除。
        rid=json.loads(path.read_text())['budget_rid']; rows=[r for r in rows if not (r['kind']=='commit' and r['rid']==rid)]
        ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif kind=='missing_field':
        r=json.loads(path.read_text()); del r['final']; path.write_text(json.dumps(r)+'\n')
    elif kind=='media_identity':
        result=Path(json.loads(path.read_text())['result']); meta=C.read(result.parent/'meta.json'); meta['identity']['key']='foreign'; C.atomic(result.parent/'meta.json',meta)
    else:
        p=ctl.gl/'queue/pp/seed7/accepted'/('ood__'+ctl.shards['pp-00']['rows'][0]['key']+'.json'); p.unlink()
    original=ledger.read_bytes()
    with pytest.raises((ValueError,KeyError,FileNotFoundError)): ctl.resume_completed_smokes('c'*40)
    assert not (ctl.gl/'published').exists() and not ctl.smoked and ledger.read_bytes()==original


def test_resume_preflight_reads_without_publishing_or_marking_smoked(tmp_path,monkeypatch):
    ctl,query=resume_fixture(tmp_path,monkeypatch); original=Path(ctl.c['budget_ledger']).read_bytes()
    proof=ctl.resume_completed_smokes('c'*40,preflight=True)
    assert proof['reused']==3 and len(proof['identities'])==3
    assert not ctl.smoked and not ctl.published and not (ctl.gl/'published').exists()
    assert not (ctl.control/'resume-completed-smokes.json').exists() and Path(ctl.c['budget_ledger']).read_bytes()==original


def test_resume_waits_for_complete_report_visibility_without_new_attempt(tmp_path,monkeypatch):
    import threading,time
    ctl,query=resume_fixture(tmp_path,monkeypatch)
    path=ctl.gl/'seats/pp/seed7/s103-pp-00-smoke/seat-results.jsonl'; complete=path.read_bytes()
    path.write_bytes(complete[:len(complete)//2]); original=Path(ctl.c['budget_ledger']).read_bytes()
    producer=threading.Thread(target=lambda:(time.sleep(.05),path.write_bytes(complete)))
    producer.start()
    try: ctl.resume_completed_smokes('c'*40)
    finally: producer.join()
    assert ctl.smoked=={'qwenvl','smvla','pp'} and Path(ctl.c['budget_ledger']).read_bytes()==original


def test_unacked_full_terminal_publication_writes_real_first_step_ack(tmp_path):
    ctl,a,row=scan_fixture(tmp_path); a['smoke']=False
    ctl.publish(a,row)
    ack=C.read(ctl.control/'first_dispatch.json')
    assert ack['proof']==row['result'] and ack['first_step']==276 and ack['client_pid']==1000
    assert ack['step_id']=='103.1' and a['acked'] and len(ctl.published)==1


def test_resume_wait_is_inside_live_heartbeat_and_thread_stops_on_failure(tmp_path,monkeypatch):
    import threading,time
    ctl=C.Controller(config(tmp_path)); ctl.control.mkdir()
    C.atomic(ctl.control/'shards.json',dict(ctl.base,shards=ctl.shards))
    entered=threading.Event(); release=threading.Event(); errors=[]
    def resume(*a,**kw):
        entered.set(); release.wait(timeout=3); raise ValueError('fixture resume stop')
    monkeypatch.setattr(ctl,'resume_completed_smokes',resume)
    def run():
        try: ctl.run(resume_completed_smokes=True,orchestration_commit='c'*40)
        except ValueError as e: errors.append(str(e))
    worker=threading.Thread(target=run); worker.start()
    try:
        assert entered.wait(timeout=2)
        deadline=time.monotonic()+2
        while not (ctl.control/'controller-heartbeat.json').exists() and time.monotonic()<deadline: time.sleep(.01)
        beat=C.read(ctl.control/'controller-heartbeat.json')
        assert beat['pid']>0 and time.time()-beat['t']<2
        assert any(t.name=='raw-ood-controller-heartbeat' and t.is_alive() for t in threading.enumerate())
    finally: release.set(); worker.join(timeout=3)
    assert errors==['fixture resume stop'] and not worker.is_alive()
    assert not any(t.name=='raw-ood-controller-heartbeat' and t.is_alive() for t in threading.enumerate())
