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


def test_shared_terminal_requires_explicit_zero_and_exact_disjoint_coverage(tmp_path):
    from raw_ood_context import validate_terminal_marker,validate_historical_errors
    c=config(tmp_path)
    doc=dict({k:c[k] for k in C.CORE},schema='raw-ood-terminal-v1',expected=4000,settled=4000,published=4000,infra=0,unresolved=0,unattempted=0,
             status='COMPLETE',published_identities=c['expected_identities'],failure_documents={},unresolved_identities=[],unattempted_identities=[])
    assert validate_terminal_marker(json.loads(json.dumps(doc)),c)==doc
    for mutation in ('zero','dup','missing','status'):
        bad=json.loads(json.dumps(doc))
        if mutation=='zero': del bad['infra']
        if mutation=='dup': bad['unresolved_identities']=[bad['published_identities'][0]]
        if mutation=='missing': bad['published_identities'].pop()
        if mutation=='status': bad['status']='INCOMPLETE'
        with pytest.raises(ValueError): validate_terminal_marker(bad,c)
    old=tmp_path/'old-error.json'; old.write_text('{"error":"历史"}')
    c['historical_errors']=[dict(path=str(old),sha256=C.sha(old))]
    assert validate_historical_errors(c)=={str(old)}
    old.write_text('{"error":"改变"}')
    with pytest.raises(ValueError): validate_historical_errors(c)


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
    with pytest.raises(RuntimeError,match='STOP'): controller.run(pause_on_stop=False)
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
    budget_ledger.BudgetLedger(c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000).check()
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


def scan_fixture(tmp_path,*,rc=0,status='success'):
    import time,budget_ledger
    c=config(tmp_path); ctl=C.Controller(c); ident=ctl.shards['pp-00']['rows'][0]
    inv=tmp_path/'control/invocations/pp-00-smoke'; inv.mkdir(parents=True)
    progress=inv/'progress.json'; results=Path(c['gl_root'])/'seats/pp/seed7/s3-pp-00-smoke/seat-results.jsonl'
    results.parent.mkdir(parents=True)
    C.atomic(progress,dict(label='pp',phase='finished',key=None,pid=1000,t=time.time()-300,slurm_job_id='103',slurm_step_id='1'))
    directory=Path(c['gl_root'])/'rollouts/pp/ood/seed7/raw/VideoUnmask_ep0_xhard1'; directory.mkdir(parents=True)
    for name in ('front.mkv','wrist.mkv','arrays.npz','trace.jsonl','meta.json','events.jsonl','frames-front.jsonl','frames-wrist.jsonl'):
        (directory/name).write_text('fixture')
    success=int(status=='success')
    result=directory/'result.json'; C.atomic(result,dict(key=ident['key'],dataset='ood',infra=False,task_success=success,exec_steps=276,status=status))
    led=budget_ledger.BudgetLedger(c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    route='pp/seed7/new'; token=route+'|'+ident['key']+'|a1'
    rid=led.reserve(resets=2,route=route,key=ident['key'],token=token,kind_of_try='first',attempt_no=1,slurm_job_id='103')
    led.claim_reset(rid,'build'); led.claim_reset(rid,'reset'); led.commit(rid,status=status,infra=False,slurm_job_id='103')
    row=dict(key=ident['key'],infra=False,final=True,accepted=True,result=str(result),exec_steps=276,
             task_success=success,budget_exhausted=False,run_blocked=False,dataset='ood',attempt=1,seat='s103-pp-00-smoke',policy='pp',policy_seed=7,
             budget_rid=rid,budget_token=token,status=status)
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


@pytest.mark.parametrize('status',['success','fail','timeout'])
def test_normal_success_failure_and_step_timeout_all_publish_and_continue(tmp_path,status):
    ctl,a,row=scan_fixture(tmp_path,status=status); a['results'].write_text(json.dumps(row)+'\n')
    before=Path(ctl.c['budget_ledger']).read_bytes(); ctl.scan()
    assert len(ctl.published)==1 and not ctl.failures and not ctl.active and ctl.smoked=={'pp'}
    result=C.read(row['result']); assert result['task_success']==int(status=='success') and result['infra'] is False
    assert Path(ctl.c['budget_ledger']).read_bytes()==before


@pytest.mark.parametrize('reason',['env_build','server_dead','protocol_error'])
def test_unapproved_infrastructure_preserves_original_unknown_values_and_stops(tmp_path,reason):
    ctl,a,row=scan_fixture(tmp_path); row=dict(key=row['key'],infra=True,infra_reason=reason,task_success=None,exec_steps=None)
    a['results'].write_text(json.dumps(row)+'\n')
    with pytest.raises(RuntimeError,match='已留'): ctl.scan()
    evidence=C.read(a['progress'].parent/'unclassified-infra.json')
    assert json.loads(evidence['report_line'])['task_success'] is None and not ctl.published and not ctl.failures and a['offset']==0
    a['log'].close()


def test_budget_binding_error_does_not_advance_and_fixed_report_can_resume(tmp_path):
    ctl,a,row=scan_fixture(tmp_path); bad=dict(row,budget_token='伪token'); a['results'].write_text(json.dumps(bad)+'\n')
    before=Path(ctl.c['budget_ledger']).read_bytes()
    with pytest.raises(ValueError): ctl.scan()
    assert a['offset']==0 and not ctl.published
    a['results'].write_text(json.dumps(row)+'\n'); ctl.scan()
    assert len(ctl.published)==1 and not ctl.active and Path(ctl.c['budget_ledger']).read_bytes()==before


@pytest.mark.parametrize('reason,flag',[('ServerDead','run_blocked'),('ProtocolError','run_blocked'),('IdentityMismatch','run_blocked'),('RESET_BUDGET_EXHAUSTED','budget_exhausted')])
def test_blocked_terminal_classifies_before_normal_score_or_missing_media(tmp_path,reason,flag):
    ctl,a,row=scan_fixture(tmp_path)
    row.update({flag:True,'task_success':None,'exec_steps':None,'result':None,'error':reason,'status':'error','final':False,'accepted':False})
    a['results'].write_text(json.dumps(row)+'\n')
    with pytest.raises(RuntimeError,match=reason): ctl.scan()
    evidence=C.read(a['progress'].parent/'blocked-terminal.json')
    assert evidence['reason']==reason and evidence['task_success'] is None and evidence['exec_steps'] is None and not ctl.published
    a['log'].close()


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


def resume_fixture(tmp_path,monkeypatch,models=('qwenvl','smvla','pp')):
    import budget_ledger, time
    c=config(tmp_path); rows=identities()
    for row in rows: row['seed']=int(row['key'].rsplit('_',1)[1])
    p=Path(c['identities']); p.write_text(''.join(json.dumps(r)+'\n' for r in rows)); c['identities_sha256']=C.sha(p)
    led=budget_ledger.BudgetLedger(c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    led.check(); creator=C.Controller(c)
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**kw:types.SimpleNamespace(pid=999))
    for index,model in enumerate(models):
        seat=c['seats'][index%3]; creator.start(seat,model+'-00',smoke=True); a=creator.active[str(seat['job_id'])]; ident=a['rows'][0]
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


def existing_fixture(tmp_path,monkeypatch):
    import time
    ctl,_=resume_fixture(tmp_path,monkeypatch,models=C.MODELS)
    C.atomic(ctl.control/'shards.json',dict(ctl.base,shards=ctl.shards))
    for index,model in enumerate(C.MODELS):
        seat=ctl.c['seats'][index%3]; inv=ctl.control/'invocations'/(model+'-00-smoke')
        row=json.loads((Path(ctl.c['gl_root'])/'seats'/ctl.c['models'][model]['label']/'seed7'/f"{seat['seat']}-{model}-00-smoke"/'seat-results.jsonl').read_text())
        a=dict(model=model,shard=model+'-00',rows=ctl.shards[model+'-00']['rows'][:1],smoke=True)
        doc=ctl.prepare_publication(a,row); C.atomic(ctl.gl/'published'/C.publication_name(doc['identity_key']),doc)
        C.atomic(ctl.gl/'mover/receipts'/C.publication_name(doc['identity_key']),dict(doc,result='moved',dest=str(tmp_path/'delivered'/doc['relative_dir']),
            bytes=sum(x['size'] for x in doc['files'].values()),decoded={'front':1,'wrist':1},trace_checked=True))
    metadata={}; account={}
    for index,shard in enumerate(('qwenvl-00','qwenvl-01','smvla-00')):
        seat=ctl.c['seats'][index]; ctl.start(seat,shard); inv=ctl.control/'invocations'/(shard+'-full')
        d=C.read(inv/'dispatch.json'); d['srun_pid']=9000+index; C.atomic(inv/'dispatch.json',d)
        model=ctl.shards[shard]['model']; step=str(seat['job_id'])+'.5'
        C.atomic(inv/'progress.json',dict(phase='context_load',key=None,t=time.time(),label=ctl.c['models'][model]['label'],
                  slurm_job_id=seat['job_id'],slurm_step_id='5',seat=f"{seat['seat']}-{shard}-full",policy_seed=7,pid=2000+index))
        metadata[9000+index]=dict(state='S',start_ticks=3000+index,start_time=d['time'],command=d['command'],cgroup='0::/slurm/job_100/step_12/task_0')
        account[step]=f'{step}|raw-ood-worker-{shard}-full|RUNNING|0:0\n'
    actual_identity=C.process_identity
    monkeypatch.setattr(C,'process_identity',lambda pid:metadata.get(pid) if 9000<=pid<=9002 else actual_identity(pid))
    def query(cmd,**kw):
        if cmd[0]=='sacct': return types.SimpleNamespace(returncode=0,stdout=account[cmd[cmd.index('-j')+1]],stderr='')
        if '--steps' in cmd:
            assert cmd[cmd.index('-o')+1]=='%i %j'
            return types.SimpleNamespace(returncode=0,stdout=''.join(' '.join(line.strip().split('|')[:2])+'\n' for line in account.values() if '|RUNNING|' in line),stderr='')
        return types.SimpleNamespace(returncode=0,stdout='PENDING\n',stderr='')
    monkeypatch.setattr(C.subprocess,'run',query)
    return C.Controller(ctl.c),metadata,account


def test_existing_dispatch_preflight_and_attach_do_not_replay_or_republish(tmp_path,monkeypatch):
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**k:pytest.fail('禁止重派'))
    proof=ctl.resume_existing_dispatches('c'*40,preflight=True)
    assert proof['active']==3 and proof['published']==5 and not ctl.active
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before
    ledger=Path(ctl.c['budget_ledger']).read_bytes()
    ctl.resume_existing_dispatches('c'*40)
    assert len(ctl.active)==3 and len(ctl.pending)==37 and ctl.smoked==set(C.MODELS) and ctl.formal
    assert ctl.initial_dispatched=={'101','102','103'} and len(ctl.published)==5
    ctl.scan(); ctl.dispatch()
    assert Path(ctl.c['budget_ledger']).read_bytes()==ledger
    for a in ctl.active.values(): a['log'].close()


@pytest.mark.parametrize('kind',['valid','bool','zero','missing_camera','extra_camera','bool_count','wrong_field','trace_false'])
def test_production_mover_receipt_serialization_is_consumed_strictly(tmp_path,monkeypatch,kind):
    import shutil
    import eval_video_mover as mover
    spec=importlib.util.spec_from_file_location('receipt_media_fixture',Path(__file__).with_name('test_raw_ood_mover.py'))
    fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    real_root=tmp_path/'real-media'; real_root.mkdir()
    stage,dest,source,unused=fixture.publication.__wrapped__(real_root)
    (tmp_path/'controller').mkdir()
    ctl,metadata,account=existing_fixture(tmp_path/'controller',monkeypatch)
    ident=ctl.shards['smvla-00']['rows'][0]
    value=mover.json.loads((source/'result.json').read_text()); value.update({k:ident[k] for k in ('dataset','task','tier','seed','key')})
    mover.raw_atomic(source/'result.json',value)
    trace=[json.loads(x) for x in (source/'trace.jsonl').read_text().splitlines()]; trace[0]['identity']={k:ident[k] for k in ('dataset','task','tier','seed','key')}
    (source/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace))
    target=ctl.gl/'rollouts/smvla/ood/seed7/raw/VideoUnmask_ep0_xhard1'
    for path in source.iterdir(): shutil.copy2(path,target/path.name)
    doc=dict(ctl.base,schema='raw-ood-v1',model='smvla',shard_id='smvla-00',identity_key=C.identity('smvla',ident),
             relative_dir=str(target.relative_to(ctl.gl)),files=mover.raw_tree(target),delete_files=['front.mkv','wrist.mkv','arrays.npz'],worker_exec_commit=ctl.c['exec_commit'])
    C.atomic(ctl.gl/'published'/C.publication_name(doc['identity_key']),doc)
    # 使用生产搬运函数、真实AV1解码与删除链路生成回执，再序列化供控制器读取。
    monkeypatch.undo()
    receipt=mover.raw_move(doc,ctl.gl,dest,min_free_gib=0)
    monkeypatch.setattr(C,'process_identity',lambda pid:metadata.get(pid))
    def query(cmd,**kw):
        if cmd[0]=='sacct': return types.SimpleNamespace(returncode=0,stdout=account[cmd[cmd.index('-j')+1]],stderr='')
        return types.SimpleNamespace(returncode=0,stdout=''.join(' '.join(x.strip().split('|')[:2])+'\n' for x in account.values()),stderr='')
    monkeypatch.setattr(C.subprocess,'run',query)
    assert receipt['decoded']=={'front':2,'wrist':2}
    if kind=='bool': receipt['decoded']=True
    elif kind=='zero': receipt['decoded']['front']=0
    elif kind=='missing_camera': del receipt['decoded']['wrist']
    elif kind=='extra_camera': receipt['decoded']['extra']=1
    elif kind=='bool_count': receipt['decoded']['front']=True
    elif kind=='wrong_field': receipt['worker_exec_commit']='f'*40
    elif kind=='trace_false': receipt['trace_checked']=False
    mover.raw_atomic(ctl.gl/'mover/receipts'/C.publication_name(doc['identity_key']),receipt)
    if kind=='valid':
        assert len(ctl.existing_publications())==5
        assert ctl.resume_existing_dispatches('c'*40,preflight=True)['active']==3
    else:
        with pytest.raises(ValueError): ctl.existing_publications()


@pytest.mark.parametrize('kind',['command','cgroup','time','tick','unknown','infra','publication','receipt','directory','identities','context','missing_pid'])
def test_existing_dispatch_rejects_before_any_write(tmp_path,monkeypatch,kind):
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch); inv=ctl.control/'invocations/qwenvl-00-full'
    if kind=='command': metadata[9000]['command']=['srun','wrong']
    elif kind=='cgroup': metadata[9000]['cgroup']='0::/slurm/job_999/step_12'
    elif kind=='time': metadata[9000]['start_time']-=10
    elif kind=='tick':
        d=C.read(inv/'dispatch.json'); d['srun_start_ticks']=999; C.atomic(inv/'dispatch.json',d)
    elif kind=='unknown': account['101.5']='101.5|raw-ood-worker-qwenvl-00-full|UNKNOWN|0:0\n'
    elif kind=='infra': account['101.5']='101.5|raw-ood-worker-qwenvl-00-full|FAILED|1:0\n'
    elif kind in ('publication','receipt'):
        path=next((ctl.gl/('published' if kind=='publication' else 'mover/receipts')).glob('*.json'))
        row=C.read(path); row['worker_exec_commit']='f'*40; C.atomic(path,row)
    elif kind=='directory': (ctl.control/'invocations/unknown').mkdir()
    elif kind=='identities': (inv/'identities.jsonl').write_text('{}\n')
    elif kind=='context':
        d=C.read(inv/'dispatch.json'); d['config_sha256']='f'*64; C.atomic(inv/'dispatch.json',d)
    else: metadata.pop(9000)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises((ValueError,RuntimeError,KeyError)): ctl.resume_existing_dispatches('c'*40,preflight=True)
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before


def test_query_unknown_window_and_pending_cooldown_never_launch(tmp_path,monkeypatch):
    c=config(tmp_path); clock=[0.]; calls=[]; states={}; mode=['timeout']
    monkeypatch.setattr(C.time,'monotonic',lambda:clock[0])
    def query(cmd,**kw):
        calls.append(cmd); assert kw['timeout']==5
        if mode[0]=='timeout': raise C.subprocess.TimeoutExpired(cmd,5)
        return types.SimpleNamespace(returncode=0,stdout=mode[0],stderr='')
    monkeypatch.setattr(C.subprocess,'run',query)
    assert C.observed_job_state(c,'104',states,'controller') is None
    clock[0]=4; assert C.observed_job_state(c,'104',states,'controller') is None and len(calls)==1
    clock[0]=5; assert C.observed_job_state(c,'104',states,'controller') is None and len(calls)==2
    clock[0]=30
    with pytest.raises(RuntimeError,match='SLURM_QUERY_WINDOW'): C.observed_job_state(c,'104',states,'controller')
    states.clear(); mode[0]='PENDING'; clock[0]=40
    assert C.observed_job_state(c,'104',states,'controller')=='PENDING'
    mode[0]='RUNNING'; clock[0]=49; assert C.observed_job_state(c,'104',states,'controller') is None
    clock[0]=50; assert C.observed_job_state(c,'104',states,'controller')=='RUNNING'
    assert C.observed_job_state(c,'104',states,'controller')=='RUNNING'


def formal_report_fixture(ctl):
    import budget_ledger,shutil
    model='qwenvl'; shard=model+'-00'; seat=ctl.c['seats'][0]; inv=ctl.control/'invocations'/(shard+'-full')
    rows=ctl.shards[shard]['rows'][:2]; ctl.shards[shard]['rows']=rows
    (inv/'identities.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows)); C.atomic(ctl.control/'shards.json',dict(ctl.base,shards=ctl.shards))
    ident=rows[1]; directory=ctl.gl/'rollouts'/ctl.c['models'][model]['label']/'ood/seed7/raw/VideoUnmask_ep1_xhard1'
    source=directory.with_name('VideoUnmask_ep0_xhard1'); shutil.copytree(source,directory)
    result=C.read(directory/'result.json'); result.update(ident); C.atomic(directory/'result.json',result)
    meta=C.read(directory/'meta.json'); meta['identity']=ident; C.atomic(directory/'meta.json',meta)
    (directory/'trace.jsonl').write_text(json.dumps(dict(kind='header',identity=ident))+'\n'+json.dumps(dict(kind='end',exec_steps=1))+'\n')
    route='groundsg/ground-sg-qwenvl/seed7/new'; token=route+'|'+ident['key']+'|a1'
    led=budget_ledger.BudgetLedger(ctl.c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    rid=led.reserve(resets=2,route=route,key=ident['key'],token=token,kind_of_try='first',attempt_no=1,slurm_job_id=seat['job_id'])
    led.claim_reset(rid,'build'); led.claim_reset(rid,'reset'); led.commit(rid,status='success',infra=False,slurm_job_id=seat['job_id'])
    smoke=ctl.gl/'seats'/ctl.c['models'][model]['label']/'seed7'/f"{seat['seat']}-{model}-00-smoke"/'seat-results.jsonl'
    row=json.loads(smoke.read_text()); row.update(key=ident['key'],seat=f"{seat['seat']}-{shard}-full",result=str(directory/'result.json'),budget_rid=rid,budget_token=token)
    report=smoke.parent.with_name(row['seat'])/'seat-results.jsonl'; report.parent.mkdir(); report.write_text(json.dumps(row)+'\n')
    return inv,report,row


def watchdog_fixture(tmp_path,monkeypatch):
    """真实序列化稀疏旧75：两运行、一正常前缀、一永久失败、一未预约尾项。"""
    import budget_ledger
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch); inv,report,normal=formal_report_fixture(ctl)
    planned=C.make_shards(ctl.rows)['qwenvl-00']['rows'][:4]; ctl.shards['qwenvl-00']['rows']=planned
    (inv/'identities.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in planned)); C.atomic(ctl.control/'shards.json',dict(ctl.base,shards=ctl.shards))
    ident=planned[2]; seat=ctl.c['seats'][0]; route='groundsg/ground-sg-qwenvl/seed7/new'; token=route+'|'+ident['key']+'|a1'
    led=budget_ledger.BudgetLedger(ctl.c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    rid=led.reserve(resets=2,route=route,key=ident['key'],token=token,kind_of_try='first',attempt_no=1,slurm_job_id=seat['job_id'],dataset='ood')
    led.claim_reset(rid,'build'); led.claim_reset(rid,'reset'); led.commit(rid,status='fail',infra=True,slurm_job_id=seat['job_id'])
    failure=dict(key=ident['key'],dataset='ood',attempt=1,attempt_id='fixture-watchdog',status='error',infra=True,infra_reason='watchdog_exit',budget_rid=rid,t=C.time.time())
    report.write_text(json.dumps(normal)+'\n'+json.dumps(failure)+'\n')
    queue=ctl.gl/'queue'/ctl.c['models']['qwenvl']['label']/'seed7/claims'/('ood__'+ident['key']+'.a1.json')
    C.atomic(queue,dict(key=ident['key'],dataset='ood',rid=rid,attempt=1,route=route,token=token,seat=normal['seat'],ended=True,end_status='infra_timeout',exit_code=75))
    p=C.read(inv/'progress.json'); p.update(phase='finished',rc=75,key=ident['key'],t=C.time.time()); C.atomic(inv/'progress.json',p)
    account['101.5']='101.5|raw-ood-worker-qwenvl-00-full|FAILED|75:0\n'; metadata.pop(9000)
    ctl.c['failure_policy']={'continue_reasons':['watchdog_exit']}
    return ctl,metadata,account,planned,report,failure


def future_watchdog(ctl,a,metadata,account):
    """将已接管运行段真实序列化为75，保持原派发与身份文件。"""
    import budget_ledger
    row=a['rows'][0]; model=a['model']; seat=a['seat']; route=ctl.c['models'][model]['policy']+('/'+C.ROUTES[model][2] if C.ROUTES[model][2] else '')+'/seed7/new'; token=route+'|'+row['key']+'|a1'
    led=budget_ledger.BudgetLedger(ctl.c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    rid=led.reserve(resets=2,route=route,key=row['key'],token=token,kind_of_try='first',attempt_no=1,slurm_job_id=seat['job_id'])
    led.claim_reset(rid,'build'); led.claim_reset(rid,'reset'); led.commit(rid,status='fail',infra=True,slurm_job_id=seat['job_id'])
    report=dict(key=row['key'],dataset='ood',attempt=1,attempt_id='future-watchdog',infra=True,status='error',infra_reason='watchdog_exit',budget_rid=rid)
    a['results'].parent.mkdir(parents=True,exist_ok=True); a['results'].write_text(json.dumps(report)+'\n')
    queue=ctl.gl/'queue'/ctl.c['models'][model]['label']/'seed7/claims'/('ood__'+row['key']+'.a1.json')
    C.atomic(queue,dict(key=row['key'],dataset='ood',rid=rid,attempt=1,route=route,token=token,seat=seat['seat']+'-'+a['inv_tag'],ended=True,end_status='infra_timeout',exit_code=75))
    p=C.read(a['progress']); p.update(phase='finished',rc=75,key=row['key'],t=C.time.time()); C.atomic(a['progress'],p)
    step=str(seat['job_id'])+'.'+str(p['slurm_step_id']); account[step]=step+'|'+a['worker_name']+'|FAILED|75:0\n'
    metadata.pop(a['process'].pid)
    return report


def test_running_worker_future_75_settles_without_repeating_reserved_identity(tmp_path,monkeypatch):
    ctl,metadata,account,planned,report,failure=watchdog_fixture(tmp_path,monkeypatch)
    ctl.resume_existing_dispatches('c'*40); active=ctl.active['102']; newfailure=future_watchdog(ctl,active,metadata,account)
    before=Path(ctl.c['budget_ledger']).read_bytes(); monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**kw:pytest.fail('scan不得启动worker'))
    ctl.scan()
    assert '102' not in ctl.active and len(ctl.failures)==2 and C.identity('qwenvl',newfailure) in ctl.failures
    assert ctl.resume_rows['qwenvl-01']==active['rows'][1:] and ctl.segment_numbers['qwenvl-01']==1
    assert Path(ctl.c['budget_ledger']).read_bytes()==before
    for a in ctl.active.values(): a['log'].close()


@pytest.mark.parametrize('claims,reason,success,steps',[(0,'episode_wall',None,None),(1,'deadline_first_infer',None,None),(2,'deadline_media_finalize',1,27)])
def test_full_watchdog_preserves_actual_partial_reset_and_observed_success(tmp_path,monkeypatch,claims,reason,success,steps):
    ctl,metadata,account,planned,report,failure=watchdog_fixture(tmp_path,monkeypatch)
    ctl.resume_existing_dispatches('c'*40); a=ctl.active['102']; row=future_watchdog(ctl,a,metadata,account)
    row.update(status='fail',exit_kind='watchdog_hard_exit',exit_code=75,infra_reason=reason,task_success=success,exec_steps=steps,result=None,final=False,accepted=False,
               budget_token='groundsg/ground-sg-qwenvl/seed7/new|'+row['key']+'|a1',budget_exhausted=False,run_blocked=False,void=False)
    a['results'].write_text(json.dumps(row)+'\n')
    path=Path(ctl.c['budget_ledger']); rows=[json.loads(x) for x in path.read_text().splitlines()]; count=0; kept=[]
    for r in rows:
        if r.get('kind')=='reset_claim' and r.get('rid')==row['budget_rid']:
            count+=1
            if count>claims: continue
        kept.append(r)
    path.write_text(''.join(json.dumps(x)+'\n' for x in kept)); before=path.read_bytes()
    ctl.scan(); doc=C.read(ctl.failures[C.identity('qwenvl',row)]['path'])
    assert doc['infra_reason']==reason and doc['task_success']==success and doc['exec_steps']==steps and len(doc['budget']['reset_claims'])==claims
    assert path.read_bytes()==before and C.identity('qwenvl',row) not in ctl.published
    for active in ctl.active.values(): active['log'].close()


def test_unowned_reserved_tail_is_unresolved_and_never_queued(tmp_path,monkeypatch):
    import budget_ledger
    ctl,metadata,account,planned,report,failure=watchdog_fixture(tmp_path,monkeypatch)
    led=budget_ledger.BudgetLedger(ctl.c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    route='groundsg/ground-sg-qwenvl/seed7/new'; key=planned[3]['key']
    led.reserve(resets=2,route=route,key=key,token=route+'|'+key+'|a1',kind_of_try='first',attempt_no=1,slurm_job_id='101')
    before=Path(ctl.c['budget_ledger']).read_bytes()
    proof=ctl.resume_existing_dispatches('c'*40,preflight=True); assert proof['unresolved']==1
    ctl.resume_existing_dispatches('c'*40)
    assert ctl.resume_rows['qwenvl-00']==[] and 'qwenvl-00' not in ctl.pending and C.identity('qwenvl',planned[3]) in ctl.unresolved
    assert ctl.report()==2 and C.read(ctl.gl/'gpu_work_complete.json')['unresolved']==1 and Path(ctl.c['budget_ledger']).read_bytes()==before
    for a in ctl.active.values(): a['log'].close()


def test_exact_mixed_history_three_locked6_two_qwen_running_pp100_and_qwen75(tmp_path,monkeypatch):
    """按现场组合序列化：旧三锁6、两Qwen运行、PP一片100终态、新Qwen75。"""
    import budget_ledger,shutil
    ctl,metadata,account,planned=paused_history_fixture(tmp_path,monkeypatch)
    ctl.c['failure_policy']={'continue_reasons':['watchdog_exit']}; ctl.resume_existing_dispatches('c'*40,paused=True)
    counter=[11000]
    def popen(cmd,**kw):
        pid=counter[0]; counter[0]+=1; return types.SimpleNamespace(pid=pid)
    monkeypatch.setattr(C.subprocess,'Popen',popen)
    current=[]
    for seat,shard in zip(ctl.c['seats'],('qwenvl-00','qwenvl-01','pp-00','qwenvl-02')):
        ctl.start(seat,shard); a=ctl.active[str(seat['job_id'])]; d=C.read(a['progress'].parent/'dispatch.json'); step=seat['job_id']+'.9'
        C.atomic(a['progress'],dict(phase='context_load',key=None,t=C.time.time(),label=ctl.c['models'][a['model']]['label'],slurm_job_id=seat['job_id'],slurm_step_id='9',seat=seat['seat']+'-'+a['inv_tag'],policy_seed=7,pid=d['srun_pid']))
        account[step]=step+'|'+a['worker_name']+'|RUNNING|0:0\n'
        metadata[d['srun_pid']]=dict(state='S',start_ticks=d['srun_pid'],start_time=d['time'],command=d['command'],cgroup='0::/slurm/job_100/step_12/task_0')
        current.append(a)
    original=C.process_identity; monkeypatch.setattr(C,'process_identity',lambda pid:metadata.get(pid) if 11000<=pid<=11003 else original(pid))
    led=budget_ledger.BudgetLedger(ctl.c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    pp=current[2]; smoke=ctl.gl/'seats/pp/seed7/s103-pp-00-smoke/seat-results.jsonl'; seed_row=json.loads(smoke.read_text()); source=Path(seed_row['result']).parent
    records=[]
    for ident in pp['rows']:
        route='pp/seed7/new'; token=route+'|'+ident['key']+'|a1'; directory=source.with_name(ident['key']); shutil.copytree(source,directory)
        data=C.read(directory/'result.json'); data.update(ident); C.atomic(directory/'result.json',data)
        C.atomic(directory/'meta.json',dict(identity=ident)); (directory/'trace.jsonl').write_text(json.dumps(dict(kind='header',identity=ident))+'\n'+json.dumps(dict(kind='end',exec_steps=1))+'\n')
        rid=led.reserve(resets=2,route=route,key=ident['key'],token=token,kind_of_try='first',attempt_no=1,slurm_job_id='103')
        led.claim_reset(rid,'build'); led.claim_reset(rid,'reset'); led.commit(rid,status='success',infra=False,slurm_job_id='103')
        records.append(dict(seed_row,key=ident['key'],seat=pp['seat']['seat']+'-'+pp['inv_tag'],result=str(directory/'result.json'),budget_rid=rid,budget_token=token))
    pp['results'].parent.mkdir(parents=True,exist_ok=True); pp['results'].write_text(''.join(json.dumps(r)+'\n' for r in records))
    p=C.read(pp['progress']); p.update(phase='finished',rc=0,t=C.time.time()); C.atomic(pp['progress'],p)
    account['103.9']='103.9|'+pp['worker_name']+'|COMPLETED|0:0\n'; metadata.pop(pp['process'].pid)
    future_watchdog(ctl,current[3],metadata,account)
    for a in ctl.active.values(): a['log'].close()
    recovered=C.Controller(ctl.c); recovered.shards=ctl.shards; C.atomic(recovered.control/'shards.json',dict(recovered.base,shards=recovered.shards))
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**kw:pytest.fail('预检与接管不能重复派发'))
    proof=recovered.resume_existing_dispatches('d'*40,preflight=True)
    assert proof['active']==2 and proof['completed']==1 and proof['drained']==4 and proof['infra']==1
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before
    recovered.resume_existing_dispatches('d'*40)
    assert set(recovered.active)=={'101','102'} and 'pp-00' in recovered.completed
    assert 'qwenvl-00' not in recovered.pending and 'qwenvl-01' not in recovered.pending
    assert recovered.resume_rows['qwenvl-02']==current[3]['rows'][1:] and len([k for k in recovered.published if k.startswith('pp:')])==100
    for a in recovered.active.values(): a['log'].close()


def test_mixed_watchdog_recovery_is_read_only_then_settles_once_and_requeues_tail(tmp_path,monkeypatch):
    from raw_ood_context import validate_terminal_marker
    ctl,metadata,account,planned,report,failure=watchdog_fixture(tmp_path,monkeypatch)
    original={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**kw:pytest.fail('恢复不得重起已运行worker'))
    proof=ctl.resume_existing_dispatches('c'*40,preflight=True)
    assert proof['active']==2 and proof['infra']==1 and proof['unresolved']==0
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==original
    ctl.resume_existing_dispatches('c'*40)
    failed=C.identity('qwenvl',failure)
    assert failed in ctl.failures and failed not in ctl.published and ctl.resume_rows['qwenvl-00']==[planned[3]]
    assert ctl.segment_numbers['qwenvl-00']==1 and len(ctl.active)==2
    doc=C.read(ctl.failures[failed]['path']); assert doc['task_success'] is None and doc['exec_steps'] is None
    budget_before=Path(ctl.c['budget_ledger']).read_bytes()
    ctl.record_failure(doc); assert Path(ctl.c['budget_ledger']).read_bytes()==budget_before
    assert ctl.report()==2
    terminal=C.read(ctl.gl/'gpu_work_complete.json'); validate_terminal_marker(terminal,ctl.c)
    assert terminal['status']=='INCOMPLETE' and terminal['infra']==1 and terminal['published']==6 and terminal['settled']==7
    results=C.read(ctl.control/'results-report.json'); assert results['observed_count']==6 and results['observed_success']==6
    for a in ctl.active.values(): a['log'].close()


@pytest.mark.parametrize('kind',['commit','route','token','queue','rc','reason','zero','repeat'])
def test_watchdog_evidence_rejects_corrupt_binding_before_any_publication(tmp_path,monkeypatch,kind):
    ctl,metadata,account,planned,report,failure=watchdog_fixture(tmp_path,monkeypatch)
    if kind=='rc': account['101.5']=account['101.5'].replace('75:0','9:0')
    elif kind=='reason': failure['infra_reason']='unknown'; report.write_text(json.dumps(failure)+'\n')
    elif kind=='zero': failure['attempt']=0; report.write_text(json.dumps(failure)+'\n')
    elif kind=='repeat': report.write_text(report.read_text()+json.dumps(failure)+'\n')
    elif kind=='queue':
        q=next((ctl.gl/'queue').rglob('*.a1.json')); value=C.read(q); value['exit_code']=0; C.atomic(q,value)
    else:
        path=Path(ctl.c['budget_ledger']); rows=[json.loads(x) for x in path.read_text().splitlines()]
        target=next(r for r in rows if r.get('rid')==failure['budget_rid'] and r['kind']==('commit' if kind=='commit' else 'reserve'))
        if kind=='commit': target['status']='success'
        elif kind=='route': target['route']='pp/seed7/new'
        else: target['token']='forged'
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises((ValueError,KeyError)): ctl.resume_existing_dispatches('c'*40,preflight=True)
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('listed',[False,True])
def test_existing_terminal_and_unpublished_rows_resume_without_budget_write(tmp_path,monkeypatch,listed):
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch); inv,report,row=formal_report_fixture(ctl)
    account['101.5']='101.5|raw-ood-worker-qwenvl-00-full|COMPLETED|0:0\n'; metadata.pop(9000)
    if listed:
        original=C.subprocess.run
        def query(cmd,**kw):
            value=original(cmd,**kw)
            if '--steps' in cmd: value.stdout+='101.5 raw-ood-worker-qwenvl-00-full\n'
            return value
        monkeypatch.setattr(C.subprocess,'run',query)
    p=C.read(inv/'progress.json'); p.update(phase='finished',rc=0,episodes_done=1); C.atomic(inv/'progress.json',p)
    before=Path(ctl.c['budget_ledger']).read_bytes(); pub_before=len(list((ctl.gl/'published').glob('*.json')))
    proof=ctl.resume_existing_dispatches('c'*40,preflight=True)
    assert proof['completed']==1 and proof['active']==2 and len(list((ctl.gl/'published').glob('*.json')))==pub_before
    ctl.resume_existing_dispatches('c'*40)
    assert 'qwenvl-00' in ctl.completed and len(ctl.published)==6 and len(ctl.active)==2
    assert Path(ctl.c['budget_ledger']).read_bytes()==before
    for a in ctl.active.values(): a['log'].close()


@pytest.mark.parametrize('kind',['unsupported_stderr_rc0','duplicate','unknown_job','wrong_name','extra_column'])
def test_step_inventory_uses_two_columns_and_rejects_invalid_cli_output(tmp_path,monkeypatch,kind):
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch)
    original=C.subprocess.run
    def query(cmd,**kw):
        if '--steps' not in cmd: return original(cmd,**kw)
        assert cmd[cmd.index('-o')+1]=='%i %j'
        stdout='101.5 raw-ood-worker-qwenvl-00-full\n102.5 raw-ood-worker-qwenvl-01-full\n103.5 raw-ood-worker-smvla-00-full\n'; stderr=''
        if kind=='unsupported_stderr_rc0': stderr='Invalid job step format specification: T'; stdout=stdout.replace('\n',' %T\n')
        elif kind=='duplicate': stdout+='101.5 raw-ood-worker-qwenvl-00-full\n'
        elif kind=='unknown_job': stdout+='999.1 raw-ood-worker-qwenvl-07-full\n'
        elif kind=='wrong_name': stdout=stdout.replace('qwenvl-00-full','qwenvl-07-full')
        else: stdout=stdout.replace('\n',' RUNNING\n')
        return types.SimpleNamespace(returncode=0,stdout=stdout,stderr=stderr)
    monkeypatch.setattr(C.subprocess,'run',query)
    before=Path(ctl.c['budget_ledger']).read_bytes()
    with pytest.raises(ValueError): ctl.resume_existing_dispatches('c'*40,preflight=True)
    assert Path(ctl.c['budget_ledger']).read_bytes()==before and not ctl.active


@pytest.mark.parametrize('kind',['completed','running','unsupported_stderr_rc0','duplicate','unknown_job'])
def test_old_smoke_resume_uses_supported_step_columns_and_true_sacct(tmp_path,monkeypatch,kind):
    ctl,original=resume_fixture(tmp_path,monkeypatch)
    def query(cmd,**kw):
        if '--steps' not in cmd: return original(cmd,**kw)
        assert cmd[cmd.index('-o')+1]=='%i %j'
        stdout='101.1 raw-ood-worker-qwenvl-00-smoke\n'; stderr=''
        if kind=='unsupported_stderr_rc0': stdout='101.1 raw-ood-worker-qwenvl-00-smoke %T\n'; stderr='Invalid job step format specification: T'
        elif kind=='duplicate': stdout*=2
        elif kind=='unknown_job': stdout='999.1 raw-ood-worker-qwenvl-00-smoke\n'
        return types.SimpleNamespace(returncode=0,stdout=stdout,stderr=stderr)
    def wrapped(cmd,**kw):
        if kind=='running' and cmd[0]=='sacct': return types.SimpleNamespace(returncode=0,stdout='101.1|raw-ood-worker-qwenvl-00-smoke|RUNNING|0:0\n',stderr='')
        return query(cmd,**kw)
    monkeypatch.setattr(C.subprocess,'run',wrapped)
    before=Path(ctl.c['budget_ledger']).read_bytes()
    if kind=='completed': assert ctl.resume_completed_smokes('c'*40,preflight=True)['reused']==3
    else:
        with pytest.raises(ValueError): ctl.resume_completed_smokes('c'*40,preflight=True)
    assert Path(ctl.c['budget_ledger']).read_bytes()==before and not ctl.published


@pytest.mark.parametrize('kind',['already_published','duplicate','half_terminal','uncommitted','cross_identity'])
def test_existing_report_rows_are_checked_before_offset_or_publish(tmp_path,monkeypatch,kind):
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch); inv,report,row=formal_report_fixture(ctl)
    a=dict(model='qwenvl',shard='qwenvl-00',rows=ctl.shards['qwenvl-00']['rows'],smoke=False)
    if kind=='already_published':
        doc=ctl.prepare_publication(a,row); C.atomic(ctl.gl/'published'/C.publication_name(doc['identity_key']),doc)
    elif kind=='duplicate': report.write_text(report.read_text()*2)
    elif kind=='half_terminal':
        report.write_text(report.read_text().rstrip('\n')); account['101.5']='101.5|raw-ood-worker-qwenvl-00-full|COMPLETED|0:0\n'
        p=C.read(inv/'progress.json'); p.update(phase='finished',rc=0); C.atomic(inv/'progress.json',p)
    elif kind=='uncommitted': row['budget_rid']='missing'; report.write_text(json.dumps(row)+'\n')
    else: row['key']='wrong'; report.write_text(json.dumps(row)+'\n')
    before=Path(ctl.c['budget_ledger']).read_bytes()
    if kind=='already_published':
        ctl.resume_existing_dispatches('c'*40); assert ctl.active['101']['offset']==1
        ctl.scan(); assert ctl.active['101']['offset']==1 and len(ctl.published)==6
        for x in ctl.active.values(): x['log'].close()
    else:
        with pytest.raises((ValueError,RuntimeError,KeyError)): ctl.resume_existing_dispatches('c'*40,preflight=True)
    assert Path(ctl.c['budget_ledger']).read_bytes()==before


def test_existing_adapter_real_pid_disappearance_requires_true_terminal_accounting(tmp_path,monkeypatch):
    import subprocess,time
    ready=tmp_path/'ready'
    child=subprocess.Popen([sys.executable,'-c',"import pathlib,time,sys; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(.3)",str(ready)])
    deadline=time.monotonic()+2
    while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)
    assert ready.exists()
    proc=C.process_identity(child.pid); adapter=C.ExistingProcess(child.pid,proc,'101.5','raw-ood-worker-qwenvl-00-full')
    try:
        assert adapter.poll() is None
        child.wait(timeout=2)
        monkeypatch.setattr(C,'worker_account',lambda *a,**kw:None)
        with pytest.raises(ValueError,match='退出未知'): adapter.poll()
        monkeypatch.setattr(C,'worker_account',lambda *a,**kw:0)
        assert adapter.poll()==0
    finally:
        if child.poll() is None: child.kill(); child.wait(timeout=2)


def test_real_process_identity_preserves_empty_gpu_argument(tmp_path):
    import subprocess,time
    ready=tmp_path/'ready'
    command=[sys.executable,'-c',"import pathlib,time,sys; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(60)",str(ready),'--gpus','','--tail','value']
    child=subprocess.Popen(command)
    try:
        deadline=time.monotonic()+2
        while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)
        assert ready.exists()
        proc=C.process_identity(child.pid)
        assert proc['command']==command
        i=proc['command'].index('--gpus'); assert proc['command'][i+1]==''
        assert C.ExistingProcess(child.pid,proc,'101.5','raw-ood-worker-qwenvl-00-full').poll() is None
    finally: child.kill(); child.wait(timeout=2)


def test_controller_lease_is_exclusive_across_real_processes(tmp_path):
    import subprocess,time
    c=config(tmp_path); (tmp_path/'control').mkdir(); cfg=tmp_path/'config.json'; C.atomic(cfg,c)
    ready=tmp_path/'lease-ready'; release=tmp_path/'release'
    code="import pathlib,sys,time; sys.path.insert(0,sys.argv[1]); import raw_ood_controller as C; ctl=C.Controller(C.read(pathlib.Path(sys.argv[2]))); lease=C.ControllerLease(ctl); lease.__enter__(); pathlib.Path(sys.argv[3]).write_text('ready');\nwhile not pathlib.Path(sys.argv[4]).exists(): time.sleep(.01)\nlease.__exit__(None,None,None)"
    child=subprocess.Popen([sys.executable,'-c',code,str(HERE),str(cfg),str(ready),str(release)])
    try:
        deadline=time.monotonic()+3
        while not ready.exists() and child.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        assert ready.exists()
        with pytest.raises(C.ControllerLeaseBusy): C.Controller(c).run()
        assert not (tmp_path/'control/controller-heartbeat.json').exists() and not (tmp_path/'control/controller-error.json').exists()
    finally:
        release.write_text('release'); child.wait(timeout=3)


def paused_history_fixture(tmp_path,monkeypatch):
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch); inv,report,row=formal_report_fixture(ctl)
    planned=C.make_shards(ctl.rows)['qwenvl-00']['rows'][:3]; ctl.shards['qwenvl-00']['rows']=planned
    (inv/'identities.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in planned)); C.atomic(ctl.control/'shards.json',dict(ctl.base,shards=ctl.shards))
    stop=ctl.control/'preserved-STOP'; stop.write_text('本夹具暂停'); workers=[]; stop_time=C.time.time()-1
    for path in sorted((ctl.control/'invocations').glob('*-full/dispatch.json')):
        d=C.read(path); p=path.parent/'progress.json'; progress=C.read(p); progress.update(phase='finished',rc=6,t=C.time.time()); C.atomic(p,progress)
        step=str(d['gpu_job_id'])+'.'+progress['slurm_step_id']; name='raw-ood-worker-'+path.parent.name
        account[step]=step+'|'+name+'|FAILED|6:0\n'; metadata.pop(d['srun_pid'])
        results=ctl.gl/'seats'/ctl.c['models'][d['model']]['label']/'seed7'/progress['seat']/'seat-results.jsonl'
        workers.append(dict(invocation=str(path.parent),dispatch_sha256=C.sha(path),progress_sha256=C.sha(p),report_sha256=C.sha(results) if results.exists() else None,
                            step_id=step,job_name=name,state='FAILED',exit_code='6:0'))
    snapshot=ctl.control/'snapshot.json'; C.atomic(snapshot,dict(schema='raw-ood-pause-v1',run=ctl.c['run'],run_name=ctl.c['run_name'],root=str(tmp_path),stop_time=stop_time,
                   stop_files=[dict(path=str(stop),sha256=C.sha(stop))],workers=workers))
    ctl.c['pause_snapshot']=dict(path=str(snapshot),sha256=C.sha(snapshot))
    return ctl,metadata,account,planned


def test_paused_complete_prefix_restores_only_unattempted_new_segment(tmp_path,monkeypatch):
    ctl,metadata,account,planned=paused_history_fixture(tmp_path,monkeypatch)
    ledger=Path(ctl.c['budget_ledger']); before=ledger.read_bytes()
    source_before={p:p.read_bytes() for p in (ctl.control/'invocations').rglob('*') if p.is_file()}
    proof=ctl.resume_existing_dispatches('c'*40,paused=True,preflight=True)
    assert proof['accepted']==6 and not ctl.published and ledger.read_bytes()==before
    ctl.resume_existing_dispatches('c'*40,paused=True)
    assert len(ctl.published)==6 and ctl.resume_rows['qwenvl-00']==[planned[2]] and ctl.segment_numbers['qwenvl-00']==1
    captured=[]; monkeypatch.setattr(C.subprocess,'Popen',lambda cmd,**kw:captured.append(cmd) or types.SimpleNamespace(pid=9900))
    ctl.start(ctl.c['seats'][0],'qwenvl-00')
    ids=ctl.control/'invocations/qwenvl-00-resume-1-full/identities.jsonl'
    assert [json.loads(x) for x in ids.read_text().splitlines()]==[planned[2]]
    assert '--job-name=raw-ood-worker-qwenvl-00-resume-1-full' in captured[0] and ledger.read_bytes()==before
    assert all(p.read_bytes()==raw for p,raw in source_before.items())
    ctl.active['101']['log'].close()


@pytest.mark.parametrize('unknown_history',[False,True])
def test_locked_drained_history_and_four_running_segments_existing_preflight(tmp_path,monkeypatch,unknown_history):
    import budget_ledger
    ctl,metadata,account,planned=paused_history_fixture(tmp_path,monkeypatch)
    ctl.resume_existing_dispatches('c'*40,paused=True)
    next_pid=[9900]
    def popen(cmd,**kw):
        pid=next_pid[0]; next_pid[0]+=1; return types.SimpleNamespace(pid=pid)
    monkeypatch.setattr(C.subprocess,'Popen',popen)
    for seat,shard in zip(ctl.c['seats'],('qwenvl-00','qwenvl-01','smvla-00','pp-00')):
        ctl.start(seat,shard); a=ctl.active[str(seat['job_id'])]; d=C.read(a['progress'].parent/'dispatch.json'); tag=a['inv_tag']; step=str(seat['job_id'])+'.6'
        C.atomic(a['progress'],dict(phase='context_load',key=None,t=C.time.time(),label=ctl.c['models'][a['model']]['label'],slurm_job_id=seat['job_id'],slurm_step_id='6',seat=seat['seat']+'-'+tag,policy_seed=7,pid=d['srun_pid']))
        account[step]=step+'|raw-ood-worker-'+tag+'|RUNNING|0:0\n'
        metadata[d['srun_pid']]=dict(state='S',start_ticks=d['srun_pid'],start_time=d['time'],command=d['command'],cgroup='0::/slurm/job_100/step_12/task_0')
    old_identity=C.process_identity
    monkeypatch.setattr(C,'process_identity',lambda pid:metadata.get(pid) if 9900<=pid<=9903 else old_identity(pid))
    # 新续段正在进行的预约不属于旧暂停前缀，不得误判旧历史半局。
    ledger=budget_ledger.BudgetLedger(ctl.c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000)
    rid=ledger.reserve(resets=2,route='groundsg/ground-sg-qwenvl/seed7/new',key=planned[2]['key'],token='当前未完成预约',kind_of_try='first',attempt_no=1,slurm_job_id='101')
    ledger.claim_reset(rid,'build'); ledger.claim_reset(rid,'reset')
    if unknown_history:
        snapshot=Path(ctl.c['pause_snapshot']['path']); doc=C.read(snapshot); doc['workers']=doc['workers'][1:]; C.atomic(snapshot,doc); ctl.c['pause_snapshot']['sha256']=C.sha(snapshot)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**kw:pytest.fail('接管禁止重复模型'))
    try:
        if unknown_history:
            with pytest.raises(ValueError): ctl.resume_existing_dispatches('c'*40,preflight=True)
        else:
            proof=ctl.resume_existing_dispatches('c'*40,preflight=True); assert proof['active']==4 and proof['drained']==3
        assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before
    finally:
        for a in ctl.active.values(): a['log'].close()


def test_scan_transient_io_and_uncertain_publish_are_idempotent(tmp_path,monkeypatch):
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch); inv,report,row=formal_report_fixture(ctl)
    seat=ctl.c['seats'][0]; a=dict(process=types.SimpleNamespace(poll=lambda:None),seat=seat,shard='qwenvl-00',model='qwenvl',smoke=False,progress=inv/'progress.json',
        results=report,rows=ctl.shards['qwenvl-00']['rows'],offset=0,started=C.time.time(),acked=True,phase_key=None,phase_start=C.time.time(),step_id='101.5',worker_name='raw-ood-worker-qwenvl-00-full',log=(inv/'worker.log').open('a'))
    ctl.active['101']=a; ctl.published=set(ctl.existing_publications()); ledger=Path(ctl.c['budget_ledger']); before=ledger.read_bytes(); progress_before=a['progress'].read_bytes()
    original_read=Path.read_text; read_once=[False]
    def read(path,*args,**kw):
        if path==a['progress'] and not read_once[0]: read_once[0]=True; raise OSError(121,'瞬态读取')
        return original_read(path,*args,**kw)
    monkeypatch.setattr(Path,'read_text',read); atomic=C.atomic; published=[]
    def write(path,obj):
        atomic(path,obj)
        if path.parent==ctl.gl/'published':
            published.append(path)
            if len(published)==1: raise OSError(5,'rename完成后不确定回执')
    monkeypatch.setattr(C,'atomic',write)
    try:
        ctl.retry_filesystem(ctl.scan)
        assert a['offset']==1 and len(ctl.published)==6 and len(published)==1
        assert ledger.read_bytes()==before and a['progress'].read_bytes()==progress_before
        ctl.scan(); assert a['offset']==1 and len(published)==1
    finally: a['log'].close()


def test_filesystem_retry_has_120_bound_without_killing_existing_child(tmp_path,monkeypatch):
    import subprocess
    ctl=C.Controller(config(tmp_path)); child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); clock=[0.]
    monkeypatch.setattr(C.time,'monotonic',lambda:clock[0]); monkeypatch.setattr(C.time,'sleep',lambda n:clock.__setitem__(0,clock[0]+n))
    def failed(): raise OSError(116,'持续失联')
    try:
        with pytest.raises(RuntimeError,match='CONTROLLER_FILESYSTEM'): ctl.retry_filesystem(failed)
        assert clock[0]==120 and child.poll() is None
        with pytest.raises(ValueError): ctl.retry_filesystem(lambda:(_ for _ in ()).throw(ValueError('坏schema')))
    finally: child.terminate(); child.wait(timeout=3)


def test_existing_resume_validation_runs_inside_real_heartbeat_thread(tmp_path,monkeypatch):
    import time,threading
    ctl,metadata,account=existing_fixture(tmp_path,monkeypatch)
    def verify(*args,**kwargs):
        path=ctl.control/'controller-heartbeat.json'; deadline=time.monotonic()+2
        while not path.exists() and time.monotonic()<deadline: time.sleep(.01)
        assert C.read(path)['config_sha256']==ctl.c['config_sha256']
        raise ValueError('夹具结束只退出控制器')
    monkeypatch.setattr(ctl,'resume_existing_dispatches',verify)
    with pytest.raises(ValueError,match='夹具结束'): ctl.run(resume_existing_dispatches=True,orchestration_commit='c'*40)
    assert not any(t.name=='raw-ood-controller-heartbeat' and t.is_alive() for t in threading.enumerate())


def test_delayed_fourth_seat_does_not_request_exhausted_initial_model(tmp_path,monkeypatch):
    import budget_ledger
    ctl=C.Controller(config(tmp_path)); ctl.formal=True; ctl.pending=['oracle-07']; ctl.initial_dispatched={'101','102','103'}
    budget_ledger.BudgetLedger(ctl.c['budget_ledger'],trajectory_cap=4000,reset_cap=8000,astra_cap=0,shared_infra_cap=0,expired_cap=0,planned_first_tries=4000).check()
    ctl.active={str(s['job_id']):dict(model='pp',rows=[]) for s in ctl.c['seats'][:3]}
    seat=ctl.c['seats'][3]; C.atomic(ctl.control/'guardian-heartbeat-104.json',dict(ctl.base,gpu_job_id='104',pid=123,t=C.time.time(),monitor_t=C.time.time(),step_id='1'))
    monkeypatch.setattr(C.subprocess,'run',lambda cmd,**k:types.SimpleNamespace(returncode=0,stdout='104.1 raw-ood-lifecycle-guard\n' if '--steps' in cmd else 'RUNNING',stderr=''))
    launched=[]; monkeypatch.setattr(ctl,'start',lambda seat,shard:launched.append(shard))
    ctl.dispatch(); assert launched==['oracle-07'] and not ctl.pending


def legacy_fixture(ctl):
    import copy
    old=copy.deepcopy(ctl.c); old.pop('config_sha256')
    path=ctl.control/'old-config.json'; C.atomic(path,old)
    old=C.load_config(path)
    for dispatch in (ctl.control/'invocations').glob('*/dispatch.json'):
        row=C.read(dispatch); row['config_sha256']=old['config_sha256']; C.atomic(dispatch,row)
    current=copy.deepcopy(old); current['exec_commit']='d'*40; current['config_sha256']='e'*64
    current['runner']='/new/runner.sh'
    current['legacy_source']=dict(config=str(path),sha256=C.sha(path),exec_commit=old['exec_commit'])
    return C.Controller(current),old


def test_legacy_resume_uses_current_publication_context_and_original_worker(tmp_path,monkeypatch):
    ctl,_=resume_fixture(tmp_path,monkeypatch); ctl,old=legacy_fixture(ctl)
    ledger=Path(ctl.c['budget_ledger']); before=ledger.read_bytes()
    proof=ctl.resume_completed_smokes('f'*40,preflight=True)
    assert len(ctl.published)==0 and ledger.read_bytes()==before
    ctl.resume_completed_smokes('f'*40)
    for path in (Path(ctl.c['gl_root'])/'published').glob('*.json'):
        row=C.read(path)
        assert row['exec_commit']=='d'*40 and row['config_sha256']=='e'*64
        assert row['worker_exec_commit']==old['exec_commit']
    assert len(ctl.published)==3 and ledger.read_bytes()==before


@pytest.mark.parametrize('kind',['sha','cpu','notification','route'])
def test_legacy_context_rejects_changed_binding(tmp_path,monkeypatch,kind):
    ctl,_=resume_fixture(tmp_path,monkeypatch); ctl,old=legacy_fixture(ctl)
    if kind=='sha': ctl.c['legacy_source']['sha256']='0'*64
    elif kind=='cpu': ctl.c['controller_job_id']='999'
    elif kind=='notification': ctl.c['notification']={'enabled':True}
    else: ctl.c['models']['pp']['policy']='dummy'
    with pytest.raises(ValueError): C.legacy_context(ctl.c)


def test_control_ready_path_requires_local_entity_without_symlink(tmp_path):
    c={'root':str(tmp_path)}; control=tmp_path/'control'; control.mkdir()
    ready=control/'new-ready.json'; C.atomic(ready,{})
    assert C.control_file(c,str(ready))==ready
    linked=control/'linked.json'; linked.symlink_to(ready)
    for path in (linked,tmp_path/'outside.json',control/'../outside.json'):
        with pytest.raises(ValueError): C.control_file(c,str(path))


def test_manual_resume_validates_all_closed_smokes_then_skips_existing(tmp_path,monkeypatch):
    ctl,query=resume_fixture(tmp_path,monkeypatch); ledger=Path(ctl.c['budget_ledger']); before=ledger.read_bytes()
    ctl.resume_completed_smokes('c'*40)
    assert ctl.smoked=={'qwenvl','smvla','pp'} and len(ctl.published)==3 and ledger.read_bytes()==before
    assert C.read(ctl.control/'resume-completed-smokes.json')['orchestration_commit']=='c'*40
    launches=[]
    def run(cmd,**kw):
        job=cmd[cmd.index('-j')+1]
        return types.SimpleNamespace(returncode=0,stdout=job+'.2 raw-ood-lifecycle-guard\n' if '--steps' in cmd else 'RUNNING\n',stderr='')
    monkeypatch.setattr(C.subprocess,'run',run)
    for seat in ctl.c['seats']:
        C.atomic(ctl.control/f"guardian-heartbeat-{seat['job_id']}.json",dict(ctl.base,gpu_job_id=seat['job_id'],pid=2000,t=C.time.time(),monitor_t=C.time.time(),step_id='2'))
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
