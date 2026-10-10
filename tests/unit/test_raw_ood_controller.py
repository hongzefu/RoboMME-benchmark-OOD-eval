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
    c=config(tmp_path); controller=C.Controller(c); Path(c['budget_ledger']).write_text('')
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
         retries=[],recovery_used=0,astra=0,claimed={'x':8000},orphan_claims=0))
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


@pytest.mark.parametrize('mutation',['astra','memer','budget','cfg_gpu'])
def test_route_or_budget_override_blocked_before_process(tmp_path,mutation):
    c=config(tmp_path)
    if mutation=='astra': c['models']['qwenvl']['policy']='astra'
    elif mutation=='memer': c['models']['qwenvl']['args'][1]='ground-sg-memer'
    elif mutation=='budget': c['models']['pp']['args']+=['--trajectory-cap','9999']
    else: c['models']['pp']['args']+=['--cfg','gpus=0']
    with pytest.raises(ValueError): C.validate_config(c)


def test_identity_same_size_mutation_and_manifest_mismatch_blocked(tmp_path):
    c=config(tmp_path); p=Path(c['identities']); old=p.read_bytes()
    p.write_bytes(old.replace(b'aaaaaaaa',b'bbbbbbbb',1)); assert p.stat().st_size==len(old)
    with pytest.raises(ValueError,match='SHA'): C.validate_config(c)
    c=config(tmp_path); c['expected_identities'][0]='wrong'
    with pytest.raises(ValueError,match='manifest'): C.validate_config(c)
