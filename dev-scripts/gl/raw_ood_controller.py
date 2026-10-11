#!/usr/bin/env python3
"""五模型原始录像控制器；只使用登记作业，零重试，不提交资源。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from raw_ood_context import CORE, match_context, read_pause_snapshot, source_configs, trusted_contexts, validate_failure_document, validate_terminal_marker, validate_historical_errors, read_budget_rows, validate_pause_proof, legacy_watchdog_progress

MODELS = ('qwenvl', 'smvla', 'pp', 'oracle', 'framesamp')
SECONDS = dict(qwenvl=311.428, smvla=152.526, pp=138.695, oracle=130.485, framesamp=86.518)
ROUTES = dict(qwenvl=('groundsg','groundsg-ground-sg-qwenvl','ground-sg-qwenvl'),
              oracle=('groundsg','groundsg-ground-sg-oracle','ground-sg-oracle'),smvla=('smvla','smvla',None),
              pp=('pp','pp',None),framesamp=('perceptual-framesamp-modul','perceptual-framesamp-modul',None))
PRIOR_SHA256='fc294d2d4dfe5d3ad00b1cd93afe7d1e3e47989ba457b0241c0695b1b1131d74'


def ledger_snapshot(path: Path):
    """严格只读，不调用会初始化文件或配置的BudgetLedger接口。"""
    from budget_ledger import BudgetState
    import fcntl
    lock=path.with_name(path.name+'.lock')
    if lock.exists():
        with lock.open('rb') as stream:
            deadline=time.monotonic()+5
            while True:
                try: fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.monotonic()>=deadline: raise RuntimeError('预算共享锁等待超过5秒')
                    time.sleep(.05)
            raw=path.read_bytes()
    else: raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'): raise ValueError('预算账本缺少完整末行')
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len([r for r in rows if r.get('kind')=='config'])!=1: raise ValueError('预算配置行必须唯一')
    state=BudgetState(rows)
    if state.bad_rows: raise ValueError('预算账本坏行')
    caps=state.config
    if any(caps[k]!=v for k,v in {'trajectory_cap':4000,'reset_cap':8000,'planned_first_tries':4000,'shared_infra_cap':0,'expired_cap':0}.items()) or caps['schema'] not in ('sgeval-budget/3','sgeval-budget/4'):
        raise ValueError('预算账本4000/8000硬上限不符')
    if caps['schema']=='sgeval-budget/4' and (type(caps['astra_cap']) is not int or caps['astra_cap']!=0): raise ValueError('当前schema4必须显式Astra0')
    return state,hashlib.sha256(raw).hexdigest()


def work_budget(c: dict, *, current=None) -> dict:
    prior=c.get('prior_work'); prior_count=0; prior_resets=0; fingerprint=None
    if prior is None:
        if c['run'].endswith('-02'): raise ValueError('恢复轮必须声明prior_work')
        tcap,rcap=4000,8000
    else:
        fields={'sha256':PRIOR_SHA256,'expected_trajectories':1,'expected_reset_claims':2,'combined_trajectory_cap':4001,'combined_reset_cap':8002}
        if any(prior[k]!=v for k,v in fields.items()): raise ValueError('跨轮预算固定锚点或1/2/4001/8002不符')
        path=Path(prior['ledger'])
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to((Path(c['root'])/'control').resolve()): raise ValueError('prior账本必须是本轮control内实体副本')
        old,fingerprint=ledger_snapshot(path)
        if fingerprint!=PRIOR_SHA256: raise ValueError('prior账本SHA不符')
        prior_count=old.trajectories; prior_resets=sum(old.claimed.values())+old.orphan_claims
        if prior_count!=1 or prior_resets!=2 or old.resets!=2 or old.retries or old.recovery_used or old.astra: raise ValueError('prior实际计量不是1/2零重试')
        tcap,rcap=4001,8002
        if tcap!=4000+prior_count or rcap!=8000+prior_resets: raise ValueError('跨轮硬上限未闭合')
    if current is None:
        if Path(c['budget_ledger']).exists(): current,_=ledger_snapshot(Path(c['budget_ledger']))
        elif prior is not None: raise ValueError('恢复轮缺少当前预算账本')
    trajectories=current.trajectories if current is not None else 0
    claims=sum(current.claimed.values())+current.orphan_claims if current is not None else 0
    resets=current.resets if current is not None else 0
    retries=len(current.retries) if current is not None else 0
    if current is not None and (current.bad_rows or current.astra or current.recovery_used): raise ValueError('当前账本损坏或有禁止的Astra/恢复尝试')
    if trajectories>4000 or claims>8000 or resets>8000 or retries: raise ValueError('当前4000/8000/零重试上限越界')
    if prior_count+trajectories>tcap or prior_resets+max(claims,resets)>rcap: raise ValueError('跨轮累计上限越界')
    return dict(prior=dict(trajectories=prior_count,reset_claims=prior_resets,sha256=fingerprint),
                current=dict(trajectories=trajectories,reset_claims=claims,reset_count=resets,retries=retries,trajectory_cap=4000,reset_cap=8000),
                combined=dict(trajectories=prior_count+trajectories,reset_claims=prior_resets+claims,reset_count=prior_resets+resets,
                              trajectory_cap=tcap,reset_cap=rcap))
TASK_COUNTS = {
    **{t: (25,25) for t in ('BinFill','VideoUnmaskSwap','ButtonUnmaskSwap','VideoPlaceButton','VideoPlaceOrder','PickHighlight','VideoRepick')},
    **{t: (13,13,12,12) for t in ('VideoUnmask','ButtonUnmask')},
    **{t: (17,17,16) for t in ('PickXtimes','RouteStick','PatternLock')},
    **{t: (10,10,10,10,10) for t in ('SwingXtimes','StopCube')},
    'MoveCube': (50,), 'InsertPeg': (50,),
}


def atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    c=read(path); digest=sha(path)
    if 'config_sha256' in c and c['config_sha256']!=digest: raise ValueError('配置文件摘要冲突')
    c['config_sha256']=digest
    return c


def control_file(c: dict, path: str) -> Path:
    """显式控制槽只允许同root/control实体路径，不跟随任何符号链。"""
    control=Path(c['root'])/'control'; target=Path(path)
    try: relative=target.relative_to(control)
    except ValueError: raise ValueError('控制槽必须在同root/control下') from None
    if not target.is_absolute() or '..' in relative.parts: raise ValueError('控制槽不可包含相对路径或上跳')
    if Path(c['root']).is_symlink() or control.is_symlink() or target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(control.resolve()):
        raise ValueError('控制槽路径越界/缺失/符号链')
    node=target.parent
    while node!=control:
        if node.is_symlink(): raise ValueError('控制槽父目录符号链')
        node=node.parent
    return target


def legacy_context(c: dict) -> dict:
    source=c.get('legacy_source')
    if source is None: return c
    if set(source)!={'config','sha256','exec_commit'}: raise ValueError('旧来源描述字段不完整或含未知项')
    path=control_file(c,source['config'])
    if sha(path)!=source['sha256']: raise ValueError('旧配置SHA不符')
    old=load_config(path)
    if old.get('legacy_source') is not None: raise ValueError('禁止递归旧来源')
    if old['exec_commit']!=source['exec_commit']: raise ValueError('旧worker源码锚不符')
    changes={'exec_commit','config_sha256','runner','legacy_source','ready_path','models','prior_sources_manifest','pause_snapshot','shared_astra_cap','failure_policy','historical_errors'}
    if c.get('prior_sources_manifest') is not None: trusted_contexts(c)
    if c.get('pause_snapshot') is not None: read_pause_snapshot(c)
    if 'shared_astra_cap' in c and (type(c['shared_astra_cap']) is not int or c['shared_astra_cap']!=0): raise ValueError('新共享Astra上限必须0')
    if {k:v for k,v in old.items() if k not in changes}!={k:v for k,v in c.items() if k not in changes}:
        raise ValueError('新旧配置运行/身份/预算/CPU/资源/通知等固定字段改变')
    if set(old['models'])!=set(c['models']): raise ValueError('新旧模型集合冲突')
    for model in old['models']:
        before=dict(old['models'][model]); after=dict(c['models'][model]); env0=before.pop('env',{}); env1=after.pop('env',{})
        allowed={'PYTHONPATH','SGEVAL_THIRD_PARTY','ROBOMME_EVAL_ROOT'}
        if before!=after or {k:v for k,v in env0.items() if k not in allowed}!={k:v for k,v in env1.items() if k not in allowed}:
            raise ValueError('新旧模型路由/权重/参数或非快照环境改变')
    return old


def identity(model: str, row: dict) -> str:
    return f"{model}:ood:{row['key']}"


def publication_name(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()+'.json'


def failure_category(row):
    if row.get('infra') is not True: return None
    if row.get('exit_kind')=='watchdog_hard_exit' and row.get('exit_code')==75:
        if row.get('infra_reason') in ('watchdog_exit','episode_wall','deadline_first_infer','deadline_media_finalize'): return 'watchdog_exit'
    if row.get('infra_reason')=='watchdog_exit' and 'exit_kind' not in row: return 'watchdog_exit'
    return row.get('infra_reason')


def validate_config(config: dict) -> list[dict]:
    """完整输入字段缺失即失败，不能把未知计数当零。"""
    for k in ('run','run_name','root','gl_root','controller_job_id','exec_commit','config_sha256','identities','identities_sha256','models','seats','python','runner','budget_ledger','cpu_end_time','expected_identities','expected_publications'):
        if k not in config: raise ValueError(f'配置缺少{k}')
    if config['run'] != config['run_name']: raise ValueError('运行名冲突')
    policy=config.get('failure_policy',{'continue_reasons':[]})
    if set(policy)!={'continue_reasons'} or policy['continue_reasons'] not in ([],['watchdog_exit']): raise ValueError('继续失败策略只允许显式watchdog_exit')
    validate_historical_errors(config)
    if set(config['models']) != set(MODELS): raise ValueError('五模型白名单不符')
    seats=config['seats']; ids=[str(s['job_id']) for s in seats]
    if len(ids)!=4 or len(set(ids))!=4 or not all(i.isdigit() for i in ids): raise ValueError('GPU登记不符')
    if str(config['controller_job_id']) in ids: raise ValueError('CPU与GPU作业重复')
    for name,m in config['models'].items():
        policy,label,variant=ROUTES[name]
        if m['policy']!=policy or m['label']!=label: raise ValueError('模型路由不符合固定白名单')
        forbidden={'--infra-retries','--client-restarts','--trajectory-cap','--reset-cap','--astra-cap','--shared-infra-cap','--expired-cap',
                   '--planned-first-tries','--policy-seed','--identities','--dataset','--out','--seat','--run','--progress-file','--gpus',
                   '--stop-file','--claim-deadline','--policies','--policy','--budget-ledger','--budget','--queue','--memer-adapter','--port-base'}
        if any(str(a).split('=')[0] in forbidden for a in m['args']): raise ValueError('模型参数覆盖本轮固定预算或身份')
        def values(flag):
            args=m['args']; vals=[]
            for i,a in enumerate(args):
                if a==flag:
                    if i+1>=len(args): raise ValueError('模型参数缺值')
                    vals.append(args[i+1])
                elif str(a).startswith(flag+'='): vals.append(str(a).split('=',1)[1])
            return vals
        if values('--groundsg-variant')!=([variant] if variant else []): raise ValueError('GroundSG变体配对错误')
        adapters=values('--qwenvl-groundsg-adapter')
        if name=='qwenvl' and (len(adapters)!=1 or not adapters[0]): raise ValueError('QwenVL adapter缺失或重复')
        if name!='qwenvl' and adapters: raise ValueError('非QwenVL模型使用adapter')
        protected={f[2:].replace('-','_') for f in forbidden}|{'groundsg_variant','qwenvl_groundsg_adapter','memer_adapter','gpu'}
        legal_special={'--groundsg-variant','--qwenvl-groundsg-adapter'}
        for arg in m['args']:
            token=str(arg).split('=',1)[0]
            if token.startswith('--') and token[2:].replace('-','_') in protected and token not in legal_special:
                raise ValueError('透传别名覆盖固定路由或预算')
        for item in values('--cfg'):
            if str(item).split('=',1)[0].replace('-','_') in protected: raise ValueError('cfg覆盖固定路由或预算')
    if sha(Path(config['identities']))!=config['identities_sha256']: raise ValueError('身份文件SHA不符')
    rows=[json.loads(line) for line in Path(config['identities']).read_text().splitlines() if line.strip()]
    expected=Counter()
    for task, counts in TASK_COUNTS.items():
        for n,count in enumerate(counts, 1): expected[(task, 'xhard4' if task in ('MoveCube','InsertPeg') else f'xhard{n}')]=count
    if len(rows)!=800 or Counter((r['task'],r['tier']) for r in rows)!=expected: raise ValueError('800身份43格不符')
    if len({r['key'] for r in rows})!=800 or any(r['dataset']!='ood' for r in rows): raise ValueError('身份重复或非OOD')
    if any(len(r['spec_sha256'])!=64 or not isinstance(r['candidate'],int) for r in rows): raise ValueError('规格指纹不符')
    keys={identity(m,r) for m in MODELS for r in rows}
    if len(config['expected_identities'])!=4000 or set(config['expected_identities'])!=keys: raise ValueError('搬运身份manifest不符')
    if len(config['expected_publications'])!=4000 or set(config['expected_publications'])!={publication_name(k) for k in keys}: raise ValueError('搬运发布manifest不符')
    work_budget(config)
    if 'shared_astra_cap' in config and (type(config['shared_astra_cap']) is not int or config['shared_astra_cap']!=0): raise ValueError('本轮必须显式共享Astra0')
    legacy_context(config)
    trusted_contexts(config)
    return rows


def make_shards(rows: list[dict]) -> dict[str,list[dict]]:
    # 冒烟身份置首，之后各模型相同的互斥100身份切片。
    smoke=[r for r in rows if r['task']=='VideoUnmask' and r['tier']=='xhard1' and r['builder_episode']==0]
    if len(smoke)!=1: raise ValueError('正式首局缺失')
    ordered=smoke+[r for r in rows if r!=smoke[0]]
    return {f'{model}-{i:02d}': {'model':model,'rows':ordered[i*100:(i+1)*100]} for model in MODELS for i in range(8)}


def query_gpu(config: dict, *, timeout: float=5) -> dict:
    """独立查询当前登记GPU状态，未知查询不视作清理成功。"""
    try:
        p=subprocess.run(['squeue','-h','-j',','.join(str(s['job_id']) for s in config['seats']),'-o','%A %T'],
                         capture_output=True,text=True,timeout=timeout)
        active=p.stdout.strip(); query_rc=p.returncode; stderr=p.stderr
    except (OSError,subprocess.TimeoutExpired) as e: active=str(e); query_rc=-1; stderr=str(e)
    return dict(active=active,query_rc=query_rc,stderr=stderr,cleanup_incomplete=bool(active) or query_rc!=0,time=time.time())


def cancel_gpu(config: dict, *, timeout: float=25) -> dict:
    """只取消本轮四GPU；每条调用有超时，返回原始回执。"""
    deadline=time.monotonic()+timeout; records=[]
    for seat in config['seats']:
        jid=str(seat['job_id'])
        try:
            p=subprocess.run(['scancel',jid],capture_output=True,text=True,timeout=max(.1,deadline-time.monotonic()))
            records.append(dict(job_id=jid,rc=p.returncode,stdout=p.stdout,stderr=p.stderr))
        except (OSError,subprocess.TimeoutExpired) as e: records.append(dict(job_id=jid,error=str(e)))
    latest=dict(active='尚未查询',query_rc=-1,cleanup_incomplete=True)
    while time.monotonic()<deadline:
        latest=query_gpu(config,timeout=max(.1,deadline-time.monotonic()))
        if not latest['cleanup_incomplete']: break
        time.sleep(min(.5,max(0,deadline-time.monotonic())))
    return dict(records=records,**latest)


def observed_job_state(c,job,failures,component):
    """每次查询仍限5秒，临时查询错误只在连续30秒观察窗内跳过该席。"""
    now=time.monotonic(); prior=failures.get(str(job))
    if prior and prior.get('error') and now-prior['start']>=30: raise RuntimeError('SLURM_QUERY_WINDOW=FAIL '+json.dumps(prior,ensure_ascii=False))
    if prior and now<prior['next_query']:
        return None
    try:
        p=subprocess.run(['squeue','-h','-j',str(job),'-o','%T'],capture_output=True,text=True,timeout=5)
        if p.returncode: raise RuntimeError(f'rc={p.returncode} stderr={p.stderr}')
    except (OSError,subprocess.TimeoutExpired,RuntimeError) as e:
        now=time.monotonic()
        first=prior if prior and prior.get('error') else dict(start=now,error=str(e),time=time.time())
        first['next_query']=now+5; failures[str(job)]=first
        evidence=dict(component=component,job_id=str(job),first_error=first['error'],last_error=str(e),first_time=first['time'],elapsed_s=now-first['start'],time=time.time())
        atomic(Path(c['root'])/'control'/f'{component}-query-{job}.json',evidence)
        if now-first['start']>=30: raise RuntimeError('SLURM_QUERY_WINDOW=FAIL '+json.dumps(evidence,ensure_ascii=False)) from e
        return None
    status=p.stdout.strip()
    if status in ('PENDING','CONFIGURING'): failures[str(job)]=dict(next_query=time.monotonic()+10)
    else: failures.pop(str(job),None)
    return status


def process_identity(pid):
    """只读精确PID的原始命令行与创建tick，不用模式匹配，也不推断退出成功。"""
    proc=Path('/proc')/str(int(pid))
    try:
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        raw=(proc/'cmdline').read_bytes()
        if not raw: return None
        if not raw.endswith(b'\0'): raise ValueError('进程命令行缺少终结NUL')
        # 只去掉终结NUL；--gpus的合法空参数必须逐字保留。
        args=[x.decode() for x in raw.split(b'\0')[:-1]]
    except FileNotFoundError: return None
    boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    return dict(state=stat[0],start_ticks=int(stat[19]),start_time=boot+int(stat[19])/os.sysconf('SC_CLK_TCK'),command=args,
                cgroup=(proc/'cgroup').read_text())


def worker_account(step,name,*,allow_drained=False,allow_watchdog=False,evidence=False):
    p=subprocess.run(['sacct','-n','-P','-j',step,'--format=JobIDRaw,JobName%128,State,ExitCode'],capture_output=True,text=True,timeout=5)
    rows=[x.split('|') for x in p.stdout.splitlines() if x.split('|')[0]==step]
    if p.returncode or len(rows)!=1 or rows[0][1]!=name: raise ValueError('已有worker记账身份或状态未知 '+step)
    state,code=rows[0][2:]
    if state=='RUNNING' and code=='0:0': return None
    if state=='COMPLETED' and code=='0:0': return dict(id=step,name=name,state=state,exit_code=code) if evidence else 0
    if allow_drained and state in ('FAILED','COMPLETED') and code=='6:0': return dict(id=step,name=name,state=state,exit_code=code) if evidence else 6
    if allow_watchdog and state in ('FAILED','COMPLETED') and code=='75:0':
        return dict(id=step,name=name,state=state,exit_code=code) if evidence else 75
    raise ValueError('已有worker异常终态或未知退出 '+str(rows[0]))


def step_inventory(c):
    """Slurm步骤格式只读取StepID与JobName，状态独立以sacct确认。"""
    jobs={str(s['job_id']) for s in c['seats']}
    q=subprocess.run(['squeue','--steps','-h','-j',','.join(sorted(jobs)),'-o','%i %j'],capture_output=True,text=True,timeout=5)
    if q.returncode or q.stderr: raise ValueError('全步骤查询未知 '+q.stderr)
    steps={}
    for line in q.stdout.splitlines():
        cols=line.split()
        parts=cols[0].split('.') if cols else []
        if len(cols)!=2 or len(parts)!=2 or parts[0] not in jobs or not (parts[1].isdigit() or parts[1] in ('batch','extern','interactive')) or cols[0] in steps:
            raise ValueError('全步骤查询格式/作业/重复不符')
        steps[cols[0]]=cols[1]
    return steps


class ExistingProcess:
    """附着已存在srun，只查询；没有wait/kill/launch能力。"""
    def __init__(self,pid,identity,step,name): self.pid=pid; self.identity=identity; self.step=step; self.name=name
    def poll(self):
        current=process_identity(self.pid)
        if current is not None and current['state']!='Z':
            if self.identity is None or any(current[k]!=self.identity[k] for k in ('start_ticks','command','cgroup')):
                raise ValueError('已有srun PID复用或命令行改变')
            return None
        rc=worker_account(self.step,self.name,allow_watchdog=True,allow_drained=True)
        if rc is None:
            began=getattr(self,'disappeared_monotonic',None)
            if began is None: self.disappeared_monotonic=time.monotonic()
            elif time.monotonic()-began>=120: raise ValueError('srun消失但Slurm仍RUNNING，退出未知超过120秒')
        return rc


class ControllerLeaseBusy(RuntimeError): pass


class ControllerLease:
    """控制器写心跳和状态之前取得排他租约，子进程不继承描述符。"""
    def __init__(self,controller): self.owner=controller; self.stream=None
    def __enter__(self):
        import fcntl
        c=self.owner; path=c.control/'controller.lease'
        if path.is_symlink(): raise ValueError('控制器租约不能是符号链')
        self.stream=path.open('a+')
        try: fcntl.flock(self.stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close(); self.stream=None
            raise ControllerLeaseBusy('全局控制器租约已占用') from None
        os.set_inheritable(self.stream.fileno(),False)
        # 旧快照没有租约，仍不能与已证明存活的同轮控制器并行。
        beat=c.control/'controller-heartbeat.json'
        if beat.exists():
            row=read(beat); pid=int(row['pid'])
            if pid!=os.getpid() and all(str(row[k])==str(c.base[k]) for k in CORE):
                proc=process_identity(pid)
                if proc is not None and proc['state']!='Z' and any(Path(x).name=='raw_ood_controller.py' for x in proc['command']):
                    self.stream.close(); self.stream=None; raise ControllerLeaseBusy('旧协议控制器仍存活，禁止并行迁移')
        self.stream.seek(0); self.stream.truncate(); self.stream.write(json.dumps(dict(c.base,pid=os.getpid(),time=time.time()))+'\n'); self.stream.flush(); os.fsync(self.stream.fileno())
        return self
    def __exit__(self,*_):
        if self.stream is not None: self.stream.close()


class Controller:
    def __init__(self, config: dict):
        self.c=config; self.rows=validate_config(config); self.root=Path(config['root']); self.control=self.root/'control'
        self.gl=Path(config['gl_root']); self.shards=make_shards(self.rows); self.pending=list(self.shards)
        self.active={}; self.published=set(); self.smoked=set(); self.completed=set(); self.formal=False
        self.failures={}; self.unresolved=set()
        self.initial_dispatched=set(); self.started=time.time()
        self.query_failures={}
        self.publication_writes={}
        self.loop_t=time.time(); self.loop_phase='initialization'; self.pausing=False; self.paused=set(); self.resume_rows={}; self.segment_numbers={}
        self.base={k:config[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
        self.sources=source_configs(config)
        old=legacy_context(config)
        if old['config_sha256'] not in {c['config_sha256'] for c in self.sources}: self.sources.append(old)
        self.deadline=min(float(config['cpu_end_time']),*(float(s['end_time']) for s in config['seats']))-7200

    def event(self, kind: str, **kw):
        row=dict(self.base,time=time.time(),kind=kind,**kw)
        self.control.mkdir(parents=True,exist_ok=True)
        with (self.control/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

    def advance(self,phase): self.loop_t=time.time(); self.loop_phase=phase

    def retry_filesystem(self,operation,*args,**kwargs):
        """只重试明确瞬态errno，最多120秒，不改worker真实t或声明完成。"""
        started=time.monotonic()
        while True:
            try: return operation(*args,**kwargs)
            except OSError as e:
                if e.errno not in (5,121,116): raise
                elapsed=time.monotonic()-started
                if elapsed>=120: raise RuntimeError(f'CONTROLLER_FILESYSTEM=FAIL errno={e.errno} elapsed_s={elapsed}') from e
                self.loop_phase='filesystem_retry'
                print(f'CONTROLLER_FILESYSTEM=RETRY errno={e.errno} elapsed_s={elapsed}',flush=True)
                time.sleep(min(1,120-elapsed))

    def source_of(self,row):
        if not set(CORE)<=row.keys(): raise ValueError('已有记录缺core字段')
        matches=[c for c in self.sources if all(type(row[k]) is type(c[k]) and row[k]==c[k] for k in CORE)]
        if len(matches)!=1: raise ValueError('已有记录没有唯一已锁来源')
        return matches[0]

    def start(self, seat: dict, shard: str, *, smoke=False):
        spec=self.shards[shard]; model=spec['model']; m=self.c['models'][model]
        rows=spec['rows'][:1] if smoke else self.resume_rows.get(shard,spec['rows'])
        selection=None
        if not smoke and self.formal:
            available={r['key'] for r in self.unreserved_rows(shard)}
            rows=[r for r in rows if r['key'] in available]; selection='unreserved-v1'
            if not rows: raise ValueError('正式片没有尚未预约身份，不得启动worker')
        number=self.segment_numbers.get(shard,0)
        tag=f"{shard}-resume-{number}-full" if not smoke and number else f"{shard}-{'smoke' if smoke else 'full'}"
        inv=self.control/'invocations'/tag
        inv.mkdir(parents=True,exist_ok=False)
        ids=inv/'identities.jsonl'; ids.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        seat_name=f"{seat['seat']}-{tag}"; progress=inv/'progress.json'
        worker_name=f'raw-ood-worker-{tag}'
        cmd=self.worker_command(seat,shard,smoke=smoke,inv_tag=tag)
        env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_','SRUN_'))}
        env.pop('CUDA_VISIBLE_DEVICES',None)
        env.update(BENCH_PY=self.c['python'],PYTHONUNBUFFERED='1'); env.update(m.get('env',{}))
        log=(inv/'worker.log').open('a'); p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        active=dict(process=p,log=log,seat=seat,shard=shard,smoke=smoke,model=model,progress=progress,
                    results=self.gl/'seats'/m['label']/'seed7'/seat_name/'seat-results.jsonl',rows=rows,offset=0,started=time.time(),acked=False,
                    phase_key=None,phase_start=time.time(),step_id=None,worker_name=worker_name,inv_tag=tag)
        self.active[str(seat['job_id'])]=active
        proc=process_identity(p.pid)
        if proc is not None and (not proc['command'] or proc['command'][1:]!=cmd[1:]): proc=None
        atomic(inv/'dispatch.json',dict(self.base,gpu_job_id=str(seat['job_id']),shard_id=shard,segment_id=tag,model=model,smoke=smoke,command=cmd,srun_pid=p.pid,
                                       srun_start_ticks=proc['start_ticks'] if proc else None,identity_selection=selection,time=time.time()))
        self.event('dispatch',gpu_job_id=str(seat['job_id']),shard_id=shard,smoke=smoke)

    def worker_command(self,seat,shard,*,smoke=False,inv_tag=None):
        model=self.shards[shard]['model']; m=self.c['models'][model]; tag=inv_tag or f"{shard}-{'smoke' if smoke else 'full'}"
        inv=self.control/'invocations'/tag; seat_name=f"{seat['seat']}-{tag}"; ids=inv/'identities.jsonl'; progress=inv/'progress.json'
        expansion=dict(root=str(self.root),gl_root=str(self.gl),seat=seat_name,shard=shard,port_base=seat['port_base'],work_dir=str(inv/'work'))
        model_args=[str(x).format_map(expansion) for x in m['args']]
        return ['srun',f"--jobid={seat['job_id']}",f'--job-name=raw-ood-worker-{tag}','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--gpu_cmode=shared',
             'bash',self.c['runner'],'--policies',m['policy'],'--policy-seed','7','--identities',str(ids),'--out',str(self.gl),
             '--run',self.c['run'],'--seat',seat_name,'--budget-ledger',self.c['budget_ledger'],
             '--trajectory-cap','4000','--reset-cap','8000','--planned-first-tries','4000','--shared-infra-cap','0','--expired-cap','0',
             '--infra-retries','0','--client-restarts','0','--gpus','','--no-render','--stop-on-env-build-error','--dataset','ood',
             '--port-base',str(seat['port_base']),'--claim-deadline',str(self.deadline),'--stop-file',str(self.control/'STOP'),
             '--progress-file',str(progress),*(['--astra-cap','0'] if 'shared_astra_cap' in self.c else []),*model_args]

    def prepare_publication(self, a: dict, row: dict,*,worker_exec_commit=None):
        if row.get('infra') is True: raise RuntimeError('基础设施失败不得发布原始媒体')
        required=('key','infra','final','accepted','result','exec_steps','task_success','budget_exhausted','run_blocked')
        if any(k not in row for k in required): raise ValueError('席位报告缺少必填字段')
        if row['infra'] or row['budget_exhausted'] or row['run_blocked'] or not row['final'] or not row['accepted']:
            raise RuntimeError(f"席位基础设施失败 {row}")
        key=f"{a['model']}:ood:{row['key']}"
        if key in self.published: raise ValueError('重复完成身份')
        if row['key'] not in {r['key'] for r in a['rows']}: raise ValueError('跨片身份')
        result=Path(row['result']); directory=result.parent
        if not directory.resolve().is_relative_to(self.gl.resolve()) or directory.is_symlink(): raise ValueError('产物越界')
        data=read(result)
        if data['key']!=row['key'] or data['dataset']!='ood' or data['infra']: raise ValueError('result身份或状态不符')
        for file in ('front.mkv','wrist.mkv','arrays.npz','trace.jsonl','meta.json','events.jsonl','frames-front.jsonl','frames-wrist.jsonl'):
            if not (directory/file).is_file(): raise ValueError(f'产物缺失{directory/file}')
        files={}
        for f in directory.rglob('*'):
            if f.is_symlink(): raise ValueError('产物不得是符号链接')
            if f.is_file(): files[f.relative_to(directory).as_posix()]=dict(sha256=sha(f),size=f.stat().st_size)
        if any(f.endswith('.mp4') for f in files): raise ValueError('出现官方视频')
        doc=dict(self.base,schema='raw-ood-v1',model=a['model'],shard_id=a['shard'],identity_key=key,
                 relative_dir=directory.relative_to(self.gl).as_posix(),files=files,delete_files=['front.mkv','wrist.mkv','arrays.npz'],
                 worker_exec_commit=worker_exec_commit or self.c['exec_commit'])
        return doc

    def publish(self, a: dict, row: dict, *, prepared=None):
        if 'key' not in row: raise ValueError('席位报告缺身份key')
        key=identity(a['model'],row)
        doc=prepared if prepared is not None else self.publication_writes.get(key)
        if doc is None: doc=self.prepare_publication(a,row)
        key=doc['identity_key']
        if key in self.published: raise ValueError('重复完成身份')
        self.publication_writes[key]=doc
        target=self.gl/'published'/publication_name(key)
        if target.exists():
            if read(target)!=doc: raise ValueError('已有发布原字节内容冲突')
        else: atomic(target,doc)
        self.published.add(key)
        self.event('published',identity_key=key,task_success=row['task_success'])
        if not a['smoke'] and not a['acked'] and row['exec_steps']>=1 and a.get('step_id') and a['progress'].exists():
            p=read(a['progress'])
            self.write_ack(a,key,p['pid'],row['exec_steps'],str(row['result']))
        self.publication_writes.pop(key,None)

    def resume_completed_smokes(self,orchestration_commit,*,preflight=False):
        """手动复用原三个已结算首局；先完整验证全部输入，再发布，绝不启动worker。"""
        import re
        if not re.fullmatch('[0-9a-f]{40}',str(orchestration_commit)): raise ValueError('恢复须记录40位orchestration_commit')
        old=legacy_context(self.c); oldbase={k:old[k] for k in self.base}
        for step,name in step_inventory(self.c).items():
            if name.startswith('raw-ood-worker-'):
                allowed={'raw-ood-worker-'+m+'-00-smoke' for m in ('qwenvl','smvla','pp')}
                if name not in allowed or worker_account(step,name)!=0: raise ValueError('原worker仍在途或异常，不可接管')
        invs=self.control/'invocations'
        if any((invs/f'{m}-00-smoke').exists() for m in ('oracle','framesamp')): raise ValueError('仅允许复用原三个smoke')
        ledger,ledger_sha=ledger_snapshot(Path(self.c['budget_ledger']))
        prepared=[]; fingerprints={}
        for model in ('qwenvl','smvla','pp'):
            inv=invs/f'{model}-00-smoke'; dispatch=read(inv/'dispatch.json'); ident=self.shards[model+'-00']['rows'][0]
            if any(str(dispatch[k])!=str(v) for k,v in oldbase.items()): raise ValueError('原dispatch版本/运行身份冲突')
            if dispatch['model']!=model or dispatch['shard_id']!=model+'-00' or dispatch['smoke'] is not True: raise ValueError('原dispatch不是指定首局')
            job=str(dispatch['gpu_job_id']); seat=next((s for s in self.c['seats'] if str(s['job_id'])==job),None)
            if seat is None: raise ValueError('原dispatch作业不在登记清单')
            command=dispatch['command']; seat_name=f"{seat['seat']}-{model}-00-smoke"
            for flag,value in {'--jobid':job,'--seat':seat_name,'--out':str(self.gl),'--policy-seed':'7','--policies':old['models'][model]['policy'],'--identities':str(inv/'identities.jsonl')}.items():
                values=[str(a).split('=',1)[1] for a in command if str(a).startswith(flag+'=')]
                values += [str(command[i+1]) for i,a in enumerate(command[:-1]) if a==flag]
                if values!=[value]: raise ValueError('原worker命令身份冲突 '+flag)
            if old['runner'] not in command: raise ValueError('原runner路径冲突')
            expansion=dict(root=str(self.root),gl_root=str(self.gl),seat=seat_name,shard=model+'-00',port_base=seat['port_base'],work_dir=str(inv/'work'))
            original_args=[str(x).format_map(expansion) for x in old['models'][model]['args']]
            if original_args and command[-len(original_args):]!=original_args: raise ValueError('原模型参数与旧配置不符')
            identities=[json.loads(x) for x in (inv/'identities.jsonl').read_text().splitlines()]
            if identities!=[ident]: raise ValueError('原smoke身份清单不一致')
            progress=read(inv/'progress.json')
            if progress['phase']!='finished' or progress['rc']!=0 or progress['episodes_done']!=1: raise ValueError('原worker未正常finished')
            if str(progress['slurm_job_id'])!=job or progress['label']!=self.c['models'][model]['label'] or progress['seat']!=seat_name or progress['policy_seed']!=7:
                raise ValueError('原progress身份冲突')
            step=job+'.'+str(progress['slurm_step_id'])
            accounting=subprocess.run(['sacct','-n','-P','-j',step,'--format=JobIDRaw,JobName%128,State,ExitCode'],capture_output=True,text=True,timeout=5)
            records=[x.split('|') for x in accounting.stdout.splitlines() if x.split('|')[0]==step]
            if accounting.returncode or len(records)!=1 or records[0][1:]!=['raw-ood-worker-'+model+'-00-smoke','COMPLETED','0:0']:
                raise ValueError('原worker退出码未知或非COMPLETED/0:0')
            path=self.gl/'seats'/self.c['models'][model]['label']/'seed7'/seat_name/'seat-results.jsonl'
            deadline=time.monotonic()+120
            while True:
                try: content=path.read_text()
                except FileNotFoundError: content=''
                if content and content.endswith('\n'): break
                if time.monotonic()>=deadline: raise ValueError('原smoke完整报告不可见 '+str(path))
                time.sleep(1)
            rows=[json.loads(x) for x in content.splitlines()]
            if len(rows)!=1: raise ValueError('原smoke报告必须完整唯一一行')
            row=rows[0]
            if row['key']!=ident['key'] or row['dataset']!='ood' or row['seat']!=seat_name or row['attempt']!=1 or row['policy_seed']!=7 or row['policy']!=self.c['models'][model]['policy'] or row['exec_steps']<1:
                raise ValueError('原报告首身份或步骤冲突')
            rid=row['budget_rid']; reserve=ledger.reserves.get(rid); commit=ledger.commits.get(rid)
            route=self.c['models'][model]['policy']+('/'+ROUTES[model][2] if ROUTES[model][2] else '')+'/seed7/new'
            if not reserve or not commit or rid in ledger.releases or reserve['route']!=route or reserve['key']!=ident['key'] or reserve['token']!=row['budget_token']:
                raise ValueError('原报告没有对应已结算预约')
            if reserve['attempt_no']!=1 or str(reserve['slurm_job_id'])!=job or str(commit['slurm_job_id'])!=job or commit['infra'] is not False or commit['status']!=row['status']:
                raise ValueError('原共享账本结算冲突')
            if row['budget_token']!=route+'|'+ident['key']+'|a1' or ledger.claimed.get(rid)!=2: raise ValueError('原首试token/reset计量冲突')
            accepted=read(self.gl/'queue'/self.c['models'][model]['label']/'seed7/accepted'/('ood__'+ident['key']+'.json'))
            if accepted['key']!=ident['key'] or accepted['dataset']!='ood' or accepted['attempt']!=1 or accepted['seat']!=seat_name or accepted['result']!=row['result'] or accepted['attempt_id']!=row['attempt_id']:
                raise ValueError('原accepted标记缺失或冲突，禁止让后续分片重新领取')
            expected_result=self.gl/'rollouts'/self.c['models'][model]['label']/'ood/seed7/raw'/f"{ident['task']}_ep{ident['builder_episode']}_{ident['tier']}"/'result.json'
            if Path(row['result'])!=expected_result: raise ValueError('原result路径冲突')
            directory=Path(row['result']).parent; result=read(Path(row['result'])); meta=read(directory/'meta.json'); summary=read(directory/'summary.json')
            for key in ('dataset','task','tier','seed','key','spec_sha256'):
                if result[key]!=ident[key] or meta['identity'][key]!=ident[key]: raise ValueError('原raw身份冲突 '+key)
            if result['policy_label']!=self.c['models'][model]['label'] or result['policy_seed']!=7 or result['recorder_verify']!='PASS' or result['exec_steps']!=row['exec_steps']:
                raise ValueError('原result未正常关闭或步骤不符')
            if result['status']!=row['status'] or result['task_success']!=row['task_success'] or meta['pid']!=progress['pid']: raise ValueError('原终态或媒体生成进程冲突')
            if summary['RECORDER_VERIFY']!='PASS' or summary['errors'] or summary['decode_mismatch']!=0 or summary['summary']['exec_steps']!=row['exec_steps']:
                raise ValueError('原录像器关闭核验失败')
            for stream in ('front','wrist'):
                info=summary['streams'][stream]
                if info['error'] is not None or info['mismatch']!=0 or not info['timestamps_ok'] or info['encoded']<=0 or info['decoded']!=info['encoded']:
                    raise ValueError('原媒体流关闭核验失败')
            trace=[json.loads(x) for x in (directory/'trace.jsonl').read_text().splitlines()]
            if trace[0]['kind']!='header' or trace[-1]['kind']!='end' or trace[-1]['exec_steps']!=row['exec_steps']: raise ValueError('原trace未关闭')
            for key in ('dataset','task','tier','seed','key'):
                if trace[0]['identity'][key]!=ident[key]: raise ValueError('原trace身份冲突')
            a=dict(model=model,shard=model+'-00',rows=[ident],smoke=True)
            doc=self.prepare_publication(a,row,worker_exec_commit=dispatch['exec_commit'])
            if (self.gl/'published'/publication_name(doc['identity_key'])).exists(): raise ValueError('原smoke已发布，拒绝重复恢复')
            prepared.append((a,row,doc))
            fingerprints[model]=dict(dispatch_sha256=sha(inv/'dispatch.json'),progress_sha256=sha(inv/'progress.json'),report_sha256=sha(path),step=step,
                                     worker_exec_commit=dispatch['exec_commit'])
        if sha(Path(self.c['budget_ledger']))!=ledger_sha: raise ValueError('复用验证期间共享账本发生改变')
        proof=dict(self.base,orchestration_commit=orchestration_commit,reused=3,fingerprints=fingerprints,budget_sha256=ledger_sha,
                   identities=[doc['identity_key'] for _,_,doc in prepared],verification='identity_closed_recorder_summary_full_decode_by_mover',time=time.time())
        if self.c.get('legacy_source') is not None: proof['legacy_source']=dict(self.c['legacy_source'])
        if preflight:
            print('RESUME_PREFLIGHT=PASS models=3 identities=3 budget_sha256='+ledger_sha,flush=True)
            print(json.dumps(proof,ensure_ascii=False,sort_keys=True),flush=True)
            return proof
        for a,row,doc in prepared:
            self.publish(a,row,prepared=doc); self.smoked.add(a['model'])
        atomic(self.control/'resume-completed-smokes.json',proof)

    def write_ack(self,a,key,pid,step,proof):
        ack=dict(self.base,gpu_job_id=str(a['seat']['job_id']),step_id=a['step_id'],model=a['model'],shard_id=a['shard'],
                 identity_key=key,client_pid=pid,first_step=step,progress_mtime=a['progress'].stat().st_mtime,proof=proof,time=time.time())
        if not (self.control/'first_dispatch.json').exists(): atomic(self.control/'first_dispatch.json',ack)
        a['acked']=True

    def validate_settled_row(self,a,row,ledger,pub=None):
        """已有报告仍须逐身份与已结算预算核验，不能因发布存在就略过坏行。"""
        model=a['model']; ids={r['key']:r for r in a['rows']}
        required={'key','dataset','policy','policy_seed','infra','accepted','final','budget_exhausted','run_blocked','attempt','seat','budget_rid','budget_token','status','task_success','exec_steps','result'}
        if not required<=row.keys(): raise ValueError('正常报告缺显式必需字段')
        if type(row['infra']) is not bool or row['infra'] or type(row['task_success']) not in (bool,int) or row['task_success'] not in (0,1) or type(row['exec_steps']) is not int or row['exec_steps']<0:
            raise ValueError('正常报告基础设施/成绩/步数类型不符')
        if row['key'] not in ids or row['dataset']!='ood' or row['policy']!=self.c['models'][model]['policy'] or row['policy_seed']!=7:
            raise ValueError('已有报告模型或身份不符')
        if row['infra'] is not False or row['accepted'] is not True or row['final'] is not True or row['budget_exhausted'] is not False or row['run_blocked'] is not False or type(row['attempt']) is not int or row['attempt']!=1:
            raise ValueError('已有报告异常或非已接受首试')
        expected_seat=f"{a['seat']['seat']}-"+a.get('inv_tag',f"{a['shard']}-{'smoke' if a['smoke'] else 'full'}")
        if row['seat']!=expected_seat: raise ValueError('已有报告席位冲突')
        reserve=ledger.reserves.get(row['budget_rid']); commit=ledger.commits.get(row['budget_rid'])
        route=self.c['models'][model]['policy']+('/'+ROUTES[model][2] if ROUTES[model][2] else '')+'/seed7/new'
        if not reserve or not commit or row['budget_rid'] in ledger.releases or reserve['key']!=row['key'] or reserve['route']!=route or reserve['token']!=row['budget_token'] or reserve['token']!=route+'|'+row['key']+'|a1' or reserve['kind_of_try']!='first' or reserve['attempt_no']!=1 or reserve['resets']!=2 or commit['infra'] is not False or commit['status']!=row['status']:
            raise ValueError('已有报告预算未结算或路由不符')
        if len([r for r in ledger.reserves.values() if r.get('route')==route and r.get('key')==row['key']])!=1: raise ValueError('同路线身份重复预约')
        if ledger.claimed.get(row['budget_rid'],0)!=2: raise ValueError('已有报告reset claims不符')
        if str(reserve['slurm_job_id'])!=str(a['seat']['job_id']) or str(commit['slurm_job_id'])!=str(a['seat']['job_id']): raise ValueError('已有报告预算Slurm作业不符')
        if pub is not None:
            result=self.gl/pub['relative_dir']/'result.json'
            if Path(row['result'])!=result or row['exec_steps']!=read(result)['exec_steps']: raise ValueError('已有报告与发布结果冲突')

    def prepare_failure(self,a,row,line):
        """先辨基础设施行，再用原派发、真实75及预算/领取证明稀疏历史行。"""
        if failure_category(row)!='watchdog_exit': raise ValueError('未获继续许可的基础设施原因')
        inv=a['progress'].parent; dispatch=inv/'dispatch.json'; d=read(dispatch); p=read(a['progress'])
        source=self.source_of(d); original=Controller(source)
        if d['command']!=original.worker_command(a['seat'],a['shard'],smoke=a['smoke'],inv_tag=inv.name): raise ValueError('失败原worker命令不符锁定配置')
        step=str(a['seat']['job_id'])+'.'+str(p['slurm_step_id']); name='raw-ood-worker-'+inv.name
        step_proof=worker_account(step,name,allow_watchdog=True,evidence=True)
        if not isinstance(step_proof,dict) or step_proof['exit_code']!='75:0': raise ValueError('watchdog报告缺真实rc75')
        if row['key'] not in {r['key'] for r in a['rows']}: raise ValueError('失败跨片身份')
        ledger,_=ledger_snapshot(Path(self.c['budget_ledger'])); rid=row['budget_rid']
        raw,_=read_budget_rows(Path(self.c['budget_ledger']))
        claim=self.gl/'queue'/self.c['models'][a['model']]['label']/'seed7/claims'/('ood__'+row['key']+'.a1.json')
        key=identity(a['model'],row)
        doc=dict({k:d[k] for k in CORE},schema='raw-ood-failure-v1',identity_key=key,model=a['model'],shard_id=a['shard'],reason='watchdog_exit',infra_reason=row['infra_reason'],
                 task_success=row.get('task_success'),exec_steps=row.get('exec_steps'),report_path=str(a['results']),report_line=line,report_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                 budget=dict(rid=rid,reserve=ledger.reserves.get(rid),commit=ledger.commits.get(rid),reset_claims=[r for r in raw if r.get('kind')=='reset_claim' and r.get('rid')==rid]),
                 dispatch=dict(path=str(dispatch),sha256=sha(dispatch)),progress=dict(path=str(a['progress']),sha256=sha(a['progress'])),
                 queue=dict(path=str(claim),sha256=sha(claim)),step=step_proof)
        validate_failure_document(doc,self.c)
        a['legacy_terminal_progress']=p['phase']=='episode' and legacy_watchdog_progress(d,row,self.c)
        return doc

    def record_failure(self,doc):
        key=doc['identity_key']
        if key in self.published: raise ValueError('失败与已发布身份冲突')
        target=self.gl/'failures'/publication_name(key)
        if target.exists():
            if read(target)!=doc: raise ValueError('失败终态证据冲突')
        else: atomic(target,doc)
        self.failures[key]=dict(path=str(target),sha256=sha(target))

    def settled(self): return self.published|set(self.failures)

    def unreserved_rows(self,shard):
        ledger,_=ledger_snapshot(Path(self.c['budget_ledger'])); model=self.shards[shard]['model']
        route=self.c['models'][model]['policy']+('/'+ROUTES[model][2] if ROUTES[model][2] else '')+'/seed7/new'
        attempted={identity(model,r) for r in ledger.reserves.values() if r.get('route')==route}
        self.unresolved.update(attempted-self.settled()-{identity(a['model'],r) for a in self.active.values() for r in a['rows']})
        return [r for r in self.shards[shard]['rows'] if identity(model,r) not in attempted]

    def validate_drained_prefix(self,a,done):
        """rc6仅允许完整已结算前缀，预约过但无完整结果的半局绝不重跑。"""
        ledger,_=ledger_snapshot(Path(self.c['budget_ledger']))
        keys={r['key'] for r in a['rows']}; model=a['model']; route=self.c['models'][model]['policy']+('/'+ROUTES[model][2] if ROUTES[model][2] else '')+'/seed7/new'
        attempted=set(); seen=set()
        for rid,reserve in ledger.reserves.items():
            if reserve['route']!=route or reserve['key'] not in keys: continue
            if 'history_until' in a and float(reserve['t'])>a['history_until']: continue
            if reserve['key'] in seen: raise ValueError('暂停前缀重复预约身份')
            seen.add(reserve['key']); commit=ledger.commits.get(rid)
            if not commit:
                self.unresolved.add(identity(model,reserve)); continue
            claims_valid=ledger.claimed.get(rid,0)==2 or commit.get('infra') is True and identity(model,reserve) in done
            if reserve['kind_of_try']!='first' or reserve['attempt_no']!=1 or rid in ledger.releases or not claims_valid or 'history_until' in a and float(commit['t'])>a['history_until']:
                raise ValueError('暂停存在预算性质错误/不足2reset半局')
            attempted.add(identity(model,reserve))
        expected_done={identity(model,r) for r in a['rows']} & done
        if attempted!=expected_done: raise ValueError('暂停已预约身份与闭合报告/发布集合不齐')
        return attempted

    def existing_publications(self):
        from eval_video_mover import raw_validate_publication,raw_validate_receipt
        pubs={}; allowed=set(self.c['expected_identities']); old=legacy_context(self.c)
        sources=self.sources
        for path in sorted((self.gl/'published').glob('*.json')):
            doc=read(path); context=self.source_of(doc); raw_validate_publication(doc,{k:context[k] for k in CORE}); key=doc['identity_key']
            if key not in allowed or path.name!=publication_name(key) or key in pubs: raise ValueError('已有发布身份或文件名冲突')
            shard=doc['shard_id']; model=doc['model']
            if shard not in self.shards or self.shards[shard]['model']!=model or key not in {identity(model,r) for r in self.shards[shard]['rows']}:
                raise ValueError('已有发布分片不符')
            source=doc['worker_exec_commit']
            legacy_key=identity(model,self.shards[model+'-00']['rows'][0])
            if source!=context['exec_commit'] and not (model in ('qwenvl','smvla','pp') and key==legacy_key and source in {old['exec_commit'],*(s['exec_commit'] for s in sources)}):
                raise ValueError('已有发布worker源码不符')
            result=self.gl/doc['relative_dir']/'result.json'
            if result.is_symlink() or not result.resolve().is_relative_to(self.gl.resolve()): raise ValueError('已有结果路径越界')
            info=doc['files']['result.json']
            if sha(result)!=info['sha256'] or result.stat().st_size!=info['size']: raise ValueError('已有小result指纹不符')
            for name,info in doc['files'].items():
                if name in doc['delete_files']: continue
                small=result.parent/name
                if small.is_symlink() or not small.resolve().is_relative_to(result.parent.resolve()) or sha(small)!=info['sha256'] or small.stat().st_size!=info['size']:
                    raise ValueError('已有保留小文件指纹不符 '+name)
            value=read(result); ident=next(r for r in self.shards[shard]['rows'] if identity(model,r)==key)
            if value['infra'] or any(value[k]!=ident[k] for k in ('dataset','task','tier','seed','key')) or value['policy_label']!=self.c['models'][model]['label']:
                raise ValueError('已有小result身份或基础设施失败')
            receipt=self.gl/'mover'/'receipts'/path.name
            if receipt.exists():
                raw_validate_receipt(read(receipt),doc,self.c,dest=None)
            pubs[key]=doc
        return pubs

    def resume_existing_dispatches(self,orchestration_commit,*,preflight=False,paused=False):
        """全量只读核验后附着既有正式派发；不调用srun，不预约或修改预算。"""
        import re
        if not re.fullmatch('[0-9a-f]{40}',str(orchestration_commit)): raise ValueError('已有派发接续须明确编排SHA')
        manifest=read(self.control/'shards.json')
        old=legacy_context(self.c)
        if manifest not in tuple(dict({k:c[k] for k in CORE},shards=self.shards) for c in self.sources):
            raise ValueError('已有分片manifest不符')
        pubs=self.existing_publications(); smoke_keys={identity(m,self.shards[m+'-00']['rows'][0]) for m in MODELS}
        ack=self.control/'first_dispatch.json'
        if ack.exists():
            value=read(ack)
            self.source_of(value)
            if value['identity_key'] not in self.c['expected_identities'] or value['first_step']<1:
                raise ValueError('已有首正式动作回执冲突')
        if not smoke_keys<=set(pubs): raise ValueError('已有五首局发布不齐')
        for key in smoke_keys:
            if not (self.gl/'mover'/'receipts'/publication_name(key)).is_file(): raise ValueError('已有五首局搬运回执不齐')
            model=pubs[key]['model']; inv=self.control/'invocations'/(model+'-00-smoke'); d=read(inv/'dispatch.json')
            source=self.source_of(d)
            if d['exec_commit']!=pubs[key]['worker_exec_commit'] or d['smoke'] is not True or d['model']!=model or d['shard_id']!=model+'-00':
                raise ValueError('已有首局dispatch来源不符')
            seat=next((s for s in self.c['seats'] if str(s['job_id'])==str(d['gpu_job_id'])),None)
            if seat is None: raise ValueError('已有首局资源不符')
            original=Controller(source)
            if d['command']!=original.worker_command(seat,model+'-00',smoke=True): raise ValueError('已有首局原命令冲突')
            report=self.gl/'seats'/self.c['models'][model]['label']/'seed7'/f"{seat['seat']}-{model}-00-smoke"/'seat-results.jsonl'
            content=report.read_text()
            if not content.endswith('\n') or len(content.splitlines())!=1: raise ValueError('已有首局报告不完整唯一')
            row=json.loads(content); ledger,_=ledger_snapshot(Path(self.c['budget_ledger']))
            self.validate_settled_row(dict(model=model,rows=self.shards[model+'-00']['rows'][:1],seat=seat,shard=model+'-00',smoke=True),row,ledger,pubs[key])
        ledger,ledger_sha=ledger_snapshot(Path(self.c['budget_ledger']))
        if ledger.retries or ledger.recovery_used or ledger.astra: raise ValueError('已有预算含重试或非本计划路线')
        gathered=[]; terminal=[]; drained=[]; newpub=[]; failure_docs={}; used_jobs=set(); dispatched=set(); reported=set(smoke_keys)
        history={s:set() for s in self.shards}; segments={}; pause_snapshot=read_pause_snapshot(self.c) if self.c.get('pause_snapshot') is not None else None
        if paused and pause_snapshot is None: raise ValueError('暂停恢复缺已锁快照')
        if paused and ((self.control/'STOP').exists() or (self.gl/'STOP').exists()) and not preflight: raise ValueError('STOP仍有效，只允许只读暂停预检，移存由主门禁负责')
        paths=sorted((self.control/'invocations').glob('*-full/dispatch.json'))
        if not paths: raise ValueError('缺少已有正式派发')
        allowed_dirs={p.parent.name for p in paths}|{m+'-00-smoke' for m in MODELS}
        if any(p.name not in allowed_dirs or p.is_symlink() for p in (self.control/'invocations').iterdir()): raise ValueError('未知或符号链invocation目录')
        def segment(path):
            m=re.fullmatch(r'(qwenvl|smvla|pp|oracle|framesamp)-(\d{2})(?:-resume-([1-9]\d*))?-full',path.parent.name)
            if m is None: raise ValueError('续段名称不合法')
            return (m[1]+'-'+m[2],int(m[3] or 0))
        paths.sort(key=segment)
        known_steps=step_inventory(self.c)
        for path in paths:
            d=read(path); shard=d['shard_id']; model=d['model']; job=str(d['gpu_job_id'])
            source=self.source_of(d); logical,number=segment(path); tag=path.parent.name
            if d['smoke'] is not False or shard not in self.shards or self.shards[shard]['model']!=model or logical!=shard or number!=segments.get(shard,-1)+1:
                raise ValueError('已有正式dispatch版本/分片不符')
            seat=next((s for s in self.c['seats'] if str(s['job_id'])==job),None)
            original=Controller(source)
            if seat is None or d['command']!=original.worker_command(seat,shard,inv_tag=tag): raise ValueError('已有正式命令或资源冲突')
            ident_path=path.parent/'identities.jsonl'; content=ident_path.read_text()
            planned=self.shards[shard]['rows']; prior_completed=history[shard]|(smoke_keys if shard.endswith('-00') else set())
            expected_rows=planned if number==0 else [r for r in planned if identity(model,r) not in prior_completed]
            if d.get('identity_selection')=='unreserved-v1':
                route=self.c['models'][model]['policy']+('/'+ROUTES[model][2] if ROUTES[model][2] else '')+'/seed7/new'
                earlier={r['key'] for r in ledger.reserves.values() if r['route']==route and float(r['t'])<float(d['time'])}
                expected_rows=[r for r in expected_rows if r['key'] not in earlier]
            elif d.get('identity_selection') is not None: raise ValueError('未知正式身份选择协议')
            actual_rows=[json.loads(x) for x in content.splitlines()]
            if not content.endswith('\n') or not actual_rows or actual_rows!=expected_rows: raise ValueError('已有正式身份文件冲突或续段含已尝试身份')
            progress=read(path.parent/'progress.json'); name='raw-ood-worker-'+tag
            if str(progress['slurm_job_id'])!=job or progress['slurm_step_id'] is None or progress['label']!=self.c['models'][model]['label'] or progress['seat']!=seat['seat']+'-'+tag or progress['policy_seed']!=7:
                raise ValueError('已有正式progress身份冲突')
            observed=next((r for r in pause_snapshot['workers'] if r['invocation']==str(path.parent)),None) if pause_snapshot is not None else None
            step=job+'.'+str(progress['slurm_step_id']); rc=worker_account(step,name,allow_drained=observed is not None,allow_watchdog=True)
            pid=int(d['srun_pid']); proc=process_identity(pid)
            if rc is None:
                if paused: raise ValueError('暂停恢复不能接管仍在途worker')
                if known_steps.pop(step,None)!=name: raise ValueError('运行worker与全步骤清单冲突')
                if proc is None or proc['state']=='Z': raise ValueError('已有运行srun PID不存在')
                args=proc['command']; expected=d['command']
                if not args or Path(args[0]).name!='srun' or args[1:]!=expected[1:] or not re.search(r'job[_/-]'+re.escape(str(self.c['controller_job_id']))+r'(?:/|$)',proc['cgroup']):
                    raise ValueError('已有srun命令行或CPU作业cgroup不符')
                if d.get('srun_start_ticks') is not None:
                    if proc['start_ticks']!=d['srun_start_ticks']: raise ValueError('已有srun创建tick冲突')
                elif abs(proc['start_time']-float(d['time']))>5: raise ValueError('旧dispatch srun创建时间不符')
                if time.time()-float(progress['t'])>120: raise ValueError('已有运行席位心跳失效')
                if job in used_jobs: raise ValueError('同席有多个在途正式worker')
                used_jobs.add(job)
            else:
                listed=known_steps.pop(step,None)
                if listed is not None and listed!=name: raise ValueError('已有终态与全步骤清单冲突')
                legacy_terminal=False
                if rc==75 and progress['phase']=='episode':
                    report=self.gl/'seats'/self.c['models'][model]['label']/'seed7'/progress['seat']/'seat-results.jsonl'
                    content=report.read_text()
                    candidates=[json.loads(x) for x in content.splitlines() if x.strip()] if content.endswith('\n') else []
                    legacy_terminal=any(r.get('key')==progress['key'] and legacy_watchdog_progress(d,r,self.c) for r in candidates)
                if not legacy_terminal and (progress['phase']!='finished' or progress.get('rc')!=rc): raise ValueError('已有终态缺finished/对应退出码证据')
                if rc==6:
                    observed=next((r for r in pause_snapshot['workers'] if r['invocation']==str(path.parent)),None)
                    stop_time=float(observed['stop_time']) if observed is not None and 'stop_time' in observed else float(pause_snapshot['stop_time'])
                    if observed is None or observed['dispatch_sha256']!=sha(path) or observed['progress_sha256']!=sha(path.parent/'progress.json') or observed['step_id']!=step or observed['job_name']!=name or observed['exit_code']!='6:0' or observed['state'] not in ('FAILED','COMPLETED') or float(progress['t'])<stop_time:
                        raise ValueError('暂停终态不符固定STOP及worker快照')
            phase_start=float(d['time'])
            if progress['key'] is not None:
                ledger,_=ledger_snapshot(Path(self.c['budget_ledger']))
                route=self.c['models'][model]['policy']+('/'+ROUTES[model][2] if ROUTES[model][2] else '')+'/seed7/new'
                matching=[r for r in ledger.reserves.values() if r['key']==progress['key'] and r['route']==route]
                if matching: phase_start=max(float(r['t']) for r in matching)
            a=dict(process=ExistingProcess(pid,proc,step,name),seat=seat,shard=shard,smoke=False,model=model,
                   progress=path.parent/'progress.json',results=self.gl/'seats'/self.c['models'][model]['label']/'seed7'/progress['seat']/'seat-results.jsonl',
                   rows=actual_rows,offset=0,started=float(d['time']),acked=(self.control/'first_dispatch.json').exists(),inv_tag=tag,
                   phase_key=(progress['phase'],progress['key']),phase_start=phase_start,step_id=step,worker_name=name,read_seen=set())
            self.check_worker(a,exited=rc in (0,6,75))
            if progress['key'] is not None and progress['key'] not in {r['key'] for r in a['rows']}: raise ValueError('已有progress key不在分片')
            try: content=a['results'].read_text()
            except FileNotFoundError: content=''
            partial=bool(content and not content.endswith('\n')); lines=content.splitlines()
            if partial: lines=lines[:-1]
            if rc in (0,6,75) and partial: raise ValueError('已有终态报告半行')
            if rc==6:
                digest=sha(a['results']) if a['results'].exists() else None
                if observed['report_sha256']!=digest: raise ValueError('暂停报告原字节不符快照')
            for line in lines:
                row=json.loads(line); key=identity(model,row)
                if key in a['read_seen']: raise ValueError('已有报告重复身份')
                if row.get('run_blocked') is True or row.get('budget_exhausted') is True: raise ValueError('已有席位阻塞终态 '+str(row.get('error') or row.get('status')))
                if row.get('infra') is True:
                    if rc!=75: raise ValueError('基础设施行无真实75终态')
                    doc=self.prepare_failure(a,row,line)
                    if doc['reason'] not in self.c.get('failure_policy',{}).get('continue_reasons',[]): raise ValueError('失败原因未获继续授权')
                    if key in pubs or key in failure_docs: raise ValueError('失败重复或已有正常发布')
                    failure_docs[key]=doc; a['read_seen'].add(key); a['offset']+=1; reported.add(key); continue
                ledger,_=ledger_snapshot(Path(self.c['budget_ledger']))
                self.validate_settled_row(a,row,ledger,pubs.get(key)); a['read_seen'].add(key); a['offset']+=1
                reported.add(key)
                if key not in pubs:
                    ident=next(r for r in a['rows'] if r['key']==row['key']); result=Path(row['result']); data=read(result)
                    meta=read(result.parent/'meta.json'); trace=[json.loads(x) for x in (result.parent/'trace.jsonl').read_text().splitlines()]
                    for k in ('dataset','task','tier','seed','key'):
                        if data[k]!=ident[k] or meta['identity'][k]!=ident[k] or trace[0]['identity'][k]!=ident[k]: raise ValueError('已有未发布raw身份冲突')
                    if trace[0]['kind']!='header' or trace[-1]['kind']!='end' or trace[-1]['exec_steps']!=row['exec_steps'] or data['exec_steps']!=row['exec_steps'] or data['recorder_verify']!='PASS':
                        raise ValueError('已有未发布raw未闭合')
                    doc=self.prepare_publication(a,row,worker_exec_commit=d['exec_commit']); doc.update({k:d[k] for k in CORE})
                    newpub.append((a,row,doc))
                self.advance('restoring_report')
            expected={identity(model,r) for r in a['rows']}
            if rc==0:
                if not expected<=set(pubs)|a['read_seen']: raise ValueError('已有终态缺完成身份')
                terminal.append(a)
            elif rc in (6,75):
                if rc==75 and not (a['read_seen']&set(failure_docs)): raise ValueError('真实75缺闭合失败报告')
                a['history_until']=float(progress['t'])
                if rc==75:
                    a['history_until']=max(a['history_until'],*(max(float(failure_docs[key]['budget']['commit']['t']),float(json.loads(failure_docs[key]['report_line']).get('t',0))) for key in a['read_seen']&set(failure_docs)))
                self.validate_drained_prefix(a,a['read_seen']|prior_completed)
                drained.append(a)
            else: gathered.append(a)
            if history[shard]&a['read_seen']: raise ValueError('不同续段重复预算身份')
            history[shard].update(a['read_seen']); segments[shard]=number
            dispatched.add(shard)
        if any(name.startswith('raw-ood-worker-') for name in known_steps.values()): raise ValueError('存在未登记或未接管的worker步骤')
        if not set(pubs)<=reported: raise ValueError('已有发布缺少对应已结算席位报告')
        fresh=[]
        for a,row,doc in newpub:
            path=self.gl/'published'/publication_name(doc['identity_key'])
            if path.exists():
                if read(path)!=doc: raise ValueError('扫描期间已有发布出现冲突')
                pubs[doc['identity_key']]=doc
            else: fresh.append((a,row,doc))
        newpub=fresh
        planned_done=set(pubs)|reported
        logical_completed={s for s in dispatched if {identity(self.shards[s]['model'],r) for r in self.shards[s]['rows']}<=planned_done}
        attempted=set()
        for reserve in ledger.reserves.values():
            matches=[m for m in MODELS if reserve['route']==self.c['models'][m]['policy']+('/'+ROUTES[m][2] if ROUTES[m][2] else '')+'/seed7/new']
            if len(matches)!=1: raise ValueError('预算预约未知路线')
            attempted.add(identity(matches[0],reserve))
        active_shards={a['shard'] for a in gathered}
        active_keys={identity(a['model'],r) for a in gathered for r in a['rows']}
        unresolved=attempted-planned_done-active_keys
        remaining={s:[r for r in self.shards[s]['rows'] if identity(self.shards[s]['model'],r) not in attempted] for s in dispatched-logical_completed-active_shards}
        if paused:
            if gathered: raise ValueError('暂停迁移存在仍运行worker')
            all_attempted={identity(next(m for m in MODELS if reserve['route']==self.c['models'][m]['policy']+('/'+ROUTES[m][2] if ROUTES[m][2] else '')+'/seed7/new'),reserve)
                           for reserve in ledger.reserves.values()}
            if all_attempted!=planned_done: raise ValueError('全轮已预约身份与暂停闭合结果不齐')
        proof=dict(self.base,orchestration_commit=orchestration_commit,active=len(gathered),completed=len(terminal),published=len(pubs),
                   budget_sha256=ledger_sha,dispatched=sorted(dispatched),steps=[a['step_id'] for a in gathered+terminal+drained],drained=len(drained),
                   logical_completed=len(logical_completed),accepted=len(planned_done),remaining=4000-len(planned_done),time=time.time())
        proof.update(infra=len(failure_docs),unresolved=len(unresolved),unresolved_identities=sorted(unresolved))
        if preflight:
            print(('PAUSED_RESUME_PREFLIGHT=PASS ' if paused else 'EXISTING_DISPATCHES_PREFLIGHT=PASS ')+json.dumps(proof,ensure_ascii=False,sort_keys=True),flush=True); return proof
        self.published=set(pubs); self.smoked=set(MODELS); self.formal=True; self.completed=logical_completed
        self.pending=[s for s in self.pending if s not in dispatched or s in remaining and remaining[s]]
        self.resume_rows=remaining; self.segment_numbers=dict(segments)
        self.segment_numbers.update({s:segments[s]+1 for s in remaining}); self.unresolved=unresolved
        for doc in failure_docs.values(): self.record_failure(doc)
        for a in gathered:
            a['log']=(a['progress'].parent/'worker.log').open('a'); self.active[str(a['seat']['job_id'])]=a
        self.initial_dispatched={str(a['seat']['job_id']) for a in gathered+terminal+drained}
        for a,row,doc in newpub: self.publish(a,row,prepared=doc)
        atomic(self.control/('paused-migration.json' if paused else 'resume-existing-dispatches.json'),proof)

    def write_paused(self):
        """生成可再次钉SHA的暂停证明，原invocation及STOP字节不覆盖。"""
        workers=[]; stop_refs=[]; stop_time=self.pause_started
        if self.c.get('pause_snapshot') is not None:
            old=read_pause_snapshot(self.c)
            stop_refs.extend(old['stop_files'])
            for row in old['workers']:
                value=dict(row); value.setdefault('stop_time',old['stop_time']); workers.append(value)
        for path in (self.control/'STOP',self.gl/'STOP'):
            if path.exists():
                saved=self.control/'pause-stops'/f'{os.getpid()}-{time.time_ns()}-{path.parent.name}.stop'
                saved.parent.mkdir(parents=True,exist_ok=True); saved.write_bytes(path.read_bytes())
                stop_refs.append(dict(path=str(saved),sha256=sha(saved)))
        existing={r['invocation'] for r in workers}
        for path in sorted((self.control/'invocations').glob('*-full/dispatch.json')):
            if str(path.parent) in existing: continue
            d=read(path); p=read(path.parent/'progress.json'); step=str(d['gpu_job_id'])+'.'+str(p['slurm_step_id'])
            name='raw-ood-worker-'+path.parent.name; account=worker_account(step,name,allow_drained=True,allow_watchdog=True,evidence=True)
            rc=int(account['exit_code'].split(':')[0]) if isinstance(account,dict) else None
            legacy_terminal=False
            if rc==75 and p['phase']=='episode':
                for ref in self.failures.values():
                    failure=read(ref['path'])
                    if failure['dispatch']['path']==str(path):
                        validate_failure_document(failure,self.c); legacy_terminal=legacy_watchdog_progress(d,json.loads(failure['report_line']),self.c)
            if rc not in (0,6,75) or not legacy_terminal and (p['phase']!='finished' or p.get('rc')!=rc): raise ValueError('暂停时worker终态未知')
            model=d['model']; seat=next(s for s in self.c['seats'] if str(s['job_id'])==str(d['gpu_job_id']))
            report=self.gl/'seats'/self.c['models'][model]['label']/'seed7'/p['seat']/'seat-results.jsonl'
            workers.append(dict(invocation=str(path.parent),dispatch_sha256=sha(path),progress_sha256=sha(path.parent/'progress.json'),
                report_sha256=sha(report) if report.exists() else None,step_id=step,job_name=name,state=account['state'],
                exit_code=f'{rc}:0',stop_time=stop_time,drained=rc==6))
        ledger,digest=ledger_snapshot(Path(self.c['budget_ledger']))
        if ledger.trajectories!=len(self.settled()) or len(ledger.commits)!=len(ledger.reserves) or ledger.releases:
            raise ValueError('暂停全轮预约与闭合终态不齐')
        claims=sum(ledger.claimed.values())+ledger.orphan_claims
        doc=dict(self.base,schema='raw-ood-pause-v1',root=str(self.root),stop_time=stop_time,stop_files=stop_refs,workers=workers,
                 pause_contract='settled-v1',published_identities=sorted(self.published),failure_documents=dict(self.failures),
                 expected=4000,accepted=len(self.published),infra=len(self.failures),settled=len(self.settled()),remaining=4000-len(self.settled()),drained=True,
                 trajectories=ledger.trajectories,reset_claims=claims,unsettled=0,budget_sha256=digest,time=time.time())
        doc['budget_bytes']=Path(self.c['budget_ledger']).stat().st_size
        validate_pause_proof(doc,self.c)
        path=self.control/f'paused-{os.getpid()}-{time.time_ns()}.json'; atomic(path,doc)
        atomic(self.control/'paused.json',dict(self.base,schema='raw-ood-pause-pointer-v1',proof_path=str(path),proof_sha256=sha(path),
              expected=4000,accepted=len(self.published),infra=len(self.failures),settled=len(self.settled()),remaining=4000-len(self.settled()),drained=True,time=time.time()))
        return doc

    def poll_ack(self, a: dict):
        if a['smoke'] or a['acked'] or not a['progress'].exists(): return
        p=read(a['progress'])
        if p['phase'] not in ('episode','done') or p['key'] is None: return
        if p['key'] not in {r['key'] for r in a['rows']}: raise ValueError('专属progress身份错误')
        # 生产逐步文件可能被另一同模型席覆盖，必须同时核key和pid；终态可作为补充证据。
        m=self.c['models'][a['model']]; ep=self.gl/'rollouts'/m['label']/'ood'/'seed7'/'progress.json'
        if not ep.exists(): return
        d=read(ep)
        if d['key']!=p['key'] or d['pid']!=p['pid'] or d['step']<1: return
        if not a.get('step_id'): return
        self.write_ack(a,f"{a['model']}:ood:{p['key']}",p['pid'],d['step'],str(ep))

    def check_worker(self,a,*,exited=False):
        now=time.time()
        if not exited and not a.get('step_id'):
            steps=subprocess.run(['squeue','--steps','-h','-j',str(a['seat']['job_id']),'-o','%i %j'],capture_output=True,text=True,timeout=5)
            candidates=[s.split()[0] for s in steps.stdout.splitlines() if len(s.split())==2 and s.split()[1]==a['worker_name'] and s.split()[0].startswith(str(a['seat']['job_id'])+'.')]
            if steps.returncode: raise RuntimeError('srun step查询失败')
            if len(candidates)==1: a['step_id']=candidates[0]
        if a['progress'].exists():
            p=read(a['progress']); phase=(p['phase'],p['key'])
            if str(p['slurm_job_id'])!=str(a['seat']['job_id']): raise ValueError('客户端SLURM作业不符')
            if p['slurm_step_id'] is None: raise ValueError('客户端缺少原始SLURM_STEP_ID')
            exact=str(p['slurm_job_id'])+'.'+str(p['slurm_step_id'])
            if a['step_id'] is not None and a['step_id']!=exact: raise ValueError('客户端SLURM StepID与调度回执冲突')
            a['step_id']=exact
            if p['label']!=self.c['models'][a['model']]['label']: raise ValueError('专属进度标签错误')
            if not exited and now-float(p['t'])>120: raise RuntimeError('专属席位心跳失效')
            if phase!=a['phase_key']: a['phase_key']=phase; a['phase_start']=now
            limits=self.c.get('phase_limits',{'context_load':3600,'claim':120,'episode':3600,'done':300,'finished':120})
            if not exited and now-a['phase_start']>float(limits[p['phase']]): raise RuntimeError(f"席位阶段超时{phase}")
            return True
        elif not exited and now-a['started']>120: raise RuntimeError('席位未建立进度')
        return False

    def read_worker_results(self,a):
        """直接打开报告；新建文件的可见性与工作进程退出分开确认。"""
        try: content=a['results'].read_text()
        except FileNotFoundError: return dict(opened=False,bytes_read=0,complete_lines=0,partial_tail=False)
        lines=content.splitlines(); partial=bool(content and not content.endswith('\n'))
        if partial: lines=lines[:-1]
        for line in lines[a['offset']:]:
            row=json.loads(line); key=identity(a['model'],row)
            seen=a.setdefault('read_seen',set())
            if key in seen: raise ValueError('席位报告重复身份')
            if row.get('run_blocked') is True or row.get('budget_exhausted') is True:
                evidence=dict(self.base,report_path=str(a['results']),report_line=line,report_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                              reason=row.get('error') or row.get('status'),task_success=row.get('task_success'),exec_steps=row.get('exec_steps'),time=time.time())
                atomic(a['progress'].parent/'blocked-terminal.json',evidence)
                raise RuntimeError('席位阻塞终态已留证 '+str(evidence['reason']))
            if row.get('infra') is True:
                if failure_category(row)!='watchdog_exit':
                    atomic(a['progress'].parent/'unclassified-infra.json',dict(self.base,report_path=str(a['results']),report_line=line,report_line_sha256=hashlib.sha256(line.encode()).hexdigest(),reason=row.get('infra_reason'),task_success=row.get('task_success'),exec_steps=row.get('exec_steps'),time=time.time()))
                    raise RuntimeError('未获继续许可的基础设施原因已留原行证据')
                if a['process'].poll() is None:
                    began=a.setdefault('failure_exit_wait_started',time.monotonic())
                    if time.monotonic()-began>=120: raise RuntimeError('看门狗原行已可见但真实worker退出未在120秒内闭合')
                    return dict(opened=True,bytes_read=len(content.encode()),complete_lines=len(lines),partial_tail=partial,failure_exit_waiting=True)
                if not a['progress'].exists():
                    return dict(opened=True,bytes_read=len(content.encode()),complete_lines=len(lines),partial_tail=partial,failure_exit_waiting=True)
                progress=read(a['progress'])
                d=read(a['progress'].parent/'dispatch.json')
                legacy_terminal=progress['phase']=='episode' and progress['key']==row['key'] and legacy_watchdog_progress(d,row,self.c)
                if not legacy_terminal and progress['phase']!='finished':
                    return dict(opened=True,bytes_read=len(content.encode()),complete_lines=len(lines),partial_tail=partial,failure_exit_waiting=True)
                if not legacy_terminal and progress.get('rc')!=75: raise ValueError('看门狗报告与finished退出码矛盾')
                step=str(a['seat']['job_id'])+'.'+str(progress['slurm_step_id'])
                if worker_account(step,a['worker_name'],allow_watchdog=True) is None:
                    return dict(opened=True,bytes_read=len(content.encode()),complete_lines=len(lines),partial_tail=partial,failure_exit_waiting=True)
                doc=self.prepare_failure(a,row,line); self.record_failure(doc)
                seen.add(key); a['offset']+=1
                if doc['reason'] not in self.c.get('failure_policy',{}).get('continue_reasons',[]): raise RuntimeError('基础设施失败已留证，继续未授权')
                continue
            ledger,_=ledger_snapshot(Path(self.c['budget_ledger']))
            self.validate_settled_row(a,row,ledger)
            if key in self.published:
                ledger,_=ledger_snapshot(Path(self.c['budget_ledger']))
                self.validate_settled_row(a,row,ledger,read(self.gl/'published'/publication_name(key)))
            else: self.publish(a,row)
            seen.add(key); a['offset']+=1
            self.publication_writes.pop(key,None)
        return dict(opened=True,bytes_read=len(content.encode()),complete_lines=len(lines),partial_tail=partial)

    def final_confirmation(self,a,rc,read_info,identity_valid):
        now=time.monotonic()
        if 'exit_observed_monotonic' not in a:
            a['exit_observed_monotonic']=now; a['exit_observed_wall']=time.time()
        expected={identity(a['model'],r) for r in a['rows']}
        progress=read(a['progress']) if a['progress'].exists() else None
        if progress is not None and progress['phase']=='finished' and progress['rc']!=rc: raise ValueError('终态finished退出码与进程矛盾')
        progress_ready=progress is not None and progress['phase']=='finished' and progress['rc']==rc or rc==75 and a.get('legacy_terminal_progress') is True
        required=expected
        if rc in (6,75):
            ledger,_=ledger_snapshot(Path(self.c['budget_ledger']))
            model=a['model']; route=self.c['models'][model]['policy']+('/'+ROUTES[model][2] if ROUTES[model][2] else '')+'/seed7/new'
            required={identity(model,r) for rid,r in ledger.reserves.items() if r.get('route')==route and rid in ledger.commits}&expected
        missing=sorted(required-self.settled())
        elapsed=now-a['exit_observed_monotonic']
        try: size=a['results'].stat().st_size
        except FileNotFoundError: size=None
        ok=not missing and identity_valid and progress_ready and not read_info['partial_tail'] and not read_info.get('failure_exit_waiting',False)
        if rc==75: ok=ok and bool(expected&set(self.failures))
        state='confirmed' if ok and elapsed<=120 else ('failed' if elapsed>=120 else 'waiting')
        evidence=dict(self.base,model=a['model'],shard_id=a['shard'],gpu_job_id=str(a['seat']['job_id']),rc=rc,state=state,
                      expected_path=str(a['results']),path_exists=a['results'].exists(),stat_size=size,read_info=read_info,
                      expected=len(expected),published=len(expected & self.published),
                      missing_identities=missing,identity_valid=identity_valid,consumed_lines=a['offset'],
                      exit_observed_wall=a['exit_observed_wall'],elapsed_s=elapsed,deadline_s=120,time=time.time())
        atomic(a['progress'].parent/'final-confirmation.json',evidence)
        if state=='failed':
            raise ValueError('WORKER_REPORT_CONFIRMATION=FAIL '+json.dumps(evidence,ensure_ascii=False,sort_keys=True))
        return state=='confirmed'

    def scan(self):
        for jid,a in list(self.active.items()):
            rc=a['process'].poll()
            if rc in (0,6,75) and 'exit_observed_monotonic' not in a:
                a['exit_observed_monotonic']=time.monotonic(); a['exit_observed_wall']=time.time()
            exited=rc in (0,75) or rc==6 and self.pausing or getattr(a['process'],'disappeared_monotonic',None) is not None
            try: identity_valid=self.check_worker(a,exited=exited)
            except RuntimeError:
                if rc is None and a['process'].poll()==0:
                    rc=0; identity_valid=self.check_worker(a,exited=True)
                else: raise
            read_info=self.read_worker_results(a)
            self.poll_ack(a)
            if rc is None: rc=a['process'].poll()
            if rc is not None:
                if rc!=0:
                    if rc==75 and not a['smoke']:
                        read_info=self.read_worker_results(a); identity_valid=self.check_worker(a,exited=True)
                        if not self.final_confirmation(a,rc,read_info,identity_valid): continue
                        self.validate_drained_prefix(a,self.settled())
                        a['log'].close(); del self.active[jid]
                        rows=self.unreserved_rows(a['shard'])
                        if rows:
                            self.resume_rows[a['shard']]=rows
                            tag=a.get('inv_tag',a['progress'].parent.name)
                            prefix=a['shard']+'-resume-'
                            current=int(tag[len(prefix):-len('-full')]) if tag.startswith(prefix) and tag.endswith('-full') else 0
                            self.segment_numbers[a['shard']]=current+1
                            if a['shard'] not in self.pending: self.pending.insert(0,a['shard'])
                        elif {identity(a['model'],r) for r in self.shards[a['shard']]['rows']}<=self.settled(): self.completed.add(a['shard'])
                        self.event('infra_settled',shard_id=a['shard'],infra=len(self.failures)); continue
                    if rc==6 and self.pausing:
                        read_info=self.read_worker_results(a); self.check_worker(a,exited=True)
                        identity_valid=self.check_worker(a,exited=True)
                        if not self.final_confirmation(a,rc,read_info,identity_valid): continue
                        self.validate_drained_prefix(a,self.settled())
                        a['log'].close(); del self.active[jid]; self.paused.add(a['shard'])
                        self.event('drained',shard_id=a['shard'],accepted=a['offset']); continue
                    a['log'].close(); del self.active[jid]
                    raise RuntimeError(f"worker失败 model={a['model']} shard={a['shard']} rc={rc}")
                # 已退出后再读一次；仍不可见则保留active等待，绝不重跑该worker。
                read_info=self.read_worker_results(a)
                identity_valid=self.check_worker(a,exited=True)
                if not self.final_confirmation(a,rc,read_info,identity_valid): continue
                a['log'].close(); del self.active[jid]
                if a['smoke']: self.smoked.add(a['model'])
                else:
                    if {identity(a['model'],r) for r in self.shards[a['shard']]['rows']}<=self.settled(): self.completed.add(a['shard'])
                    else: raise ValueError('续段退出但logical shard未完整')
            self.advance('worker_scanned')

    def dispatch(self):
        for seat in self.c['seats']:
            if str(seat['job_id']) in self.active: continue
            state=observed_job_state(self.c,seat['job_id'],self.query_failures,'controller')
            if state is None or state in ('PENDING','CONFIGURING'): continue
            if state!='RUNNING': raise RuntimeError(f'GPU异常状态{state}')
            guardian=self.control/f"guardian-heartbeat-{seat['job_id']}.json"
            if not guardian.exists(): continue
            ready=read(guardian)
            if any(str(ready[k])!=str(v) for k,v in self.base.items()) or str(ready['gpu_job_id'])!=str(seat['job_id']):
                raise ValueError('附加独立守卫就绪身份不符')
            if type(ready['pid']) is not int or ready['pid']<=0 or ready['step_id'] is None or time.time()-float(ready['t'])>=30 or float(ready['t'])>time.time()+5 or time.time()-float(ready['monitor_t'])>=30:
                raise ValueError('附加独立守卫缺少有效PID/时间/StepID')
            steps=step_inventory(dict(seats=[seat]))
            if steps.get(str(seat['job_id'])+'.'+str(ready['step_id']))!='raw-ood-lifecycle-guard': raise ValueError('附加独立守卫步骤已不存活')
            if not self.formal:
                busy={a['model'] for a in self.active.values()}
                todo=[m for m in MODELS if m not in self.smoked and m not in busy]
                if todo: self.start(seat,todo[0]+'-00',smoke=True)
            elif self.pending:
                remaining=Counter(self.shards[s]['model'] for s in self.pending)
                index=self.c['seats'].index(seat)
                if str(seat['job_id']) not in self.initial_dispatched:
                    model=('qwenvl','qwenvl','smvla','pp')[index]; self.initial_dispatched.add(str(seat['job_id']))
                else: model=max(remaining,key=lambda m:(remaining[m]*100*SECONDS[m],-MODELS.index(m)))
                if model not in remaining: model=max(remaining,key=lambda m:(remaining[m]*100*SECONDS[m],-MODELS.index(m)))
                shard=next(s for s in self.pending if self.shards[s]['model']==model)
                self.pending.remove(shard)
                rows=self.unreserved_rows(shard)
                if not rows:
                    if {identity(model,r) for r in self.shards[shard]['rows']}<=self.settled(): self.completed.add(shard)
                    continue
                self.resume_rows[shard]=rows; self.start(seat,shard)

    def report(self):
        """读取保留的小result与共享账本，自动产生实测报告；任务失败单列。"""
        ledger,current_sha=ledger_snapshot(Path(self.c['budget_ledger']))
        if ledger.bad_rows or ledger.trajectories>4000 or ledger.resets>8000 or ledger.retries or ledger.recovery_used or ledger.astra:
            raise ValueError('最终共享预算不符合4000首试/8000reset/零重试')
        groups={}; model_stats={m:dict(accepted=0,success=0,failed=0,infra=0,episode_wall_s=0.,recorder_finalize_s=0.,chunks=0,decision_wall_ms=0.,
                                      episode_wall_measured=0,recorder_finalize_measured=0,policy_timing_missing=0,decision_wall_missing=0) for m in MODELS}
        for key in self.published:
            pub=read(self.gl/'published'/publication_name(key)); r=read(self.gl/pub['relative_dir']/'result.json')
            model=pub['model']; stat=model_stats[model]; task=r['task']; tier=r['tier']
            success=int(bool(r['task_success'])); stat['accepted']+=1; stat['success']+=success; stat['failed']+=1-success
            g=groups.setdefault(f'{model}:{task}:{tier}',dict(model=model,task=task,tier=tier,accepted=0,success=0,failed=0,infra=0))
            g['accepted']+=1; g['success']+=success; g['failed']+=1-success
            timing=r.get('timing'); timing=timing if isinstance(timing,dict) else {}
            if timing.get('episode_wall_s') is not None:
                stat['episode_wall_s']+=float(timing['episode_wall_s']); stat['episode_wall_measured']+=1
            recorder=timing.get('recorder'); recorder=recorder if isinstance(recorder,dict) else {}
            if recorder.get('finalize_s') is not None:
                stat['recorder_finalize_s']+=float(recorder['finalize_s']); stat['recorder_finalize_measured']+=1
            policy=timing.get('policy'); policy=policy if isinstance(policy,dict) else {}
            chunks=policy.get('chunks')
            if not isinstance(chunks,list): stat['policy_timing_missing']+=1; chunks=[]
            for index,chunk in enumerate(chunks):
                if index<3: continue
                if isinstance(chunk,dict) and chunk.get('decision_wall_ms') is not None:
                    stat['chunks']+=1; stat['decision_wall_ms']+=float(chunk['decision_wall_ms'])
                else: stat['decision_wall_missing']+=1
        for model,stat in model_stats.items():
            tasks={}
            for g in groups.values():
                if g['model']==model:
                    t=tasks.setdefault(g['task'],[0,0]); t[0]+=g['success']; t[1]+=g['accepted']
            stat['infra']=sum(key.startswith(model+':') for key in self.failures)
            stat['episode_success_rate']=stat['success']/stat['accepted'] if stat['accepted'] else None
            stat['task_macro_success_rate']=sum(s/n for s,n in tasks.values())/len(tasks) if tasks else None
            stat['steady_decision_wall_ms_mean']=stat['decision_wall_ms']/stat['chunks'] if stat['chunks'] else None
        budget=dict(trajectories=ledger.trajectories,reset_claims=sum(ledger.claimed.values())+ledger.orphan_claims,
                    reset_count=ledger.resets,retries=len(ledger.retries),recovery=ledger.recovery_used,astra=ledger.astra,
                    trajectory_cap=4000,reset_cap=8000,bad_rows=ledger.bad_rows)
        combined=work_budget(self.c,current=ledger)
        combined['current']['sha256']=current_sha
        expected=set(self.c['expected_identities']); settled=self.settled(); unresolved=self.unresolved-settled
        unattempted=expected-settled-unresolved
        terminal=dict(self.base,schema='raw-ood-terminal-v1',expected=len(expected),settled=len(settled),published=len(self.published),infra=len(self.failures),
                      unresolved=len(unresolved),unattempted=len(unattempted),published_identities=sorted(self.published),failure_documents=self.failures,
                      unresolved_identities=sorted(unresolved),unattempted_identities=sorted(unattempted),
                      status='COMPLETE' if self.published==expected and not self.failures and not unresolved else 'INCOMPLETE',time=time.time())
        validate_terminal_marker(terminal,self.c)
        atomic(self.control/'results-report.json',dict(self.base,expected=len(expected),accepted=len(self.published),missing=len(expected-self.published),infra=len(self.failures),status=terminal['status'],
              settled=len(settled),unresolved=len(unresolved),unattempted=len(unattempted),failure_reasons={key:read(refpath['path']).get('infra_reason',read(refpath['path'])['reason']) for key,refpath in self.failures.items()},
              infra_observations={key:{field:read(refpath['path'])[field] for field in ('task_success','exec_steps')} for key,refpath in self.failures.items()},
              observed_success=sum(s['success'] for s in model_stats.values()),observed_count=len(self.published),budget=budget,work_budget=combined,
              models=model_stats,by_task_tier=list(groups.values()),wall_s=time.time()-self.started,time=time.time(),
              limitations=['未测GPU利用率','各片加载总耗时由worker日志提供，不能从局timing重复累加']))
        print(f"RUN_BUDGET=PASS trajectories={budget['trajectories']}/4000 reset_claims={budget['reset_claims']}/8000 retries=0",flush=True)
        print(f"OOD_RESULTS={'PASS' if terminal['status']=='COMPLETE' else 'INCOMPLETE'} models=5 expected=4000 accepted={len(self.published)} missing={len(expected-self.published)} infra={len(self.failures)}",flush=True)
        print(f"WORK_BUDGET=PASS trajectories={combined['combined']['trajectories']}/{combined['combined']['trajectory_cap']} reset_claims={combined['combined']['reset_claims']}/{combined['combined']['reset_cap']} prior_trajectories={combined['prior']['trajectories']} prior_reset_claims={combined['prior']['reset_claims']}",flush=True)
        atomic(self.gl/'gpu_work_complete.json',terminal); atomic(self.control/'gpu_work_complete.json',terminal)
        return 0 if terminal['status']=='COMPLETE' else 2

    def run(self,*,resume_completed_smokes=False,resume_existing_dispatches=False,resume_paused=False,orchestration_commit=None,pause_on_stop=True):
        self.control.mkdir(parents=True,exist_ok=True)
        with ControllerLease(self):
            try:
                return self._run_locked(resume_completed_smokes=resume_completed_smokes,resume_existing_dispatches=resume_existing_dispatches,
                                        resume_paused=resume_paused,orchestration_commit=orchestration_commit,pause_on_stop=pause_on_stop)
            except BaseException as e:
                path=self.control/f'controller-error.current-{os.getpid()}-{time.time_ns()}.json'
                atomic(path,dict(self.base,component='controller',pid=os.getpid(),error=type(e).__name__,message=str(e),time=time.time()))
                raise

    def _run_locked(self,*,resume_completed_smokes=False,resume_existing_dispatches=False,resume_paused=False,orchestration_commit=None,pause_on_stop=True):
        self.control.mkdir(parents=True,exist_ok=True)
        manifest=dict(self.base,shards=self.shards)
        stop_beat=threading.Event()
        heartbeat_error=[]
        def heartbeat():
            while not stop_beat.is_set():
                proc=process_identity(os.getpid())
                try:
                    self.retry_filesystem(atomic,self.control/'controller-heartbeat.json',dict(self.base,t=time.time(),loop_t=self.loop_t,phase=self.loop_phase,
                          pid=os.getpid(),start_ticks=proc['start_ticks'],active=len(self.active)))
                except BaseException as e: heartbeat_error.append(e); return
                stop_beat.wait(5)
        beat=threading.Thread(target=heartbeat,name='raw-ood-controller-heartbeat',daemon=True); beat.start()
        try:
            if sum((resume_completed_smokes,resume_existing_dispatches,resume_paused))>1: raise ValueError('手动恢复模式互斥')
            if resume_paused: self.retry_filesystem(self.resume_existing_dispatches,orchestration_commit,paused=True)
            elif resume_existing_dispatches: self.retry_filesystem(self.resume_existing_dispatches,orchestration_commit)
            elif resume_completed_smokes:
                old=legacy_context(self.c); oldmanifest=dict({k:old[k] for k in self.base},shards=self.shards)
                if read(self.control/'shards.json')!=oldmanifest: raise ValueError('原分片manifest冲突')
                self.resume_completed_smokes(orchestration_commit)
            else: atomic(self.control/'shards.json',manifest)
            while True:
                if heartbeat_error: raise RuntimeError('控制器心跳写入失败') from heartbeat_error[0]
                if (self.control/'STOP').exists() or (self.gl/'STOP').exists():
                    if not pause_on_stop: raise RuntimeError('全局STOP（显式关闭暂停排空）')
                    if not self.pausing:
                        self.pausing=True; self.pause_started=time.time(); self.event('pause_requested')
                self.advance('scanning')
                self.retry_filesystem(self.scan)
                if self.pausing:
                    if not self.active:
                        self.write_paused(); print(f'CONTROLLER_PAUSED=PASS accepted={len(self.published)} remaining={4000-len(self.published)}',flush=True); return 6
                    self.advance('pause_draining'); time.sleep(1); continue
                if len(self.smoked)==5 and not self.formal:
                    # 完整本机解码及SHA回执确认才允许全量；正常任务失败不阻塞。
                    names=[publication_name(identity(m,self.shards[m+'-00']['rows'][0])) for m in MODELS]
                    if all((self.gl/'mover'/'receipts'/n).exists() for n in names):
                        for n in names:
                            from eval_video_mover import raw_validate_receipt
                            raw_validate_receipt(read(self.gl/'mover'/'receipts'/n),read(self.gl/'published'/n),self.c,dest=None)
                        self.formal=True; atomic(self.control/'smoke-completed.json',dict(self.base,models=5,infra=0,time=time.time()))
                if time.time()<self.deadline: self.dispatch()
                elif not self.active and (self.pending or not self.formal): raise RuntimeError('已安全排空，剩余身份不重跑')
                if self.formal and not self.pending and not self.active:
                    return self.retry_filesystem(self.report)
                time.sleep(1)
                self.advance('loop_completed')
        finally:
            # 异常只结束控制器自身；已启动的GPU step继续，不主动TERM或scancel。
            stop_beat.set(); beat.join(timeout=6)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['run','validate','resume-preflight','existing-dispatches-preflight','paused-resume-preflight']); ap.add_argument('--config',required=True)
    ap.add_argument('--resume-completed-smokes',action='store_true'); ap.add_argument('--resume-existing-dispatches',action='store_true'); ap.add_argument('--resume-paused',action='store_true'); ap.add_argument('--orchestration-commit',default=None)
    ap.add_argument('--pause-on-stop',action=argparse.BooleanOptionalAction,default=True)
    args=ap.parse_args(); c=load_config(Path(args.config))
    if args.command=='validate':
        rows=validate_config(c); print(f'OOD_SCOPE=PASS models=5 identities_per_model={len(rows)} planned=4000'); return 0
    controller=Controller(c)
    if args.command=='resume-preflight':
        controller.resume_completed_smokes(args.orchestration_commit,preflight=True); return 0
    if args.command=='existing-dispatches-preflight':
        controller.resume_existing_dispatches(args.orchestration_commit,preflight=True); return 0
    if args.command=='paused-resume-preflight':
        controller.resume_existing_dispatches(args.orchestration_commit,preflight=True,paused=True); return 0
    def term(*_): raise RuntimeError('控制器收到终止信号')
    signal.signal(signal.SIGTERM,term); signal.signal(signal.SIGINT,term)
    try: return controller.run(resume_completed_smokes=args.resume_completed_smokes,resume_existing_dispatches=args.resume_existing_dispatches,resume_paused=args.resume_paused,
                               orchestration_commit=args.orchestration_commit,pause_on_stop=args.pause_on_stop)
    except ControllerLeaseBusy as e:
        print('CONTROLLER_LEASE=FAIL '+str(e),flush=True); return 1
    except BaseException as e:
        print('CONTROLLER_EXIT=FAIL '+str(e),flush=True)
        return 1

if __name__=='__main__': raise SystemExit(main())
