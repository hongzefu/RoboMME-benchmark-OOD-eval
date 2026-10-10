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

MODELS = ('qwenvl', 'smvla', 'pp', 'oracle', 'framesamp')
SECONDS = dict(qwenvl=311.428, smvla=152.526, pp=138.695, oracle=130.485, framesamp=86.518)
ROUTES = dict(qwenvl=('groundsg','groundsg-ground-sg-qwenvl','ground-sg-qwenvl'),
              oracle=('groundsg','groundsg-ground-sg-oracle','ground-sg-oracle'),smvla=('smvla','smvla',None),
              pp=('pp','pp',None),framesamp=('perceptual-framesamp-modul','perceptual-framesamp-modul',None))
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
    return json.loads(path.read_text())


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


def identity(model: str, row: dict) -> str:
    return f"{model}:ood:{row['key']}"


def publication_name(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()+'.json'


def validate_config(config: dict) -> list[dict]:
    """完整输入字段缺失即失败，不能把未知计数当零。"""
    for k in ('run','run_name','root','gl_root','controller_job_id','exec_commit','config_sha256','identities','identities_sha256','models','seats','python','runner','budget_ledger','cpu_end_time','expected_identities','expected_publications'):
        if k not in config: raise ValueError(f'配置缺少{k}')
    if config['run'] != config['run_name']: raise ValueError('运行名冲突')
    if set(config['models']) != set(MODELS): raise ValueError('五模型白名单不符')
    seats=config['seats']; ids=[str(s['job_id']) for s in seats]
    if len(ids)!=4 or len(set(ids))!=4 or not all(i.isdigit() for i in ids): raise ValueError('GPU登记不符')
    if str(config['controller_job_id']) in ids: raise ValueError('CPU与GPU作业重复')
    for name,m in config['models'].items():
        policy,label,variant=ROUTES[name]
        if m['policy']!=policy or m['label']!=label: raise ValueError('模型路由不符合固定白名单')
        forbidden={'--infra-retries','--client-restarts','--trajectory-cap','--reset-cap','--shared-infra-cap','--expired-cap',
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
    return rows


def make_shards(rows: list[dict]) -> dict[str,list[dict]]:
    # 冒烟身份置首，之后各模型相同的互斥100身份切片。
    smoke=[r for r in rows if r['task']=='VideoUnmask' and r['tier']=='xhard1' and r['builder_episode']==0]
    if len(smoke)!=1: raise ValueError('正式首局缺失')
    ordered=smoke+[r for r in rows if r!=smoke[0]]
    return {f'{model}-{i:02d}': {'model':model,'rows':ordered[i*100:(i+1)*100]} for model in MODELS for i in range(8)}


def cancel_gpu(config: dict, *, timeout: float=25) -> dict:
    """只取消本轮四GPU；每条调用有超时，返回原始回执。"""
    deadline=time.monotonic()+timeout; records=[]
    for seat in config['seats']:
        jid=str(seat['job_id'])
        try:
            p=subprocess.run(['scancel',jid],capture_output=True,text=True,timeout=max(.1,deadline-time.monotonic()))
            records.append(dict(job_id=jid,rc=p.returncode,stdout=p.stdout,stderr=p.stderr))
        except (OSError,subprocess.TimeoutExpired) as e: records.append(dict(job_id=jid,error=str(e)))
    active='尚未查询'; query_rc=-1
    while time.monotonic()<deadline:
        try:
            p=subprocess.run(['squeue','-h','-j',','.join(str(s['job_id']) for s in config['seats']),'-o','%A %T'],capture_output=True,text=True,timeout=max(.1,deadline-time.monotonic()))
            active=p.stdout.strip(); query_rc=p.returncode
        except (OSError,subprocess.TimeoutExpired) as e: active=str(e); query_rc=-1
        if not active and query_rc==0: break
        time.sleep(min(.5,max(0,deadline-time.monotonic())))
    return dict(records=records,active=active,query_rc=query_rc,cleanup_incomplete=bool(active) or query_rc!=0)


class Controller:
    def __init__(self, config: dict):
        self.c=config; self.rows=validate_config(config); self.root=Path(config['root']); self.control=self.root/'control'
        self.gl=Path(config['gl_root']); self.shards=make_shards(self.rows); self.pending=list(self.shards)
        self.active={}; self.published=set(); self.smoked=set(); self.completed=set(); self.formal=False
        self.initial_dispatched=set(); self.started=time.time()
        self.base={k:config[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
        self.deadline=min(float(config['cpu_end_time']),*(float(s['end_time']) for s in config['seats']))-7200

    def event(self, kind: str, **kw):
        row=dict(self.base,time=time.time(),kind=kind,**kw)
        self.control.mkdir(parents=True,exist_ok=True)
        with (self.control/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

    def start(self, seat: dict, shard: str, *, smoke=False):
        spec=self.shards[shard]; model=spec['model']; m=self.c['models'][model]
        rows=spec['rows'][:1] if smoke else spec['rows']
        tag=f"{shard}-{'smoke' if smoke else 'full'}"; inv=self.control/'invocations'/tag
        inv.mkdir(parents=True,exist_ok=False)
        ids=inv/'identities.jsonl'; ids.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        seat_name=f"{seat['seat']}-{tag}"; progress=inv/'progress.json'
        expansion=dict(root=str(self.root),gl_root=str(self.gl),seat=seat_name,shard=shard,port_base=seat['port_base'],work_dir=str(inv/'work'))
        model_args=[str(x).format_map(expansion) for x in m['args']]
        worker_name=f'raw-ood-worker-{tag}'
        cmd=['srun',f"--jobid={seat['job_id']}",f'--job-name={worker_name}','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--gpu_cmode=shared',
             'bash',self.c['runner'],'--policies',m['policy'],'--policy-seed','7','--identities',str(ids),'--out',str(self.gl),
             '--run',self.c['run'],'--seat',seat_name,'--budget-ledger',self.c['budget_ledger'],
             '--trajectory-cap','4000','--reset-cap','8000','--planned-first-tries','4000','--shared-infra-cap','0','--expired-cap','0',
             '--infra-retries','0','--client-restarts','0','--gpus','','--no-render','--stop-on-env-build-error','--dataset','ood',
             '--port-base',str(seat['port_base']),'--claim-deadline',str(self.deadline),'--stop-file',str(self.control/'STOP'),
             '--progress-file',str(progress),*model_args]
        env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_','SRUN_'))}
        env.pop('CUDA_VISIBLE_DEVICES',None)
        env.update(BENCH_PY=self.c['python'],PYTHONUNBUFFERED='1'); env.update(m.get('env',{}))
        log=(inv/'worker.log').open('a'); p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        active=dict(process=p,log=log,seat=seat,shard=shard,smoke=smoke,model=model,progress=progress,
                    results=self.gl/'seats'/m['label']/'seed7'/seat_name/'seat-results.jsonl',rows=rows,offset=0,started=time.time(),acked=False,
                    phase_key=None,phase_start=time.time(),step_id=None,worker_name=worker_name)
        self.active[str(seat['job_id'])]=active
        atomic(inv/'dispatch.json',dict(self.base,gpu_job_id=str(seat['job_id']),shard_id=shard,model=model,smoke=smoke,command=cmd,srun_pid=p.pid,time=time.time()))
        self.event('dispatch',gpu_job_id=str(seat['job_id']),shard_id=shard,smoke=smoke)

    def publish(self, a: dict, row: dict):
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
                 relative_dir=directory.relative_to(self.gl).as_posix(),files=files,delete_files=['front.mkv','wrist.mkv','arrays.npz'])
        atomic(self.gl/'published'/publication_name(key),doc); self.published.add(key)
        self.event('published',identity_key=key,task_success=row['task_success'])
        if not a['smoke'] and not a['acked'] and row['exec_steps']>=1 and a.get('step_id') and a['progress'].exists():
            p=read(a['progress'])
            self.write_ack(a,key,p['pid'],row['exec_steps'],str(result))

    def write_ack(self,a,key,pid,step,proof):
        ack=dict(self.base,gpu_job_id=str(a['seat']['job_id']),step_id=a['step_id'],model=a['model'],shard_id=a['shard'],
                 identity_key=key,client_pid=pid,first_step=step,progress_mtime=a['progress'].stat().st_mtime,proof=proof,time=time.time())
        if not (self.control/'first_dispatch.json').exists(): atomic(self.control/'first_dispatch.json',ack)
        a['acked']=True

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

    def check_worker(self,a):
        now=time.time()
        if not a.get('step_id'):
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
            if now-float(p['t'])>120: raise RuntimeError('专属席位心跳失效')
            if phase!=a['phase_key']: a['phase_key']=phase; a['phase_start']=now
            limits=self.c.get('phase_limits',{'context_load':3600,'claim':120,'episode':3600,'done':300,'finished':120})
            if now-a['phase_start']>float(limits[p['phase']]): raise RuntimeError(f"席位阶段超时{phase}")
        elif now-a['started']>120: raise RuntimeError('席位未建立进度')

    def scan(self):
        for jid,a in list(self.active.items()):
            self.check_worker(a)
            if a['results'].exists():
                content=a['results'].read_text(); lines=content.splitlines()
                if content and not content.endswith('\n'): lines=lines[:-1]
                for line in lines[a['offset']:]:
                    self.publish(a,json.loads(line)); a['offset']+=1
            self.poll_ack(a)
            rc=a['process'].poll()
            if rc is not None:
                a['log'].close(); del self.active[jid]
                if rc!=0:
                    if rc==6 and time.time()>=self.deadline:
                        self.event('drained',shard_id=a['shard'],accepted=a['offset']); continue
                    raise RuntimeError(f"worker失败 model={a['model']} shard={a['shard']} rc={rc}")
                expected={identity(a['model'],r) for r in a['rows']}
                if not expected<=self.published: raise ValueError('worker正常退出但缺结果')
                if a['smoke']: self.smoked.add(a['model'])
                else: self.completed.add(a['shard'])

    def dispatch(self):
        for seat in self.c['seats']:
            if str(seat['job_id']) in self.active: continue
            state=subprocess.run(['squeue','-h','-j',str(seat['job_id']),'-o','%T'],capture_output=True,text=True,timeout=5)
            if state.returncode: raise RuntimeError('Slurm查询失败')
            if state.stdout.strip() in ('PENDING','CONFIGURING'): continue
            if state.stdout.strip()!='RUNNING': raise RuntimeError(f'GPU异常状态{state.stdout}')
            guardian=self.control/f"guardian-ready-{seat['job_id']}.json"
            if not guardian.exists(): continue
            ready=read(guardian)
            if any(str(ready[k])!=str(v) for k,v in self.base.items()) or str(ready['gpu_job_id'])!=str(seat['job_id']):
                raise ValueError('附加独立守卫就绪身份不符')
            if not isinstance(ready['pid'],int) or ready['pid']<=0 or not 0<float(ready['time'])<=time.time()+30 or ready['step_id'] is None:
                raise ValueError('附加独立守卫缺少有效PID/时间/StepID')
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
                shard=next(s for s in self.pending if self.shards[s]['model']==model)
                self.pending.remove(shard); self.start(seat,shard)

    def stop_children(self):
        for a in self.active.values():
            if a['process'].poll() is None:
                try: os.killpg(a['process'].pid,signal.SIGTERM)
                except ProcessLookupError: pass

    def report(self):
        """读取保留的小result与共享账本，自动产生实测报告；任务失败单列。"""
        from budget_ledger import BudgetState
        ledger=BudgetState([json.loads(x) for x in Path(self.c['budget_ledger']).read_text().splitlines() if x.strip()])
        if ledger.bad_rows or ledger.trajectories!=4000 or ledger.resets>8000 or ledger.retries or ledger.recovery_used or ledger.astra:
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
            if stat['accepted']!=800: raise ValueError('逐模型800终态不齐')
            tasks={}
            for g in groups.values():
                if g['model']==model:
                    t=tasks.setdefault(g['task'],[0,0]); t[0]+=g['success']; t[1]+=g['accepted']
            stat['episode_success_rate']=stat['success']/stat['accepted']
            stat['task_macro_success_rate']=sum(s/n for s,n in tasks.values())/len(tasks)
            stat['steady_decision_wall_ms_mean']=stat['decision_wall_ms']/stat['chunks'] if stat['chunks'] else None
        budget=dict(trajectories=ledger.trajectories,reset_claims=sum(ledger.claimed.values())+ledger.orphan_claims,
                    reset_count=ledger.resets,retries=len(ledger.retries),recovery=ledger.recovery_used,astra=ledger.astra,
                    trajectory_cap=4000,reset_cap=8000,bad_rows=ledger.bad_rows)
        atomic(self.control/'results-report.json',dict(self.base,expected=4000,accepted=4000,missing=0,infra=0,budget=budget,
              models=model_stats,by_task_tier=list(groups.values()),wall_s=time.time()-self.started,time=time.time(),
              limitations=['未测GPU利用率','各片加载总耗时由worker日志提供，不能从局timing重复累加']))
        print(f"RUN_BUDGET=PASS trajectories=4000/4000 reset_claims={budget['reset_claims']}/8000 retries=0",flush=True)
        print('OOD_RESULTS=PASS models=5 expected=4000 accepted=4000 missing=0 infra=0',flush=True)

    def run(self):
        self.control.mkdir(parents=True,exist_ok=True)
        atomic(self.control/'shards.json',dict(self.base,shards=self.shards))
        stop_beat=threading.Event()
        def heartbeat():
            while not stop_beat.is_set():
                atomic(self.control/'controller-heartbeat.json',dict(self.base,t=time.time(),pid=os.getpid(),active=len(self.active)))
                stop_beat.wait(5)
        beat=threading.Thread(target=heartbeat,daemon=True); beat.start()
        try:
            while True:
                if (self.control/'STOP').exists() or (self.gl/'STOP').exists(): raise RuntimeError('全局STOP')
                self.scan()
                if len(self.smoked)==5 and not self.formal:
                    # 完整本机解码及SHA回执确认才允许全量；正常任务失败不阻塞。
                    names=[publication_name(identity(m,self.shards[m+'-00']['rows'][0])) for m in MODELS]
                    if all((self.gl/'mover'/'receipts'/n).exists() for n in names):
                        for n in names:
                            r=read(self.gl/'mover'/'receipts'/n)
                            if r['result']!='moved' or not r['decoded'] or not r['trace_checked']: raise ValueError('冒烟搬运未验收')
                        self.formal=True; atomic(self.control/'smoke-completed.json',dict(self.base,models=5,infra=0,time=time.time()))
                if time.time()<self.deadline: self.dispatch()
                elif not self.active and (self.pending or not self.formal): raise RuntimeError('已安全排空，剩余身份不重跑')
                if self.formal and not self.pending and not self.active:
                    if len(self.published)!=4000 or len(self.completed)!=40: raise ValueError('4000终态不齐')
                    self.report()
                    doc=dict(self.base,expected=4000,published=4000,time=time.time())
                    atomic(self.gl/'gpu_work_complete.json',doc); atomic(self.control/'gpu_work_complete.json',doc)
                    atomic(self.control/'gpu-cleanup.json',cancel_gpu(self.c)); return 0
                time.sleep(1)
        finally:
            stop_beat.set(); beat.join(timeout=6); self.stop_children()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['run','validate']); ap.add_argument('--config',required=True)
    args=ap.parse_args(); c=load_config(Path(args.config))
    if args.command=='validate':
        rows=validate_config(c); print(f'OOD_SCOPE=PASS models=5 identities_per_model={len(rows)} planned=4000'); return 0
    controller=Controller(c)
    def term(*_): raise RuntimeError('控制器收到终止信号')
    signal.signal(signal.SIGTERM,term); signal.signal(signal.SIGINT,term)
    try: return controller.run()
    except BaseException as e:
        atomic(Path(c['root'])/'control'/'controller-error.json',dict(controller.base,component='controller',error=type(e).__name__,message=str(e),time=time.time()))
        return 1

if __name__=='__main__': raise SystemExit(main())
