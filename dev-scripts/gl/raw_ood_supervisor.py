#!/usr/bin/env python3
"""CPU独立监督：真实子进程、原子错误回执、有限清理、异常非零退出。"""
from __future__ import annotations

import argparse
from email.message import EmailMessage
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from raw_ood_controller import Controller, atomic, control_file, ledger_snapshot, legacy_context, load_config, observed_job_state, process_identity, query_gpu, read, sha, validate_config
MAIL_TO='hongzefu@umich.edu'
MAIL_COMMANDS={'/usr/sbin/sendmail','/usr/bin/sendmail','/usr/bin/mailx','/bin/mailx'}
GUARD_QUERY_FAILURES={}


def notification_config(c: dict) -> dict:
    cfg=c.get('notification',{'enabled':False})
    if not isinstance(cfg,dict) or not isinstance(cfg.get('enabled',False),bool): raise ValueError('notification配置错误')
    if not cfg.get('enabled',False): return dict(enabled=False)
    if cfg['to']!=MAIL_TO or cfg['command'] not in MAIL_COMMANDS or cfg['timeout']!=30: raise ValueError('通知收件人/命令/30秒上限不符')
    return cfg


def notify(c: dict, event: str, body: dict) -> dict:
    """只提交一次，不重试；submitted只说明本机MTA接纳，不代表邮箱收到。"""
    if event not in ('probe','failure','completed'): raise ValueError('未知通知事件')
    control=Path(c['root'])/'control'; target=control/f'notification-{event}.json'
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    if target.exists():
        old=read(target)
        if any(str(old[k])!=str(v) for k,v in base.items()) or old['event']!=event: raise ValueError('已有通知回执身份冲突')
        return old
    cfg=notification_config(c)
    subject=f"RoboMME {c['run']} {event} CPU {c['controller_job_id']} {c['config_sha256'][:12]}"
    record=dict(base,event=event,to=MAIL_TO,subject=subject,time=time.time(),state='disabled',rc=None,stderr='',stdout='',
                delivery_verified=False,notification_established=False)
    if cfg['enabled']:
        message=EmailMessage(); message['To']=MAIL_TO; message['Subject']=subject
        message.set_content(json.dumps(body,ensure_ascii=False,sort_keys=True,indent=2))
        command=cfg['command']; args=[command,'-i','-t'] if Path(command).name=='sendmail' else [command,'-s',subject,MAIL_TO]
        payload=message.as_bytes() if Path(command).name=='sendmail' else message.get_content().encode()
        record.update(state='submitting',command=args,timeout=cfg['timeout'])
        # 提交前持久化，崩溃后不会重复发信，也不会把未知提交结果当成功。
        atomic(target,record)
        try:
            p=subprocess.run(args,input=payload,capture_output=True,timeout=cfg['timeout'])
            record.update(state='submitted' if p.returncode==0 else 'failed',rc=p.returncode,
                          stdout=p.stdout.decode(errors='replace') if isinstance(p.stdout,bytes) else p.stdout,
                          stderr=p.stderr.decode(errors='replace') if isinstance(p.stderr,bytes) else p.stderr)
        except (OSError,subprocess.TimeoutExpired) as e:
            record.update(state='failed',error_kind=type(e).__name__,stderr=str(e))
    atomic(target,record)
    print(f"NOTIFICATION state={record['state']} event={event} rc={record['rc']} delivery_verified=0",flush=True)
    return record


def require_probe(c: dict) -> dict | None:
    if not notification_config(c)['enabled']: return
    old=legacy_context(c); report=read(Path(c['root'])/'control'/'notification-probe.json')
    for key in ('run','run_name','exec_commit','config_sha256','controller_job_id'):
        if str(report[key])!=str(old[key]): raise ValueError('通知预检身份不符')
    if report['event']!='probe' or report['state']!='submitted' or report['rc']!=0 or report['to']!=MAIL_TO or not report['subject']:
        raise ValueError('邮件预检未提交成功')
    if not 0<float(report['time'])<=time.time()+30: raise ValueError('邮件预检时间错误')
    if c.get('legacy_source') is not None:
        path=control_file(c,str(Path(c['root'])/'control'/'mail-delivery-confirmed.json')); delivery=read(path)
        for key in ('run','run_name','exec_commit','config_sha256','controller_job_id'):
            if str(delivery[key])!=str(old[key]): raise ValueError('旧实收证据身份冲突')
        if delivery['received'] is not True or delivery['source']!='user_confirmation' or delivery['to']!=MAIL_TO or delivery['subject']!=report['subject']:
            raise ValueError('旧邮件实收证据缺失或冲突')
        # 旧回执不改写成新发送；复用来源独立记录。
        return dict(reused=True,legacy_config_sha256=old['config_sha256'],probe_sha256=sha(Path(c['root'])/'control'/'notification-probe.json'),
                    delivery_proof_sha256=sha(path),delivery_verified=True)
    return dict(reused=False,delivery_verified=False)


def notification_probe(config_path: Path) -> int:
    c=load_config(config_path); validate_config(c)
    if str(os.environ.get('SLURM_JOB_ID',''))!=str(c['controller_job_id']): raise ValueError('邮件预检必须在登记CPU作业')
    report=notify(c,'probe',dict(purpose='邮件投递链路预检',run=c['run'],controller_job_id=c['controller_job_id']))
    return 0 if report['state']=='submitted' else 1


def start_guardians(c: dict, guardians: list[dict] | None=None) -> list[dict]:
    """在已登记四hold各追加一个纯CPU守卫，不申请资源。"""
    control=Path(c['root'])/'control'; guardians=[] if guardians is None else guardians
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_','SRUN_')) and k!='CUDA_VISIBLE_DEVICES'}
    env.update(BENCH_PY=c['python'],PYTHONUNBUFFERED='1')
    for seat in c['seats']:
        job=str(seat['job_id'])
        if job in {g['job_id'] for g in guardians}: continue
        status=observed_job_state(c,job,GUARD_QUERY_FAILURES.setdefault((c['root'],c['config_sha256']),{}),'supervisor')
        if status is None or status in ('PENDING','CONFIGURING'): continue
        if status!='RUNNING': raise RuntimeError(f'附加守卫GPU状态异常 {job} {status}')
        log=(control/f'guardian-{job}.log').open('a')
        cmd=['srun',f'--jobid={job}','--gres=none','--overlap','--exact','--ntasks=1','--cpus-per-task=1',
             '--job-name=raw-ood-lifecycle-guard','bash',str(Path(__file__).with_name('raw_ood_hold_guard.sh')),
             c['root'],c['run'],str(c['controller_job_id']),str(int(c['cpu_end_time'])),
             str(control_file(c,c.get('ready_path',str(control/'launch.ready'))))]
        try: p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        except BaseException: log.close(); raise
        guardians.append(dict(job_id=job,process=p,log=log,command=cmd))
    atomic(control/'guardian-launches.json',dict(run=c['run'],run_name=c['run_name'],exec_commit=c['exec_commit'],
           config_sha256=c['config_sha256'],controller_job_id=c['controller_job_id'],
           guardians=[dict(job_id=g['job_id'],pid=g['process'].pid,command=g['command']) for g in guardians],time=time.time()))
    return guardians


def check_guardians(guardians: list[dict], c: dict) -> None:
    failures=[]
    for seat in c['seats']:
        job=str(seat['job_id']); path=Path(c['root'])/'control'/f'guard-failure-{job}.json'
        if not path.exists(): continue
        d=read(path)
        if d['run']!=c['run'] or str(d['controller_job_id'])!=str(c['controller_job_id']) or str(d['gpu_job_id'])!=job:
            raise ValueError('守卫原始失败回执身份错误')
        for k in ('exec_commit','config_sha256'):
            if k in d and d[k]!=c[k]: raise ValueError('守卫原始失败回执版本错误')
        if not 0<float(d['time'])<=time.time()+30: raise ValueError('守卫原始失败回执时间错误')
        failures.append((float(d['time']),d))
    if failures:
        d=min(failures,key=lambda pair:pair[0])[1]
        raise RuntimeError('独立守卫原始失败: '+json.dumps(d,ensure_ascii=False,sort_keys=True))
    marker=Path(c['root'])/'control'/'gpu_work_complete.json'; normal=False
    if marker.exists():
        d=read(marker)
        normal=all(str(d[k])==str(c[k]) for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'))
        if not normal: raise ValueError('GPU正常完成标记身份错误')
    paused=Path(c['root'])/'control/paused.json'
    if paused.exists() and (Path(c['root'])/'control/STOP').exists():
        pointer=read(paused); path=control_file(c,pointer['proof_path']); proof=read(path)
        if any(proof[k]!=c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')) or sha(path)!=pointer['proof_sha256'] or proof['unsettled']!=0 or proof['drained'] is not True:
            raise ValueError('守卫暂停释放证明不符')
        normal=True
    for g in guardians:
        rc=g['process'].poll()
        if rc is not None and not normal: raise RuntimeError(f"独立守卫提前退出 gpu={g['job_id']} rc={rc}")


def final_cleanup_evidence(c: dict, base: dict) -> dict:
    """保留最初取消回执；最终独立状态查询必须实际为空才通过。"""
    control=Path(c['root'])/'control'; initial=control/'gpu-cleanup.json'
    original=read(initial); current=query_gpu(c,timeout=5)
    result=dict(base,initial_receipt=str(initial),initial_sha256=sha(initial),initial=original,final_query=current,
                cleanup_incomplete=current['cleanup_incomplete'],time=time.time())
    atomic(control/'gpu-cleanup-final.json',result)
    with (control/'gpu-final-checks.jsonl').open('a') as f: f.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+'\n')
    return result


def check_heartbeat(path: Path, base: dict, *, now: float, timeout: float=120) -> dict:
    d=read(path)
    for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'):
        if str(d[k])!=str(base[k]): raise ValueError(f'心跳身份错误{k}')
    t=d['time'] if 'time' in d else d['t']
    if now-float(t)>timeout or float(t)>now+30: raise RuntimeError(f'心跳失效{path}')
    return d


def controller_terminal_proof(c,base):
    """PID消失本身没有成功含义；只接受完整4000终态协议，最终搬运及清理仍由监督主循环验。"""
    control=Path(c['root'])/'control'; gl=Path(c['gl_root'])
    if (control/'controller-error.json').exists(): raise RuntimeError('附着控制器已记录错误')
    for path in (control/'gpu_work_complete.json',gl/'gpu_work_complete.json'):
        d=read(path)
        if any(str(d[k])!=str(v) for k,v in base.items()) or d['expected']!=4000 or d['published']!=4000:
            raise ValueError('附着控制器完成标记不符')
    result=read(control/'results-report.json')
    if any(str(result[k])!=str(v) for k,v in base.items()) or result['expected']!=4000 or result['accepted']!=4000 or result['missing']!=0 or result['infra']!=0:
        raise ValueError('附着控制器4000终态报告不符')
    ledger,_=ledger_snapshot(Path(c['budget_ledger']))
    budget=result['budget']
    if ledger.trajectories!=4000 or ledger.retries or ledger.recovery_used or ledger.astra or budget['trajectories']!=4000 or budget['retries']!=0 or budget['bad_rows']!=0:
        raise ValueError('附着控制器终态预算不符')
    claims=sum(ledger.claimed.values())+ledger.orphan_claims
    if claims>8000 or budget['reset_claims']!=claims or budget['reset_count']!=ledger.resets: raise ValueError('附着控制器终态reset计量不符')
    if len(c['expected_publications'])!=4000 or {p.name for p in (gl/'published').glob('*.json')}!=set(c['expected_publications']):
        raise ValueError('附着控制器4000发布集合不符')
    return dict(results_sha256=sha(control/'results-report.json'),budget_trajectories=4000,reset_claims=claims)


class AttachedController:
    """只读附着现存控制器，不创建、停止或等待其它子进程。"""
    def __init__(self,c,pid,identity):
        self.c=c; self.pid=pid; self.identity=identity
        self.base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    def poll(self):
        current=process_identity(self.pid)
        if current is not None and current['state']!='Z':
            if any(current[k]!=self.identity[k] for k in ('start_ticks','command','cgroup')): raise ValueError('附着控制器PID复用或来源变化')
            return None
        try: proof=controller_terminal_proof(self.c,self.base)
        except (FileNotFoundError,KeyError) as e: raise RuntimeError('附着控制器消失且终态未知，不当作成功') from e
        atomic(Path(self.c['root'])/'control/controller-attach-terminal.json',dict(self.base,pid=self.pid,exit_code=None,status_source='validated_complete_protocol',**proof,time=time.time()))
        return 0


def attach_controller(c,config_path,pid,source,manifest_path,expected_manifest_sha):
    """先核独立源码锁与原启动证明，再绑定真实PID；此函数没有任何Popen。"""
    import re
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    root=Path(c['root']); control=root/'control'; source=Path(source)
    if source.name!='raw_ood_controller.py' or not source.is_absolute() or source.is_symlink() or not source.resolve().is_relative_to(root.resolve()): raise ValueError('附着控制器源码路径不符')
    for path in source.parents:
        if path==root.parent: break
        if path.is_symlink(): raise ValueError('附着源码目录符号链')
    manifest_path=control_file(c,str(manifest_path))
    if not re.fullmatch('[0-9a-f]{64}',str(expected_manifest_sha)) or sha(manifest_path)!=expected_manifest_sha: raise ValueError('附着原源码锁SHA不符')
    manifest=read(manifest_path); origin=read(control/'orchestration-started.json'); started=read(control/'supervisor-started.json')
    if any(str(origin[k])!=str(v) or str(started[k])!=str(v) for k,v in base.items()) or int(started['controller_pid'])!=int(pid): raise ValueError('附着原启动证明身份/PID不符')
    if origin['resume_existing_dispatches'] is not True or origin['resume_completed_smokes'] is not False or manifest['orchestration_commit']!=origin['orchestration_commit'] or manifest['worker_exec_commit']!=c['exec_commit'] or manifest['config_sha256']!=c['config_sha256']:
        raise ValueError('附着原编排来源不符')
    info=manifest['files']['raw_ood_controller.py']
    if source.stat().st_size!=info['bytes'] or sha(source)!=info['sha256']: raise ValueError('附着控制器源码不符原锁')
    heartbeat=check_heartbeat(control/'controller-heartbeat.json',base,now=time.time(),timeout=120)
    if int(heartbeat['pid'])!=int(pid): raise ValueError('附着心跳PID不符')
    proc=process_identity(pid)
    expected=[c['python'],str(source),'run','--config',str(config_path),'--resume-existing-dispatches','--orchestration-commit',origin['orchestration_commit']]
    if proc is None or proc['state']=='Z' or proc['command']!=expected or not re.search(r'job[_/-]'+re.escape(str(c['controller_job_id']))+r'(?:/|$)',proc['cgroup']): raise ValueError('附着PID命令/CPU作业不符或不存在')
    if abs(proc['start_time']-float(started['time']))>5: raise ValueError('附着PID创建时间与原启动证明不符')
    proof=dict(base,controller_pid=int(pid),source=str(source),source_sha256=info['sha256'],manifest_sha256=expected_manifest_sha,
               original_orchestration_commit=origin['orchestration_commit'],start_ticks=proc['start_ticks'],command=proc['command'],cgroup=proc['cgroup'],
               original_start_sha256=sha(control/'supervisor-started.json'),original_orchestration_sha256=sha(control/'orchestration-started.json'),
               controller_action='attach_only',budget_action='none',time=time.time())
    return AttachedController(c,int(pid),proc),proof


def fail(config: dict, error: BaseException, *, process=None) -> int:
    control=Path(config['root'])/'control'; base={k:config[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    outcome=dict(base,error=type(error).__name__,message=str(error),time=time.time(),supervisor_exit_code=1,
                 controller_pid=process.pid if process else None,controller_action='none',gpu_action='none',
                 cpu_hold=True,notification_established=False)
    atomic(control/'supervisor-failure.json',outcome)
    atomic(control/'intentional-supervisor-stop.json',dict(outcome,reason='exception_stop_only_supervisor'))
    try: outcome['notification']=notify(config,'failure',outcome)
    except BaseException as e: outcome['notification']=dict(state='failed',error_kind=type(e).__name__,stderr=str(e),delivery_verified=False)
    atomic(control/'supervisor-failure.json',outcome)
    print('RAW_SUPERVISOR=FAIL cpu_hold=1 gpu_cancel=0 controller_kill=0 notification_established=0 error='+str(error),flush=True)
    return 1


def run(config_path: Path,*,resume_completed_smokes=False,resume_existing_dispatches=False,resume_paused=False,orchestration_commit=None,
        attach_controller_pid=None,attach_controller_source=None,attach_controller_manifest=None,attach_manifest_sha256=None) -> int:
    c=load_config(config_path)
    control=Path(c['root'])/'control'; gl=Path(c['gl_root'])
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    if str(os.environ.get('SLURM_JOB_ID',''))!=str(c['controller_job_id']): raise ValueError('监督器不在登记CPU作业')
    if int(os.environ.get('SLURM_RESTART_COUNT','0')): raise ValueError('禁止重排队')
    controller=None; guardians=[]; started=time.time(); last_moved=None; last_progress=started; cleanup_settle_started=None
    loop_state={'t':started}; controller_started_t=started
    guard_monitor_stop=threading.Event(); guard_monitor=None
    cpu_heart_stop=threading.Event(); cpu_heart=None
    def term(signum,*_):
        if signum==signal.SIGUSR1: raise RuntimeError('独立守卫异常: '+(control/'guardian-error.json').read_text())
        raise RuntimeError('CPU监督收到终止信号')
    signal.signal(signal.SIGTERM,term); signal.signal(signal.SIGINT,term); signal.signal(signal.SIGUSR1,term)
    try:
        if sum((resume_completed_smokes,resume_existing_dispatches,resume_paused))>1: raise ValueError('手动恢复模式互斥')
        if attach_controller_pid is not None and (resume_completed_smokes or resume_existing_dispatches or resume_paused): raise ValueError('附着与重建恢复模式互斥')
        validate_config(c)
        probe_provenance=require_probe(c)
        ready=read(control_file(c,c.get('ready_path',str(control/'launch.ready')))); jobs=read(control/'jobs.json')
        for k,v in base.items():
            if k=='run_name' and k not in ready: continue
            if str(ready[k])!=str(v): raise ValueError('ready版本或身份不符')
        if ready['root']!=str(Path(c['root'])) or jobs['run']!=c['run'] or str(jobs['controller_job_id'])!=str(c['controller_job_id']):
            raise ValueError('资源清单身份不符')
        if set(map(str,jobs['gpu_job_ids']))!={str(s['job_id']) for s in c['seats']}: raise ValueError('资源清单GPU不符')
        if probe_provenance is not None and probe_provenance['reused']:
            atomic(control/'notification-probe-reused.json',dict(base,**probe_provenance,time=time.time()))
        mover=Path(c.get('mover_heartbeat',control/'mover-heartbeat.json'))
        mover_error=Path(c.get('mover_error',control/'mover-error.json'))
        check_heartbeat(mover,base,now=time.time())
        if attach_controller_pid is not None:
            controller,attach_proof=attach_controller(c,config_path,attach_controller_pid,attach_controller_source,attach_controller_manifest,attach_manifest_sha256)
            atomic(control/f'controller-attach-{os.getpid()}.json',dict(attach_proof,new_orchestration_commit=orchestration_commit))
        atomic(control/'cpu-heartbeat.json',dict(base,t=time.time(),loop_t=loop_state['t'],pid=os.getpid(),controller_pid=controller.pid if controller else None,controller_started_t=controller_started_t))
        def heartbeat():
            while not cpu_heart_stop.is_set():
                atomic(control/'cpu-heartbeat.json',dict(base,t=time.time(),loop_t=loop_state['t'],pid=os.getpid(),controller_pid=controller.pid if controller else None,controller_started_t=controller_started_t))
                cpu_heart_stop.wait(5)
        cpu_heart=threading.Thread(target=heartbeat,name='raw-ood-cpu-heartbeat',daemon=True); cpu_heart.start()
        # 接续先全量只读核验；任何新守卫/控制器子进程均在此门之后启动。
        if resume_existing_dispatches: Controller(c).resume_existing_dispatches(orchestration_commit,preflight=True)
        if resume_paused: Controller(c).resume_existing_dispatches(orchestration_commit,preflight=True,paused=True)
        def watch_guards():
            while not guard_monitor_stop.wait(1):
                try: check_guardians(list(guardians),c)
                except BaseException as e:
                    atomic(control/'guardian-error.json',dict(base,error=str(e),time=time.time()))
                    os.kill(os.getpid(),signal.SIGUSR1)
                    return
        guard_monitor=threading.Thread(target=watch_guards,daemon=True); guard_monitor.start()
        start_guardians(c,guardians)
        log=None if attach_controller_pid is not None else (control/'controller.log').open('a')
        controller_command=[c['python'],str(Path(__file__).with_name('raw_ood_controller.py')),'run','--config',str(config_path)]
        if resume_completed_smokes or resume_existing_dispatches or resume_paused:
            import re
            if not re.fullmatch('[0-9a-f]{40}',str(orchestration_commit)): raise ValueError('手动恢复须明确orchestration_commit')
            controller_command+=['--resume-paused' if resume_paused else '--resume-existing-dispatches' if resume_existing_dispatches else '--resume-completed-smokes','--orchestration-commit',orchestration_commit]
            atomic(control/'orchestration-started.json',dict(base,orchestration_commit=orchestration_commit,
                   resume_completed_smokes=resume_completed_smokes,resume_existing_dispatches=resume_existing_dispatches,resume_paused=resume_paused,time=time.time()))
        if attach_controller_pid is None:
            controller_started_t=time.time()
            controller=subprocess.Popen(controller_command,
                                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
            atomic(control/'supervisor-started.json',dict(base,pid=os.getpid(),controller_pid=controller.pid,time=time.time()))
        else: atomic(control/f'supervisor-attached-{os.getpid()}.json',dict(base,pid=os.getpid(),controller_pid=controller.pid,time=time.time()))
        while True:
            now=time.time()
            check_guardians(guardians,c)
            for f in (control/'controller-error.json',control/'guardian-error.json',mover_error,gl/'mover'/'error.json',control/'STOP',gl/'STOP'):
                if f.exists(): raise RuntimeError(f'已发布错误{f}: {f.read_text()[:3000]}')
            h=check_heartbeat(mover,base,now=now)
            # 零值必须实际写入JSON；缺键属于错误，不能默认成功。
            pending=int(h['pending']); moved=int(h['moved']); progress=float(h['last_progress'])
            if pending<0 or moved<0: raise ValueError('搬运计数负数')
            if moved!=last_moved or progress>last_progress: last_progress=now; last_moved=moved
            if not pending: last_progress=now
            elif now-last_progress>1800: raise RuntimeError('搬运积压30分钟无进展')
            if now>=float(c['cpu_end_time'])-120: raise RuntimeError('CPU到期前清理')
            rc=controller.poll()
            if rc is None:
                heartbeat=control/'controller-heartbeat.json'
                if heartbeat.exists():
                    d=check_heartbeat(heartbeat,base,now=now,timeout=120)
                    if d['pid']!=controller.pid or now-float(d['loop_t'])>=120: raise RuntimeError('控制器主循环无进展或PID错误')
                elif now-started>120: raise RuntimeError('控制器未建立心跳')
            elif rc==6:
                d=read(control/'paused.json'); proof=read(Path(d['proof_path']))
                if any(proof[k]!=v or d[k]!=v for k,v in base.items()) or sha(Path(d['proof_path']))!=d['proof_sha256'] or proof['unsettled']!=0 or proof['drained'] is not True or proof['accepted']+proof['remaining']!=4000:
                    raise ValueError('控制器暂停证明不完整')
                atomic(control/'supervisor-paused.json',dict(base,controller_pid=controller.pid,accepted=proof['accepted'],remaining=proof['remaining'],paused_proof=d,time=now))
                print('SUPERVISOR_PAUSED=PASS accepted='+str(proof['accepted'])+' remaining='+str(proof['remaining']),flush=True); return 6
            elif rc!=0: raise RuntimeError(f'控制器退出{rc}')
            else:
                complete=gl/'mover'/'completed.json'
                if not (control/'gpu_work_complete.json').exists(): raise RuntimeError('控制器退出缺GPU完成标记')
                if complete.exists():
                    d=read(complete)
                    for k,v in base.items():
                        if str(d[k])!=str(v): raise ValueError('搬运完成身份错误')
                    for k in ('expected','moved','missing','decode_fail','sha_mismatch','pending'):
                        if k not in d: raise ValueError('搬运完成缺计数'+k)
                    if d['expected']!=4000 or d['moved']!=4000 or any(d[k]!=0 for k in ('missing','decode_fail','sha_mismatch','pending')):
                        raise ValueError('搬运完整验收失败')
                    cleanup=final_cleanup_evidence(c,base)
                    if cleanup['cleanup_incomplete']:
                        if cleanup_settle_started is None: cleanup_settle_started=time.monotonic()
                        if time.monotonic()-cleanup_settle_started>=30: raise RuntimeError('最终GPU状态查询仍非空或未知，清理未完成')
                        time.sleep(5); continue
                    atomic(control/'completed.json',dict(base,expected=4000,moved=4000,infra=0,time=now))
                    atomic(control/'final-report.json',dict(base,results=read(control/'results-report.json'),delivery=d,cleanup=cleanup,
                          supervisor_exit_code=0,cpu_hold=True,notification_established=False,time=now))
                    atomic(control/'intentional-supervisor-stop.json',dict(base,reason='supervisor_completed',supervisor_exit_code=0,
                          cpu_hold=True,gpu_action='already_completed_release',controller_action='none',notification_established=False,time=now))
                    report=read(control/'final-report.json'); report['notification']=notify(c,'completed',report)
                    atomic(control/'final-report.json',report)
                    print('RAW_DELIVERY=PASS episodes=4000 videos=8000 hash_mismatch=0 decode_fail=0 missing=0 nfs_large_left=0',flush=True)
                    print('RUN_CLEANUP=PASS gpu_jobs=4 active_gpu=0 supervisor_exit_code=0 cpu_hold=1 notification_established=0',flush=True)
                    print('RAW_SUPERVISOR=PASS expected=4000 moved=4000 infra=0',flush=True); return 0
            if rc is None: start_guardians(c,guardians)
            loop_state['t']=time.time()
            time.sleep(5)
    except BaseException as e:
        # 清理会使守卫srun退出；先关闭独立监控，不能用二次信号打断首次错误清理。
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        guard_monitor_stop.set()
        if guard_monitor is not None: guard_monitor.join(timeout=2)
        return fail(c,e,process=controller)
    finally:
        cpu_heart_stop.set()
        if cpu_heart is not None: cpu_heart.join(timeout=6)
        guard_monitor_stop.set()
        if guard_monitor is not None: guard_monitor.join(timeout=2)
        for g in guardians:
            g['log'].close()


def self_test(root: Path) -> int:
    """不触发仿真的真实子进程与序列化夹具，供CPU节点运行。"""
    root.mkdir(parents=True,exist_ok=False)
    base=dict(run='fixture',run_name='fixture',exec_commit='a'*40,config_sha256='b'*64,controller_job_id='123')
    fixture=root/'heartbeat.json'
    p=subprocess.Popen([sys.executable,'-c','import json,sys,time;json.dump(json.loads(sys.argv[2]),open(sys.argv[1],"w"))',str(fixture),json.dumps(dict(base,time=time.time(),pending=0,moved=0,last_progress=time.time()))])
    assert p.wait(timeout=5)==0
    check_heartbeat(fixture,base,now=time.time())
    broken=dict(base,time=time.time()-121); atomic(fixture,broken)
    try: check_heartbeat(fixture,base,now=time.time())
    except RuntimeError: pass
    else: raise AssertionError('失效心跳未阻断')
    detached=root/'detached.json'
    worker='import json,sys,time;time.sleep(.1);json.dump({"complete":1},open(sys.argv[1],"w"))'
    launcher='import subprocess,sys;subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)'
    parent=subprocess.run([sys.executable,'-c',launcher,worker,str(detached)],timeout=5)
    assert parent.returncode==0
    end=time.monotonic()+5
    while not detached.exists() and time.monotonic()<end: time.sleep(.01)
    assert read(detached)['complete']==1
    bindir=root/'bin'; bindir.mkdir(); calls=root/'cancelled.txt'
    for name,body in [('scancel','echo "$1" >> "'+str(calls)+'"'),('squeue','exit 0')]:
        f=bindir/name; f.write_text('#!/bin/bash\n'+body+'\n'); f.chmod(0o700)
    old=os.environ['PATH']; os.environ['PATH']=str(bindir)+':'+old
    try:
        c=dict(base,root=str(root),seats=[dict(job_id=str(i)) for i in range(124,128)])
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        assert fail(c,RuntimeError('夹具监督异常'),process=child)==1
        assert child.poll() is None and not calls.exists() and not (root/'control'/'STOP').exists()
        assert read(root/'control'/'supervisor-failure.json')['message']=='夹具监督异常'
        assert read(root/'control'/'intentional-supervisor-stop.json')['notification_established'] is False
        child.terminate(); child.wait(timeout=5)  # 仅清理本夹具自己创建的CPU子进程。
        crashed=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],start_new_session=True)
        assert crashed.wait(timeout=5)==7
        assert fail(c,RuntimeError('控制器退出7'),process=crashed)==1
        assert read(root/'control'/'supervisor-failure.json')['message']=='控制器退出7'
        # CPU消失时只结束此guard，不取消GPU allocation或其它step。
        guard_root=root/'guard'; (guard_root/'control').mkdir(parents=True)
        (guard_root/'control'/'gpu-job-ids.txt').write_text('124\n125\n126\n127\n')
        guard=subprocess.run(['bash',str(Path(__file__).with_name('raw_ood_hold_guard.sh')),str(guard_root),'fixture','123',str(int(time.time()+60))],
                             env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='124',SLURM_STEP_ID='0'),capture_output=True,text=True,timeout=5)
        assert guard.returncode!=0 and not (guard_root/'control'/'STOP').exists() and not calls.exists()
    finally: os.environ['PATH']=old
    atomic(root/'result.json',dict(supervisor_exit=1,controller_alive=1,guard_self_exit=1,gpu_cancel=0,global_stop=0,parent_exit=1,notification_established=False))
    print('STOP_ISOLATION=PASS supervisor_exit=1 controller_alive=1 guard_self_exit=1 gpu_cancel=0 global_stop=0 parent_exit=1 notification_established=0 simulated_slurm=1')
    return 0


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run'); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume-completed-smokes',action='store_true'); p.add_argument('--resume-existing-dispatches',action='store_true'); p.add_argument('--orchestration-commit',default=None)
    p.add_argument('--resume-paused',action='store_true')
    p.add_argument('--attach-controller-pid',type=int); p.add_argument('--attach-controller-source',type=Path)
    p.add_argument('--attach-controller-manifest',type=Path); p.add_argument('--attach-manifest-sha256')
    p=sub.add_parser('self-test'); p.add_argument('--root',type=Path,required=True)
    p=sub.add_parser('notification-probe'); p.add_argument('--config',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='notification-probe': return notification_probe(args.config)
    return run(args.config,resume_completed_smokes=args.resume_completed_smokes,resume_existing_dispatches=args.resume_existing_dispatches,resume_paused=args.resume_paused,orchestration_commit=args.orchestration_commit,
               attach_controller_pid=args.attach_controller_pid,attach_controller_source=args.attach_controller_source,
               attach_controller_manifest=args.attach_controller_manifest,attach_manifest_sha256=args.attach_manifest_sha256) if args.command=='run' else self_test(args.root)

if __name__=='__main__': raise SystemExit(main())
