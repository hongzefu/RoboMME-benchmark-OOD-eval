#!/usr/bin/env python3
"""CPU独立监督：真实子进程、原子错误回执、有限清理、异常非零退出。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from raw_ood_controller import atomic, cancel_gpu, load_config, read, validate_config


def start_guardians(c: dict, guardians: list[dict] | None=None) -> list[dict]:
    """在已登记四hold各追加一个纯CPU守卫，不申请资源。"""
    control=Path(c['root'])/'control'; guardians=[] if guardians is None else guardians
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_','SRUN_')) and k!='CUDA_VISIBLE_DEVICES'}
    env.update(BENCH_PY=c['python'],PYTHONUNBUFFERED='1')
    for seat in c['seats']:
        job=str(seat['job_id'])
        if job in {g['job_id'] for g in guardians}: continue
        state=subprocess.run(['squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True,timeout=5)
        if state.returncode: raise RuntimeError('附加守卫GPU作业查询失败')
        status=state.stdout.strip()
        if status in ('PENDING','CONFIGURING'): continue
        if status!='RUNNING': raise RuntimeError(f'附加守卫GPU状态异常 {job} {status}')
        log=(control/f'guardian-{job}.log').open('a')
        cmd=['srun',f'--jobid={job}','--gres=none','--overlap','--exact','--ntasks=1','--cpus-per-task=1',
             '--job-name=raw-ood-lifecycle-guard','bash',str(Path(__file__).with_name('raw_ood_hold_guard.sh')),
             c['root'],c['run'],str(c['controller_job_id']),str(int(c['cpu_end_time']))]
        try: p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        except BaseException: log.close(); raise
        guardians.append(dict(job_id=job,process=p,log=log,command=cmd))
    atomic(control/'guardian-launches.json',dict(run=c['run'],run_name=c['run_name'],exec_commit=c['exec_commit'],
           config_sha256=c['config_sha256'],controller_job_id=c['controller_job_id'],
           guardians=[dict(job_id=g['job_id'],pid=g['process'].pid,command=g['command']) for g in guardians],time=time.time()))
    return guardians


def check_guardians(guardians: list[dict], c: dict) -> None:
    marker=Path(c['root'])/'control'/'gpu_work_complete.json'; normal=False
    if marker.exists():
        d=read(marker)
        normal=all(str(d[k])==str(c[k]) for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'))
        if not normal: raise ValueError('GPU正常完成标记身份错误')
    for g in guardians:
        rc=g['process'].poll()
        if rc is not None and not normal: raise RuntimeError(f"独立守卫提前退出 gpu={g['job_id']} rc={rc}")


def check_heartbeat(path: Path, base: dict, *, now: float, timeout: float=120) -> dict:
    d=read(path)
    for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'):
        if str(d[k])!=str(base[k]): raise ValueError(f'心跳身份错误{k}')
    t=d['time'] if 'time' in d else d['t']
    if now-float(t)>timeout or float(t)>now+30: raise RuntimeError(f'心跳失效{path}')
    return d


def fail(config: dict, error: BaseException, *, process=None) -> int:
    control=Path(config['root'])/'control'; base={k:config[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    atomic(control/'STOP',dict(base,component='supervisor',message=str(error),time=time.time()))
    atomic(control/'failure.json',dict(base,error=type(error).__name__,message=str(error),time=time.time()))
    if process and process.poll() is None:
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError: pass
    cleanup=cancel_gpu(config)
    atomic(control/'cleanup.json',dict(base,**cleanup,time=time.time()))
    if process and process.poll() is None:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    print('RAW_SUPERVISOR=FAIL error='+str(error),flush=True)
    return 1


def run(config_path: Path) -> int:
    c=load_config(config_path); validate_config(c)
    control=Path(c['root'])/'control'; gl=Path(c['gl_root'])
    base={k:c[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
    if str(os.environ.get('SLURM_JOB_ID',''))!=str(c['controller_job_id']): raise ValueError('监督器不在登记CPU作业')
    if int(os.environ.get('SLURM_RESTART_COUNT','0')): raise ValueError('禁止重排队')
    controller=None; guardians=[]; started=time.time(); last_moved=None; last_progress=started
    guard_monitor_stop=threading.Event(); guard_monitor=None
    def term(signum,*_):
        if signum==signal.SIGUSR1: raise RuntimeError('独立守卫异常: '+(control/'guardian-error.json').read_text())
        raise RuntimeError('CPU监督收到终止信号')
    signal.signal(signal.SIGTERM,term); signal.signal(signal.SIGINT,term); signal.signal(signal.SIGUSR1,term)
    try:
        ready=read(control/'launch.ready'); jobs=read(control/'jobs.json')
        for k,v in base.items():
            if k=='run_name' and k not in ready: continue
            if str(ready[k])!=str(v): raise ValueError('ready版本或身份不符')
        if ready['root']!=str(Path(c['root'])) or jobs['run']!=c['run'] or str(jobs['controller_job_id'])!=str(c['controller_job_id']):
            raise ValueError('资源清单身份不符')
        if set(map(str,jobs['gpu_job_ids']))!={str(s['job_id']) for s in c['seats']}: raise ValueError('资源清单GPU不符')
        mover=Path(c.get('mover_heartbeat',control/'mover-heartbeat.json'))
        mover_error=Path(c.get('mover_error',control/'mover-error.json'))
        check_heartbeat(mover,base,now=time.time())
        atomic(control/'cpu-heartbeat.json',dict(base,t=time.time(),pid=os.getpid()))
        def watch_guards():
            while not guard_monitor_stop.wait(1):
                try: check_guardians(list(guardians),c)
                except BaseException as e:
                    atomic(control/'guardian-error.json',dict(base,error=str(e),time=time.time()))
                    atomic(control/'STOP',dict(base,component='guardian-monitor',message=str(e),time=time.time()))
                    os.kill(os.getpid(),signal.SIGUSR1)
                    return
        guard_monitor=threading.Thread(target=watch_guards,daemon=True); guard_monitor.start()
        start_guardians(c,guardians)
        log=(control/'controller.log').open('a')
        controller=subprocess.Popen([c['python'],str(Path(__file__).with_name('raw_ood_controller.py')),'run','--config',str(config_path)],
                                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        atomic(control/'supervisor-started.json',dict(base,pid=os.getpid(),controller_pid=controller.pid,time=time.time()))
        while True:
            now=time.time(); atomic(control/'cpu-heartbeat.json',dict(base,t=now,pid=os.getpid(),controller_pid=controller.pid))
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
                if heartbeat.exists(): check_heartbeat(heartbeat,base,now=now,timeout=120)
                elif now-started>120: raise RuntimeError('控制器未建立心跳')
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
                    cleanup=read(control/'gpu-cleanup.json')
                    if cleanup['cleanup_incomplete']: raise RuntimeError('GPU清理未完成')
                    atomic(control/'completed.json',dict(base,expected=4000,moved=4000,infra=0,time=now))
                    atomic(control/'final-report.json',dict(base,results=read(control/'results-report.json'),delivery=d,cleanup=cleanup,
                          cpu_exit_code=0,time=now))
                    print('RAW_DELIVERY=PASS episodes=4000 videos=8000 hash_mismatch=0 decode_fail=0 missing=0 nfs_large_left=0',flush=True)
                    print('RUN_CLEANUP=PASS gpu_jobs=4 cpu_jobs=1 active=0 cpu_exit_code=0',flush=True)
                    print('RAW_SUPERVISOR=PASS expected=4000 moved=4000 infra=0',flush=True); return 0
            if rc is None: start_guardians(c,guardians)
            time.sleep(5)
    except BaseException as e:
        # 清理会使守卫srun退出；先关闭独立监控，不能用二次信号打断首次错误清理。
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        guard_monitor_stop.set()
        if guard_monitor is not None: guard_monitor.join(timeout=2)
        return fail(c,e,process=controller)
    finally:
        guard_monitor_stop.set()
        if guard_monitor is not None: guard_monitor.join(timeout=2)
        for g in guardians:
            if g['process'].poll() is None:
                try: os.killpg(g['process'].pid,signal.SIGTERM)
                except ProcessLookupError: pass
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
        assert fail(c,RuntimeError('夹具控制器崩溃'),process=child)==1
        child.wait(timeout=5)
        assert calls.read_text().splitlines()==['124','125','126','127']
        assert read(root/'control'/'failure.json')['message']=='夹具控制器崩溃'
        assert not read(root/'control'/'cleanup.json')['cleanup_incomplete']
        crashed=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],start_new_session=True)
        assert crashed.wait(timeout=5)==7
        assert fail(c,RuntimeError('控制器退出7'),process=crashed)==1
        assert read(root/'control'/'failure.json')['message']=='控制器退出7'
        # 整个CPU作业从squeue消失时，真实GPU守卫须在ready尚不存在时也停止。
        guard_root=root/'guard'; (guard_root/'control').mkdir(parents=True)
        (guard_root/'control'/'gpu-job-ids.txt').write_text('124\n125\n126\n127\n')
        guard=subprocess.run(['bash',str(Path(__file__).with_name('raw_ood_hold_guard.sh')),str(guard_root),'fixture','123',str(int(time.time()+60))],
                             env=dict(os.environ,BENCH_PY=sys.executable,SLURM_JOB_ID='124'),capture_output=True,text=True,timeout=5)
        assert guard.returncode!=0 and (guard_root/'control'/'STOP').exists()
        assert calls.read_text().splitlines()[8:]==['125','126','127','124']
    finally: os.environ['PATH']=old
    atomic(root/'result.json',dict(success=1,global_stop=1,cpu_lost=1,mover_lost=1,parent_exit=1))
    print('DETACHED_FLOW=PASS success=1 global_stop=1 cpu_lost=1 mover_lost=1 parent_exit=1 simulated_slurm=1')
    return 0


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run'); p.add_argument('--config',type=Path,required=True)
    p=sub.add_parser('self-test'); p.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    return run(args.config) if args.command=='run' else self_test(args.root)

if __name__=='__main__': raise SystemExit(main())
