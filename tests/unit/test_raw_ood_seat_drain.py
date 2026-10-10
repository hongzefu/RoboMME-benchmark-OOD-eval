"""席位排空只阻断新领取；专属进度隔离其它席心跳。"""
import importlib.util
from pathlib import Path
import subprocess
import types

HERE=Path(__file__).resolve().parents[2]/'dev-scripts/gl'


def test_claim_stop_deadline_and_file(tmp_path):
    import sys
    spec=importlib.util.spec_from_file_location('raw_drain_seat',HERE/'seat.py')
    seat=importlib.util.module_from_spec(spec); sys.modules[spec.name]=seat; spec.loader.exec_module(seat)
    runner=seat.SeatRunner.__new__(seat.SeatRunner)
    runner.args=types.SimpleNamespace(claim_deadline=20,stop_file=str(tmp_path/'STOP')); runner.now=lambda:10
    assert not runner.claims_stopped()
    runner.now=lambda:20; assert runner.claims_stopped()
    runner.args.claim_deadline=None; runner.now=lambda:10
    (tmp_path/'STOP').write_text('stop'); assert runner.claims_stopped()


def test_watchdog_explicit_path_ignores_other_seat(tmp_path):
    text=(HERE/'run_eval_gl.sh').read_text()
    func=text[text.index('newest_progress()'):text.index('\nstop_left_servers()')]
    explicit=tmp_path/'own.json'; explicit.write_text('{}')
    other=tmp_path/'seats'/'same'/'other'/'progress.json'; other.parent.mkdir(parents=True); other.write_text('{}')
    import os
    os.utime(explicit,(100,100)); os.utime(other,(10000,10000))
    cmd=f'PROGRESS_FILE={explicit}; OUT={tmp_path}; POLICY_SEED=7\n'+func+'\nnewest_progress same\n'
    p=subprocess.run(['bash','-c',cmd],capture_output=True,text=True)
    assert p.returncode==0 and p.stdout.strip()=='100'


def test_exclusive_heartbeat_keeps_latest_phase_and_identity(tmp_path):
    import sys, threading, json
    spec=importlib.util.spec_from_file_location('raw_progress_seat',HERE/'seat.py')
    seat=importlib.util.module_from_spec(spec); sys.modules[spec.name]=seat; spec.loader.exec_module(seat)
    runner=seat.SeatRunner.__new__(seat.SeatRunner)
    runner.args=types.SimpleNamespace(progress_file=str(tmp_path/'progress.json'),policy='smvla')
    runner._lock=threading.Lock(); runner._progress_lock=threading.Lock(); runner._last_progress=None
    runner._current=None; runner.progress_path=tmp_path/'progress.json'
    runner.seat='s'; runner.label='smvla'; runner.policy_seed=7; runner.episodes_done=0
    runner.progress('context_load'); runner.heartbeat_once()
    runner._current=types.SimpleNamespace(key='real-key',ident={'dataset':'ood'},attempt=1)
    runner.queue=types.SimpleNamespace(heartbeat=lambda _:None); runner._current.path=tmp_path/'claim'
    runner.progress('episode'); runner.heartbeat_once()
    doc=json.loads(runner.progress_path.read_text())
    assert doc['phase']=='episode' and doc['key']=='real-key' and doc['pid']>0
    runner._current=None; runner.progress('done'); runner.heartbeat_once()
    assert json.loads(runner.progress_path.read_text())['phase']=='done'


def test_real_constructor_initializes_progress_and_concurrent_updates(tmp_path):
    import sys, threading, json
    spec=importlib.util.spec_from_file_location('raw_real_progress_seat',HERE/'seat.py')
    seat=importlib.util.module_from_spec(spec); sys.modules[spec.name]=seat; spec.loader.exec_module(seat)
    args=seat.build_parser().parse_args(['run','--policy','smvla','--policy-seed','7','--out',str(tmp_path/'stage'),
         '--identities',str(tmp_path/'ids'),'--budget-ledger',str(tmp_path/'budget'),'--trajectory-cap','4000','--reset-cap','8000',
         '--shared-infra-cap','0','--expired-cap','0','--planned-first-tries','4000','--progress-file',str(tmp_path/'progress.json')])
    runner=seat.SeatRunner(args,shared=None)
    assert runner._last_progress is None
    runner.progress('context_load')
    worker=threading.Thread(target=lambda:[runner.heartbeat_once() for _ in range(30)])
    worker.start()
    for _ in range(30): runner.progress('episode')
    worker.join(); runner.progress('done'); runner.heartbeat_once()
    doc=json.loads(runner.progress_path.read_text())
    assert doc['phase']=='done' and doc['label']=='smvla' and doc['key'] is None
    runner.close()
