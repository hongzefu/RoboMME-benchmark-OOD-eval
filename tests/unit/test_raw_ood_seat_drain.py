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
