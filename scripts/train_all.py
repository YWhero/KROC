"""Run the three approved experiments sequentially, detached from SSH sessions."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kroc_act.paths import PROJECT_ROOT, atomic_json, project_path
from kroc_act.train import resolve_config

EXPERIMENTS = ['01_baseline', '02_concat', '03_window']


def now():
    return datetime.now(timezone.utc).isoformat()


def last_metrics(run):
    path = project_path(run/'metrics.jsonl')
    if not path.is_file():
        return {}
    with path.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell()-65536))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            return {key: row[key] for key in ['step','loss','val_l1_zero_latent','best_val_l1','best_step'] if key in row}
        except (ValueError, UnicodeDecodeError):
            continue
    return {}


def transport_ready():
    run = project_path(PROJECT_ROOT/'outputs/relay_transport_test_20261115')
    path = project_path(run/'upload_receipts/00000001.json')
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    return (receipt.get('verified') is True and receipt.get('cleanup_complete') is True
            and receipt.get('repo_id') == 'HYHero/kroc-act-emg-checkpoints'
            and not project_path(run/'checkpoints/00000001/checkpoint.pt').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--relay-start-timeout',type=float,default=900)
    args = parser.parse_args()
    configs = [(name,resolve_config(PROJECT_ROOT/'configs'/f'{name}.json')) for name in EXPERIMENTS]
    if args.dry_run:
        print(json.dumps({'order':EXPERIMENTS,'steps_each':[cfg['steps'] for _,cfg in configs],
                          'transport_ready':transport_ready(),'root':str(PROJECT_ROOT)}))
        return
    logs = project_path(PROJECT_ROOT/'logs/training_20261115')
    logs.mkdir(parents=True,exist_ok=True)
    lock_path = project_path(logs/'runner.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('The KROC training sequence is already running')
        for name,cfg in configs:
            if project_path(cfg['output_dir']).exists():
                raise FileExistsError(f'Refusing to overwrite existing experiment: {name}')
        status = {'runner_pid':os.getpid(),'started_at_utc':now(),'updated_at_utc':now(),
                  'state':'waiting_for_relay_test','experiments':[
                      {'name':name,'state':'queued','target_steps':cfg['steps']} for name,cfg in configs]}
        status_path = project_path(logs/'status.json')
        def publish():
            status['updated_at_utc'] = now()
            atomic_json(status_path,status)
        publish()
        deadline = time.monotonic()+args.relay_start_timeout
        while not transport_ready():
            if time.monotonic()>=deadline:
                status['state']='blocked_relay_test';publish()
                raise SystemExit('Diagnostic upload not confirmed; training not started')
            time.sleep(5)
        status['transport_verified_at_utc']=now()
        env = dict(os.environ)
        for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','KROC_RELAY_HF_TOKEN']:
            env.pop(key,None)
        for index,(name,cfg) in enumerate(configs):
            entry = status['experiments'][index]
            log = project_path(logs/f'{name}.log')
            status.update(state='running',active_experiment=name)
            entry.update(state='running',started_at_utc=now(),log=str(log))
            with log.open('ab',buffering=0) as output:
                process = subprocess.Popen(['bash',str(PROJECT_ROOT/'scripts/train.sh'),
                    '--config',str(PROJECT_ROOT/'configs'/f'{name}.json')],cwd=PROJECT_ROOT,
                    env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                    start_new_session=True)
                entry['training_pid']=process.pid;publish()
                while process.poll() is None:
                    entry['last_metrics']=last_metrics(Path(cfg['output_dir']))
                    publish()
                    time.sleep(15)
                code=process.wait()
            entry.update(exit_code=code,finished_at_utc=now(),last_metrics=last_metrics(Path(cfg['output_dir'])))
            if code!=0:
                entry['state']='failed';status['state']='failed';publish()
                raise SystemExit(f'{name} failed; subsequent experiments remain queued. See {log}')
            if entry['last_metrics'].get('step')!=cfg['steps']:
                entry['state']='failed';status['state']='failed';publish()
                raise SystemExit(f'{name} did not reach its requested step count')
            entry['state']='completed';publish()
        status.update(state='completed',active_experiment=None,finished_at_utc=now());publish()


if __name__=='__main__':
    main()
