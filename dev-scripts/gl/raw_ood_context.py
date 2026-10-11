"""从固定原字节清单读取可信前驱上下文；不写旧配置或媒体。"""
import hashlib
import json
from pathlib import Path

CORE=('run','run_name','exec_commit','config_sha256','controller_job_id')


def _entity(root,path):
    root=Path(root); path=Path(path)
    if not path.is_absolute() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()): raise ValueError('来源文件缺失或越界')
    for node in (path,*path.parents):
        if node==root.parent: break
        if node.is_symlink(): raise ValueError('来源文件不可有符号链')
    return path


def _read_locked(root,ref):
    if set(ref)!={'path','sha256'}: raise ValueError('来源引用必须显式path和sha256')
    path=_entity(root,ref['path']); raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']: raise ValueError('来源原字节SHA不符')
    return json.loads(raw)


def _source_configs(c):
    """返回当前配置及独立SHA锁定的前驱配置，拒绝任意上下文字典自证。"""
    ref=c.get('prior_sources_manifest')
    if ref is None: return [c]
    manifest=_read_locked(c['root'],ref)
    if manifest['schema']!='raw-ood-sources-v1' or manifest['run']!=c['run'] or manifest['run_name']!=c['run_name'] or not manifest['sources']:
        raise ValueError('前驱来源清单身份或协议不符')
    contexts=[c]; seen={c['config_sha256']}
    fixed=('run','run_name','root','gl_root','identities','identities_sha256','budget_ledger','expected_identities','expected_publications','prior_work')
    for source in manifest['sources']:
        if set(source)!={'config','sha256'}: raise ValueError('前驱必须引用原完整配置')
        path=_entity(c['root'],source['config']); raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
        if digest!=source['sha256'] or digest in seen: raise ValueError('前驱配置SHA冲突或重复')
        old=json.loads(raw)
        if 'config_sha256' in old and old['config_sha256']!=digest: raise ValueError('前驱配置内SHA不符')
        old['config_sha256']=digest
        if any(old[k]!=c[k] for k in fixed): raise ValueError('前驱运行/身份/预算固定字段改变')
        if {str(s['job_id']):(s['seat'],s['port_base']) for s in old['seats']}!={str(s['job_id']):(s['seat'],s['port_base']) for s in c['seats']}:
            raise ValueError('前驱GPU/席位/端口改变')
        if set(old['models'])!=set(c['models']): raise ValueError('前驱模型集合改变')
        for model in old['models']:
            before=dict(old['models'][model]); after=dict(c['models'][model]); env0=before.pop('env',{}); env1=after.pop('env',{})
            allowed={'PYTHONPATH','SGEVAL_THIRD_PARTY','ROBOMME_EVAL_ROOT'}
            if before!=after or {k:v for k,v in env0.items() if k not in allowed}!={k:v for k,v in env1.items() if k not in allowed}:
                raise ValueError('前驱模型路由/标签/权重/参数改变')
        contexts.append(old); seen.add(digest)
    return contexts


def source_configs(c):
    try: return _source_configs(c)
    except (KeyError,TypeError,json.JSONDecodeError) as e: raise ValueError('来源清单/配置必需字段错误') from e


def trusted_contexts(c):
    configs=source_configs(c)
    if any(not set(CORE)<=source.keys() for source in configs): raise ValueError('来源缺必需core字段')
    return [{k:source[k] for k in CORE} for source in configs]


def match_context(row,c):
    if not set(CORE)<=row.keys(): raise ValueError('记录缺必需core字段')
    matches=[source for source in trusted_contexts(c) if all(type(row[k]) is type(source[k]) and row[k]==source[k] for k in CORE)]
    if len(matches)!=1: raise ValueError('产物上下文没有唯一已锁来源')
    return matches[0]


def read_pause_snapshot(c):
    value=_read_locked(c['root'],c['pause_snapshot'])
    if value['schema']!='raw-ood-pause-v1' or any(value[k]!=c[k] for k in ('run','run_name','root')):
        raise ValueError('暂停快照身份不符')
    if not value['workers'] or not value['stop_files'] or float(value['stop_time'])<=0: raise ValueError('暂停快照缺停止证据')
    for ref in value['stop_files']:
        path=_entity(c['root'],ref['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']: raise ValueError('暂停STOP证据原字节改变')
    return value


def validate_historical_errors(c):
    """只承认配置显式锁定的历史错误，缺失或改字节均拒绝。"""
    refs=c.get('historical_errors',[])
    if not isinstance(refs,list): raise ValueError('历史错误须为来源引用列表')
    paths=set()
    for ref in refs:
        _read_locked(c['root'],ref)
        if ref['path'] in paths: raise ValueError('历史错误引用重复')
        paths.add(ref['path'])
    return paths


def is_historical_error(path,c):
    return str(path) in validate_historical_errors(c)


def read_budget_rows(path):
    """共享锁下读取原预算，消费者不创建锁、不触发配置升级。"""
    import fcntl,time
    path=Path(path); lock=path.with_name(path.name+'.lock')
    if lock.exists():
        with lock.open('rb') as stream:
            deadline=time.monotonic()+5
            while True:
                try: fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.monotonic()>=deadline: raise ValueError('预算只读共享锁超过5秒')
                    time.sleep(.05)
            raw=path.read_bytes()
    else: raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'): raise ValueError('预算原账本半行')
    return [json.loads(x) for x in raw.decode().splitlines() if x.strip()],hashlib.sha256(raw).hexdigest()


def validate_failure_document(doc,c):
    """失败证据与原报告、预算、领取、派发原字节逐项闭合；不把未知成绩编成零。"""
    from budget_ledger import BudgetState
    try:
        match_context(doc,c)
        if doc['schema']!='raw-ood-failure-v1' or doc['identity_key'] not in c['expected_identities']: raise ValueError('失败身份未知')
        model, dataset, key=doc['identity_key'].split(':',2)
        if dataset!='ood' or model!=doc['model'] or model not in c['models']: raise ValueError('失败模型身份不符')
        if not doc['shard_id'].startswith(model+'-') or doc['reason']!='watchdog_exit': raise ValueError('失败分片或原因不符')
        if doc['task_success'] is not None and (type(doc['task_success']) not in (bool,int) or doc['task_success'] not in (0,1)): raise ValueError('未知或非法成绩')
        if doc['exec_steps'] is not None and (type(doc['exec_steps']) is not int or doc['exec_steps']<0): raise ValueError('非法步数')
        raw_line=doc['report_line']
        if hashlib.sha256(raw_line.encode()).hexdigest()!=doc['report_line_sha256']: raise ValueError('报告原行SHA不符')
        report=_entity(c['root'],doc['report_path'])
        if raw_line not in report.read_text().splitlines(): raise ValueError('原报告不含失败原行')
        row=json.loads(raw_line)
        if row['key']!=key or row['dataset']!='ood' or row['infra'] is not True or type(row['attempt']) is not int or row['attempt']!=1 or row['infra_reason']!=doc.get('infra_reason',doc['reason']): raise ValueError('失败原行身份不符')
        full=row.get('exit_kind')=='watchdog_hard_exit'
        if full:
            required={'exit_code','task_success','exec_steps','result','final','accepted','budget_token','budget_exhausted','run_blocked','void','status'}
            if not required<=row.keys() or row['status']!='fail' or row['exit_code']!=75 or row['infra_reason'] not in ('watchdog_exit','episode_wall','deadline_first_infer','deadline_media_finalize') or row['final'] is not False or row['accepted'] is not False or row['void'] is not False or row['budget_exhausted'] is not False or row['run_blocked'] is not False: raise ValueError('完整看门狗协议不符')
        elif row['infra_reason']!='watchdog_exit' or 'exit_kind' in row: raise ValueError('稀疏历史失败不是已证watchdog')
        if doc['task_success']!=row.get('task_success') or doc['exec_steps']!=row.get('exec_steps'): raise ValueError('失败观察值与原行不符')
        dispatch=_read_locked(c['root'],doc['dispatch']); progress=_read_locked(c['root'],doc['progress']); claim=_read_locked(c['root'],doc['queue'])
        match_context(dispatch,c)
        seat=next(s for s in c['seats'] if str(s['job_id'])==str(dispatch['gpu_job_id']))
        tag=Path(doc['dispatch']['path']).parent.name
        if dispatch['model']!=model or dispatch['shard_id']!=doc['shard_id'] or dispatch['smoke'] is not False: raise ValueError('失败派发不符')
        expected_report=Path(c['gl_root'])/'seats'/c['models'][model]['label']/'seed7'/(seat['seat']+'-'+tag)/'seat-results.jsonl'
        expected_claim=Path(c['gl_root'])/'queue'/c['models'][model]['label']/'seed7/claims'/('ood__'+key+'.a1.json')
        if report!=expected_report or Path(doc['queue']['path'])!=expected_claim or Path(doc['progress']['path'])!=Path(doc['dispatch']['path']).parent/'progress.json': raise ValueError('失败证据路径绑定不符')
        identities=[json.loads(x) for x in _entity(c['root'],str(Path(doc['dispatch']['path']).parent/'identities.jsonl')).read_text().splitlines()]
        if len([r for r in identities if r['key']==key and r['dataset']=='ood'])!=1: raise ValueError('失败不在原派发身份清单')
        if progress['phase']!='finished' or progress['rc']!=75 or str(progress['slurm_job_id'])!=str(seat['job_id']) or progress['seat']!=seat['seat']+'-'+tag: raise ValueError('失败进度终态不符')
        step=doc['step']
        if step['id']!=str(seat['job_id'])+'.'+str(progress['slurm_step_id']) or step['name']!='raw-ood-worker-'+tag or step['state'] not in ('FAILED','COMPLETED') or step['exit_code']!='75:0': raise ValueError('失败真实步骤不符')
        budget=doc['budget']; rid=budget['rid']; reserve=budget['reserve']; commit=budget['commit']
        rows,_=read_budget_rows(_entity(c['root'],c['budget_ledger']))
        state=BudgetState(rows)
        claims=[r for r in rows if r.get('kind')=='reset_claim' and r.get('rid')==rid]
        if state.bad_rows or state.reserves.get(rid)!=reserve or state.commits.get(rid)!=commit or claims!=budget['reset_claims'] or rid in state.releases: raise ValueError('失败预算证据改变或未闭合')
        route=c['models'][model]['policy']+('/'+{'qwenvl':'ground-sg-qwenvl','oracle':'ground-sg-oracle'}[model] if model in ('qwenvl','oracle') else '')+'/seed7/new'
        token=route+'|'+key+'|a1'
        if row['budget_rid']!=rid or row.get('budget_token',token)!=token or reserve['key']!=key or reserve['route']!=route or reserve['token']!=token or reserve['kind_of_try']!='first' or type(reserve['attempt_no']) is not int or reserve['attempt_no']!=1 or reserve['resets']!=2: raise ValueError('失败首试预算绑定不符')
        if len([r for r in state.reserves.values() if r.get('route')==route and r.get('key')==key])!=1 or commit['infra'] is not True or commit['status']!='fail': raise ValueError('失败重复首试或结算状态不符')
        expected_claims=['build','reset'][:len(claims)]
        if len(claims)>2 or not full and len(claims)!=2 or [r['what'] for r in claims]!=expected_claims or any(str(r['slurm_job_id'])!=str(seat['job_id']) for r in (reserve,commit)): raise ValueError('失败reset或GPU作业不符')
        if claim['rid']!=rid or claim['key']!=key or claim['dataset']!='ood' or type(claim['attempt']) is not int or claim['attempt']!=1 or claim['route']!=route or claim['token']!=token or claim['seat']!=progress['seat'] or claim['ended'] is not True or claim['end_status']!='infra_timeout' or claim['exit_code']!=75: raise ValueError('失败领取未闭合')
        if (Path(c['gl_root'])/'queue'/c['models'][model]['label']/'seed7/accepted'/('ood__'+key+'.json')).exists(): raise ValueError('失败与正常接受身份冲突')
    except (KeyError,TypeError,StopIteration,json.JSONDecodeError) as e: raise ValueError('失败文档缺必需证据字段') from e
    return doc


def validate_terminal_marker(doc,c):
    """全部消费者共用严格终态：失败可结算，但不构成完整原始交付。"""
    try:
        match_context(doc,c)
        if doc['schema']!='raw-ood-terminal-v1': raise ValueError('终态协议不符')
        names=('expected','settled','published','infra','unresolved','unattempted')
        if any(type(doc[k]) is not int or doc[k]<0 for k in names): raise ValueError('终态缺显式非负计数')
        lists=[doc['published_identities'],list(doc['failure_documents']),doc['unresolved_identities'],doc['unattempted_identities']]
        if any(not isinstance(x,list) or len(x)!=len(set(x)) for x in lists): raise ValueError('终态身份列表重复')
        flat=[key for group in lists for key in group]
        if len(flat)!=len(set(flat)) or set(flat)!=set(c['expected_identities']) or doc['expected']!=len(flat): raise ValueError('终态身份互斥覆盖不符')
        if [doc[k] for k in ('published','infra','unresolved','unattempted')]!=[len(x) for x in lists] or doc['settled']!=doc['published']+doc['infra']: raise ValueError('终态计数不闭合')
        status='COMPLETE' if doc['published']==doc['expected'] and doc['infra']==doc['unresolved']==doc['unattempted']==0 else 'INCOMPLETE'
        if doc['status']!=status: raise ValueError('终态状态与证据不符')
        for key,ref in doc['failure_documents'].items():
            failure=_read_locked(c['root'],ref)
            if failure['identity_key']!=key or Path(ref['path'])!=Path(c['gl_root'])/'failures'/(hashlib.sha256(key.encode()).hexdigest()+'.json'): raise ValueError('失败引用身份不符')
            validate_failure_document(failure,c)
    except (KeyError,TypeError,json.JSONDecodeError) as e: raise ValueError('终态缺必需字段') from e
    return doc
