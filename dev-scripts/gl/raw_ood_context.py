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
