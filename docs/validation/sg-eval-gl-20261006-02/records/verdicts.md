# sg-eval-gl-20261006-02 判定行原文（随运行追加）

## 第一档（生成回归，GL A40，执行副本 robomme_benchmark-noise@b869a3df）

```
GEN_REGRESS=NEED_RERUN flip=1 set=v9 n=129 match=126 jitter=2 flip=1 structural=0 unknown=0 missing=0 invalid=0 rerun_rows=4 filler=3
GEN_REGRESS=PASS set=v9 n=129 match=126 jitter=2 flip=1 structural=0 unknown=0 missing=0 invalid=0 rerun_rows=4 filler=3 noise=1 regression=0 env_changed=0 unstable=0 noise_max=2
GEN_REGRESS=PASS set=xhard0 n=48 match=48 jitter=0 flip=0 structural=0 unknown=0 missing=0 invalid=0
```

## 第二档（原侧 vs 新侧，五模型各 16 任务 × 1 档 × 12 局 = 192）

```
GATE2_INPUTS=PASS expected=192 missing=0 extra=0 unaccepted=0 ambiguous=0 trace_binding_mismatch=0      （五模型均如此）
GATE2_PROVENANCE=PASS local_rows=0 unknown_rows=0                                                       （五模型均如此）
GATE2_SUPPLEMENT=PASS old=39 supplement=153 expected=153 missing=0 extra=0 overlap=0                    （QwenVL）
GATE2_SUPPLEMENT=PASS old=96 supplement=96 expected=96 missing=0 extra=0 overlap=0                      （PonderPounce）
GATE2=INFO policy=mmesg-oracle compared=192 same_terminal=188 s2f=2 f2s=2 sr_orig=0.7500 sr_new=0.7500 sr_diff_pp=0.00 mcnemar_p=1 not_observed=0 identical_trace=144 … matrix=ss:142,sf:1,st:1,fs:2,ff:20,ft:0,ts:0,tf:0,tt:26 server_epoch_first=4
GATE2=INFO policy=mmesg-qwenvl compared=192 same_terminal=177 s2f=3 f2s=5 sr_orig=0.2292 sr_new=0.2396 sr_diff_pp=1.04 mcnemar_p=0.7266 not_observed=0 identical_trace=86 … matrix=ss:41,sf:2,st:1,fs:3,ff:121,ft:4,ts:2,tf:3,tt:15 server_epoch_first=8
GATE2=INFO policy=pp compared=192 same_terminal=192 s2f=0 f2s=0 sr_orig=0.4167 sr_new=0.4167 sr_diff_pp=0.00 mcnemar_p=1 not_observed=82598 identical_trace=0 … matrix=ss:80,ff:90,tt:22
GATE2=INFO policy=smvla compared=192 same_terminal=192 s2f=0 f2s=0 sr_orig=0.7344 sr_new=0.7344 sr_diff_pp=0.00 mcnemar_p=1 not_observed=250164 identical_trace=0 … matrix=ss:141,ff:36,tt:15
GATE2=INFO policy=mme compared=192 same_terminal=174 s2f=7 f2s=10 sr_orig=0.2552 sr_new=0.2708 sr_diff_pp=1.56 mcnemar_p=0.6291 not_observed=131154 identical_trace=0 … matrix=ss:42,sf:7,st:0,fs:9,ff:122,ft:0,ts:1,tf:1,tt:10
GATE2_PROJ policy=mmesg-oracle action=72268/22002/0 obs=218648/42548/0 state=108596/22002/0 logic=20034/4686/0 text=90571/3891/0 stop=282928/109/0 episodes_diff=action:48,obs:48,state:48,logic:48,text:27,stop:35
GATE2_PROJ policy=mmesg-qwenvl action=33747/42482/0 obs=143480/81634/0 state=70074/42483/0 logic=11699/10741/0 text=58209/18212/0 stop=228702/258/0 episodes_diff=action:106,obs:106,state:106,logic:106,text:65,stop:81
GATE2_PROJ policy=pp action=82598/0/0 obs=237468/192/0 state=118734/192/0 logic=165580/0/82598 text=6025/76765/0 stop=247986/0/0 episodes_diff=action:0,obs:192,state:192,logic:0,text:192,stop:0
GATE2_PROJ policy=smvla action=85192/0/0 obs=243040/0/0 state=121520/0/0 logic=10824/0/0 text=85384/0/0 stop=5604/0/250164 episodes_diff=action:0,obs:0,state:0,logic:0,text:0,stop:0
GATE2_PROJ policy=mme action=20196/45381/0 obs=117068/86742/0 state=56524/45381/0 logic=8501/10195/0 text=65769/0/0 stop=65662/195/131154 episodes_diff=action:133,obs:133,state:133,logic:133,text:0,stop:89
ORIG_RERUN_VS_E0=INFO policy=smvla compared=192 flips=0 s2f=0 f2s=0
ORIG_RERUN_VS_E0=INFO policy=mme compared=192 flips=15 s2f=8 f2s=7
GATE2_NOISE=INFO policy=smvla pair=E0-O1 flips=0；E0-O2 flips=0；O1-O2 flips=0（sr 均 0.7344）
GATE2_NOISE=INFO policy=mme pair=E0-O1 s2f=4 f2s=3 flips=7；E0-O2 s2f=9 f2s=7 flips=16；O1-O2 s2f=10 f2s=9 flips=19
OBSERVER_COMPLETE=PASS（批次 3a 十片 × SimpleMemVLA、MME 各一行；MME 每片 conns=episodes mismatch=0）
```

GATE2_PROJ 各维度为 same/diff/not_observed 步数。

## 第二档新侧覆盖与官方视频（五模型）

```
EVAL_COVERAGE=PASS dataset=test-hard0 expected=192 missing=0 extra=0 duplicate=0 conflicting_terminal=0 error_final=0   （五模型）
EVAL_REPORT=PASS dataset=test-hard0 count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=skip          （五模型）
EVAL_VIDEOS=PASS dataset=test-hard0 expected=192 videos=192 missing=0 decode_fail=0                                    （五模型）
OFFICIAL_MEDIA_INPUTS=PASS missing=0 extra=0 ambiguous=0 identity_mismatch=0 attempt_mismatch=0 provenance_missing=0   （五模型）
OFFICIAL_MEDIA=PASS total=192 skip=0 fail=0 no_frame_error=0                                                           （五模型，共 960）
```

## 批次 5 Astra（本机新侧 smoke）

```
ASTRA_GUARD=PASS cap=5 committed=0.0000 heartbeat_age=0.7s
ASTRA_CHECK=PASS dataset=test-hard0 max_steps=1300 cases=1 port=18763
ASTRA_SUMMARY dataset=test-hard0 expected=1 success=1 fail=0 timeout=0 error=0 pending=0
OFFICIAL_RENDER=PASS dir=VideoUnmask_xhard0_560300.a1 frames=349 demo=66 steps=282 omitted=0 source_kind=raw-new
OFFICIAL_MEDIA=PASS total=1 skip=0 fail=0 no_frame_error=0
ASTRA_COST usd=0.3043 calls=3 pending=0 projected=0.5072 cap=5
```

## 第三档（用户指令中止时的部分结果，--partial）

```
EVAL_COVERAGE=FAIL dataset=test-hard policy=mmesg:ground-sg-oracle expected=800 missing=258 extra=0 duplicate=0 conflicting_terminal=0 error_final=0 partial=1
EVAL_REPORT=FAIL dataset=test-hard policy=mmesg:ground-sg-oracle count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=0 partial=1
EVAL_VIDEOS=FAIL dataset=test-hard policy=mmesg:ground-sg-oracle expected=800 videos=541 missing=259 decode_fail=0 partial=1
EVAL_COVERAGE=FAIL dataset=test-hard policy=mmesg:ground-sg-qwenvl expected=800 missing=637 extra=0 duplicate=0 conflicting_terminal=0 error_final=0 partial=1
EVAL_REPORT=FAIL dataset=test-hard policy=mmesg:ground-sg-qwenvl count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=0 partial=1
EVAL_VIDEOS=FAIL dataset=test-hard policy=mmesg:ground-sg-qwenvl expected=800 videos=160 missing=640 decode_fail=0 partial=1
BUDGET_DETAIL reserves=2084 committed=2056 released=20 open=8 expired=0/500 reset_soft_exceeded=0
BUDGET_ENFORCEMENT=PASS trajectories=2064/6366 resets=4908/141430 astra=0/2 shared_infra=0/50
```
