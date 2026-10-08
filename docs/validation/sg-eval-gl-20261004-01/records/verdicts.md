# 判定行原文（收尾时点汇总）

## 预检（本机）
```
PREFLIGHT_SUMMARY=PASS routes=13 failed=0 stopped_routes=none step_cap_ok=1 video_ok=1 site=sled-vail gpu=RTX6000Ada
VIDEO_SAVED=PASS route=mmesg-oracle-new-hard0 episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=485616 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=mmesg-oracle-new-v9 episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=405116 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=mmesg-oracle-orig episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=483891 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=mmesg-qwenvl-new-hard0 episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=493000 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=mmesg-qwenvl-new-v9 episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=582961 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=mmesg-qwenvl-orig episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=476051 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=pp-new-hard0 episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=467431 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=pp-new-v9 episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=939008 codec_bad=0 multi_mp4=0 key_mismatch=0
VIDEO_SAVED=PASS route=pp-orig episodes=1 missing=0 decode_fail=0 frame_mismatch=0 raw_left=0 bytes=467431 codec_bad=0 multi_mp4=0 key_mismatch=0
```

## 第一档
```
GEN_REGRESS=PASS set=v9 n=129 match=127 jitter=2 flip=0 structural=0 unknown=0 missing=0 invalid=0
GEN_REGRESS=PASS set=xhard0 n=48 match=48 jitter=0 flip=0 structural=0 unknown=0 missing=0 invalid=0
LOCAL_GEN_DIFF set=v9 n=129 byte_equal=128 diverge=0 gen_fail=1 structural=0 unknown=0
LOCAL_GEN_DIFF set=x0 n=48 byte_equal=47 diverge=0 gen_fail=1 structural=0 unknown=0
```

## 第二档
```
GATE2=INFO policy=mmesg-oracle compared=192 same_terminal=189 identical_trace=101 first_diverge_obs=0 first_diverge_state=0 first_diverge_text=0 first_diverge_action=91 missing=0 first_diverge_stop=0 first_diverge_request=0 duplicate=0 missing_trace=0 bad_trace=0 first_episode_identical=1/3 first_episode_aligned=1 expect_total=192 site=local
GATE2=INFO policy=pp compared=192 same_terminal=192 identical_trace=192 first_diverge_obs=0 first_diverge_state=0 first_diverge_text=0 first_diverge_action=0 missing=0 first_diverge_stop=0 first_diverge_request=0 duplicate=0 missing_trace=0 bad_trace=0 expect_total=192 site=local
GATE2=INFO policy=mmesg-qwenvl compared=192 same_terminal=176 identical_trace=141 first_diverge_obs=0 first_diverge_state=0 first_diverge_text=0 first_diverge_action=51 missing=0 first_diverge_stop=0 first_diverge_request=0 duplicate=0 missing_trace=0 bad_trace=0 first_episode_identical=0/2 first_episode_aligned=1 expect_total=192 site=local
GATE2=INFO policy=mmesg-oracle compared=192 same_terminal=192 identical_trace=180 first_diverge_obs=0 first_diverge_state=0 first_diverge_text=0 first_diverge_action=12 missing=0 first_diverge_stop=0 first_diverge_request=0 duplicate=0 missing_trace=0 bad_trace=0 first_episode_identical=1/1 first_episode_aligned=1 expect_total=192 site=gl
GATE2=INFO policy=pp compared=96 same_terminal=96 identical_trace=96 first_diverge_obs=0 first_diverge_state=0 first_diverge_text=0 first_diverge_action=0 missing=0 first_diverge_stop=0 first_diverge_request=0 duplicate=0 missing_trace=0 bad_trace=0 expect_total=96 site=gl
GATE2=INCOMPLETE policy=mmesg-qwenvl compared=37 same_terminal=37 identical_trace=37 first_diverge_obs=0 first_diverge_state=0 first_diverge_text=0 first_diverge_action=0 missing=2 first_diverge_stop=0 first_diverge_request=0 duplicate=0 missing_trace=0 bad_trace=0 first_episode_identical=1/1 first_episode_aligned=1 expect_total=39 site=gl
```

## 第三档本机复刻对照
```
LG3_COMPARE model=oracle n=800 gl_rows=800 local_rows=800 gl_sr=0.5288 local_sr=0.5175 agree=0.8438 gl_only_success=54 local_only_success=45 mcnemar_p=0.4215 gl_nonterminal=0 local_nonterminal=0
LG3_COMPARE model=pp-shard00-partial n=184 gl_rows=400 local_rows=184 gl_sr=0.1087 local_sr=0.1359 agree=0.8261 gl_only_success=3 local_only_success=8 mcnemar_p=0.2266 gl_nonterminal=1 local_nonterminal=0
LG3_COMPARE model=qwenvl-shard00-partial n=9 gl_rows=9 local_rows=137 gl_sr=0.0000 local_sr=0.0000 agree=0.7778 gl_only_success=0 local_only_success=0 mcnemar_p=1 gl_nonterminal=0 local_nonterminal=0
LG3_RUNTIME_DIFF=PASS（a299fcdc..4b8b4636 运行路径零差异，仅 trace_writer.validate_trace 与 eval_manifest）
LG3_SHARD_SHA=PASS n=8
```

## 席位收尾行（GL）
```
CHAIN_STEP seat=00 step=gate1-v9 rc=0 2026-10-05T00:11:12-04:00
CHAIN_STEP seat=00 step=gate1-x0 rc=0 2026-10-05T00:20:32-04:00
SEAT_REC_SYNC=PASS n=192 bytes=287775523 left=0 seat=00 side=orig transcoded=192 frame_mismatch=0 transcode_fail=0
OFFICIAL_SEAT_DONE seat=00 policy=mmesg-ground-sg-oracle outcome=pass rc=0 dataset=test-hard0 run_name=g2-oracle-s00 host=gl1525.arc-ts.umich.edu 2026-10-05T01:17:27-04:00
SEAT_DONE policy=mmesg-ground-sg-oracle cond=SGEVAL seat=00 done=192 errors=12 infra=0 queue_check=NA loop_exit_status=0
SEAT_REC_SYNC=PASS n=192 bytes=594786596 left=0 seat=00 transcoded=192 frame_mismatch=0 transcode_fail=0
V8_SEAT_DONE seat=00 outcome=pass rc=0 run_name=g2-oracle-s00 dataset=test-hard0 host=gl1525.arc-ts.umich.edu 2026-10-05T02:44:57-04:00
PAIR_SEAT_DONE seat=00 policy=mmesg-ground-sg-oracle orig_rc=0 new_rc=0 rc=0 outcome=pass 2026-10-05T02:44:57-04:00
CHAIN_STEP seat=00 step=gate2 rc=0 2026-10-05T02:44:57-04:00
SEAT_DONE policy=mmesg-ground-sg-oracle cond=V8 seat=00 done=1 errors=1 infra=0 queue_check=NA loop_exit_status=3
SEAT_REC_SYNC=PASS n=0 bytes=0 left=0 seat=00 transcoded=0 frame_mismatch=0 transcode_fail=0
V8_SEAT_DONE seat=00 outcome=fail rc=3 run_name=g3-oracle-s00 dataset=test-hard host=gl1525.arc-ts.umich.edu 2026-10-05T02:45:51-04:00
CHAIN_STEP seat=00 step=gate3 rc=3 2026-10-05T02:45:51-04:00
CHAIN_STOP seat=00 step=gate3 rc=3
SEAT_DONE policy=mmesg-ground-sg-oracle cond=V8 seat=00 done=12 errors=0 infra=0 queue_check=NA loop_exit_status=0
SEAT_REC_SYNC=PASS n=12 bytes=32339580 left=0 seat=00 transcoded=12 frame_mismatch=0 transcode_fail=0
V8_SEAT_DONE seat=00 outcome=pass rc=0 run_name=g2-oracle-swing-s00 dataset=test-hard0 host=gl1525.arc-ts.umich.edu 2026-10-05T13:39:11-04:00
SEAT_DONE policy=mmesg-ground-sg-oracle cond=V8 seat=00 done=800 errors=0 infra=0 queue_check=NA loop_exit_status=0
SEAT_REC_SYNC=PASS n=800 bytes=3767761349 left=0 seat=00 transcoded=800 frame_mismatch=0 transcode_fail=0
V8_SEAT_DONE seat=00 outcome=pass rc=0 run_name=g3-oracle-s00 dataset=test-hard host=gl1525.arc-ts.umich.edu 2026-10-05T13:33:19-04:00
CHAIN_STEP seat=00 step=gate3 rc=0 2026-10-05T13:33:19-04:00
SEAT_CHAIN_DONE seat=00 model=oracle 2026-10-05T13:33:19-04:00
SEAT_REC_SYNC=PASS n=39 bytes=55487747 left=0 seat=03 side=orig transcoded=39 frame_mismatch=0 transcode_fail=0
OFFICIAL_SEAT_DONE seat=03 policy=mmesg-ground-sg-qwenvl outcome=pass rc=0 dataset=test-hard0 run_name=g2-qwenvl-s03 host=gl1525.arc-ts.umich.edu 2026-10-05T17:03:38-04:00
SEAT_DONE policy=mmesg-ground-sg-qwenvl cond=SGEVAL seat=03 done=37 errors=0 infra=4 queue_check=NA loop_exit_status=6
SEAT_REC_SYNC=FAIL n=41 bytes=3954221950 left=0 seat=03 failed=0 transcoded=37 frame_mismatch=0 transcode_fail=4
V8_SEAT_DONE seat=03 outcome=fail rc=6 run_name=g2-qwenvl-s03 dataset=test-hard0 host=gl1525.arc-ts.umich.edu 2026-10-05T21:21:33-04:00
PAIR_SEAT_DONE seat=03 policy=mmesg-ground-sg-qwenvl orig_rc=0 new_rc=6 rc=6 outcome=fail 2026-10-05T21:21:33-04:00
CHAIN_STEP seat=03 step=gate2 rc=6 2026-10-05T21:21:33-04:00
SEAT_REC_SYNC=PASS n=96 bytes=149687618 left=0 seat=01 side=orig transcoded=96 frame_mismatch=0 transcode_fail=0
OFFICIAL_SEAT_DONE seat=01 policy=pp outcome=pass rc=0 dataset=test-hard0 run_name=g2-pp-s01 host=gl1512.arc-ts.umich.edu 2026-10-05T07:49:57-04:00
SEAT_DONE policy=pp cond=SGEVAL seat=01 done=96 errors=0 infra=0 queue_check=NA loop_exit_status=0
SEAT_REC_SYNC=PASS n=96 bytes=294054877 left=0 seat=01 transcoded=96 frame_mismatch=0 transcode_fail=0
V8_SEAT_DONE seat=01 outcome=pass rc=0 run_name=g2-pp-s01 dataset=test-hard0 host=gl1512.arc-ts.umich.edu 2026-10-05T09:54:12-04:00
PAIR_SEAT_DONE seat=01 policy=pp orig_rc=0 new_rc=0 rc=0 outcome=pass 2026-10-05T09:54:12-04:00
CHAIN_STEP seat=01 step=gate2 rc=0 2026-10-05T09:54:12-04:00
SEAT_DONE policy=pp cond=V8 seat=01 done=399 errors=0 infra=3 queue_check=NA loop_exit_status=6
SEAT_REC_SYNC=PASS n=402 bytes=1777764517 left=0 seat=01 transcoded=402 frame_mismatch=0 transcode_fail=0
V8_SEAT_DONE seat=01 outcome=fail rc=6 run_name=g3-pp-s01 dataset=test-hard host=gl1512.arc-ts.umich.edu 2026-10-05T22:08:42-04:00
CHAIN_STEP seat=01 step=gate3 rc=6 2026-10-05T22:08:42-04:00
SEAT_CHAIN_DONE seat=01 model=pp 2026-10-05T22:08:42-04:00
SEAT_REC_SYNC=PASS n=0 bytes=0 left=0 seat=03 side=orig transcoded=0 frame_mismatch=0 transcode_fail=0
OFFICIAL_SEAT_DONE seat=03 policy=mmesg-ground-sg-qwenvl outcome=fail rc=4 dataset=test-hard0 run_name=g2-qwenvl-s03 host=gl1512.arc-ts.umich.edu 2026-10-05T05:27:41-04:00
CHAIN_STOP seat=03 policy=mmesg-ground-sg-qwenvl side=orig rc=4（基础设施／额度／信号，不跑新侧）
PAIR_SEAT_DONE seat=03 policy=mmesg-ground-sg-qwenvl orig_rc=4 new_rc=na rc=4 outcome=fail 2026-10-05T05:27:41-04:00
CHAIN_STEP seat=03 step=gate2 rc=4 2026-10-05T05:27:41-04:00
CHAIN_STOP seat=03 step=gate2 rc=4
```

## 席位收尾行（本机第三档复刻）
```
LG3_START card=1 seat=91 queue=qwenvl:00 qwenvl:01 qwenvl:02 qwenvl:03 qwenvl:04 git=4b8b463678cb9cab90eda61fe702a9b186c6129c 2026-10-05T11:57:36-04:00
LG3_SEG_START item=qwenvl:00 n=160 card=1 2026-10-05T11:57:36-04:00
LG3_START card=0 seat=90 queue=oracle:00 pp:00 pp:01 git=4b8b463678cb9cab90eda61fe702a9b186c6129c 2026-10-05T11:57:36-04:00
LG3_SEG_START item=oracle:00 n=800 card=0 2026-10-05T11:57:36-04:00
[oracle-00] SEAT_REC_SYNC=PASS n=800 bytes=3784431445 left=0 seat=90 transcoded=800 frame_mismatch=0 transcode_fail=0
[oracle-00] V8_SEAT_DONE seat=90 outcome=pass rc=0 run_name=lg3-oracle-00 dataset=test-hard host=sled-vail 2026-10-05T17:57:58-04:00
LG3_SEG_END item=oracle:00 rc=0 2026-10-05T17:57:58-04:00
LG3_SEG_START item=pp:00 n=400 card=0 2026-10-05T17:57:58-04:00
```

## Astra 费用（本机预检）
```
ASTRA_COST usd=0.5149 calls=6 pending=0 projected=0.7179 cap=30
```
