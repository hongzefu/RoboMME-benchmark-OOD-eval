# 恢复轮结果与接续

## 当前结论

原三个正式首局均正常完成并完整核验，控制器第一次接续失败已留存；CPU 和原 GPU 占位保留，没有重跑。修补后的接续正在验证，完整五模型结果尚未完成。

| 模型 | 范围 | 实际步数 | task_success | infra |
| --- | --- | --- | --- | --- |
| SMVLA | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 269 | 1 | false |
| GroundSG QwenVL | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 440 | 1 | false |
| PP | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 276 | 1 | false |

新轮当前 3模型 × 1任务 × 1档 × 1首试 = 3轨迹／6 reset；加旧轮 PP × 1任务 × 1档 × 1首试 = 1／2，累计 4／8，上限4001／8002。余下两个模型的正式首局和正式分片尚未起跑；没有额外基础设施轨迹重试。

## 第一次接续失败

旧执行源 `656c591f32238703c54967d0f550220cf0cec074`、配置SHA `5cc29c377a607186a6a21b27791b02eec480b2fd7bd2f390ff64043a07f4c9b3`。PP结果及文件mtime早于控制器异常约19.35秒；控制器在 `Controller.scan` 对退出0立即检查已发布集合，没有最终补读确认窗口，报“worker正常退出但缺结果”。路径和标签契约正确；具体NFS可见延迟与扫描竞态尚不能区分。

控制器自然退出，监督器随后退出1，独立故障邮件提交退出0；CPU bootstrap持续占位，四 GPU 未取消。SMVLA和QwenVL既有步骤继续自行完成，三个原步骤 `63664391.1 / 63664392.1 / 63664393.1` 的sacct均 `COMPLETED|0:0`。未建立首正式分片交接，不把后台占位当完成。

## 数据核验与录像口径

三个原始目录已rsync到本机 `artifacts/ood-five-raw-seed7-20261010-02/preserved-gl/`，冻结MOVER完整解码、轨迹与数组闭环及逐文件GL源SHA比对退出0：`INITIAL_RAW=PASS models=3 task=VideoUnmask tier=xhard1 first_tries=3 source_hashes=matched`。解码前／腕帧数：PP248／286，SMVLA248／279，Qwen412／450；不把帧数与动作步数强行等同。

真实CPU只读恢复预检发现Qwen `raw/official/*.mp4` 后拒绝，未发布或启动任何新worker。外层 `--no-render` 仅跳过网站渲染，GroundSG的原外层调用固定 `keep_official=True`；上游原循环始终临时编码MP4，没有原生禁编码开关。

用户明确选择：「只交付原始录像，允许内部临时 MP4」。原Qwen官方目录两个文件已逐文件SHA／字节数核验后移存到本轮NFS `legacy-official/`，未删除、未作为原始交付；原本机全体备份保留。随后旧上下文真实CPU只读预检退出0：`RESUME_PREFLIGHT=PASS models=3 identities=3 budget_sha256=3d8b1e492f5538cfbaf5bd33aaba45973757c6cc0b271d5cf3b7a7ab90b56404`，原账本字节不变。

## 用户授权与修补状态

用户原话：「已收到该探针邮件」，投递链路已由用户实收验证；「可以再次启动这个CPU的任务。你可以随意地启动CPU的任务，不需要每次都问用户。」必要CPU接续按本授权直接执行，不反复问许可；GPU重跑和预算不因此扩展。

私有修补B4/B5及修正提交已合并推送 `dee47267a1cd85193e647824815a8a6e82d3530b`。退出后报告确认、复用已结束首局和补证／恢复期心跳闭环通过固定审查；主会话定向回归 `143 passed, 2 deselected, 23 warnings in 36.72s`，退出0，`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。确认循环120秒无法抢占内核阻塞IO，未声称已重现真实NFS触发。

外层GroundSG保存开关已完成独立审查：默认True兼容旧行为；GL `--no-render` 强制False，原循环临时视频生成清理不改，官方视频不保留到raw。接续新worker版本及原三个旧来源分开绑定，旧执行快照与原配置保持原字节；最终执行版本、配置与交接实测完成后继续追加。

接续编排B7同时绑定新增配置／ready路径和旧三局来源，守卫明确读取新ready，复用旧已实收探针仅在通知通道完全一致时允许，原回执不改。固定 `11e22c32982b9d7eb5bac9786fdbef244bc8a066` 的 `INPLACE_NEW_CONTEXT_REVIEW=PASS`；操作前必须移存旧guard-ready和旧failure通知，并给搬运器新配置上下文。主会话整合验证：`177 passed, 2 deselected, 23 warnings in 50.41s`、另官方适配 `34 passed in 23.45s`，两者退出0、资源守卫全零；`PUBLIC_LANG=PASS files=1 cjk_hits=0 binary_skipped=0`。未把静态审查代替真实接续预检。

## 新上下文第一次恢复失败

实际新执行快照为 `8c3c84575e2a07727931c698b6ca17bedf495bac`，配置SHA为 `bbe580d4948cd6e07b4ded82f633b0786b0d1d245c9dfeaa01b7de9b09e0f7f2`；快照2224个文件与5个gitlink核验通过，真实CPU新上下文只读预检复用三个已结束首局通过。原配置和执行快照未修改。

人工监督启动后，健康检查本身返回0，但守卫Shell仅凭旧 `guardian-intent-accepted-<GPU>.json` 文件存在就退出0。该回执仍属于旧配置／旧监督PID；先前移存清单漏列此文件，导致监督器报“独立守卫提前退出”。失败邮件提交退出0，不代表再次实收。监督器直接作为CPU的srun任务退出后，Slurm也清理了同一步骤内的控制器；真实CPU `ps -p 649098,649147` 无进程，不能把Python没有主动kill表述为控制器仍存活。

本次没有发布结果或派发新worker，仍只有原三个smoke调用目录。预算SHA仍为 `3d8b1e492f5538cfbaf5bd33aaba45973757c6cc0b271d5cf3b7a7ab90b56404`。四GPU占位与CPU占位保留，新搬运会话 `ev-ood-five-raw-mover-resume-20261010-02` 持续运行。第二次恢复先把15个明确失败／旧回执活动槽逐文件原字节移存：`FAILURE_ARCHIVE=PASS files=15 original_bytes=preserved`，存于运行根 `control/attempts/resume-01-failure/`；本机镜像为 `control/resume-01-failure/`。

两个机制已修复：健康守卫忽略历史接受回执，仅由本次严格验证的停止信号返回专用状态10后退出；CPU恢复入口只启动一次监督器，退出后保持srun步骤，以保留控制器。两处均不重跑原三局、不增加reset／轨迹预算、不停止GPU。固定代码 `657db86b29d82f24471d202cbfa2525525a5aaeb` 的 `GUARD_CPU_HOLD_REVIEW=PASS`；10项夹具20.89秒及3项原超时回归35.14秒退出0，资源计数全零。用户主动终止CPU步骤／CPU作业仍会由Slurm清理其子进程，不承诺这类情况下保留控制器。后续真实运行结果另行记录。

主整合定向命令 `uv run --no-sync python -m pytest tests/unit/test_raw_ood_supervisor.py tests/unit/test_raw_ood_controller.py tests/static/test_inventory.py -q` 首次得到99通过、2失败，88.72秒；两项均为新增wrapper在契约总表缺文件索引，控制链路未失败。补齐总表索引和delta后仅复跑该门禁：`11 passed in 4.36s`、退出0，`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。未放宽判据，未重复无关仿真。
