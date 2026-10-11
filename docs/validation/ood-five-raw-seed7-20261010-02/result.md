# 恢复轮结果与接续

## 当前结论

五个正式首局均完成执行与原始媒体核验，正常任务失败独立记分；原三局复用没有重跑。正式分片已派发，之后控制器／监督器因第四排队GPU的调度查询超时退出，三个正式worker保留运行；附着恢复与真实首步交接待验。CPU与原GPU占位保留，完整五模型结果尚未完成。

| 模型 | 范围 | 实际步数 | task_success | infra |
| --- | --- | --- | --- | --- |
| SMVLA | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 269 | 1 | false |
| GroundSG QwenVL | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 440 | 1 | false |
| PP | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 276 | 1 | false |
| GroundSG Oracle | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 329 | 1 | false |
| FrameSamp＋Modulation | 1任务 VideoUnmask × 1档 xhard1 × 1首试 | 107 | 0 | false |

首局阶段为新轮5模型 × 1任务 × 1档 × 1首试 = 5轨迹／10 reset；加旧轮 PP × 1任务 × 1档 × 1首试 = 1／2，首局阶段累计6／12，上限4001／8002。正式分片另计在同一获批预算，交接时的完整账本快照见后文；没有额外基础设施轨迹重试。

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

## 第二次人工恢复真实起跑

编排提交 `ae44271248931eb7e1aa9eb9e7ab93dd171b3db7` 已推送，独立五文件束在本机／NFS与锁逐SHA匹配：`RECOVERY_BUNDLE=PASS copies=2 files=5 config=1`。真实CPU预检退出0：`RESUME_PREFLIGHT=PASS models=3 identities=3 budget_sha256=3d8b1e492f5538cfbaf5bd33aaba45973757c6cc0b271d5cf3b7a7ab90b56404`。

新CPU步骤 `63664373.12` 在gl3010运行，wrapper PID665775、监督器665797、控制器665851；三个新守卫在gl1505的 `.3` 步骤持续运行，PID2235012／2235028／2235058。原三个已完成首局成功复用发布并获得本机完整媒体搬运回执。Oracle `63664391.4`、FrameSamp `63664392.4` 分别起跑新模型首局；尚未完成，不把健康服务或context_load心跳当推理通过。新搬运PID2794202保持、无活动错误，没有自动重试或资源取消。

随后Oracle329步、FrameSamp107步正常完成，任务成功分别为1／0、infra均false；五份回执均完成SHA、前／腕解码和数组／动作闭环。Oracle前／腕302／339帧，FrameSamp82／117帧；与原三个首局同样不以帧数等同步数。新Oracle `official_source=disabled`，原始交付目录无MP4，符合允许内部临时MP4、仅交付原始录像的用户裁决。FrameSamp正常任务失败保留，不重试。

同CPU真实Slurm纯CPU夹具退出0：`SLURM_CPU_HOLD=PASS supervisor_rc=1 wrapper_alive=1 child_alive=1 retries=0 gpu_actions=0`。夹具未调用仿真、邮件或账本；仅结束自身精确PID，未触碰真实CPU步骤／GPUworker。日志在records。本机夹具和实际Slurm两种证据分开保留。

五首局搬运齐后，控制器自动派发三个正式分片：QwenVL的qwenvl-00、qwenvl-01，SMVLA的smvla-00；第四GPU仍排队。正式首步／交接验收另记，不把已派发当已执行动作。

## 正式派发后的调度查询失败

控制器对第四个排队GPU63664404执行 `squeue -h -j 63664404 -o %T` 超过单次5秒，记录 `TimeoutExpired` 并退出；监督器也退出1并提交故障邮件。CPU步骤仍由wrapper665775保持，不取消任何GPU、不结束现有formal worker。此次证明了与第一次人工恢复不同：监督器退出后，三个正式srun进程仍存活。

真实只读定位：CPU侧srun PID683623／683630／683637均由PID1接管，精确cmdline与dispatch逐项一致；对应GPU步骤为qwenvl-00的63664391.5、qwenvl-01的63664392.5、smvla-00的63664393.4，clientPID2247115／2247129／2247043，均仍存活。观察时全部context_load，不能当作已有动作。预算当时仍只有五首局5reserve／10reset_claim／5commit。

15个明确活动失败／守卫回执已逐文件原字节移存至运行根 `control/attempts/resume-02-query-failure/`，原派发、progress、结果、账本及CPU旧wrapper不动：`QUERY_FAILURE_ARCHIVE=PASS files=15 original_bytes=preserved active_dispatches=preserved`。恢复不得重新走首局发布或重新启动现有三个正式分片，须附着其现有进程，重新构建发布／完成／待派发状态，并保持查询超时不把未知状态当可派发。

在途接管代码固定 `b0861b2c4d7bf5369a5df7b9187173ca4eceb5e6`，独立 `EXISTING_DISPATCHES_REVIEW=PASS`；113项回归86.65秒和最终30项定向3.78秒退出0、资源全零。新增只读预检与明确接管入口，严格复用已有发布和原派发，附着仅监测；单次查询5秒保持，连续查询失败观察最长30秒并冷却，未知状态不派发。真实gl3010预检与接管结果另行追加。

主整合控制器／监督器／契约门禁真实回归：`126 passed in 91.21s`、退出0，`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。无需仿真，未消耗运行预算。

两次真实接管预检均拒绝，未启动新监督器：03私有文件束漏列控制器新增依赖eval_video_mover，报ModuleNotFoundError；主保留原五文件束，另从同7d60ab6提交导出包含该依赖的六文件束04，独立导入闭包通过。04随后发现消费者把真实 `decoded={front:帧数,wrist:帧数}` 错读为布尔True；已有五媒体回执全为合法计数字典，不能改回执迁就消费者，须修消费者并用生产回执序列化补验证。两次预检日志保留，未重启GPU、未重领轨迹或reset。

补充修复固定 `22c5cb56e088bdb7239bef511d793f64f435313c`：decoded必须是正好front／wrist两键、各为非布尔正整数，trace_checked仍必须True；其余字段严格绑定保持。同时修正 `/proc/cmdline` 解析仅剔除末尾NUL、保留 `--gpus ''` 的合法内部空参数，不改原派发或对比规则。生产搬运生成真AV1／NPZ／trace、rsync核验删除、原子JSON写读再预检的正负链路，以及真实CPU子PID空argv验证，29项10.22秒通过；独立 `CONSUMER_ARGV_REVIEW=PASS` 直接核对生产协议。主整合控制器及契约门禁 `97 passed in 17.91s`、退出0、资源计数全零。下一私有文件束05完整包含六文件并验证独立导入，真实CPU再预检，不覆盖03／04失败束。

提交1.112前的默认diff检查报告观察日志四行末尾空格；该命令序列没有及时阻止提交，不应表述为当时默认diff检查通过。此处已将Git内该日志副本四行末尾空格去掉，原日志字节仍在运行根control完整保留（26333字节、SHA256 `192482a0215aa92acc64867316b91b21a40f40bb9a71135a284881be3ceb5c5c`），JSON／失败回执不改；本次 `git diff --check` 退出0。没有改写已有提交或原始运行证据。
