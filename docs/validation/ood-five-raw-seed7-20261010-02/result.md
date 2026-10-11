# 恢复轮结果与接续

## 当前结论

评估已沿原四个GPU及CPU作业恢复，四条正式worker均有新版本真实动作。暂停前已领取结果复用，原三局补发布和搬运完成。监督器退出时GPU继续运行；随后原控制器也因共享盘读取错误自行退出，正在按用户限定的第六次最终修复恢复在途接管。完整五模型交付仍未完成，下文历史状态均为当时快照。

## 暂停后的修复与审查

用户原话：「最多进行五次审查，第五次审查结束后最多进行第六次修复，第六次修复完就直接开始，不能无限期的审查。」「直接开始进入全体的evaluation，继续沿用原来的job。」「不要阻塞不要问用户，尽可能一路走到尾。」依此只修实质阻断，沿原预算、原作业恢复未尝试身份，不再追措辞或重复申请授权。

第一轮六路固定锚审查发现暂停恢复、守卫接续、账本配置、媒体完整性及模型终态协议问题；第二轮复审发现新账本Astra上限与席位CLI不一致，以及SMVLA最后观察故障丢失已发生的成功。两项已修复，第三轮定向静态复审为 `FIX_R5=PASS`，锚点 `e01abc75befb4ff902d0f197e83bfa0b44d6d82a`。独立媒体静态复审 `FIX_R3=PASS`，锚点 `d80d46f59365350bab2d79567470c6fa3c291599`。静态复审不替代真实恢复验证。

原数据和源码快照不覆盖。PP历史S2合成量显式标为 `pp_s2_composite`；连续完整墙钟时间仍为未验证，不把任务执行通过写为完整计时通过。

主分支整合后定向测试退出0：`226 passed in 61.00s`，覆盖账本、原始媒体、席位暂停、SMVLA／GroundSG／FrameSamp／PP协议、计时及外层episode；`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。新增来源模块登记后契约清单测试退出0：`11 passed in 4.53s`，资源守卫全零。命令为 `UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 280s uv run --no-sync python -m pytest -q` 后接上述链路的既有测试文件；清单测试入口为 `tests/static/test_inventory.py`。尚未覆盖待整合生命周期修复与真实集群恢复。

## 暂停后真实恢复及有限修复

生命周期旧夹具首次回归65失败／70通过，未被记为通过。适配真实零Astra、暂停及守卫新契约后，两个测试文件 `137 passed in 108.66s`，退出0且资源守卫全零；新增租约及暂停续段核心夹具 `2 passed in 0.46s`。第三轮固定03e00e4静态 `R1_ROUND3=PASS`。

原CPU两次真实只读预检退出0：`PAUSED_RESUME_PREFLIGHT=PASS accepted=60 active=0 drained=3 published=57 remaining=3940`。第二次核验schema4零Astra，账本仅追加元数据，原字节前缀完整保留。完整执行锚 `33bef4f485ca95b6c43017369220eec7eb7fd668` 与配置SHA `6eb5cdcdf6c0de32df2323f28231b9b25278377be8cac677dce87f186737028e` 固定：`INPUTS_LOCK=PASS files=2277 gitlinks=5 identities=800 tokenizer=1 client_origins=3 server_origins=5 gpu_initializations=0`。NFS强制执行位不能chmod复原，冻结副本沿既有core.filemode=false；SMVLA沿既有cat过滤器，逐文件字节仍等于固定Git blob与SHA，未改模型源码。

监督器07在原CPU步骤24启动控制器1011559及四正式步骤63664391.10／63664392.8／63664393.7／63664404.1。旧三个未发布结果经新媒体及完整状态数组验收搬运，发布／回执补齐60；旧57份仅复核回执、身份与本机全量SHA，最终交付仍会完整复核所有媒体及数组，不把复用写为全部重新解码。

实跑后第四守卫因Slurm查询连续超时30秒而误退出，尽管CPU心跳约1.8秒旧；监督器退出但控制器和GPU继续。旧通知回执阻挡新失败通知，未称邮件送达。第四轮最小修补与静态 `R1_ROUND4=PASS` 后，监督器08严格附着原控制器；查询未知由原心跳有效期裁定，通知改为监督实例独立幂等。真实shell7项／45.43秒、附着及替身MTA12项／0.11秒，退出0且资源全零。

附着约146秒后，共享盘ready_publish写入超过5秒又使守卫及监督器退出。第五轮最后静态 `R1_ROUND5=PASS`，四明确文件阶段的TimeoutError仅在已验证原t／loop_t加120秒内重试；无缓存、过期及身份错仍拒绝。真实write／rename慢夹具5项／45.96秒，意图回归3项／20.21秒，退出0且资源全零。未继续增加审查轮次。

与此同时原控制器因读取pp进度出现 `OSError errno=121` 自行退出，CPU实际只读进程查询证实 `OLD_CONTROLLER_GONE`；09附着正确拒绝失效心跳，没有启动重复控制器。其故障通知通过同CPU投递链提交：`state=submitted rc=0 delivery_verified=false`，原完整回执按实例保留，未声称收件人实收。

四正式worker步骤及PID保持，新动作快照为QwenVL的PickXtimes × xhard1 × 1当前局432步、SwingXtimes × xhard3 × 1当前局503步、BinFill × xhard2 × 1当前局464步，及PP的PickXtimes × xhard1 × 1当前局981步；后续快照仍继续增长，没有基础设施失败结算或重复reserve。它们是进展，不是完整成绩。总授权仍为5模型 × 16任务 × 每任务50身份（依原难度表分配）=4000首试／8000 reset，加旧PP × VideoUnmask × xhard1 × 1首试的1／2，累计上限4001／8002。

后台阶段接续由控制器负责，媒体交付由搬运器负责；原CPU占位始终保留。宿主未提供可建立的自动唤醒工具，代理自动唤醒未建立，不把后台进程存活写为代理无人值守自动修复保证。

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

05实际CPU预检继续拒绝，未启动监督器：步骤清单使用了作业状态字段 `%T`，但当前Slurm步骤模式不支持它。主实查 `squeue --steps -o '%i %j %T'` 给出stderr错误且退出0，stdout第三列为字面 `%T`，不能用于状态判定。改用受支持的 `%i %j` 两列清单，stderr空、合法登记StepID与唯一名称均必检；实际RUNNING／COMPLETED／退出码继续严格取精确sacct，不从列表推断状态。原三个formal仍运行，未重复启动或领取预算；05失败日志保留。

步骤格式补充修复固定 `52b5b2f43d597ed8ae51ab9a5f4b524b95fbe545`，`SLURM_STEP_FORMAT_REVIEW=PASS`；主整合控制器及契约门禁 `108 passed in 18.68s`、退出0，资源守卫全零。旧首局和新在途恢复共用正确步骤清单，原预算／来源／进程绑定不放宽。

## 06真实接管与交接边界

实际编排为已推送 `e39c6d4a407bb81f49f6f293a0be86b4f9a03cf3`，六文件束 `orchestration-recovery-06/` 两端SHA和四模块独立导入通过。原CPU真实预检退出0：`EXISTING_DISPATCHES_PREFLIGHT=PASS active=3 completed=0 published=5`，严格复用原步骤63664391.5／63664392.5／63664393.4。新监督器734024、控制器734113启动，三个新守卫分别为63664391.7／63664392.6／63664393.5；原srun683623／683630／683637与client2247115／2247129／2247043保持，没有重复启动。

本机读取真实已序列化回执、保留小result、预算共享锁快照与最新心跳，短检查退出0：`RAW_SMOKE=PASS models=5 task=VideoUnmask tier=xhard1 first_tries=5 infra=0 media=5 success=4`；`FIRST_DISPATCH=PASS model=qwenvl step=63664391.5 actions=1379`；`RUN_HEARTBEATS=PASS fresh=3 errors=0 mover_alive=1 supervisor_session_alive=1 old_cpu_wrapper_session_alive=1`。首动作证明为原PickXtimes首局result，正常任务失败不重试。

快照尝试乘式：Oracle／FrameSamp／PP各1任务VideoUnmask × 1档xhard1 × 1首试；QwenVL为1任务PickXtimes × 1档xhard1 × 2首试 + 1任务StopCube × 1档xhard5 × 1首试 + 1任务SwingXtimes × 1档xhard1 × 6首试 + 1任务VideoUnmask × 1档xhard1 × 1首试；SMVLA为1任务PickXtimes × 1档xhard1 × 5首试 + 1任务VideoUnmask × 1档xhard1 × 1首试。新轮合计19／38，加旧PP × 1任务VideoUnmask × 1档xhard1 × 1首试的1／2，累计20轨迹／40reset，零重试；只是该时点快照，运行会正常增长，上限仍4001／8002。

外部SSH追加查询出现 `Session open refused by peer`，随后普通客户端尝试新认证并报失败。核对greatlakes.md及当前状态：`ssh -O check greatlakes` 退出0、Master PID1709683存活，TCP到GreatLakes22端口仍ESTAB；主连接并未消失。问题是新增命令会话被拒，容量／服务器策略原因尚未确证，不将其表述为连接过期或要求用户先重新认证。未关闭主连接或任何既有会话。最后Slurm状态刷新尚未取得：`AGENT_HANDOFF=INCOMPLETE missing=fresh_slurm_job_query ssh_session_refused=1 automatic_agent_wakeup=0`，不伪造全通过。

后台CPU调度与搬运继续；故障邮件通道已由用户实收探针验证，后续故障邮件仅证提交，不逐封声称实收。当前宿主未建立代理自动唤醒，不能承诺代理无人值守自行修复异常。恢复可用新SSH命令会话后补最后查询；不重跑GPU或重复申请CPU许可。

用户追问：「为什么会有这个问题？你检查一下 Great Lakes Markdown 里面的 SSH 的这个链接是不是现在已经没有了吗？就是链接已经没有了吗。」「活动连接已经没有了吗？」据greatlakes.md的ControlMaster规则重新只读核验：主进程、控制socket和TCP ESTAB均在，不属于主连接丢失。当前/proc扫描发现主进程10个唯一输出pipe，其中8条暂无可见本机reader、2条有本轮ssh/tee存活端点；不能把10pipe直接换算10SSH会话，也不能据此确认远端MaxSessions设置或擅自清理。先前请用户恢复登录不够准确，已纠正为无需先重新认证；新会话被拒的具体远端原因尚未确证。
