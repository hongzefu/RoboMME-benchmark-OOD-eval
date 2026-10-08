# gen-regress 逐局报告：v9

- 判定行：`GEN_REGRESS=PASS set=v9 n=129 match=127 jitter=2 flip=0 structural=0 unknown=0 missing=0 invalid=0`
- 参照：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-sgeval/scripts/configs/noise-ref-20261003.json`（sha256 `90c0ea527dd7`）
- 首跑：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261004/gate1/g1-new-v9`；改后第二次：`—`；旧代码：`—`

| 任务 | 档 | seed | 参照类 | 陪跑 | 第一次 | 改后第二次 | 旧代码 | 一=二 | 一=旧 | 二=旧 | 分叉步 | 子目标 | 定性 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BinFill | xhard1 | 16400000 | jitter |  | `26c4d449632e` 失败 | — | — | — | — | — | — | — | 已知抖动（只报告：fail） |
| MoveCube | xhard4 | 23400200 | jitter |  | `aa414cf9f8dc` 成功 | — | — | — | — | — | — | — | 已知抖动（只报告：same_as_a） |
