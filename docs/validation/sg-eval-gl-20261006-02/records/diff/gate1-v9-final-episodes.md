# gen-regress 逐局报告：v9

- 判定行：`GEN_REGRESS=PASS set=v9 n=129 match=126 jitter=2 flip=1 structural=0 unknown=0 missing=0 invalid=0 rerun_rows=4 filler=3 noise=1 regression=0 env_changed=0 unstable=0 noise_max=2`
- 参照：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-noise/scripts/configs/noise-ref-20261003.json`（sha256 `90c0ea527dd7`）
- 首跑：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/gate1/g1-new-v9`；改后第二次：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/gate1/g1-new2-v9`；旧代码：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/gate1/g1-old2-v9`

| 任务 | 档 | seed | 参照类 | 陪跑 | 第一次 | 改后第二次 | 旧代码 | 一=二 | 一=旧 | 二=旧 | 分叉步 | 子目标 | 定性 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| StopCube | xhard4 | 22200000 | stable | 是 | `d4b7aaab77c2` 成功 | `d4b7aaab77c2` 成功 | `d4b7aaab77c2` 成功 | 同 | 同 | 同 | — | — | 陪跑（只报告）：改后同基线、旧码同基线 |
| StopCube | xhard4 | 22200100 | stable | 是 | `dd1bae03cd98` 成功 | `dd1bae03cd98` 成功 | `dd1bae03cd98` 成功 | 同 | 同 | 同 | — | — | 陪跑（只报告）：改后同基线、旧码同基线 |
| StopCube | xhard4 | 22200200 | stable | 是 | `732002d8cfe1` 成功 | `732002d8cfe1` 成功 | `732002d8cfe1` 成功 | 同 | 同 | 同 | — | — | 陪跑（只报告）：改后同基线、旧码同基线 |
| BinFill | xhard1 | 16400000 | jitter |  | `26c4d449632e` 失败 | — | — | — | — | — | — | — | 已知抖动（只报告：fail） |
| InsertPeg | xhard4 | 23300100 | stable |  | `019f7a98bd2d` 成功 | `2e368975398e` 成功 | `2e368975398e` 成功 | 异 | 异 | 同 | — | — | 噪声 |
| MoveCube | xhard4 | 23400200 | jitter |  | `aa414cf9f8dc` 成功 | — | — | — | — | — | — | — | 已知抖动（只报告：same_as_a） |
