# audio-noise-gate

## 题面

给直播加噪声门后，说话开头被吃掉、底噪还一抽一抽：电平按整段均值计算，门限对突发噪声失效；单阈值开关，临界点每秒抖 10 次；开关增益瞬时切换，出现咔哒；attack/release 用同一时间常数，关闭太快；保持时间按样本数，采样率变化时保持时长变；侧链 depth 不钳制，超过 1 后反相；底噪估计用最小值，安静段被判成底噪；open_db < close_db 不校验，状态机死锁；保持计数可为负，提前关闭；批量共用状态机，轨道之间串。gate/noisegate.py 是核心，tests/test_noisegate.py 15个红。要修的是：把整段均值改为按窗滑动、把单阈值改为双阈值滞回、把瞬时增益改为 attack/release 平滑、让 attack 与 release 使用各自的时间常数、把保持时间从样本数改为毫秒换算、把侧链 depth 钳制到 0..1、把底噪最小值改为分位数估计、校验 open_db > close_db、把保持计数钳制为非负、让批量每轨使用独立状态机并保证失败时整体回滚。

## 运行方式

```powershell
cd D:\gsb-20261119\audio-noise-gate
python -m pytest tests -q
```

初始失败环境应恰好输出 `15 failed`，没有 collection/import/syntax error。15 个测试全部直接调用 `gate/noisegate.py` 里的桩函数，修复实现后应全部转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/` 不可修改。实现只使用标准库，离线、无随机、不 sleep；信号统一为 `list[float]`，电平单位为 dBFS，时间单位为毫秒，浮点比较容差为 `1e-6`。
