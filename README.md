# DRL_Warehouse

Paper Title: **The Impact of Dynamic Workforce Allocation in Human-Robot Collaborative Warehouses: A Human-Centric Perspective**

本项目用于研究人机协同仓库中的动态资源配置问题，包含统一配置、订单算例生成、PPO 训练入口，以及 A2C / DDPG / TD3 / SAC / DQN / DDQN 对比算法。

## Configuration

`configs/default.json` 是项目中已有配置参数的唯一数值来源。训练模式、算例、episode、随机种子、路径、设备、PPO 参数、仓库参数、成本参数和 `fixed_hybrid` 固定长租参数都只能通过修改该文件控制。

运行入口会打印当前配置来源：

```text
Using config: <repo>/configs/default.json
```

不再支持 `--config`、`DRL_WAREHOUSE_CONFIG`，也不再支持通过 CLI 覆盖 `default.json` 中已有的参数。

## Project Structure

```text
DRL_Warehouse/
├── configs/
│   ├── default.json                  # 唯一默认配置文件
│   └── fixed_hybrid_batch.json       # fixed_hybrid 批量训练的逐月长租参数
├── agent/
│   ├── ppo/                          # PPO 训练入口与网络
│   └── baselines/                    # A2C / DDPG / TD3 / SAC / DQN / DDQN
├── data/
│   ├── generate_orders.py            # 订单算例生成入口
│   └── instances/                    # items_<n>/orders_m<month>.pkl
├── environment/                      # 仓库仿真环境与实体定义
├── test/                             # 轻量测试
└── result/                           # 实验结果
```

## Setup

```powershell
cd DRL_Warehouse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy gymnasium visdom pygame
```

## Generate Orders

订单商品数范围、月份范围、输出目录和随机种子均来自 `configs/default.json`：

```powershell
python -m data.generate_orders
```

默认输出结构：

```text
data/instances/items_2/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_4/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_6/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_10/orders_m01.pkl ... orders_m12.pkl
```

## Train PPO

训练模式、算例、episode、seed、输出目录和设备均来自 `configs/default.json`：

```powershell
python -m agent.ppo.train
```

## Train Baselines

baseline 的训练模式、算例、episode、seed、输出目录和设备同样来自 `configs/default.json`。各算法独有超参数仍可通过 CLI 调整：

```powershell
python agent\baselines\a2c.py
python agent\baselines\a2c.py --lr 0.0003
python agent\baselines\ddpg.py --actor-lr 0.0001 --critic-lr 0.001
python agent\baselines\td3.py --policy-delay 2
python agent\baselines\sac.py --alpha-lr 0.0003
python agent\baselines\dqn.py --learning-rate 0.0003
python agent\baselines\ddqn.py --target-update-interval 100
```

六个对比算法（A2C / DDPG / TD3 / SAC / DQN / DDQN）使用与 PPO 相同的状态归一化尺度，但不会修改或覆盖 PPO 实现及其已有结果。长租模式的主要训练规则如下：

- A2C 跨 64 个单步 episode 累积一个更新批次，避免单样本 advantage 归零。
- DDPG / TD3 / SAC / DQN / DDQN 在 64 个 transition 后结束纯随机探索阶段，第 65 个长租 episode 起使用已学习策略并按各算法的探索规则继续采样。
- DQN / DDQN 使用多离散分支 Q 网络，分别选择机器人和各区域拣货员动作，避免枚举全部联合动作；DDQN 使用在线网络选取下一动作、目标网络评估该动作。
- long 模式和 hybrid 第一次决策的动作表示绝对资源数；机器人范围为 1~15，每个区域的拣货员范围为 1~5。
- 最后 10% episode 关闭探索并冻结参数更新，训练 CSV 尾段直接反映最终确定性策略。
- baseline 不执行独立评估，也不保存模型 checkpoint；结果来自逐 episode CSV 和训练过程生成的 `_best_config.csv`。
- `_best_config.csv` 只从最后 10% 的确定性收敛段中选择，不使用前期随机探索的偶然最小成本。

`--learning-starts` 可调整 DDPG / TD3 / SAC / DQN / DDQN 的纯随机探索 transition 数；旧参数名 `--warmup-steps` 作为兼容别名继续可用。

## Modes

`configs/default.json` 中的 `experiment.mode` 支持：

```text
short | long | hybrid | fixed_hybrid
```

`fixed_hybrid` 模式下，第一天按 `experiment.fixed_hybrid.long_term_robots` 和 `experiment.fixed_hybrid.long_term_pickers_area` 创建固定长租资源，后续天数由算法调整短租资源。

## Batch Training

当需要遍历多个月份或多个模式进行实验时，可使用批量训练脚本。两个批量脚本均基于 `ProcessPoolExecutor` 实现多进程并行，每轮训练在独立进程中运行，互不干扰 PyTorch/CUDA 状态。

### PPO Batch: `agent.ppo.train_batch`

遍历 `月份 × 模式` 的组合，对每种组合调用单次 `train()`。默认覆盖 1~12 月 × short / long / hybrid 共 36 个任务。

```powershell
# 顺序执行（默认）
python -m agent.ppo.train_batch

# 4 进程并行，每个进程限 2 个 CPU 线程
python -m agent.ppo.train_batch --workers 4 --cpu-threads 2

# 12 进程并行，单线程（适合纯 CPU 环境，避免线程竞争）
python -m agent.ppo.train_batch --workers 12 --cpu-threads 1

# 指定月份和模式
python -m agent.ppo.train_batch --months 1,3,5 --modes short,long

# 预览任务列表（不执行训练）
python -m agent.ppo.train_batch --dry-run

# 强制 CPU 模式（避免多进程抢占同一 GPU）
python -m agent.ppo.train_batch --workers 4 --force-cpu
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workers` | `2` | 并行进程数 |
| `--cpu-threads` | `2` | 每个进程的 PyTorch 线程数；高并发时建议设为 `1` |
| `--force-cpu` | 关闭 | 强制所有进程使用 CPU，通过 `CUDA_VISIBLE_DEVICES=""` 屏蔽 GPU |
| `--months` | `1,2,...,12` | 逗号分隔的月份列表 |
| `--modes` | `short,long,hybrid` | 逗号分隔的模式列表（不含 `fixed_hybrid`） |
| `--dry-run` | 关闭 | 仅打印任务列表，不执行训练 |

进程数 × 线程数超过 CPU 核心数时，脚本会打印警告提示。训练结束后输出汇总：成功/失败数量及总耗时。

### PPO Fixed Hybrid Batch: `agent.ppo.train_fixed_hybrid`

专门针对 `fixed_hybrid` 模式，遍历 1~12 月，每个月份可使用不同的长租资源参数。逐月的 `long_term_robots` 和 `long_term_pickers_area` 在 `configs/fixed_hybrid_batch.json` 中配置。

```powershell
# 使用默认配置文件
python -m agent.ppo.train_fixed_hybrid

# 自定义配置文件
python -m agent.ppo.train_fixed_hybrid --config path/to/my_config.json

# 4 进程并行
python -m agent.ppo.train_fixed_hybrid --workers 4 --cpu-threads 1

# 指定月份
python -m agent.ppo.train_fixed_hybrid --months 1,6,11,12

# 预览
python -m agent.ppo.train_fixed_hybrid --dry-run
```

不同于 `train_batch`，此脚本仅遍历月份（模式固定为 `fixed_hybrid`），未提供 `--modes` 和 `--force-cpu` 参数。

### Baseline Batch: `agent.baselines.train_batch`

遍历指定算法 + 模式下的 1~12 月，每轮训练通过 `importlib` 动态加载对应算法模块并调用其 `train()` 入口。

```powershell
# DDPG long 模式，全部 12 个月，2 进程
python -m agent.baselines.train_batch --algorithm ddpg --mode long

# A2C short 模式，仅 6 月和 9 月，4 进程
python -m agent.baselines.train_batch --algorithm a2c --mode short --months 6,9 --workers 4

# TD3 hybrid 模式，自定义 episodes
python -m agent.baselines.train_batch --algorithm td3 --mode hybrid --months 9 --episodes 1000

# SAC fixed_hybrid 模式，预览
python -m agent.baselines.train_batch --algorithm sac --mode fixed_hybrid --dry-run

# DDQN long 模式，全部 12 个月
python -m agent.baselines.train_batch --algorithm ddqn --mode long
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algorithm` | **必填** | 算法名：`a2c` / `ddpg` / `td3` / `sac` / `dqn` / `ddqn` |
| `--mode` | **必填** | 训练模式：`short` / `long` / `hybrid` / `fixed_hybrid` |
| `--months` | `1,2,...,12` | 逗号分隔的月份列表 |
| `--episodes` | 配置文件中的值 | 覆盖训练 episode 数 |
| `--workers` | `2` | 并行进程数 |
| `--cpu-threads` | `2` | 每个进程的 PyTorch 线程数 |
| `--force-cpu` | 关闭 | 强制 CPU 模式 |
| `--dry-run` | 关闭 | 仅预览不执行 |

### 并行与线程说明

所有批量脚本使用 Python `multiprocessing` 的 `"spawn"` 上下文启动子进程（Windows 必须，Linux/Mac 同样安全）。每个子进程拥有独立的 Python 解释器和 PyTorch 上下文，避免了 GIL 和 CUDA 状态共享问题。

每个子进程的 PyTorch 线程数通过 `torch.set_num_threads()` 和 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 环境变量控制。建议 `workers × cpu-threads ≤ CPU 核心数`，超出时可能因线程过订阅导致性能下降。

单 GPU 环境建议加 `--force-cpu` 避免多进程争抢同一 GPU；多 GPU 环境可自行修改代码按进程分配 `CUDA_VISIBLE_DEVICES`。

## Outputs

- PPO: `result/ppo/<mode>_i<items>_m<month>_seed<seed>.csv`
- PPO checkpoint: `result/ppo/<mode>_i<items>_m<month>_seed<seed>.pth`
- PPO best resource configuration: `result/ppo/<mode>_i<items>_m<month>_seed<seed>_best_config.csv`
- Baselines: `result/baselines/<algorithm>_<mode>_i<items>_m<month>_seed<seed>.csv`
- Baseline best resource configuration: `result/baselines/<algorithm>_<mode>_i<items>_m<month>_seed<seed>_best_config.csv`

批量运行 baseline 时，月份 `m` 使用 `experiment.seed + m - 1` 作为训练 seed；五个 off-policy 算法和 A2C 在同一月份仍使用相同 seed，保证横向比较可复现。

## Tests

```powershell
python -m py_compile environment\class_public.py environment\class_warehouse.py environment\warehouse_env.py data\generate_orders.py agent\training_utils.py agent\ppo\train.py agent\baselines\common.py agent\baselines\dqn.py agent\baselines\ddqn.py agent\baselines\train_batch.py
python -m unittest discover -s test -p "test_*.py"
```
