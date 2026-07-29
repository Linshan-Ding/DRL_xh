# DRL_Warehouse

Paper Title: **The Impact of Dynamic Workforce Allocation in Human-Robot Collaborative Warehouses: A Human-Centric Perspective**

本项目用于研究人机协同仓库中的动态资源配置问题，包含统一配置、订单算例生成、PPO 训练入口，以及 A2C / DDPG / TD3 / SAC 对比算法。

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
│   └── default.json             # 唯一默认配置文件
├── agent/
│   ├── ppo/                     # PPO 训练入口与网络
│   └── baselines/               # A2C / DDPG / TD3 / SAC 对比算法
├── data/
│   ├── generate_orders.py       # 订单算例生成入口
│   └── instances/               # items_<n>/orders_m<month>.pkl
├── environment/                 # 仓库仿真环境与实体定义
├── test/                        # 轻量测试
└── result/                      # 实验结果
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
```

## Modes

`configs/default.json` 中的 `experiment.mode` 支持：

```text
short | long | hybrid | fixed_hybrid
```

`fixed_hybrid` 模式下，第一天按 `experiment.fixed_hybrid.long_term_robots` 和 `experiment.fixed_hybrid.long_term_pickers_area` 创建固定长租资源，后续天数由算法调整短租资源。

## Outputs

- PPO: `result/ppo/<mode>_i<items>_m<month>_seed<seed>.csv`
- PPO checkpoint: `result/ppo/<mode>_i<items>_m<month>_seed<seed>.pth`
- PPO best resource configuration: `result/ppo/<mode>_i<items>_m<month>_seed<seed>_best_config.csv`
- Baselines: `result/baselines/<algorithm>_<mode>_i<items>_m<month>_seed<seed>.csv`
- Baseline best resource configuration: `result/baselines/<algorithm>_<mode>_i<items>_m<month>_seed<seed>_best_config.csv`

## Tests

```powershell
python -m py_compile environment\class_public.py environment\class_warehouse.py environment\warehouse_env.py data\generate_orders.py agent\training_utils.py agent\ppo\train.py agent\baselines\common.py
python -m unittest discover -s test -p "test_*.py"
```
