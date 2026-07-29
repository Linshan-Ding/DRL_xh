# DRL_Warehouse

> **Paper:** *The Impact of Dynamic Workforce Allocation in Human-Robot Collaborative Warehouses: A Human-Centric Perspective*

基于深度强化学习的人机协同仓库动态资源配置研究。项目以拣选仓库为背景，将机器人搬运与拣货员协助相结合的拣选流程建模为序列决策问题，使用 PPO 及其变体（A2C / DDPG / TD3 / SAC）联合优化各作业区域的机器人数量和拣货员数量，在最小化运营成本（机器人运行 + 拣货员雇佣 + 订单延期惩罚）的同时保证订单履约率。

---

## 仓库环境建模

### 物理布局

仓库由 3 个作业区域（`area_num=3`）组成，每个区域含 3 条巷道（`aisle_num=3`），每条巷道两侧各有 20 个储货位（`shelf_capacity=20`）× 3 层货架（`shelf_levels=3`），即每个拣货位对应 6 个拣货面（`left` / `right` × 3 层），仓库共有 **540 个可拣选货位**。

```
 ┌─────────────────────────────────────────────────────┐
 │  Area 1          Area 2          Area 3             │
 │  ┌─┬─┬─┐        ┌─┬─┬─┐        ┌─┬─┬─┐             │
 │  │ │ │ │  ...   │ │ │ │  ...   │ │ │ │  ...        │
 │  └─┴─┴─┘        └─┴─┴─┘        └─┴─┴─┘             │
 │                    ↑ 巷道 (aisle)                    │
 │  ████████████████████████████████████████  Depot     │
 └─────────────────────────────────────────────────────┘
```

### 作业流程

1. **订单到达** — 按泊松过程随机生成，每个订单包含 2~10 件商品，分布在仓库不同拣货位
2. **机器人分配订单** — 空闲机器人被分配未分配订单，规划拣货位序列，移动到目标拣货位排队
3. **拣货员协助** — 拣货员在各区域的待分配拣货位与机器人之间协调，依次为排队机器人拣选商品
4. **返回 Depot** — 机器人完成所有商品拣选后返回 Depot 打包出库
5. **订单延期惩罚** — 超过交期（due_time）的订单按秒累计延期成本

### 资源类型

| 资源 | 租赁模式 | 成本构成 |
|------|----------|----------|
| **机器人 (Robot)** | 长租 / 短租 | 单位运行时间成本 × 工作时长 |
| **拣货员 (Picker)** | 长租 / 短租 | 单位时间雇佣成本 × 工作时长 + 解雇固定成本 |

- **长租**：成本低（约短租的 35%~65%），但整个周期不可解雇
- **短租**：成本高，但可按天灵活增减

### 状态空间

PPO 智能体的观测由两部分组成：

| 输入类型 | 维度 | 含义 |
|----------|------|------|
| **矩阵输入** (CNN) | `3 × N_w × N_l` | 通道 0: 各拣货位机器人排队长度; 通道 1: 各拣货位是否有拣货员; 通道 2: 各拣货位未拣选商品数 |
| **标量输入** (MLP) | `1 + N_a` | 当前机器人总数 + 各区域拣货员数量 |

### 动作空间

连续动作向量维度 `1 + N_a`（4 维），每维 ∈ ℝ，经舍入后映射为整数资源调整量：

- `action[0]`: 机器人增减量（正→新增，负→解雇短租）
- `action[1]~action[3]`: 各区域拣货员增减量

### 奖励函数

$$r_t = C_{t-1} - C_t$$

其中 $C_t = C_{\text{delay}} + C_{\text{robot}} + C_{\text{picker}}$，即相邻决策步的总成本降低量作为即时奖励。

---

## 算法架构

### PPO 网络结构

```
                  矩阵输入 (3×N_w×N_l)              标量输入 (1+N_a)
                        │                                │
                   ┌─ Conv2d(3→16, 3×3) ─┐                │
                   │      ReLU            │                │
                   └─ Conv2d(16→32, 3×3) ─┘               │
                        │  ReLU                           │
                        │  MaxPool2d(2×2)                 │
                        │  Flatten                        │
                        ▼                                │
                   Visual FC (cnn_out→32)                 │
                        │  ReLU                           │
                        └────────┬────────────────────────┘
                                 ▼
                        FC Backbone (32+4 → 128)
                        LayerNorm + ReLU
                        FC (128 → 128) + ReLU
                                 │
                     ┌───────────┴───────────┐
                     │  MultiHeadAttention   │
                     │  (4 heads, dim=128)   │
                     │  + Residual + LN      │
                     └───────────┬───────────┘
                    ┌────────────┴────────────┐
                    ▼                         ▼
            Action Mean (128→4)        Value Head (128→1)
            Action LogStd (可学习参数)
                    │
                    ▼
            Normal(mean, std) → 采样动作
```

- **CNN 编码器**: 双层卷积提取拣货位空间特征
- **Multi-Head Self-Attention**: 捕获各拣货位之间的全局依赖关系
- **Orthogonal 初始化**: 所有线性层使用正交初始化，保证训练初期的梯度稳定性
- **可学习 LogStd**: 策略标准差随训练自适应调节

### Baseline 算法对比

| 算法 | 类型 | Actor | Critic |
|------|------|-------|--------|
| **A2C** | On-Policy, Stochastic | 共享 CNN + Gaussian Policy | 共享 CNN + Value Head |
| **DDPG** | Off-Policy, Deterministic | 共享 CNN + Tanh 输出 | 共享 CNN + Q-Network（含动作输入） |
| **TD3** | Off-Policy, Deterministic | 双 Q 网络 + 目标网络平滑 | 策略延迟更新 |
| **SAC** | Off-Policy, Stochastic | Squashed Gaussian Policy | 双 Q 网络 + 可学习 α |

所有 Baseline 共享 `StateEncoder`（与 PPO 的 CNN Backbone 结构一致），使用 Replay Buffer 和 Soft Update。

---

## 训练模式

`configs/default.json` 中 `experiment.mode` 支持 4 种模式：

| 模式 | 决策次数 | 说明 |
|------|----------|------|
| `short` | 每天 1 次 × `max_days` | 纯短租，每日调整所有资源 |
| `long` | 仅 1 次 | 一次性决定全周期资源，资源租赁贯穿始终 |
| `hybrid` | 每天 1 次 × `max_days` | 首日按算法输出分配资源（可混合长/短租），后续每日仅调整短租资源 |
| `fixed_hybrid` | 每天 1 次 | 首日按 `fixed_hybrid` 预设值固定创建长租，后续每日由算法调整短租资源 |

---

## 项目结构

```text
DRL_Warehouse/
├── configs/
│   └── default.json              # 唯一全局配置文件（所有参数均在此维护）
├── environment/
│   ├── class_public.py           # Config 基类、配置加载/校验
│   ├── class_warehouse.py        # 实体类：Order, Item, Robot, Picker, PickPoint, StorageBin
│   └── warehouse_env.py          # Gymnasium 环境：WarehouseEnv
├── agent/
│   ├── ppo/
│   │   ├── train.py              # PPO 训练入口
│   │   ├── train_batch.py        # PPO 批量训练（多算例×多月份）
│   │   ├── train_fixed_hybrid.py  # Fixed Hybrid 模式训练入口
│   │   ├── agent.py              # PPOAgent：select_action / update / GAE
│   │   ├── networks.py           # PolicyNetwork + ValueNetwork
│   │   └── buffer.py             # RolloutBuffer（On-Policy 经验存储）
│   ├── baselines/
│   │   ├── common.py             # 共享模块：StateEncoder, ReplayBuffer, Q-Network 等
│   │   ├── a2c.py                # Advantage Actor-Critic
│   │   ├── ddpg.py               # Deep Deterministic Policy Gradient
│   │   ├── td3.py                # Twin Delayed DDPG
│   │   ├── sac.py                # Soft Actor-Critic
│   │   └── train_batch.py        # Baseline 批量训练入口
│   └── training_utils.py         # 训练工具：指标收集、CSV 日志、Visdom、checkpoint
├── data/
│   ├── generate_orders.py        # 订单生成器（泊松到达过程）
│   └── instances/                # items_<n>/orders_m<month>.pkl（48 个实例）
├── test/
│   ├── test_PPO_I.py             # 配置文件完整性测试
│   ├── test_PPO_II.py            # 环境基础功能测试
│   ├── test_PPO_III.py           # PPO 训练集成测试
│   └── test_best_config.py       # 最佳配置输出测试
└── result/                       # 实验结果输出目录（gitignore）
    ├── ppo/                      # PPO 训练结果
    └── baselines/                # Baseline 训练结果
```

---

## 环境配置

### 依赖安装

```powershell
cd DRL_Warehouse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy gymnasium visdom
```

### 配置文件

`configs/default.json` 是唯一的参数来源，包含以下 8 个部分：

| 节 | 说明 |
|----|------|
| `warehouse` | 仓库物理参数（区域数、巷道数、货架容量、尺寸） |
| `robot` | 机器人成本与速度参数 |
| `picker` | 拣货员成本、速度与解雇费用 |
| `order` | 订单延迟成本、打包时间、到达过程参数 |
| `item` | 单件商品拣选时间 |
| `ppo` | PPO 超参数（学习率、GAE、裁剪范围、熵系数等） |
| `experiment` | 训练模式、算例、episode、seed、设备 |
| `paths` | 数据与输出路径配置 |

运行入口自动打印当前配置来源：

```text
Using config: <repo>/configs/default.json
```

---

## 使用指南

### 1. 生成订单实例

```powershell
python -m data.generate_orders
```

默认生成 4 种商品数 × 12 个月 = 48 个订单实例（每个 `.pkl` 约 50-90 MB）：

```text
data/instances/items_2/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_4/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_6/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_10/orders_m01.pkl ... orders_m12.pkl
```

各月份的订单到达过程由真实月度参数（`generate_orders.MONTH_PARAMS`）驱动，模拟不同季节/月份的订单密度变化。

### 2. 训练 PPO

```powershell
# 单算例单月训练（参数来自 configs/default.json）
python -m agent.ppo.train

# 批量训练（遍历所有 items × months 组合）
python -m agent.ppo.train_batch

# Fixed Hybrid 模式训练
python -m agent.ppo.train_fixed_hybrid
```

训练过程中通过 Visdom 实时可视化总成本曲线（需先启动 `python -m visdom.server`）。

### 3. 训练 Baseline

```powershell
# 单独运行（超参数可选 CLI 覆盖）
python agent\baselines\a2c.py --lr 0.0003
python agent\baselines\ddpg.py --actor-lr 0.0001 --critic-lr 0.001
python agent\baselines\td3.py --policy-delay 2
python agent\baselines\sac.py --alpha-lr 0.0003

# 批量训练
python -m agent.baselines.train_batch
```

### 4. 运行测试

```powershell
python -m unittest discover -s test -p "test_*.py"
```

---

## 输出格式

### 训练指标 CSV (`<stem>.csv`)

| 列名 | 含义 |
|------|------|
| `episode` | 训练 episode 编号 |
| `total_cost` | 总成本 = 延期成本 + 机器人成本 + 拣货员成本 |
| `delay_cost` | 订单延期惩罚成本 |
| `robot_cost` | 机器人运行成本 |
| `picker_cost` | 拣货员雇佣成本 |
| `completed_orders` | 已完成订单数 |
| `on_time_completed_orders` | 按时完成订单数 |
| `total_orders` | 到达订单总数 |
| `completion_rate` | 按时履约率 |
| `average_picking_time` | 平均拣选耗时 (s) |

### 最佳配置 CSV (`<stem>_best_config.csv`)

记录最优 episode 中每个决策步的完整资源配置快照，包括：

- 算法宣称动作 vs 实际生效动作（`action_*_delta` vs `effective_*_delta`）
- 各区域长/短租赁机器人/拣货员实时存量
- 各成本分量与履约率

### 模型权重 (`<stem>.pth`)

保存最优 episode 对应的 `policy` 和 `value` 网络参数及训练配置。

---

## 关键设计决策

- **统一配置管理**: 所有参数集中在 `default.json`，通过 `Config` 基类加载，保证实验可复现
- **模块化 CNN + Attention**: 视觉编码器对所有算法共享相同结构，保证对比公平性
- **Orthogonal 初始化**: 全连接层使用 $\sqrt{2}$ 正交初始化，改善深层网络训练稳定性
- **GAE 优势估计**: PPO 使用广义优势估计（λ=0.95），平衡偏差-方差权衡
- **熵正则衰减**: 初始熵系数 0.05，每次更新后 ×0.995 衰减至最低 0.001，实现从探索到利用的平滑过渡
