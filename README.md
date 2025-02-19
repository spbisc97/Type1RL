# T1D Reinforcement Learning Project

This project implements Reinforcement Learning agents to control insulin delivery in a Type-1 Diabetes simulator (simglucose). The project uses Stable Baselines3 for RL algorithms and simglucose for the T1D simulation environment.

## Features

- Multiple RL algorithms (SAC, PPO) implementations
- Custom environment wrapper for the simglucose simulator
- Configurable neural network architectures and hyperparameters
- Training and evaluation scripts
- Visualization tools for training progress and evaluation results

## Installation

1. Clone this repository and the simglucose submodule:
```bash
git clone https://github.com/your-username/t1d-rl.git
cd t1d-rl
git submodule add https://github.com/jxx123/simglucose.git
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate glucose-rl-env
```

3. Install PyTorch (choose the appropriate version for your system):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only and cuda 12.4
pip install torch torchvision torchaudio
```

4. Install the remaining requirements(should not be needed):
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── agents/                 # RL agent implementations
│   ├── sac_agent.py       # Soft Actor-Critic agent
│   └── ppo_agent.py       # Proximal Policy Optimization agent
├── utils/                 
│   └── plotting.py        # Visualization utilities
├── config.py              # Configuration parameters
├── environment.py         # Environment wrapper
├── main.py               # Main training/evaluation script
├── requirements.txt      # Python dependencies
└── environment.yml       # Conda environment specification
```

## Usage

### Training

To train an agent:
```bash
# Train SAC agent
python main.py --mode train --agent sac --timesteps 1000000

# Train PPO agent
python main.py --mode train --agent ppo --timesteps 1000000
```

### Evaluation

To evaluate a trained agent:
```bash
# Evaluate SAC agent
python main.py --mode evaluate --agent sac --model-path models/sac_best_model/best_model.zip

# Evaluate PPO agent
python main.py --mode evaluate --agent ppo --model-path models/ppo_best_model/best_model.zip
```

## Environment

The environment wraps the simglucose simulator with the following specifications:

- **Observation Space**: CGM readings history (configurable length)
- **Action Space**: Continuous insulin dosing (0 to max_basal)
- **Reward Function**: Custom reward based on blood glucose levels
  - +1 for maintaining glucose in target range (70-180 mg/dL)
  - -1 for mild hypo/hyperglycemia
  - -20 for severe hypo/hyperglycemia

## Results Visualization

The project includes several visualization tools:
- Training progress plots (rewards and episode lengths)
- Evaluation episode plots (CGM values, rewards, and actions)
- Performance metrics over time

Results are automatically saved in the `results/` directory.

## Configuration

Key parameters can be modified in `config.py`:
- Network architectures
- Learning rates
- Batch sizes
- Buffer sizes
- Other algorithm-specific parameters

## Bergman Minimal Model Parameters

The Bergman minimal model uses the following parameters to simulate glucose-insulin dynamics:

| Parameter | Description | Value and Unit | Dimensions (M,L,T) |
|-----------|-------------|----------------|-------------------|
| p1 | Insulin-independent glucose disappearance rate (glucose effectiveness, SG) | ~10⁻² min⁻¹ | (0,0,-1) |
| p2 | Rate constant of tissue glucose uptake ability decrease | ~10⁻² min⁻¹ | (0,0,-1) |
| p3 | Insulin-dependent increase in tissue glucose uptake ability | ~10⁻⁵ min⁻² (μU/ml)⁻¹ | (-1,3,-2) |
| n | Disappearance rate of endogenous insulin | ~10⁻¹ min⁻¹ | (0,0,-1) |
| γ | Rate of second phase endogenous insulin secretion | ~10⁻²–10⁻³ (μU/ml) min⁻² | (0,0,-2) |
| Gb | Baseline plasma glucose | ~100 mg/dl | (1,-3,0) |
| G0 | Initial glucose concentration during FSIVGTT | ~300 mg/dl | (1,-3,0) |
| h | Threshold value for second phase insulin secretion | ~100 mg/dl | (1,-3,0) |
| Ib | Baseline plasma insulin | ~10 μU/ml | (1,-3,0) |
| I0 | Initial insulin concentration | ~30 μU/ml | (1,-3,0) |

Where dimensions are represented as:
- M: Mass
- L: Length
- T: Time

## Testing Agents

The project includes a test environment (`test_env.py`) that uses the LunarLander environment from Gymnasium to verify that the RL algorithms are working correctly. This is useful for:

1. Validating agent implementations
2. Testing training pipeline
3. Debugging environment wrappers
4. Verifying algorithm hyperparameters

To run the tests:
```bash
# Install Box2D dependencies first
pip install swig
pip install "gymnasium[box2d]"

# Run the test environment
python test_env.py
```

The test script will:
- Train each agent (SAC, PPO, TD3, DQN) on LunarLander
- Use shorter training duration (100k steps)
- Print training progress and rewards
- Save trained models to the models directory

A successful test indicates that:
- The agent implementation is correct
- The training pipeline works
- The environment wrappers are functioning
- The hyperparameters are reasonable

This is particularly useful when implementing new features or debugging issues with the T1D environment.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Environments

The project includes two different simulation environments:

### 1. Simglucose Environment
Based on the UVA/Padova T1D Simulator, this environment provides:
- Realistic glucose-insulin dynamics
- Multiple patient scenarios
- Meal disturbances
- CGM sensor noise
- Insulin pump dynamics

### 2. Bergman Minimal Model Environment
A simpler mathematical model based on the Bergman equations:
- Three state variables (Glucose, Insulin effect, Plasma insulin)
- Configurable model parameters
- Meal disturbances
- More computationally efficient

Both environments follow the Gymnasium interface and provide:
- Observation: CGM history (configurable length)
- Action: Insulin infusion rate
- Reward: Based on blood glucose control
- Info: Additional state information

## Project Structure

```
├── agents/                 # RL agent implementations
│   ├── sac_agent.py       # Soft Actor-Critic
│   ├── ppo_agent.py       # Proximal Policy Optimization
│   ├── td3_agent.py       # Twin Delayed DDPG
│   └── dqn_agent.py       # Deep Q-Network
├── utils/                 
│   ├── plotting.py        # Visualization utilities
│   └── callbacks.py       # Training callbacks
├── environment.py         # Simglucose environment wrapper
├── bergman_env.py        # Bergman model environment
├── test_env.py           # Test environment (LunarLander)
├── config.py             # Configuration parameters
├── main.py               # Main training/evaluation script
└── requirements.txt      # Python dependencies
```

## Monitoring Training with TensorBoard

The project uses TensorBoard for monitoring training progress. To view training metrics:

1. Start TensorBoard:
```bash
# From the project root directory
tensorboard --logdir logs

# Or specify a specific agent's logs
tensorboard --logdir logs/sac_results
```

2. Open your browser and navigate to:
```
http://localhost:6006
```

TensorBoard will show:
- Training rewards
- Episode lengths
- Learning rates
- Policy/Value losses
- Network gradients
- Other agent-specific metrics

You can compare different runs and agents by selecting them in the TensorBoard interface.

Note: Training logs are saved in the `logs/` directory with the following structure:
```
logs/
├── sac_results/
├── ppo_results/
├── td3_results/
└── dqn_results/
```