[![Build Status](https://travis-ci.org/btaba/yarlp.svg?branch=master)](https://travis-ci.org/btaba/yarlp)

## yarlp

**Yet Another Reinforcement Learning Package**

Implementations of [`CEM`](/yarlp/agent/cem_agent.py), [`REINFORCE`](/yarlp/agent/pg_agents.py), [`TRPO`](/yarlp/agent/trpo_agent.py), [`DDQN`](/yarlp/agent/ddqn_agent.py), benchmarked against OpenAI [baselines](https://github.com/openai/baselines), mostly done for educational purposes.

Quick example:

```python
from yarlp.agent.trpo_agent import TRPOAgent
from yarlp.utils.env_utils import NormalizedGymEnv

env = NormalizedGymEnv('MountainCarContinuous-v0')
agent = TRPOAgent(env, seed=123)
agent.train(max_timesteps=1000000)
```

## Benchmarks

We benchmark against Openai [`baselines`](https://github.com/openai/baselines) using [`run_benchmark`](/yarlp/experiment/experiment.py#334). Benchmark scripts for Openai `baselines` were made ad-hoc, such as [this one](https://github.com/btaba/baselines/blob/master/baselines/trpo_mpi/run_trpo_experiment.py).

### Mujoco1M

#### TRPO

We average over 5 random seeds instead of 3 for both `baselines` and `yarlp`. More seeds probably wouldn't hurt here, we report 95th percent confidence intervals.

|   |   |   |   |
|---|---|---|---|
|![Hopper-v1](/assets/mujoco1m/trpo/Hopper-v1.png)|![HalfCheetah-v1](/assets/mujoco1m/trpo/HalfCheetah-v1.png)|![Reacher-v1](/assets/mujoco1m/trpo/Reacher-v1.png)|![Swimmer-v1](/assets/mujoco1m/trpo/Swimmer-v1.png)|
|![InvertedDoublePendulum-v1](/assets/mujoco1m/trpo/InvertedDoublePendulum-v1.png)|![Walker2d-v1](/assets/mujoco1m/trpo/Walker2d-v1.png)|![InvertedPendulum-v1](/assets/mujoco1m/trpo/InvertedPendulum-v1.png)|

### Atari10M

#### DDQN with dueling networks and prioritized replay


I don't compare to OpenAI baselines because the OpenAI DDQN implementation is **not** currently able to reproduce published results as of 2018-01-20. See [this github issue](https://github.com/openai/baselines/issues/176).


## CLI scripts

CLI convenience scripts will be installed with the package:

* Run a benchmark:
	* `run_benchmark`
* Plot `yarlp` compared to Openai `baselines` benchmarks:
	* `compare_benchmark <yarlp-experiment-dir> <baseline-experiment-dir>`
* Experiments:
	* Experiments can be defined using json, validated with `jsonschema`. See [here](/experiment_configs) for sample experiment configs. You can do a grid search if multiple parameters are specified, which will run in parallel.
	* Example: `run_yarlp_experiment --spec-file experiment_configs/trpo_experiment_mult_params.json`
* Experiment plots:
	* `make_plots <experiment-dir>`


##### TODO:
* DDQN
	- benchmark plots, ablation plots
* A2C
	- run a2c on breakout...can i get good results???!
* PPO2
* docs
* pypi

