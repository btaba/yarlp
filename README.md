[![Build Status](https://travis-ci.org/btaba/yarlp.svg?branch=master)](https://travis-ci.org/btaba/yarlp)

## yarlp

**Yet Another Reinforcement Learning Package**

Implementations of [`CEM`](/yarlp/agent/cem_agent.py), [`REINFORCE`](/yarlp/agent/pg_agents.py), [`TRPO`](/yarlp/agent/trpo_agent.py), [`DDQN`](/yarlp/agent/ddqn_agent.py) with reproducible benchmarks. Experiments are templated using `jsonschema` and are compared to published results. This is meant to be a starting point for working implementations of classic RL algorithms. Unfortunately even implementations from OpenAI baselines are [not always reproducible](https://github.com/openai/baselines/issues/176).

A working Dockerfile with `yarlp` installed can be run with:

* `docker build -t "yarlpd" .`
* `docker run -it yarlpd bash`

To run a benchmark, simply:

`python yarlp/experiment/experiment.py --help`


If you want to run things manually:

```python
from yarlp.agent.trpo_agent import TRPOAgent
from yarlp.utils.env_utils import NormalizedGymEnv

env = NormalizedGymEnv('MountainCarContinuous-v0')
agent = TRPOAgent(env, seed=123)
agent.train(max_timesteps=1000000)
```

## Benchmarks

We benchmark against published results and Openai [`baselines`](https://github.com/openai/baselines) where available using [`yarlp/experiment/experiment.py`](/yarlp/experiment/experiment.py). Benchmark scripts for Openai `baselines` were made ad-hoc, such as [this one](https://github.com/btaba/baselines/blob/master/baselines/trpo_mpi/run_trpo_experiment.py).

### Mujoco1M

#### TRPO

`python yarlp/experiment/experiment.py run_mujoco1m_benchmark`

We average over 5 random seeds instead of 3 for both `baselines` and `yarlp`. More seeds probably wouldn't hurt here, we report 95th percent confidence intervals.

|   |   |   |   |
|---|---|---|---|
|![Hopper-v1](/assets/mujoco1m/trpo/Hopper-v1.png)|![HalfCheetah-v1](/assets/mujoco1m/trpo/HalfCheetah-v1.png)|![Reacher-v1](/assets/mujoco1m/trpo/Reacher-v1.png)|![Swimmer-v1](/assets/mujoco1m/trpo/Swimmer-v1.png)|
|![InvertedDoublePendulum-v1](/assets/mujoco1m/trpo/InvertedDoublePendulum-v1.png)|![Walker2d-v1](/assets/mujoco1m/trpo/Walker2d-v1.png)|![InvertedPendulum-v1](/assets/mujoco1m/trpo/InvertedPendulum-v1.png)|

### Atari10M

#### DDQN with dueling networks and prioritized replay

`python yarlp/experiment/experiment.py run_atari10m_benchmark`


I trained 6 Atari environments for 10M time-steps (**40M frames**), using 1 random seed, since I only have 1 GPU and limited time on this Earth. I used DDQN with dueling networks, but no prioritized replay (although it's implemented). I compare the final mean 100 episode raw scores for yarlp (with exploration of 0.01) with results from [Hasselt et al, 2015](https://arxiv.org/pdf/1509.06461.pdf) and [Wang et al, 2016](https://arxiv.org/pdf/1511.06581.pdf) which train for **200M frames** and evaluate on 100 episodes (exploration of 0.05).

I don't compare to OpenAI baselines because the OpenAI DDQN implementation is **not** currently able to reproduce published results as of 2018-01-20. See [this github issue](https://github.com/openai/baselines/issues/176), although I found [these benchmark plots](https://github.com/openai/baselines-results/blob/master/dqn_results.ipynb) to be pretty helpful.

|env|yarlp DUEL 40M Frames|Hasselt et al DDQN 200M Frames|Wang et al DUEL 200M Frames|
|---|---|---|---|
|BeamRider|8705|7654|12164|
|Breakout|423.5|375|345|
|Pong|20.73|21|21|
|QBert|5410.75|14875|19220.3|
|Seaquest|5300.5|7995|50245.2|
|SpaceInvaders|1978.2|3154.6|6427.3|


|   |   |   |   |
|---|---|---|---|
|![BeamRiderNoFrameskip-v4](/assets/atari10m/ddqn/BeamRiderNoFrameskip-v4.png)|![BreakoutNoFrameskip-v4](/assets/atari10m/ddqn/BreakoutNoFrameskip-v4.png)|![PongNoFrameskip-v4](/assets/atari10m/ddqn/PongNoFrameskip-v4.png)|![QbertNoFrameskip-v4](/assets/atari10m/ddqn/QbertNoFrameskip-v4.png)|
|![SeaquestNoFrameskip-v4](/assets/atari10m/ddqn/SeaquestNoFrameskip-v4.png)|![SpaceInvadersNoFrameskip-v4](/assets/atari10m/ddqn/SpaceInvadersNoFrameskip-v4.png)||


||||
|---|---|---|
|![BeamRider](/assets/atari10m/ddqn/beamrider.gif)|![Breakout](/assets/atari10m/ddqn/breakout.gif)|![Pong](/assets/atari10m/ddqn/pong.gif)|
|![QBert](/assets/atari10m/ddqn/qbert.gif)|![Seaquest](/assets/atari10m/ddqn/seaquest.gif)|![SpaceInvaders](/assets/atari10m/ddqn/spaceinvaders.gif)|

## CLI scripts

CLI convenience scripts will be installed with the package:

* Run a benchmark:
	* `python yarlp/experiment/experiment.py --help`
* Plot `yarlp` compared to Openai `baselines` benchmarks:
	* `compare_benchmark <yarlp-experiment-dir> <baseline-experiment-dir>`
* Experiments:
	* Experiments can be defined using json, validated with `jsonschema`. See [here](/experiment_configs) for sample experiment configs. You can do a grid search if multiple parameters are specified, which will run in parallel.
	* Example: `run_yarlp_experiment --spec-file experiment_configs/trpo_experiment_mult_params.json`
* Experiment plots:
	* `make_plots <experiment-dir>`


##### TODO:

* A2C
* docs
* pypi
