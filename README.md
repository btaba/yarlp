[![Build Status](https://travis-ci.org/btaba/yarlp.svg?branch=master)](https://travis-ci.org/btaba/yarlp)

## yarlp

**Yet Another Reinforcement Learning Package**

Implementations of `CEM`, `REINFORCE`, and `TRPO`, benchmarked against OpenAI [baselines](https://github.com/openai/baselines).

Example:

```python
from yarlp.agent.trpo_agent import TRPOAgent
from yarlp.utils.env_utils import NormalizedGymEnv

env = NormalizedGymEnv(
    'MountainCarContinuous-v0',
    normalize_obs=True)

agent = TRPOAgent(
    env, discount_factor=0.99,
    policy_network=mlp, seed=0)
agent.train(500, 0, n_steps=2048)
```

##### TODO:

* benchmarks with experiment dir
	- make yarlp perform same as baselines
        - then replace policy model
    - then do atari tasks
* DQ, DDQ
* serialization/deserialization of agents
* docs
* pypi

