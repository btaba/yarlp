|Build Status|

yarlp
-----

**Yet Another Reinforcement Learning Package**

Implementations of ```CEM`` </yarlp/agent/cem_agent.py>`__,
```REINFORCE`` </yarlp/agent/pg_agents.py>`__,
```TRPO`` </yarlp/agent/trpo_agent.py>`__,
```DDQN`` </yarlp/agent/ddqn_agent.py>`__,
```A2C`` </yarlp/agent/a2c_agent.py>`__ with reproducible benchmarks.
Experiments are templated using ``jsonschema`` and are compared to
published results. This is meant to be a starting point for working
implementations of classic RL algorithms. Unfortunately even
implementations from OpenAI baselines are `not always
reproducible <https://github.com/openai/baselines/issues/176>`__.

A working Dockerfile with ``yarlp`` installed can be run with:

-  ``docker build -t "yarlpd" .``
-  ``docker run -it yarlpd bash``

To run a benchmark, simply:

``python yarlp/experiment/experiment.py --help``

If you want to run things manually, look in ``examples`` or look at
this:

.. code:: python

    from yarlp.agent.trpo_agent import TRPOAgent
    from yarlp.utils.env_utils import NormalizedGymEnv

    env = NormalizedGymEnv('MountainCarContinuous-v0')
    agent = TRPOAgent(env, seed=123)
    agent.train(max_timesteps=1000000)

Benchmarks
----------

We benchmark against published results and Openai
```baselines`` <https://github.com/openai/baselines>`__ where available
using
```yarlp/experiment/experiment.py`` </yarlp/experiment/experiment.py>`__.
Benchmark scripts for Openai ``baselines`` were made ad-hoc, such as
`this
one <https://github.com/btaba/baselines/blob/master/baselines/trpo_mpi/run_trpo_experiment.py>`__.

Atari10M
~~~~~~~~

+---------------+--------------+-------------------+
| |BeamRider|   | |Breakout|   | |Pong|            |
+---------------+--------------+-------------------+
| |QBert|       | |Seaquest|   | |SpaceInvaders|   |
+---------------+--------------+-------------------+

DDQN with dueling networks and prioritized replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``python yarlp/experiment/experiment.py run_atari10m_ddqn_benchmark``

I trained 6 Atari environments for 10M time-steps (**40M frames**),
using 1 random seed, since I only have 1 GPU and limited time on this
Earth. I used DDQN with dueling networks, but no prioritized replay
(although it's implemented). I compare the final mean 100 episode raw
scores for yarlp (with exploration of 0.01) with results from `Hasselt
et al, 2015 <https://arxiv.org/pdf/1509.06461.pdf>`__ and `Wang et al,
2016 <https://arxiv.org/pdf/1511.06581.pdf>`__ which train for **200M
frames** and evaluate on 100 episodes (exploration of 0.05).

I don't compare to OpenAI baselines because the OpenAI DDQN
implementation is **not** currently able to reproduce published results
as of 2018-01-20. See `this github
issue <https://github.com/openai/baselines/issues/176>`__, although I
found `these benchmark
plots <https://github.com/openai/baselines-results/blob/master/dqn_results.ipynb>`__
to be pretty helpful.

+------+------+------+------+
| env  | yarl | Hass | Wang |
|      | p    | elt  | et   |
|      | DUEL | et   | al   |
|      | 40M  | al   | DUEL |
|      | Fram | DDQN | 200M |
|      | es   | 200M | Fram |
|      |      | Fram | es   |
|      |      | es   |      |
+======+======+======+======+
| Beam | 8705 | 7654 | 1216 |
| Ride |      |      | 4    |
| r    |      |      |      |
+------+------+------+------+
| Brea | 423. | 375  | 345  |
| kout | 5    |      |      |
+------+------+------+------+
| Pong | 20.7 | 21   | 21   |
|      | 3    |      |      |
+------+------+------+------+
| QBer | 5410 | 1487 | 1922 |
| t    | .75  | 5    | 0.3  |
+------+------+------+------+
| Seaq | 5300 | 7995 | 5024 |
| uest | .5   |      | 5.2  |
+------+------+------+------+
| Spac | 1978 | 3154 | 6427 |
| eInv | .2   | .6   | .3   |
| ader |      |      |      |
| s    |      |      |      |
+------+------+------+------+

+------+------+------+------+
| |Bea | |Bre | |Pon | |Qbe |
| mRid | akou | gNoF | rtNo |
| erNo | tNoF | rame | Fram |
| Fram | rame | skip | eski |
| eski | skip | -v4| | p-v4 |
| p-v4 | -v4| |      | |    |
| |    |      |      |      |
+------+------+------+------+
| |Sea | |Spa |      |      |
| ques | ceIn |      |      |
| tNoF | vade |      |      |
| rame | rsNo |      |      |
| skip | Fram |      |      |
| -v4| | eski |      |      |
|      | p-v4 |      |      |
|      | |    |      |      |
+------+------+------+------+

A2C
^^^

``python yarlp/experiment/experiment.py run_atari10m_a2c_benchmark``

A2C on 10M time-steps (**40M frames**) with 1 random seed. Results
compared to learning curves from `Mnih et al,
2016 <https://arxiv.org/pdf/1602.01783.pdf>`__ extracted at 10M
time-steps from Figure 3. You are invited to run for multiple seeds and
the full 200M frames for a better comparison.

+-----------------+-----------------+---------------------------------+
| env             | yarlp A2C 40M   | Mnih et al A3C 40M 16-threads   |
+=================+=================+=================================+
| BeamRider       | 3150            | ~3000                           |
+-----------------+-----------------+---------------------------------+
| Breakout        | 418             | ~150                            |
+-----------------+-----------------+---------------------------------+
| Pong            | 20              | ~20                             |
+-----------------+-----------------+---------------------------------+
| QBert           | 3644            | ~1000                           |
+-----------------+-----------------+---------------------------------+
| SpaceInvaders   | 805             | ~600                            |
+-----------------+-----------------+---------------------------------+

+------+------+------+------+
| |Bea | |Bre | |Pon | |Qbe |
| mRid | akou | gNoF | rtNo |
| erNo | tNoF | rame | Fram |
| Fram | rame | skip | eski |
| eski | skip | -v4| | p-v4 |
| p-v4 | -v4| |      | |    |
| |    |      |      |      |
+------+------+------+------+
| |Sea | |Spa |      |      |
| ques | ceIn |      |      |
| tNoF | vade |      |      |
| rame | rsNo |      |      |
| skip | Fram |      |      |
| -v4| | eski |      |      |
|      | p-v4 |      |      |
|      | |    |      |      |
+------+------+------+------+

Here are some `more
plots <https://github.com/openai/baselines-results/blob/master/acktr_ppo_acer_a2c_atari.ipynb>`__
from OpenAI to compare against.

Mujoco1M
~~~~~~~~

TRPO
^^^^

``python yarlp/experiment/experiment.py run_mujoco1m_benchmark``

We average over 5 random seeds instead of 3 for both ``baselines`` and
``yarlp``. More seeds probably wouldn't hurt here, we report 95th
percent confidence intervals.

+-------------------------------+--------------------+-------------------------+----------------+
| |Hopper-v1|                   | |HalfCheetah-v1|   | |Reacher-v1|            | |Swimmer-v1|   |
+-------------------------------+--------------------+-------------------------+----------------+
| |InvertedDoublePendulum-v1|   | |Walker2d-v1|      | |InvertedPendulum-v1|   |                |
+-------------------------------+--------------------+-------------------------+----------------+

CLI scripts
-----------

CLI convenience scripts will be installed with the package:

-  Run a benchmark:

   -  ``python yarlp/experiment/experiment.py --help``

-  Plot ``yarlp`` compared to Openai ``baselines`` benchmarks:

   -  ``compare_benchmark <yarlp-experiment-dir> <baseline-experiment-dir>``

-  Experiments:

   -  Experiments can be defined using json, validated with
      ``jsonschema``. See `here </experiment_configs>`__ for sample
      experiment configs. You can do a grid search if multiple
      parameters are specified, which will run in parallel.
   -  Example:
      ``run_yarlp_experiment --spec-file experiment_configs/trpo_experiment_mult_params.json``

-  Experiment plots:

   -  ``make_plots <experiment-dir>``

.. |Build Status| image:: https://travis-ci.org/btaba/yarlp.svg?branch=master
   :target: https://travis-ci.org/btaba/yarlp
.. |BeamRider| image:: /assets/atari10m/ddqn/beamrider.gif
.. |Breakout| image:: /assets/atari10m/ddqn/breakout.gif
.. |Pong| image:: /assets/atari10m/ddqn/pong.gif
.. |QBert| image:: /assets/atari10m/ddqn/qbert.gif
.. |Seaquest| image:: /assets/atari10m/ddqn/seaquest.gif
.. |SpaceInvaders| image:: /assets/atari10m/ddqn/spaceinvaders.gif
.. |BeamRiderNoFrameskip-v4| image:: /assets/atari10m/ddqn/BeamRiderNoFrameskip-v4.png
.. |BreakoutNoFrameskip-v4| image:: /assets/atari10m/ddqn/BreakoutNoFrameskip-v4.png
.. |PongNoFrameskip-v4| image:: /assets/atari10m/ddqn/PongNoFrameskip-v4.png
.. |QbertNoFrameskip-v4| image:: /assets/atari10m/ddqn/QbertNoFrameskip-v4.png
.. |SeaquestNoFrameskip-v4| image:: /assets/atari10m/ddqn/SeaquestNoFrameskip-v4.png
.. |SpaceInvadersNoFrameskip-v4| image:: /assets/atari10m/ddqn/SpaceInvadersNoFrameskip-v4.png
.. |BeamRiderNoFrameskip-v4| image:: /assets/atari10m/a2c/BeamRiderNoFrameskip-v4.png
.. |BreakoutNoFrameskip-v4| image:: /assets/atari10m/a2c/BreakoutNoFrameskip-v4.png
.. |PongNoFrameskip-v4| image:: /assets/atari10m/a2c/PongNoFrameskip-v4.png
.. |QbertNoFrameskip-v4| image:: /assets/atari10m/a2c/QbertNoFrameskip-v4.png
.. |SeaquestNoFrameskip-v4| image:: /assets/atari10m/a2c/SeaquestNoFrameskip-v4.png
.. |SpaceInvadersNoFrameskip-v4| image:: /assets/atari10m/a2c/SpaceInvadersNoFrameskip-v4.png
.. |Hopper-v1| image:: /assets/mujoco1m/trpo/Hopper-v1.png
.. |HalfCheetah-v1| image:: /assets/mujoco1m/trpo/HalfCheetah-v1.png
.. |Reacher-v1| image:: /assets/mujoco1m/trpo/Reacher-v1.png
.. |Swimmer-v1| image:: /assets/mujoco1m/trpo/Swimmer-v1.png
.. |InvertedDoublePendulum-v1| image:: /assets/mujoco1m/trpo/InvertedDoublePendulum-v1.png
.. |Walker2d-v1| image:: /assets/mujoco1m/trpo/Walker2d-v1.png
.. |InvertedPendulum-v1| image:: /assets/mujoco1m/trpo/InvertedPendulum-v1.png

