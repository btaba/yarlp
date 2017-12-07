"""
    Base Agent class, which takes in a PolicyModel object
"""

import gym
import numpy as np
from abc import ABCMeta, abstractmethod
from yarlp.utils.env_utils import GymEnv
from yarlp.utils.metric_logger import MetricLogger
from yarlp.utils.replay_buffer import Rollout
from yarlp.utils import tf_utils
from yarlp.model.linear_baseline import LinearFeatureBaseline


ABC = ABCMeta('ABC', (object,), {})


class Agent(ABC):
    """
    Abstract class for an agent.
    """

    def __init__(self, env, discount_factor=0.99,
                 logger=None, seed=None,
                 state_featurizer=lambda x: x):
        """
        discount_factor : float
            Discount rewards by this factor
        """
        # Discount factor
        assert discount_factor >= 0 and discount_factor <= 1
        self._discount = discount_factor

        if logger is None:
            self.logger = MetricLogger()
        else:
            self.logger = logger

        if seed is not None:
            self.logger._logger.info('Seed: {}'.format(seed))
            tf_utils.set_global_seeds(seed)
            env.seed(seed)
        self._env = env
        self._env_id = '{}_gym{}'.format(
            env.spec.id, gym.__version__)

        self._state_featurizer = state_featurizer

    def save_models(self, path):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        pass

    @property
    def env(self):
        return self._env

    @property
    def num_actions(self):
        return GymEnv.get_env_action_space_dim(self._env)

    def rollout_n_steps(self, n_steps=1000, truncate=False, **kwargs):
        """
        Do rollouts until we have have achieved `n_steps` steps in the env.

        Parameters
        ----------
        n_steps : int, the number of steps to sample in the environment
        truncate : bool, whether to truncate the last episode to the
            exact number of steps specified in `n_steps`. If False, we could
            have more steps than `n_steps` sampled.

        Returns
        ----------
        List(Rollout) : list of Rollout
        """

        steps_sampled = 0
        rollouts = []
        while steps_sampled < n_steps:
            r = self.rollout(**kwargs)
            steps_sampled += len(r.rewards)
            rollouts.append(r)

        if truncate and steps_sampled > 0 and len(rollouts[-1]) > 1:
            steps_to_remove = steps_sampled - n_steps
            rollouts[-1] = self._truncate_rollout(
                rollouts[-1], steps_to_remove)

        return rollouts

    def _truncate_rollout(self, rollout, steps_to_remove):
        r = Rollout([], [], [], [], [])
        r.rewards.extend(rollout.rewards[:-steps_to_remove])
        r.actions.extend(rollout.actions[:-steps_to_remove])
        r.states.extend(rollout.states[:-steps_to_remove])
        r.done.extend(rollout.done[:-steps_to_remove])
        r.baseline_pred.extend(
            rollout.baseline_pred[:-steps_to_remove])
        return r

    def get_baseline_pred(self, obs, done):
        if self._baseline_model and done is False:
            return self._baseline_model.predict(
                [obs]).flatten()[0]
        elif done is True:
            return 0
        return None

    def rollout(self, n_steps=None, render=False, render_freq=5,
                greedy=False):
        """
        Performs actions on the environment
        based on the agent's current weights for 1 single rollout

        render: bool, whether to render episodes in a video

        Returns
        ----------
        dict
        """
        # r = Rollout([], [], [], [], [])

        # observation = self._env.reset()
        # observation = self.get_state(observation)
        # baseline_pred = self.get_baseline_pred(observation, False)
        # for t in range(self._env.spec.timestep_limit):
        #     r.states.append(observation)
        #     r.baseline_pred.append(baseline_pred)
        #     action = self.get_action(observation, greedy=greedy)
        #     (observation, reward, done, _) = self._env.step(action)

        #     if render and t and t % render_freq == 0:
        #         self._env.render()

        #     observation = self.get_state(observation)
        #     baseline_pred = self.get_baseline_pred(observation, done)
        #     r.done.append(done)
        #     r.rewards.append(reward)
        #     r.actions.append(action)
        #     if done:
        #         break

        # return r
        
        t = 0
        reward_sum = 0
        episode_length_sum = 0
        episode_returns = []
        episode_lengths = []

        observation = self._env.reset()
        observation = self.get_state(observation)
        baseline_pred = self.get_baseline_pred(observation, False)
        action = self._env.action_space.sample()
        done = False

        # Initialize history arrays
        observations = np.array([ob for _ in range(horizon)])
        rewards = np.zeros(horizon, 'float32')
        baseline_preds = np.zeros(horizon, 'float32')
        dones = np.zeros(horizon, 'int32')
        actions = np.array([action for _ in range(horizon)])

        while True:
            action = self.get_action(observation, greedy=greedy)

            rollout = {
                "ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                "ep_rets" : ep_rets, "ep_lens" : ep_lens
            }
            if n_steps is None and done:
                yield rollout
            elif n_steps is not None:
                if t > 0 and t % n_steps == 0:
                    yield rollout
                _, vpred = pi.act(stochastic, ob)            
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, new, _ = env.step(ac)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1


    def get_discounted_cumulative_reward(self, rewards):
        """
        Parameters
        ----------
        r : list

        Returns
        ----------
        cumulative_reward : list
        """
        cumulative_reward = [0]
        for t, r in enumerate(rewards):
            temp = cumulative_reward[-1] + self._discount ** t * r
            cumulative_reward.append(temp)

        return np.sum(cumulative_reward[1:])

    def get_discounted_reward_list(self, rewards, discount=None):
        """
        Given a list of rewards, return the discounted rewards
        at each time step, in linear time
        """
        if discount is None:
            discount = self._discount

        rt = 0
        discounted_rewards = []
        for t in range(len(rewards) - 1, -1, -1):
            rt = rewards[t] + discount * rt
            discounted_rewards.append(rt)

        return list(reversed(discounted_rewards))

    def get_action(self, state, greedy=False):
        """
        Generate an action from our policy model

        Returns
        ----------
        action : numpy array or integer
        """
        batch = np.array([state])
        with self._policy.G._session.as_default():
            # a = self._policy.policy.predict(
            #     # self._policy.get_session(),
            #     batch[0], greedy)
            a, _ = self._policy.pi.act(not greedy, batch[0])
        return a

    def argmax_break_ties(self, probs):
        """
        Breaks ties randomly in an array of probabilities

        Parameters
        ----------
        probs : numpy array, shape = (1, ?)

        Returns
        ----------
        integer indicating the action that should be taken
        """
        return np.random.choice(np.where(probs == probs.max())[0])

    def get_state(self, state):
        """
        Get the state, allows for building state featurizers here
        like tile coding
        """
        return self._state_featurizer(state)


def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)            
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    # vpred = seg["vpred"]
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

import numpy as np
def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x


class BatchAgent(Agent):
    """
    Abstract class for an agent.
    """

    def __init__(self, *args, **kwargs):
        """
        discount_factor : float
            Discount rewards by this factor
        """
        super().__init__(*args, **kwargs)

    @abstractmethod
    def update(self, path):
        """
        Parameters
        ----------
        path : dict
        """
        pass

    def train(self, num_train_steps=0, num_test_steps=0,
              n_steps=1024, max_timesteps=0,
              render=False,
              whiten_advantages=True,
              truncate_rollouts=False):
        """
        Parameters
        ----------
        num_train_steps : integer
            Total number of training iterations.

        num_test_steps : integer
            Number of testing iterations per training iteration.

        n_steps : integer
            Total number of samples from the environment for each
            training iteration.

        max_timesteps : integer
            maximum number of total steps to execute in the environment

        whiten_advantages : bool, whether to whiten the advantages

        render : bool, whether to render episodes in a video

        Returns
        ----------
        None
        """
        with self._policy.get_session().as_default():
            from baselines.common import explained_variance, zipsame, dataset
            timesteps_per_batch = n_steps
            max_episodes = num_train_steps

            episodes_so_far = 0
            timesteps_so_far = 0

            assert sum([max_timesteps>0, max_episodes>0])==1

            # U.initialize()
            th_init = self._policy.get_flat()
            # MPI.COMM_WORLD.Bcast(th_init, root=0)
            self._policy.set_from_flat(th_init)
            # vfadam.sync()
            print("Init param sum", th_init.sum(), flush=True)

            # Prepare for rollouts
            # ----------------------------------------
            # seg_gen = traj_segment_generator(self._policy.pi, self.env, timesteps_per_batch, stochastic=True)
            episodes_so_far = 0
            timesteps_so_far = 0
            iters_so_far = 0

            while True:
                if max_timesteps and timesteps_so_far >= max_timesteps:
                    break
                elif max_episodes and episodes_so_far >= max_episodes:
                    break
                print("********** Iteration %i ************"%iters_so_far)

                # seg = seg_gen.__next__()
                # add_vtarg_and_adv(seg, self._gae_lambda, self._discount)

                # execute an episode
                rollouts = self.rollout_n_steps(
                    n_steps, render=render, truncate=True)

                actions = []
                states = []
                rewards = []
                advantages = []
                td_returns = []
                new = []
                baselines = []

                for rollout in rollouts:
                    print(len(rollout.rewards))
                    print(len(rollout.actions))
                    print(len(rollout.done))
                    baseline_pred = np.zeros((len(rollout.rewards)))
                    if self._baseline_model:
                        baseline_pred = self._baseline_model.predict(
                            np.array(rollout.states)).flatten()

                    is_terminal = rollout.done[-1] == 1
                    if not is_terminal:
                        # the episode did not terminate,
                        # so we truncate the last step so that we can use
                        # baseline_pred[-1] as the discounted future reward
                        rollout = self._truncate_rollout(rollout, 1)
                    else:
                        # the episode terminated, so the future reward is 0

                        # BASELINE PRED ISNT GETTING TRUNCATED THE SAME
                        # WAY AS OTHERS
                        baseline_pred = np.append(baseline_pred, 0)

                    # advantage = rollout.rewards + self._discount *\
                    #     baseline_pred[1:] - baseline_pred[:-1]
                    # advantage = self.get_discounted_reward_list(
                    #     advantage, discount=self._discount * self._gae_lambda)

                    # advantages = np.concatenate([advantages, advantage])
                    rewards.extend(rollout.rewards)
                    states.extend(rollout.states)
                    actions.extend(rollout.actions)
                    new.extend(rollout.done)
                    baselines.extend(baseline_pred)
                    # td_returns = np.concatenate(
                    #     [td_returns, baseline_pred[:-1] + advantage])

                # states = np.concatenate([s for s in states])
                # actions = np.concatenate([a for a in actions])
                # rewards = np.concatenate([a for a in rewards])
                # new = np.concatenate([a for a in new])
                # baselines = np.concatenate([a for a in baselines])

                print(len(baselines))
                print(len(new))
                print(len(rewards))


                seg = {
                    'rew': rewards,
                    'new': new,
                    'vpred': baselines[:-1],
                    'nextvpred': baselines[-1]
                }
                add_vtarg_and_adv(seg, self._gae_lambda, self._discount)


                if whiten_advantages:
                    advantages = (advantages - np.mean(advantages)) /\
                        (np.std(advantages) + 1e-8)

                # batch update the baseline model
                if isinstance(self._baseline_model, LinearFeatureBaseline):
                    self._baseline_model.fit(states, td_returns)
                elif hasattr(self._baseline_model, 'G'):
                    self._baseline_model.update(
                        states, td_returns)

                ob = states
                ac = actions
                atarg = seg['atarg']
                tdlamret = seg['tdlamret']

                # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                # ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
                # vpredbefore = seg["vpred"] # predicted value function before udpate
                # atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

                print('\nREWARDS', seg["rew"].sum(), len(atarg), '\n')

                if hasattr(self._policy.pi, "ret_rms"): self._policy.pi.ret_rms.update(tdlamret)
                if hasattr(self._policy.pi, "ob_rms"): self._policy.pi.ob_rms.update(ob) # update running mean/std for policy

                # args = seg["ob"], seg["ac"], atarg
                args = ob, ac, atarg
                fvpargs = [arr[::5] for arr in args]
                def fisher_vector_product(p):
                    return self._policy.compute_fvp(p, *fvpargs) + self.cg_damping * p

                self._policy.assign_old_eq_new() # set old parameter values to new parameter values

                *lossbefore, g = self._policy.compute_lossandgrad(*args)
                lossbefore = np.array(lossbefore)
                print(g)
                if np.allclose(g, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters, verbose=True)
                    assert np.isfinite(stepdir).all()
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / self.max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expectedimprove = g.dot(fullstep)
                    surrbefore = lossbefore[0]
                    stepsize = 1.0
                    thbefore = self._policy.get_flat()
                    for _ in range(10):
                        thnew = thbefore + fullstep * stepsize
                        self._policy.set_from_flat(thnew)
                        meanlosses = surr, kl, *_ = np.array(self._policy.compute_losses(*args))
                        improve = surr - surrbefore
                        print("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                        if not np.isfinite(meanlosses).all():
                            print("Got non-finite value of losses -- bad!")
                        elif kl > self.max_kl * 1.5:
                            print("violated KL constraint. shrinking step.")
                        elif improve < 0:
                            print("surrogate didn't improve. shrinking step.")
                        else:
                            print("Stepsize OK!")
                            break
                        stepsize *= .5
                    else:
                        print("couldn't compute a good step")
                        self._policy.set_from_flat(thbefore)

                for _ in range(3):
                    for (mbob, mbret) in dataset.iterbatches((ob, tdlamret), 
                    include_final_partial_batch=False, batch_size=64):
                        g = self._policy.compute_vflossandgrad(mbob, mbret)
                        self._policy.vfadam.update(g, 1e-3)


                def flatten_lists(listoflists):
                    return [el for list_ in listoflists for el in list_]

                # lrlocal = seg["ep_lens"]
                # lens, rews = map(flatten_lists, zip(*lrlocal))

                episodes_so_far += len(rollouts)
                timesteps_so_far += len(ob)
                iters_so_far += 1


                # print("Entropy", entbonus)
                # print("Surrgain", surrgain)
                # print("KL between old and new distribution", kloldnew)
                print("Surrogate loss before", lossbefore)
                # print("Surrogate loss after", l)


        # while True:
        #     if max_timesteps and timesteps_so_far >= max_timesteps:
        #         break
        #     elif max_episodes and episodes_so_far >= max_episodes:
        #         break

        #     seg = seg_gen.__next__()
        #     add_vtarg_and_adv(seg, self._discount, self._gae_lambda)

        #     # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        #     ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        #     vpredbefore = seg["vpred"] # predicted value function before udpate
        #     atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        #     args = seg["ob"], seg["ac"], atarg
        #     print('\nREWARDS', seg["rew"].sum() / sum(seg["new"]), len(atarg), '\n')

        #     fvpargs = [arr[::5] for arr in args]
        #     fvp_feed = self._policy.build_update_feed_dict(
        #         self._policy,
        #         fvpargs[0], fvpargs[2],
        #         fvpargs[1])
        #     # def fisher_vector_product(p):
        #     #     return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        #     def fisher_vector_product(p):
        #         fvp_feed[self._policy.flat_tangent] = p
        #         return self._policy.G(self._policy.fvp, fvp_feed) +\
        #             self.cg_damping * p

        #     # assign_old_eq_new() # set old parameter values to new parameter values
        #     self._policy.G(self._policy.set_old_pi_eq_new_pi)

        #     # with timed("computegrad"):
        #     #     *lossbefore, g = compute_lossandgrad(*args)
        #     # lossbefore = allmean(np.array(lossbefore))
        #     # g = allmean(g)
        #     feed = self._policy.build_update_feed_dict(
        #         self._policy,
        #         seg["ob"], atarg,
        #         seg["ac"])

        #     def set_from_flat(th):
        #         feed[self._policy.theta] = th
        #         self._policy.G(self._policy.sff, feed)

        #     def get_loss():
        #         # feed[self._policy.theta] = th
        #         # self._policy.G(self._policy.sff, feed)
        #         return self._policy.G(self._policy.losses[0], feed)

        #     lossbefore = get_loss()

        #     for v in self._policy.var_list:
        #         print(v, self._policy.G(v, feed).std())
        #         print(v, self._policy.G(v, feed).mean())

        #     # print(self._policy.G(self._policy.klgrads, feed))
        #     # print(self._policy.G(self._policy.kl, feed))
        #     # print(self._policy.G(self._policy.ratio, feed))
        #     g = self._policy.G(self._policy.pg, feed)
        #     print(g)
        #     print(g.max())
        #     if np.allclose(g, 0):
        #         print('Gradient zero, skipping update.')
        #     else:

        #         # if np.allclose(g, 0):
        #         #     logger.log("Got zero gradient. not updating")
        #         # else:
        #         # stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
        #         stepdir = cg(fisher_vector_product, g, self.cg_iters)

        #         assert np.isfinite(stepdir).all()
        #         shs = .5*stepdir.dot(fisher_vector_product(stepdir))
        #         lm = np.sqrt(shs / self.max_kl)
        #         # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
        #         fullstep = stepdir / lm
        #         expectedimprove = g.dot(fullstep)
        #         surrbefore = lossbefore
        #         stepsize = 1.0
        #         # thbefore = get_flat()
        #         thprev = self._policy.G(self._policy.gf, feed)
        #         for _ in range(10):
        #             thnew = thprev + fullstep * stepsize
        #             set_from_flat(thnew)
        #             # meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
        #             # improve = surr - surrbefore
        #             surr = get_loss()
        #             improve = surr - surrbefore
        #             # improve = surr - surrbefore
        #             kl = self._policy.G(self._policy.kl, feed)
        #             meanlosses = (surr, kl)

        #             print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
        #             if not np.isfinite(meanlosses).all():
        #                 print("Got non-finite value of losses -- bad!")
        #             elif kl > self.max_kl * 1.5:
        #                 print("violated KL constraint. shrinking step.")
        #             elif improve < 0:
        #                 print("surrogate didn't improve. shrinking step.")
        #             else:
        #                 print("Stepsize OK!")
        #                 break
        #             stepsize *= .5
        #         else:
        #             print("couldn't compute a good step")
        #             set_from_flat(thprev)


        #     surrafter, kloldnew, entbonus, surrgain, entropy = self._policy.G(
        #         self._policy.losses, feed_dict=feed)

        #     print("Entropy", entbonus)
        #     print("Surrgain", surrgain)
        #     print("KL between old and new distribution", kloldnew)
        #     print("Surrogate loss before", lossbefore)
        #     print("Surrogate loss after", surrafter)

        #     # for _ in range(vf_iters):
        #     #     for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]), 
        #     #     include_final_partial_batch=False, batch_size=64):
        #     #         g = allmean(compute_vflossandgrad(mbob, mbret))
        #     #         vfadam.update(g, vf_stepsize)
        #     # batch update the baseline model
        #     self._baseline_model.fit(seg["ob"], seg["tdlamret"])


        return


    # def train(self, num_train_steps=0, num_test_steps=0,
    #           n_steps=1024, max_timesteps=0,
    #           render=False,
    #           whiten_advantages=True,
    #           truncate_rollouts=False):
    #     """
    #     Parameters
    #     ----------
    #     num_train_steps : integer
    #         Total number of training iterations.

    #     num_test_steps : integer
    #         Number of testing iterations per training iteration.

    #     n_steps : integer
    #         Total number of samples from the environment for each
    #         training iteration.

    #     max_timesteps : integer
    #         maximum number of total steps to execute in the environment

    #     whiten_advantages : bool, whether to whiten the advantages

    #     render : bool, whether to render episodes in a video

    #     Returns
    #     ----------
    #     None
    #     """
    #     assert sum([num_train_steps > 0,
    #                 max_timesteps > 0]) >= 1,\
    #         "Must provide at least one limit to training"

    #     timesteps_so_far = 0
    #     train_steps_so_far = 0

    #     while True:

    #         if max_timesteps and timesteps_so_far >= max_timesteps:
    #             break
    #         elif num_train_steps and train_steps_so_far >= num_train_steps:
    #             break

    #         # execute an episode
    #         rollouts = self.rollout_n_steps(
    #             n_steps, render=render, truncate=truncate_rollouts)

    #         actions = []
    #         states = []
    #         advantages = []
    #         td_returns = []

    #         for rollout in rollouts:

    #             baseline_pred = np.zeros((len(rollout.rewards)))
    #             if self._baseline_model:
    #                 baseline_pred = self._baseline_model.predict(
    #                     np.array(rollout.states)).flatten()

    #             is_terminal = rollout.done[-1] == 1
    #             if not is_terminal:
    #                 # the episode did not terminate,
    #                 # so we truncate the last step so that we can use
    #                 # baseline_pred[-1] as the discounted future reward
    #                 rollout = self._truncate_rollout(rollout, 1)
    #             else:
    #                 # the episode terminated, so the future reward is 0
    #                 baseline_pred = np.append(baseline_pred, 0)

    #             advantage = rollout.rewards + self._discount *\
    #                 baseline_pred[1:] - baseline_pred[:-1]
    #             advantage = self.get_discounted_reward_list(
    #                 advantage, discount=self._discount * self._gae_lambda)

    #             advantages = np.concatenate([advantages, advantage])
    #             states.append(rollout.states)
    #             actions.append(rollout.actions)
    #             td_returns = np.concatenate(
    #                 [td_returns, baseline_pred[:-1] + advantage])

    #         states = np.concatenate([s for s in states])
    #         actions = np.concatenate([a for a in actions])

    #         if whiten_advantages:
    #             advantages = (advantages - np.mean(advantages)) /\
    #                 (np.std(advantages) + 1e-8)

    #         # batch update the baseline model
    #         if isinstance(self._baseline_model, LinearFeatureBaseline):
    #             self._baseline_model.fit(states, td_returns)
    #         elif hasattr(self._baseline_model, 'G'):
    #             self._baseline_model.update(
    #                 states, td_returns)

    #         # update the policy
    #         path_dict = {
    #             'states': states,
    #             'actions': actions,
    #             'td_returns': td_returns,
    #             'advantages': advantages
    #         }
    #         self.update(path_dict)

    #         timesteps_so_far += advantages.shape[0]
    #         train_steps_so_far += 1

    #         if not is_terminal:
    #             rollouts = rollouts[:-1]
    #         self.logger.add_metric('timesteps_so_far', timesteps_so_far)
    #         self.logger.add_metric('env_id', self._env_id)
    #         self.logger.set_metrics_for_rollout(rollouts, train=True)
    #         self.logger.log()

    #         if num_test_steps > 0:
    #             r = []
    #             for t_test in range(num_test_steps):
    #                 rollout = self.rollout(greedy=True)
    #                 r.append(rollout)
    #             self.logger.set_metrics_for_rollout(r, train=False)
    #             self.logger.log()

    #         if self.logger._log_dir is not None:
    #             self.save_models(self.logger._log_dir)

    #     return
