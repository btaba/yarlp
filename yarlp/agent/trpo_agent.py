"""
TRPO
"""

import numpy as np
import tensorflow as tf

from yarlp.agent import base_agent
from yarlp.model.networks import normc_initializer
from yarlp.model.model_factories import trpo_model_factory
from yarlp.model.linear_baseline import LinearFeatureBaseline
from yarlp.utils.experiment_utils import get_network
from yarlp.model.model_factories import value_function_model_factory


from mpi4py import MPI


class TRPOAgent(base_agent.BatchAgent):
    """
    TRPO

    Parameters
    ----------
    env : gym.env

    policy_network : model.Model

    baseline_network : if None, we us no baseline
        otherwise we use a LinearFeatureBaseline as default
        you can also pass in a function as a tensorflow network which
        gets built by the value_function_model_factory

    model_file_path : str, file path for the policy_network
    """

    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_network_params={},
                 baseline_network=None,
                 baseline_model_learning_rate=1e-3,
                 baseline_train_iters=3,
                 baseline_network_params={'final_weights_initializer': normc_initializer(1.0)},
                 model_file_path=None,
                 adaptive_std=False,
                 gae_lambda=0.98, cg_iters=10,
                 cg_damping=1e-1, max_kl=1e-2,
                 input_shape=None,
                 init_std=1.0, min_std=1e-6,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        import baselines.common.tf_util as U
        from baselines.common import set_global_seeds
        from baselines.common.mpi_adam import MpiAdam
        from baselines.ppo1.mlp_policy import MlpPolicy
        from baselines.common import explained_variance, zipsame
        self.sess = sess = U.single_threaded_session()
        sess.__enter__()

        set_global_seeds(5 + 1000)

        def policy_func(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                hid_size=32, num_hid_layers=2)

        env = self._env
        env.seed(1000 + 5)

        ob_space = env.observation_space
        ac_space = env.action_space
        pi = policy_func("pi", ob_space, ac_space)
        self._policy = pi
        oldpi = policy_func("oldpi", ob_space, ac_space)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

        ob = U.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = U.mean(kloldnew)
        meanent = U.mean(ent)
        entbonus = 0 * meanent

        vferr = U.mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
        surrgain = U.mean(ratio * atarg)

        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

        dist = meankl

        all_var_list = pi.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        self.vfadam = vfadam = MpiAdam(vf_var_list)

        self.get_flat = get_flat = U.GetFlat(var_list)
        self.set_from_flat = set_from_flat = U.SetFromFlat(var_list)
        klgrads = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
        fvp = U.flatgrad(gvp, var_list)

        self.assign_old_eq_new = assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        self.compute_losses = compute_losses = U.function([ob, ac, atarg], losses)
        self.compute_lossandgrad = compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
        self.compute_fvp = compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
        self.compute_vflossandgrad = compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
        self.compute_vfloss = U.function([ob, ret], vferr)


        optimizer = tf.train.AdamOptimizer(
            learning_rate=1e-2)
        vfadam2 = optimizer.minimize(vferr)
        self.compute_vflossandgrad2 = U.function([ob, ret], vfadam2)

        U.initialize()
        th_init = get_flat()
        set_from_flat(th_init)
        vfadam.sync()
        print("Init param sum", th_init.sum(), flush=True)

        # policy_network = get_network(policy_network, policy_network_params)

        # self._policy = trpo_model_factory(
        #     env, network=policy_network, network_params=policy_network_params,
        #     min_std=min_std, init_std=init_std, adaptive_std=adaptive_std,
        #     input_shape=input_shape, model_file_path=model_file_path)

        # policy_weight_sums = sum(
        #     [np.sum(a) for a in self._policy.get_weights()])
        # self.logger._logger.info(
        #     'Policy network weight sums: {}'.format(policy_weight_sums))

        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.max_kl = max_kl
        self._gae_lambda = gae_lambda
        self.baseline_train_iters = baseline_train_iters

        if isinstance(baseline_network, LinearFeatureBaseline):
            self._baseline_model = baseline_network
        elif baseline_network is None:
            self._baseline_model = LinearFeatureBaseline()
        else:
            baseline_network = get_network(baseline_network, baseline_network_params)
            self._baseline_model = value_function_model_factory(
                env, baseline_network,
                learning_rate=baseline_model_learning_rate)

    # def update(self, path):
    #     # update the policy
    #     feed = self._policy.build_update_feed_dict(
    #         self._policy,
    #         path['states'], path['advantages'],
    #         path['actions'])
    #     thprev = self._policy.G(self._policy.gf, feed)
    #     self._policy.G(self._policy.set_old_pi_eq_new_pi)

    #     fvp_feed = self._policy.build_update_feed_dict(
    #         self._policy,
    #         path['states'][::5], path['advantages'][::5],
    #         path['actions'][::5])

    #     def fisher_vector_product(p):
    #         fvp_feed[self._policy.flat_tangent] = p
    #         return self._policy.G(self._policy.fvp, fvp_feed) +\
    #             self.cg_damping * p

    #     g = self._policy.G(self._policy.pg, feed)
    #     if np.allclose(g, 0):
    #         print('Gradient zero, skipping update.')
    #         return

    #     # descent direciton
    #     stepdir = conjugate_gradient(fisher_vector_product, g, self.cg_iters)

    #     def get_loss(th):
    #         feed[self._policy.theta] = th
    #         self._policy.G(self._policy.sff, feed)
    #         return self._policy.G(self._policy.losses[0], feed)
    #     lossbefore = get_loss(thprev)

    #     assert np.isfinite(stepdir).all()
    #     shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
    #     lm = np.sqrt(shs / self.max_kl)
    #     fullstep = stepdir / lm
    #     expectedimprove = g.dot(fullstep)
    #     surrbefore = lossbefore
    #     stepsize = 1.0

    #     for _ in range(10):
    #         thnew = thprev - fullstep * stepsize  # plus or minus?

    #         surr = get_loss(thnew)
    #         kl = self._policy.G(self._policy.kl, feed)

    #         improve = surrbefore - surr
    #         print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
    #         if not np.isfinite(surr).all():
    #             print("Got non-finite value of losses -- bad!")
    #         elif kl > self.max_kl * 1.5:
    #             print("violated KL constraint. shrinking step.")
    #         elif improve < 0:
    #             print("surrogate didn't improve. shrinking step.")
    #         else:
    #             print("Stepsize OK!")
    #             break
    #         stepsize *= .5
    #     else:
    #         print("couldn't compute a good step")
    #         get_loss(thprev)

    #     surrafter, kloldnew, entropy = self._policy.G(
    #         self._policy.losses, feed_dict=feed)

    #     print("Entropy", entropy)
    #     print("KL between old and new distribution", kloldnew)
    #     print("Surrogate loss before", lossbefore)
    #     print("Surrogate loss after", surrafter)

    #     return

    def update(self, rollout):
        """
        Parameters
        ----------

        """
        seg = rollout
        pi = self._policy
        from baselines.common import explained_variance, zipsame, dataset
        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["observations"], seg["actions"], seg["advantages"], seg["discounted_future_reward"]
        vpredbefore = seg["baseline_preds"] # predicted value function before udpate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        # print('Ob RMS', self.sess.run(pi.ob_rms.mean))
        # print('Ob RMS', self.sess.run(pi.ob_rms.std))
        # print('My Ob RMS', self._env._obs_rms._mean)
        # print('My Ob RMS', self._env._obs_rms._std)

        args = seg["observations"], seg["actions"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return self.compute_fvp(p, *fvpargs) + self.cg_damping * p

        self.assign_old_eq_new() # set old parameter values to new parameter values

        *lossbefore, g = self.compute_lossandgrad(*args)
        lossbefore = np.array(lossbefore)
        if np.allclose(g, 0):
            print("Got zero gradient. not updating")
        else:
            stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters, verbose=True)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.max_kl)
            # print("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                meanlosses = surr, kl, *_ = np.array(self.compute_losses(*args))
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
                self.set_from_flat(thbefore)

        # for (lossname, lossval) in zip(loss_names, meanlosses):
        #     logger.record_tabular(lossname, lossval)

        # print("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))


        # with self._policy.get_session().as_default():
        #     from baselines.common import explained_variance, zipsame, dataset
        #     n_steps = n_steps
        #     num_train_steps = num_train_steps

        #     episodes_so_far = 0
        #     timesteps_so_far = 0

        #     assert sum([max_timesteps>0, num_train_steps>0])==1

        #     # U.initialize()
        #     th_init = self._policy.get_flat()
        #     # MPI.COMM_WORLD.Bcast(th_init, root=0)
        #     self._policy.set_from_flat(th_init)
        #     # vfadam.sync()
        #     print("Init param sum", th_init.sum(), flush=True)

        #     # Prepare for rollouts
        #     # ----------------------------------------
        #     gen = traj_segment_generator(self._policy.pi, self.env, n_steps, stochastic=True)
        #     episodes_so_far = 0
        #     timesteps_so_far = 0
        #     iters_so_far = 0
        #     # gen2 = rollout(self._policy.pi, self._env, n_steps, greedy=False)

        #     while True:
        #         if max_timesteps and timesteps_so_far >= max_timesteps:
        #             break
        #         elif num_train_steps and episodes_so_far >= num_train_steps:
        #             break
        #         print("********** Iteration %i ************"%iters_so_far)

        #         seg = next(gen)
        #         # seg = next(gen2)

        #         # print(len(seg['episode_lengths']))
        #         # print(seg[])
        #         # print(seg2)
        #         # seg = seg_gen.__next__()
        #         add_vtarg_and_adv2(seg, self._gae_lambda, self._discount)

        #         ob, ac, atarg, tdlamret = seg["observations"], seg["actions"], seg["adv"], seg["tdlamret"]
        #         vpredbefore = seg["baseline_preds"] # predicted value function before udpate
        #         atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        #         # if whiten_advantages:
        #         #     # advantages = (advantages - np.mean(advantages)) /\
        #         #     #     (np.std(advantages) + 1e-8)
        #         #     seg['atarg'] = (seg['adv'] - np.mean(seg['adv'])) / (np.std(seg['adv']) + 1e-8)

        #         # # batch update the baseline model
        #         # if isinstance(self._baseline_model, LinearFeatureBaseline):
        #         #     self._baseline_model.fit(seg["observations"], seg["tdlamret"])
        #         # elif hasattr(self._baseline_model, 'G'):
        #         #     # self._baseline_model.update(
        #         #     #     states, td_returns)
        #         #     raise

        #         # ob = seg["observations"]
        #         # ac = seg["actions"]
        #         # atarg = seg['atarg']
        #         # tdlamret = seg['tdlamret']

        #         # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        #         # ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        #         # vpredbefore = seg["vpred"] # predicted value function before udpate
        #         # atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        #         print('\nREWARDS', np.mean(seg["episode_returns"]), len(seg['episode_lengths']), '\n')

        #         if hasattr(self._policy.pi, "ret_rms"): self._policy.pi.ret_rms.update(tdlamret)
        #         if hasattr(self._policy.pi, "ob_rms"): self._policy.pi.ob_rms.update(ob) # update running mean/std for policy

        #         # args = seg["ob"], seg["ac"], atarg
        #         args = ob, ac, atarg
        #         fvpargs = [arr[::5] for arr in args]
        #         def fisher_vector_product(p):
        #             return self._policy.compute_fvp(p, *fvpargs) + self.cg_damping * p

        #         self._policy.assign_old_eq_new() # set old parameter values to new parameter values

        #         *lossbefore, g = self._policy.compute_lossandgrad(*args)
        #         lossbefore = np.array(lossbefore)
        #         print(g)
        #         if np.allclose(g, 0):
        #             print("Got zero gradient. not updating")
        #         else:
        #             stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters, verbose=True)
        #             assert np.isfinite(stepdir).all()
        #             shs = .5*stepdir.dot(fisher_vector_product(stepdir))
        #             lm = np.sqrt(shs / self.max_kl)
        #             # print("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
        #             fullstep = stepdir / lm
        #             expectedimprove = g.dot(fullstep)
        #             surrbefore = lossbefore[0]
        #             stepsize = 1.0
        #             thbefore = self._policy.get_flat()
        #             for _ in range(10):
        #                 thnew = thbefore + fullstep * stepsize
        #                 self._policy.set_from_flat(thnew)
        #                 meanlosses = surr, kl, *_ = np.array(self._policy.compute_losses(*args))
        #                 improve = surr - surrbefore
        #                 print("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
        #                 if not np.isfinite(meanlosses).all():
        #                     print("Got non-finite value of losses -- bad!")
        #                 elif kl > self.max_kl * 1.5:
        #                     print("violated KL constraint. shrinking step.")
        #                 elif improve < 0:
        #                     print("surrogate didn't improve. shrinking step.")
        #                 else:
        #                     print("Stepsize OK!")
        #                     break
        #                 stepsize *= .5
        #             else:
        #                 print("couldn't compute a good step")
        #                 self._policy.set_from_flat(thbefore)

        #         for _ in range(3):
        #             for (mbob, mbret) in dataset.iterbatches((ob, tdlamret), 
        #             include_final_partial_batch=False, batch_size=64):
        #                 g = self._policy.compute_vflossandgrad(mbob, mbret)
        #                 self._policy.vfadam.update(g, 1e-3)


        #         def flatten_lists(listoflists):
        #             return [el for list_ in listoflists for el in list_]

        #         # lrlocal = seg["ep_lens"]
        #         # lens, rews = map(flatten_lists, zip(*lrlocal))

        #         # episodes_so_far += len(rollouts)
        #         # timesteps_so_far += len(ob)
        #         episodes_so_far += len(seg['episode_lengths'])
        #         timesteps_so_far += sum(seg['episode_lengths'])
        #         iters_so_far += 1


        #         # print("Entropy", entbonus)
        #         # print("Surrgain", surrgain)
        #         # print("KL between old and new distribution", kloldnew)
        #         print("Surrogate loss before", lossbefore)
        #         # print("Surrogate loss after", l)


        # while True:
        #     if max_timesteps and timesteps_so_far >= max_timesteps:
        #         break
        #     elif num_train_steps and episodes_so_far >= num_train_steps:
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
        #         #     print("Got zero gradient. not updating")
        #         # else:
        #         # stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
        #         stepdir = cg(fisher_vector_product, g, self.cg_iters)

        #         assert np.isfinite(stepdir).all()
        #         shs = .5*stepdir.dot(fisher_vector_product(stepdir))
        #         lm = np.sqrt(shs / self.max_kl)
        #         # print("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
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

import tensorflow as tf


def conjugate_gradient(f_Ax, b, cg_iters=10, callback=None,
                       verbose=True, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)

    if verbose:
        print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))

    return x


def test_cg():
    A = np.random.randn(5, 5)
    A = A.T.dot(A)
    b = np.random.randn(5)
    x = conjugate_gradient(lambda x: A.dot(x), b, cg_iters=5, verbose=True)
    assert np.allclose(A.dot(x), b)


def linesearch(f, x, fullstep, expected_improve_rate):
    backtrack_ratio = 0.8
    accept_ratio = .1
    max_backtracks = 15
    fval = f(x)

    for (_n_backtracks, stepfrac) in \
            enumerate(backtrack_ratio ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew
    return False, x
