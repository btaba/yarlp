
## yarlp

Yet Another Reinforcement Learning Package

##### TODO:

* TRPO
    - fix TRPO on mountain car!!!!!!!
        - add pi.ob_rms.update(ob) # update running mean/std for policy
        - Action should be able to reshape as necessary
        - fix Categorical dist
    - benchmark

* CEM - add policy objects

* DQ, DDQ
* load model from file with args and model weights
* Classic policy and value based methods
* ddpg
* parallel sampler for rollouts
* handle reshaping of arrays for different action/obs spaces
* black box optimizer on params
