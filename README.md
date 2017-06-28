
## yarlp

##### TODO:

- Make an experimentation method
    - compare performance to other packages for cem/reinforce, rllab
        - make sure keras policy is computing pi and loss correctly -> compare to rllab
        - run rllab reinforce on different envs with similar specs and compare outputs to my package
        - submit to openai
    - add continuous action spaces for CEM/Reinforce


- make a method to load model from file with all the args (including models)
- add method for different networks as args
- Add policy classes continuous gaussian, categorical, and deterministic, random uniform
    - Make generic q-function and policy models handling both continuous/deterministic action/state spaces

- DQ, DDQ
- TRPO
- A3C
- ddpg, dpg, COPDAC-Q
    - Add batch-norm to ddpg, http://ruishu.io/2016/12/27/batchnorm/
    - add action to nth-layer of Q
- Classic policy and value based methods
- Covariance Matrix Adaptation
