# Non-linear inverse optimal control (nioc)

Contains implementations of 

- Control algorithms `nioc.control`
    - `ilqr` iterative LQR (Li & Todorov, 2004) 
    - `gilqr` with `lqr=glqr` generalized iterative LQR (Li & Todorov, 2005) with signal-dependent noise, equation numbers in code comments are from Li's PhD
      thesis (2006)

- Environments `nioc.envs`
    - nonlinear environments, for which our new inference methods have been tested
      - `nonlinear_reaching.py` non-linear reaching task (Li & Todorov, 2007)
      - 'navigation.py' navigation task
      - `classic_control.pendulum.py` classic inverted pendulum control problem
      - `classic_control.cartpole.py` classic cartpole control problem

- Environment wrappers `nioc.envs.wrappers`
    - `FullyObservedWrapper` turns a partially observed problem into a fully observed problem
    - `EKFWrapper` wraps a partially observed problem with an extended Kalman filter (EKF) to turn it into a belief space problem

- Paramter inference algorithms `nioc.infer`
    - `inv_ilqr.py` inverse iterative (generalized) LQR (fully observable)
    - `inv_ilqg.py` inverse iterative (generalized) LQG (partially observable)
