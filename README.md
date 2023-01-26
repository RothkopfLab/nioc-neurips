# Non-linear inverse optimal control (nioc)

This repository contains code for the ICML submission "Probabilistic inverse optimal control with local linearization for partially-observable systems" (Anonymous Authors).

The package `nioc` contains implementations of 

- Control algorithms `nioc.control`
    - `lqr` linear quadratic regulator
    - `lqg` linear quadratic Gaussian (LQG) control
    - `glqg` generalized LQG with signal-dependent noise (Todorov, 2005)
    - `ilqr` iterative LQR (Li & Todorov, 2004) 
    - `gilqr` generalized iterative LQR with signal-dependent noise, also known as iLQG (Li & Todorov, 2005) , equation numbers in code comments are from Li's PhD
      thesis (2006)
    - `ilqg_fixed` and `ilqr_fixed` compute one iteration of ilqg or ilqr given a fixed nominal trajectory (see Section 3.3 in the paper)

- Environments `nioc.envs`
  - `nonlinear_reaching.py` non-linear reaching task (Li & Todorov, 2007)
  - `navigation.py` navigation task
  - `classic_control.pendulum.py` classic inverted pendulum control problem
  - `classic_control.cartpole.py` classic cartpole control problem

- Environment wrappers `nioc.envs.wrappers`
    - `FullyObservedWrapper` turns a partially observed problem into a fully observed problem
    - `EKFWrapper` wraps a partially observed problem with an extended Kalman filter (EKF) to turn it into a belief space problem

- Paramter inference algorithms `nioc.infer`
    - `inv_ilqr.py` inverse iterative (generalized) LQR (fully observable)
    - `inv_ilqg.py` inverse iterative (generalized) LQG (partially observable)
