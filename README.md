# Non-linear inverse optimal control (nioc)

This repository contains code for the NeurIPS submission "Probabilistic inverse optimal control for non-linear partially observable systems disentangles perceptual uncertainty and behavioral costs" (Anonymous Authors).

## Install requirements

The easiest way is to create a fresh virtual environment and install the `nioc` package using pip:

```bash
python -m venv env
source env/bin/activate
python -m pip -e .
```

## Running the example script


```bash
python reaching_example.py
```

runs an example that simulates trajectories from the reaching task, estimates the parameters using our method and the baseline, and plots simulated trajectories using the parameter estimates.

## Reproducing figures

- Figure 2 (trajectories and log likelihood): `fig2-likelihood.py`
- Figure 4 (light-dark domain): `fig4-infoseeking.py`

## Package `nioc`
The package `nioc` contains implementations of 

- Control algorithms `nioc.control`
    - `lqr` linear quadratic regulator
    - `lqg` linear quadratic Gaussian (LQG) control
    - `glqg` generalized LQG with signal-dependent noise (Todorov, 2005)
    - `ilqr` iterative LQR (Li & Todorov, 2004) 
    - `gilqr` generalized iterative LQR with signal-dependent noise, also known as iLQG (Li & Todorov, 2005) , equation numbers in code comments are from Li's PhD
      thesis (2006)
    - `gilqg` generalized iterative LQG with signal-dependent noise, also known as iLQG (Todorov & Li, 2007), equation numbers in comments are from Li's PhD thesis (2007)
    - `ilqg_fixed` and `ilqr_fixed` compute one iteration of ilqg or ilqr given a fixed nominal trajectory (see Section 3.3 in the paper)

- Environments `nioc.envs`
  - `nonlinear_reaching.py` non-linear reaching task (Li & Todorov, 2007)
  - `navigation.py` navigation task
  - `classic_control.pendulum.py` classic inverted pendulum control problem
  - `classic_control.cartpole.py` classic cartpole control problem
  - `lightdark.py` light-dark domain (Platt et al., 2010)

- Environment wrappers `nioc.envs.wrappers`
    - `FullyObservedWrapper` turns a partially observed problem into a fully observed problem
    - `EKFWrapper` wraps a partially observed problem with an extended Kalman filter (EKF) to turn it into a belief space problem

- Paramter inference algorithms `nioc.infer`
    - `inv_ilqr.py` inverse iterative (generalized) LQR (fully observable)
    - `inv_ilqg.py` inverse iterative (generalized) LQG (partially observable)
    - `inv_maxent.py` maximum entropy IOC baseline (Section 4.1)
