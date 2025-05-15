## Particle Collision Avoidance

This repository implements the *Safe Planning in Dynamic Environment using Conformal Prediction* framework in a simplified particle avoidance environment.

### File Structure

- `/env` contains the environment implementation, which includes the environment dynamic, time delta, and adversarial configurations.
- `/conformal_region` contains implementation of calculating CP regions from a trajectory prediction dataset. 
- `/mpc` contains implementation of MPC solver
- `/scripts` contains scripts for playning and visualizing the environment. `collect.py` is used to generate trajectory datasets
- `/notebook` contains experiments and interactive scripts. `generate_prediction_dataset.ipynb` contains scripts for training the trajectory predictor and generate calibration dataset. `demo_generation.ipynb` contains the experiments.


### Dependencies
- numpy
- cvxpy
- gymnasium
- matplotlib
- jupyterlab

### Play
- `python scripts/demo.py`