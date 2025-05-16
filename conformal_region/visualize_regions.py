import matplotlib.pyplot as plt
import numpy as np
from conformal_region.get_regions import ConformalRegion

def plot_scores_and_regions(
    score_t_tau_paris: list[tuple[int, int]],
    region_t_tau_pairs: list[tuple[int, int]],
    traj_path = "data/random_5_p3_p1.npy",
    pred_path = "data/random_5_p3_p1_pred_20.npy",
    pred_horizon = 20,
    failure_rate = 0.1,
):
    """
    Plot the nonconformity scores and valid prediction regions for the given time step and prediction horizon pairs.

    Args:
        score_t_tau_pairs (list[tuple[int, int]]): List of tuples containing time step and prediction horizon pairs.
        region_t_tau_pairs (list[tuple[int, int]]): List of tuples containing time step and prediction horizon pairs.
        traj_path (str): Path to the trajectory data file.
        pred_path (str): Path to the prediction data file.
        pred_horizon (int): Prediction horizon.
        failure_rate (float): Failure rate for the prediction regions.
    """
    conformal_region = ConformalRegion(traj_path, pred_path, pred_horizon, failure_rate, single_agent=True)

    t_tau_pair_to_scores = {}
    t_tau_pair_to_regions = {}

    for t, tau in score_t_tau_paris:
        nonconformity_scores = conformal_region._get_conformity_scores(t, tau)
        t_tau_pair_to_scores[(t, tau)] = nonconformity_scores

    for t, tau in region_t_tau_pairs:
        valid_region = conformal_region.construct_valid_prediction_regions(t, tau)
        t_tau_pair_to_regions[(t, tau)] = valid_region

    # Plotting
    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for idx, ((t, tau), scores) in enumerate(t_tau_pair_to_scores.items()):
        label = f"t={t}, τ={t+tau}"
        plt.hist(
            scores,
            bins=30,
            alpha=0.4,
            label=f"Scores R ({label})",
            color=colors[idx % len(colors)],
            density=True,
        )

        if (t, tau) in t_tau_pair_to_regions:
            region = t_tau_pair_to_regions[(t, tau)]
            plt.axvline(region, linestyle="--", color=colors[idx % len(colors)], label=f"Region C ({label})")

    plt.xlabel("Nonconformity Score")
    plt.ylabel("Density")
    plt.title("Nonconformity Scores and Conformal Prediction Regions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("nonconformity_scores_and_regions.png")

if __name__ == "__main__":
    traj_path = "dataset/random_5_p1_1_p5.npy"
    pred_path = "dataset/random_5_p1_1_p5_pred_20.npy"
    pred_horizon = 20
    failure_rate = 0.1

    # Example time step and prediction horizon pairs
    score_t_tau_pairs = [(10, 5), (10, 10)]
    region_t_tau_pairs = [(10, 5), (10, 10)]

    plot_scores_and_regions(
        score_t_tau_pairs,
        region_t_tau_pairs,
        traj_path,
        pred_path,
        pred_horizon,
        failure_rate,
    )

    