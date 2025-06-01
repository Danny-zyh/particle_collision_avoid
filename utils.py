import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from IPython.display import HTML


def animate_trajectory(
    simulation_trajectory, xlim=(0, 5), ylim=(0, 5), interval=200, radii=0.4
):
    n_steps = len(simulation_trajectory)
    agent_pos = np.array(
        [step["agent"][:2] for step in simulation_trajectory]
    )  # (n_steps, 2)
    landmark = simulation_trajectory[0]["landmark"][:2]
    m = len(simulation_trajectory[0]["adversaries"])
    adv_pos = np.stack(
        [
            [step["adversaries"][i][:2] for step in simulation_trajectory]
            for i in range(m)
        ],
        axis=1,
    )  # shape (n_steps, m, 2)

    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    # scatter plots
    agent_scatter = ax.scatter([], [], c="blue", s=50, label="Agent")
    adv_scatter = ax.scatter([], [], c="red", s=50, label="Adversaries")
    landmark_scatter = ax.scatter(
        *landmark, c="green", marker="*", s=100, label="Landmark"
    )
    ax.legend(loc="upper left")

    # create one Circle patch per adversary
    circles = []
    for i in range(m):
        c = Circle(
            (np.nan, np.nan),  # initial off‐screen
            radii,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
        ax.add_patch(c)
        circles.append(c)

    def init():
        agent_scatter.set_offsets(np.empty((0, 2)))
        adv_scatter.set_offsets(np.empty((0, 2)))
        # hide all circles initially
        for c in circles:
            c.center = (np.nan, np.nan)
        return [agent_scatter, adv_scatter, landmark_scatter] + circles

    def update(frame):
        # update scatter positions
        agent_scatter.set_offsets(agent_pos[frame])
        adv_scatter.set_offsets(adv_pos[frame])

        # update each circle's center
        for i, c in enumerate(circles):
            c.center = tuple(adv_pos[frame, i])
        return [agent_scatter, adv_scatter, landmark_scatter] + circles

    anim = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        init_func=init,
        blit=True,
        interval=interval,
    )
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_with_predictions(
    simulation_trajectory,
    obs_preds_log,
    obs_radii_log,
    xlim=(0, 5),
    ylim=(0, 5),
    interval=200,
):
    """
    Animate actual trajectories along with predicted obstacle circles, prediction paths,
    and historical trajectories of agent and adversaries over the full prediction horizon.

    - simulation_trajectory: list of dict with keys 'agent','adversaries','landmark'
    - obs_preds_log: np.ndarray, shape (T_log, m, H, 2)
    - obs_radii_log: np.ndarray, shape (T_log, m, H)
    """
    # determine common frame count
    T_sim = len(simulation_trajectory)
    T_pred = obs_preds_log.shape[0]
    T_rad = obs_radii_log.shape[0]
    T = min(T_sim, T_pred, T_rad)
    if T < max(T_sim, T_pred, T_rad):
        print(
            f"Warning: clipping animation to {T} frames "
            f"(sim {T_sim}, preds {T_pred}, radii {T_rad})"
        )

    m, H = obs_preds_log.shape[1], obs_preds_log.shape[2]
    # actual agent positions (T,2)
    agent_all = np.array([step["agent"][:2] for step in simulation_trajectory[:T]])
    # adversary positions history (m,T,2)
    adv_all = np.stack(
        [
            np.array([step["adversaries"][i][:2] for step in simulation_trajectory[:T]])
            for i in range(m)
        ],
        axis=0,
    )
    landmark = simulation_trajectory[0]["landmark"][:2]

    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    # scatter for current positions
    agent_scatter = ax.scatter([], [], c="blue", s=50, label="Agent")
    adv_scatter = ax.scatter([], [], c="red", s=50, label="Adversaries")
    landmark_scatter = ax.scatter(
        *landmark, c="green", marker="*", s=100, label="Landmark"
    )

    # history lines for agent and adversaries
    (agent_hist_line,) = ax.plot(
        [], [], "-", linewidth=2, alpha=0.6, label="Agent History"
    )
    adv_hist_lines = []
    for i in range(m):
        (line,) = ax.plot(
            [],
            [],
            "--",
            linewidth=1.5,
            alpha=0.6,
            label="Adversary History" if i == 0 else "",
        )
        adv_hist_lines.append(line)

    # predicted path lines
    pred_path_lines = []
    for i in range(m):
        (line,) = ax.plot(
            [],
            [],
            "-",
            linewidth=1.5,
            alpha=0.7,
            color="orange",
            label="Prediction Path" if i == 0 else "",
        )
        pred_path_lines.append(line)

    # predicted obstacle circles
    circles = []
    for _ in range(m * H):
        circ = Circle(
            (0, 0),
            radius=0,
            fill=False,
            edgecolor="orange",
            linewidth=1,
            linestyle="--",
            alpha=0.6,
        )
        ax.add_patch(circ)
        circles.append(circ)

    # legend in upper left
    # ax.legend(loc='upper left')

    def init():
        agent_scatter.set_offsets(np.empty((0, 2)))
        adv_scatter.set_offsets(np.empty((0, 2)))
        agent_hist_line.set_data([], [])
        for line in adv_hist_lines + pred_path_lines:
            line.set_data([], [])
        for c in circles:
            c.set_center((0, 0))
            c.set_radius(0)
        return (
            [agent_scatter, adv_scatter, landmark_scatter, agent_hist_line]
            + adv_hist_lines
            + pred_path_lines
            + circles
        )

    def update(frame):
        # current positions
        agent_scatter.set_offsets(agent_all[frame])
        adv_scatter.set_offsets(adv_all[:, frame, :])

        # historical trajectories
        agent_hist_line.set_data(agent_all[: frame + 1, 0], agent_all[: frame + 1, 1])
        for i, line in enumerate(adv_hist_lines):
            hist = adv_all[i, : frame + 1, :]
            line.set_data(hist[:, 0], hist[:, 1])

        # predicted paths and circles
        preds = obs_preds_log[frame]  # (m, H, 2)
        rats = obs_radii_log[frame]  # (m, H)
        idx = 0
        for i in range(m):
            path = preds[i]  # (H,2)
            xs, ys = path[:, 0], path[:, 1]
            pred_path_lines[i].set_data(xs, ys)
            for h in range(H):
                x_c, y_c = path[h]
                r = rats[i, h]
                circles[idx].set_center((float(x_c), float(y_c)))
                circles[idx].set_radius(float(r))
                idx += 1

        return (
            [agent_scatter, adv_scatter, landmark_scatter, agent_hist_line]
            + adv_hist_lines
            + pred_path_lines
            + circles
        )

    anim = FuncAnimation(
        fig, update, frames=T, init_func=init, blit=True, interval=interval
    )
    plt.close(fig)
    return HTML(anim.to_jshtml())
