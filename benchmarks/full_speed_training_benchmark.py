import argparse
import csv
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from async_gym_agents.agents.async_agent import (
    get_fast_injected_agent,
    get_injected_agent,
)
from async_gym_agents.envs.multi_env import IndexableMultiEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class EpisodeRow:
    mode: str
    seed: int
    timestep: int
    wall_seconds: float
    cpu_seconds: float
    episode_reward: float
    episode_length: int


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, mode: str, seed: int):
        super().__init__()
        self.mode = mode
        self.seed = seed
        self.rows: list[EpisodeRow] = []
        self._wall_start = 0.0
        self._cpu_start = 0.0

    def _on_training_start(self) -> None:
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            self.rows.append(
                EpisodeRow(
                    mode=self.mode,
                    seed=self.seed,
                    timestep=self.model.num_timesteps,
                    wall_seconds=time.perf_counter() - self._wall_start,
                    cpu_seconds=time.process_time() - self._cpu_start,
                    episode_reward=float(np.asarray(episode["r"]).item()),
                    episode_length=int(np.asarray(episode["l"]).item()),
                )
            )
        return True


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)
    return env


def build_model(mode: str, env_id: str, seed: int, workers: int, total_timesteps: int):
    torch.manual_seed(seed)
    env_fns = [partial(make_env, env_id, seed * 1000 + i) for i in range(workers)]
    env = IndexableMultiEnv(env_fns, env_fns[0]())
    is_full_speed = mode == "full_speed"
    Agent = get_fast_injected_agent(DQN) if is_full_speed else get_injected_agent(DQN)
    speed_kwargs = (
        dict(
            full_speed_collect_steps=32,
            full_speed_train_steps=2,
            full_speed_max_train_bursts=8,
        )
        if is_full_speed
        else {}
    )

    return Agent(
        "MlpPolicy",
        env,
        seed=seed,
        use_mp=False,
        max_episodes_in_buffer=workers * 2,
        learning_starts=256,
        buffer_size=max(20_000, total_timesteps * 2),
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.35,
        exploration_final_eps=0.05,
        verbose=0,
        **speed_kwargs,
    )


def run_one(
    mode: str,
    env_id: str,
    seed: int,
    workers: int,
    total_timesteps: int,
) -> list[EpisodeRow]:
    callback = EpisodeMetricsCallback(mode, seed)
    model = build_model(mode, env_id, seed, workers, total_timesteps)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        return callback.rows
    finally:
        model.shutdown()


def write_rows(rows: list[EpisodeRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EpisodeRow.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def step_curve(
    df: pd.DataFrame, x_col: str, y_col: str, grid: np.ndarray
) -> np.ndarray:
    if df.empty:
        return np.full_like(grid, np.nan, dtype=float)
    ordered = df.sort_values(x_col)
    x = ordered[x_col].to_numpy(dtype=float)
    y = ordered[y_col].to_numpy(dtype=float)
    idx = np.searchsorted(x, grid, side="right") - 1
    out = np.full_like(grid, np.nan, dtype=float)
    valid = idx >= 0
    out[valid] = y[idx[valid]]
    return out


def mean_ci(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(curves)
    count = np.sum(~np.isnan(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    centered = np.where(np.isnan(stacked), 0.0, stacked - mean)
    variance = np.divide(
        np.sum(centered * centered, axis=0),
        count - 1,
        out=np.zeros_like(mean),
        where=count > 1,
    )
    std = np.sqrt(variance)
    ci = np.divide(
        1.96 * std,
        np.sqrt(count),
        out=np.zeros_like(mean),
        where=count > 1,
    )
    return mean, ci


def plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    grid_size: int = 160,
) -> None:
    max_x = float(data[x_col].max())
    grid = np.linspace(0.0, max_x, grid_size)
    for mode, mode_df in data.groupby("mode"):
        curves = [
            step_curve(seed_df, x_col, y_col, grid)
            for _, seed_df in mode_df.groupby("seed")
        ]
        mean, ci = mean_ci(curves)
        ax.plot(grid, mean, label=mode)
        ax.fill_between(grid, mean - ci, mean + ci, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_results(rows_path: Path, plot_path: Path) -> None:
    data = pd.read_csv(rows_path)
    if data.empty:
        raise RuntimeError("No completed episodes were recorded.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_panel(
        axes[0, 0],
        data,
        "timestep",
        "episode_reward",
        "Reward vs Env Steps",
        "environment steps",
        "episode reward",
    )
    plot_panel(
        axes[0, 1],
        data,
        "wall_seconds",
        "episode_reward",
        "Reward vs Wall Time",
        "wall seconds",
        "episode reward",
    )
    plot_panel(
        axes[1, 0],
        data,
        "cpu_seconds",
        "episode_reward",
        "Reward vs CPU Time",
        "CPU seconds",
        "episode reward",
    )
    plot_panel(
        axes[1, 1],
        data,
        "wall_seconds",
        "timestep",
        "Throughput",
        "wall seconds",
        "environment steps",
    )
    axes[0, 1].legend()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def write_summary(rows_path: Path, summary_path: Path) -> None:
    data = pd.read_csv(rows_path)
    summary_rows = []
    for (mode, seed), run in data.groupby(["mode", "seed"]):
        last = run.sort_values("timestep").iloc[-1]
        summary_rows.append(
            {
                "mode": mode,
                "seed": seed,
                "episodes": len(run),
                "final_reward": last.episode_reward,
                "mean_last_5_reward": run.tail(5).episode_reward.mean(),
                "wall_seconds": last.wall_seconds,
                "cpu_seconds": last.cpu_seconds,
                "steps_per_wall_second": last.timestep / last.wall_seconds,
                "steps_per_cpu_second": last.timestep / last.cpu_seconds,
            }
        )

    summary = pd.DataFrame(summary_rows)
    aggregate = (
        summary.groupby("mode")
        .agg(["mean", "sem"])
        .drop(columns=[("seed", "mean"), ("seed", "sem")])
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(summary_path.with_name(summary_path.stem + "_aggregate.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("dist/full_speed_benchmark")
    )
    args = parser.parse_args()

    rows: list[EpisodeRow] = []
    for seed in range(args.seeds):
        for mode in ("baseline_async", "full_speed"):
            print(f"running mode={mode} seed={seed}")
            rows.extend(
                run_one(
                    mode,
                    args.env_id,
                    seed,
                    args.workers,
                    args.total_timesteps,
                )
            )

    rows_path = args.out_dir / "episodes.csv"
    summary_path = args.out_dir / "summary.csv"
    plot_path = args.out_dir / "learning_curves.png"
    write_rows(rows, rows_path)
    write_summary(rows_path, summary_path)
    plot_results(rows_path, plot_path)
    print(f"wrote {rows_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {summary_path.with_name(summary_path.stem + '_aggregate.csv')}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
