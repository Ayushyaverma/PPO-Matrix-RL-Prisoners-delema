import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def load_latest_phase_rows(csv_path, phase_prefix=None):
    """
    For a metrics file with a 'phase' column (e.g. no_rl_mapped, with_rl_mapped),
    return a small DataFrame with the LAST row for each phase.
    If phase_prefix is given ("no_rl" / "with_rl"), filter phases that contain it.
    """
    if not os.path.isfile(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if "phase" not in df.columns:
        print(f"[WARN] 'phase' column not found in {csv_path}")
        return None

    if phase_prefix is not None:
        df = df[df["phase"].str.contains(phase_prefix)]

    if df.empty:
        print(f"[WARN] No rows matching phase_prefix={phase_prefix} in {csv_path}")
        return None

    # Take the last row for each phase
    latest = df.sort_index().groupby("phase").tail(1)
    return latest


def plot_ppo_learning_curves():
    """
    Plot learning curves (reward, invalid rate, cooperation) for all
    metrics_ppo_*.csv files in the current directory.
    """
    ppo_files = glob.glob("metrics_ppo_*.csv")
    if not ppo_files:
        print("[WARN] No PPO metrics files found matching 'metrics_ppo_*.csv'")
        return

    # --- 1) Reward vs Update ---
    plt.figure()
    for path in ppo_files:
        df = pd.read_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]  # e.g. metrics_ppo_mapped
        if "update" not in df.columns or "avg_reward" not in df.columns:
            print(f"[WARN] Missing 'update' or 'avg_reward' in {path}, skipping reward plot for this file.")
            continue
        plt.plot(df["update"], df["avg_reward"], label=label)
    plt.xlabel("PPO Update")
    plt.ylabel("Average Reward")
    plt.title("PPO Learning Curve: Avg Reward vs Update")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_reward_vs_update.png", dpi=300)

    # --- 2) Invalid rate vs Update ---
    plt.figure()
    for path in ppo_files:
        df = pd.read_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        if "update" not in df.columns or "invalid_rate" not in df.columns:
            print(f"[WARN] Missing 'update' or 'invalid_rate' in {path}, skipping invalid-rate plot for this file.")
            continue
        plt.plot(df["update"], df["invalid_rate"], label=label)
    plt.xlabel("PPO Update")
    plt.ylabel("Invalid Action Rate")
    plt.title("PPO: Invalid Action Rate vs Update")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_invalidrate_vs_update.png", dpi=300)

    # --- 3) Cooperation / Mutual cooperation vs Update ---
    plt.figure()
    for path in ppo_files:
        df = pd.read_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        if "update" not in df.columns or "coop_rate" not in df.columns:
            print(f"[WARN] Missing 'update' or 'coop_rate' in {path}, skipping coop-rate plot for this file.")
            continue

        # coop_rate line
        plt.plot(df["update"], df["coop_rate"], label=label + " (coop)")

        # if mutual_coop_rate exists, plot it too
        if "mutual_coop_rate" in df.columns:
            plt.plot(df["update"], df["mutual_coop_rate"], label=label + " (mutual_coop)")

    plt.xlabel("PPO Update")
    plt.ylabel("Rate")
    plt.title("PPO: Cooperation and Mutual Cooperation vs Update")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_coop_vs_update.png", dpi=300)

    print("[INFO] PPO learning curve plots saved as:")
    print("  - ppo_reward_vs_update.png")
    print("  - ppo_invalidrate_vs_update.png")
    print("  - ppo_coop_vs_update.png")


def plot_before_after_bar():
    """
    Make a bar chart comparison of:
      - avg_reward
      - coop_rate
      - invalid_rate
    before vs after RL (no_rl vs with_rl).
    """
    no_rl = load_latest_phase_rows("metrics_no_rl.csv", phase_prefix="no_rl")
    with_rl = load_latest_phase_rows("metrics_with_rl.csv", phase_prefix="with_rl")

    if no_rl is None or with_rl is None:
        print("[WARN] Cannot create before/after RL bar plot (missing metrics_no_rl.csv or metrics_with_rl.csv).")
        return

    # Assume we just take the first row for each (in case multiple phases)
    no_rl_row = no_rl.iloc[0]
    with_rl_row = with_rl.iloc[0]

    metrics = ["avg_reward", "coop_rate", "invalid_rate"]
    labels = ["Avg Reward", "Coop Rate", "Invalid Rate"]

    no_rl_vals = [no_rl_row[m] for m in metrics]
    with_rl_vals = [with_rl_row[m] for m in metrics]

    x = range(len(metrics))  # 0,1,2

    plt.figure()
    width = 0.35
    x_left = [i - width / 2 for i in x]
    x_right = [i + width / 2 for i in x]

    plt.bar(x_left, no_rl_vals, width=width, label="No RL (baseline)")
    plt.bar(x_right, with_rl_vals, width=width, label="With RL (adapted)")

    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Before vs After RL: Avg Reward, Coop Rate, Invalid Rate")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("before_after_rl_bar.png", dpi=300)

    print("[INFO] Before/after RL bar plot saved as: before_after_rl_bar.png")


def main():
    """
    Run all plots.
    Assumes you have already run your training script and generated:
      - metrics_ppo_*.csv
      - metrics_no_rl.csv
      - metrics_with_rl.csv
    in the same directory.
    """
    plot_ppo_learning_curves()
    plot_before_after_bar()
    print("[DONE] All plots generated.")


if __name__ == "__main__":
    main()
