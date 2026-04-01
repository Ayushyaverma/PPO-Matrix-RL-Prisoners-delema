import re
import matplotlib.pyplot as plt

updates = []
avg_rewards = []
ma_rewards = []
invalid_rates = []
ma_invalids = []

pattern = re.compile(
    r"\[Update\s+(\d+)\]\s+avg_reward=([+-]?\d+\.\d+)\s+ma_reward=([+-]?\d+\.\d+)\s+invalid_rate=(\d+\.\d+)\s+ma_invalid=(\d+\.\d+)"
)

with open("log_mapped2.txt") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            u, ar, mar, ir, mir = m.groups()
            updates.append(int(u))
            avg_rewards.append(float(ar))
            ma_rewards.append(float(mar))
            invalid_rates.append(float(ir))
            ma_invalids.append(float(mir))

print(f"Parsed {len(updates)} updates.")

plt.figure()
plt.plot(updates, ma_rewards, marker="o")
plt.xlabel("PPO Update")
plt.ylabel("Moving Avg Reward")
plt.title("LLM Agent – Moving Average Reward vs PPO Updates")
plt.grid(True)
plt.tight_layout()
plt.savefig("ma_reward_vs_updates.png", dpi=200)

plt.figure()
plt.plot(updates, ma_invalids, marker="o")
plt.xlabel("PPO Update")
plt.ylabel("Moving Avg Invalid Rate")
plt.title("LLM Agent – Moving Avg Invalid Actions vs PPO Updates")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig("ma_invalid_vs_updates.png", dpi=200)

print("Saved plots: ma_reward_vs_updates.png, ma_invalid_vs_updates.png")
