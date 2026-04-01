#!/usr/bin/env python
import os
from dataclasses import dataclass
import csv

import torch
from transformers import AutoTokenizer
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Toggle this between False and True for Run A / Run B
USE_MAPPED_ACTIONS = False   # False = naive, True = mapped to {C,D}


# -------------------------------------------------------------------
# 1. Simple Matrix Game Environment
# -------------------------------------------------------------------
PAYOFF = {
    ("C", "C"): (+2.0, +2.0),
    ("C", "D"): (-1.0, +3.0),
    ("D", "C"): (+3.0, -1.0),
    ("D", "D"): (0.0, 0.0),
}


@dataclass
class StepResult:
    obs: str
    reward: float
    done: bool
    info: dict


class MatrixGameEnv:
    def __init__(self, opponent_policy: str = "tit_for_tat"):
        self.opponent_policy = opponent_policy
        self.history = []

    def reset(self) -> str:
        self.history = []
        return self._observation()

    def _opponent_action(self) -> str:
        if self.opponent_policy == "always_defect":
            return "D"
        if self.opponent_policy == "always_cooperate":
            return "C"
        if not self.history:
            return "C"
        last_agent_action, _ = self.history[-1]
        return last_agent_action

    def _observation(self) -> str:
        if not self.history:
            history_txt = "No previous rounds."
        else:
            rounds = []
            for i, (a, b) in enumerate(self.history, start=1):
                rounds.append(f"Round {i}: You={a}, Opponent={b}")
            history_txt = "Previous rounds:\n" + "\n".join(rounds)

        prompt = (
            history_txt
            + "\n\nYou are Player A in a Prisoner's Dilemma.\n"
              "Choose your next move.\n"
              "Reply with a single letter: C (cooperate) or D (defect).\n"
              "Action:"
        )
        return prompt

    def step(self, agent_action: str) -> StepResult:
        agent_action = agent_action.upper()
        opp_action = self._opponent_action()

        if agent_action not in ("C", "D"):
            reward_agent = -2.0
            opp_action = "D"
        else:
            reward_agent, _ = PAYOFF[(agent_action, opp_action)]

        self.history.append((agent_action, opp_action))
        obs = self._observation()
        done = True
        info = {"opp_action": opp_action, "agent_action": agent_action}
        return StepResult(obs=obs, reward=reward_agent, done=done, info=info)


# -------------------------------------------------------------------
# 2. Action parser – naive vs mapped
# -------------------------------------------------------------------
def parse_action_from_text(text: str) -> str:
    """
    If USE_MAPPED_ACTIONS is False:
        - Only 'C' or 'D' are valid; everything else -> 'INVALID'
    If True:
        - Any alphabetic char is deterministically mapped to C or D.
    """
    text = text.strip().upper()
    for ch in text:
        if ch in ("C", "D"):
            return ch
        if ch.isalpha():
            if USE_MAPPED_ACTIONS:
                # simple deterministic mapping for demo:
                # even ASCII -> 'C', odd ASCII -> 'D'
                return "C" if ord(ch) % 2 == 0 else "D"
            else:
                return "INVALID"
    return "INVALID"

def append_metrics_row(csv_path: str, row: dict):
    """
    Append a single row of metrics to a CSV file.
    If the file doesn't exist, write a header first.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_policy(
    model,
    tokenizer,
    env,
    num_episodes: int,
    csv_path: str,
    tag: str,
):
    """
    Evaluate the current model as a policy (no PPO updates), log summary metrics.
    Used both BEFORE RL (baseline) and AFTER RL (adapted).
    """
    model.eval()

    coop_count = 0
    mutual_coop_count = 0
    mutual_defect_count = 0
    exploit_count = 0
    exploited_count = 0
    invalid_count = 0
    rewards = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()

            encoded = tokenizer(
                obs,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            q_tensor = encoded["input_ids"][0].to(DEVICE)

            # 1-step generation, same style as PPO loop
            response_tensor = model.generate(
                q_tensor.unsqueeze(0),
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

            prompt_len = q_tensor.shape[0]
            new_tokens = response_tensor[prompt_len:]
            action_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            action = parse_action_from_text(action_text)
            step_res = env.step(action)
            reward = step_res.reward
            rewards.append(reward)

            opp_action = step_res.info["opp_action"]

            # Behaviour counters
            if action not in ("C", "D"):
                invalid_count += 1
            if action == "C":
                coop_count += 1
            if action == "C" and opp_action == "C":
                mutual_coop_count += 1
            if action == "D" and opp_action == "D":
                mutual_defect_count += 1
            if action == "D" and opp_action == "C":
                exploit_count += 1
            if action == "C" and opp_action == "D":
                exploited_count += 1

    total_steps = num_episodes  # 1-step episodes
    avg_reward = sum(rewards) / total_steps
    coop_rate = coop_count / total_steps
    mutual_coop_rate = mutual_coop_count / total_steps
    mutual_defect_rate = mutual_defect_count / total_steps
    exploit_rate = exploit_count / total_steps
    exploited_rate = exploited_count / total_steps
    invalid_rate = invalid_count / total_steps

    append_metrics_row(
        csv_path,
        {
            "phase": tag,
            "num_episodes": num_episodes,
            "avg_reward": avg_reward,
            "coop_rate": coop_rate,
            "mutual_coop_rate": mutual_coop_rate,
            "mutual_defect_rate": mutual_defect_rate,
            "exploit_rate": exploit_rate,
            "exploited_rate": exploited_rate,
            "invalid_rate": invalid_rate,
            "use_mapped": int(USE_MAPPED_ACTIONS),
        },
    )

    print(
        f"[EVAL {tag}] "
        f"avg_reward={avg_reward:+.3f} "
        f"coop_rate={coop_rate:.2f} "
        f"mutual_coop_rate={mutual_coop_rate:.2f} "
        f"exploit_rate={exploit_rate:.2f} "
        f"invalid_rate={invalid_rate:.2f}"
    )

# -------------------------------------------------------------------
# 3. PPO + LLM training loop
# -------------------------------------------------------------------
def main():
    print(">>> ENTERING MAIN <<<")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # important for 7–13B
        device_map="auto",            # let HF shard across GPUs if needed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For TRL models, base LM is usually at model.pretrained_model
    model.pretrained_model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False

    # ---------------- PPO CONFIG ----------------
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=4,        # or even 2 if you get OOM
        mini_batch_size=2,
        target_kl=0.05,      # often good to reduce slightly for large LLMs
        log_with=None,
    )

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(DEVICE)
    ref_model.to(DEVICE)

    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=None,
    )

    env = MatrixGameEnv(opponent_policy="tit_for_tat")

    # ---------------- METRICS CONFIG ----------------
    moving_avg_reward = 0.0
    moving_avg_invalid = 0.0
    alpha = 0.1
    num_updates = 100

    tag_mode = "mapped" if USE_MAPPED_ACTIONS else "naive"
    ppo_metrics_csv = f"metrics_ppo_{tag_mode}.csv"

    print(f"Starting PPO training. USE_MAPPED_ACTIONS={USE_MAPPED_ACTIONS}\n")

    # -------- 1) BASELINE EVAL (NO RL) --------
    evaluate_policy(
        model,
        tokenizer,
        env,
        num_episodes=500,  # adjust if you want
        csv_path="metrics_no_rl.csv",
        tag=f"no_rl_{tag_mode}",
    )

    # -------- 2) PPO TRAINING LOOP --------
    for update_idx in range(num_updates):
        batch_queries = []
        batch_query_tensors = []
        batch_responses = []
        batch_response_tensors = []
        batch_rewards = []

        invalid_actions_in_batch = 0

        # Behaviour counters for this update
        coop_count = 0
        mutual_coop_count = 0
        mutual_defect_count = 0
        exploit_count = 0
        exploited_count = 0

        for _ in range(ppo_config.batch_size):
            obs = env.reset()

            encoded = tokenizer(
                obs,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            q_tensor = encoded["input_ids"][0].to(DEVICE)

            with torch.no_grad():
                response_tensor = ppo_trainer.generate(
                    [q_tensor],
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]

            prompt_len = q_tensor.shape[0]
            new_tokens = response_tensor[prompt_len:]
            action_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            action = parse_action_from_text(action_text)
            step_res = env.step(action)
            reward = step_res.reward

            opp_action = step_res.info["opp_action"]

            # Invalid + behaviour stats
            if action not in ("C", "D"):
                invalid_actions_in_batch += 1
            if action == "C":
                coop_count += 1
            if action == "C" and opp_action == "C":
                mutual_coop_count += 1
            if action == "D" and opp_action == "D":
                mutual_defect_count += 1
            if action == "D" and opp_action == "C":
                exploit_count += 1
            if action == "C" and opp_action == "D":
                exploited_count += 1

            batch_queries.append(obs)
            batch_query_tensors.append(q_tensor)
            batch_responses.append(action_text)
            batch_response_tensors.append(response_tensor)
            batch_rewards.append(
                torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            )

        # PPO update step
        stats = ppo_trainer.step(
            batch_query_tensors,
            batch_response_tensors,
            batch_rewards,
        )
        ppo_trainer.log_stats(
            stats,
            batch={"query": batch_queries, "response": batch_responses},
            rewards=batch_rewards,
        )

        # Basic metrics
        avg_reward = torch.stack(batch_rewards).mean().item()
        invalid_rate = invalid_actions_in_batch / ppo_config.batch_size

        moving_avg_reward = (1 - alpha) * moving_avg_reward + alpha * avg_reward
        moving_avg_invalid = (1 - alpha) * moving_avg_invalid + alpha * invalid_rate

        total_steps = ppo_config.batch_size  # 1-step episodes per update
        coop_rate = coop_count / total_steps
        mutual_coop_rate = mutual_coop_count / total_steps
        mutual_defect_rate = mutual_defect_count / total_steps
        exploit_rate = exploit_count / total_steps
        exploited_rate = exploited_count / total_steps

        # Try to extract KL from stats (key names vary across TRL versions)
        kl_value = float("nan")
        for k, v in stats.items():
            if "kl" in k.lower():
                try:
                    kl_value = float(v)
                except Exception:
                    pass
                break

        # Save per-update metrics to CSV
        append_metrics_row(
            ppo_metrics_csv,
            {
                "update": update_idx + 1,
                "avg_reward": avg_reward,
                "ma_reward": moving_avg_reward,
                "invalid_rate": invalid_rate,
                "ma_invalid": moving_avg_invalid,
                "coop_rate": coop_rate,
                "mutual_coop_rate": mutual_coop_rate,
                "mutual_defect_rate": mutual_defect_rate,
                "exploit_rate": exploit_rate,
                "exploited_rate": exploited_rate,
                "kl": kl_value,
                "use_mapped": int(USE_MAPPED_ACTIONS),
            },
        )

        # Console log
        print(
            f"[Update {update_idx + 1:03d}] "
            f"avg_reward={avg_reward:+.3f} "
            f"ma_reward={moving_avg_reward:+.3f} "
            f"invalid_rate={invalid_rate:.2f} "
            f"ma_invalid={moving_avg_invalid:.2f} "
            f"coop_rate={coop_rate:.2f} "
            f"mutual_coop_rate={mutual_coop_rate:.2f} "
            f"kl={kl_value:.4f}"
        )

    # -------- 3) EVAL AFTER RL --------
    evaluate_policy(
        model,
        tokenizer,
        env,
        num_episodes=500,
        csv_path="metrics_with_rl.csv",
        tag=f"with_rl_{tag_mode}",
    )

    # -------- 4) SAVE ADAPTED MODEL --------
    out_dir = f"ppo_llm_matrix_game_{tag_mode}"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved adapted model to: {out_dir}")

if __name__ == "__main__":
    main()

