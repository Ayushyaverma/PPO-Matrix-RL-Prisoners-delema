#!/usr/bin/env python
import os
import random
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)

# -------------------------------------------------------------------
# Safety: avoid torchvision issues for text-only use
# -------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# -------------------------------------------------------------------
# 1. Simple 1-step Matrix Game Environment (Prisoner's Dilemma style)
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
    """
    Simple text-based 1-step matrix game.

    - Agent is Player A.
    - Opponent is scripted (e.g., tit-for-tat).
    - Actions: "C" (cooperate) or "D" (defect).
    """

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
        # tit-for-tat
        if not self.history:
            return "C"
        last_agent_action, _ = self.history[-1]
        return last_agent_action

    def _observation(self) -> str:
        """
        Returns a textual description of the game history plus instruction.
        This is the input prompt to the LLM.
        """
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
            # Heavy penalty for invalid actions
            reward_agent = -2.0
            opp_action = "D"
        else:
            reward_agent, _ = PAYOFF[(agent_action, opp_action)]

        self.history.append((agent_action, opp_action))
        obs = self._observation()
        done = True  # 1-step episode

        info = {"opp_action": opp_action, "agent_action": agent_action}
        return StepResult(obs=obs, reward=reward_agent, done=done, info=info)


# -------------------------------------------------------------------
# 2. Utility: parse action from LLM text
# -------------------------------------------------------------------
def parse_action_from_text(text: str) -> str:
    """
    Extracts the first alphabetic character and interprets it as C/D.
    Any other character -> INVALID.
    """
    text = text.strip().upper()
    for ch in text:
        if ch in ("C", "D"):
            return ch
        if ch.isalpha():
            return "INVALID"
    return "INVALID"


# -------------------------------------------------------------------
# 3. PPO + LLM training loop
# -------------------------------------------------------------------
def main():
    # -----------------------------
    # 3.1. Config & model setup
    # -----------------------------
    model_name = "gpt2"  # small, fast baseline; fine for demo

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=8,        # episodes per PPO update
        mini_batch_size=4,
        target_kl=0.1,
        log_with=None,       # set to "wandb" if you want W&B logs
    )

    print("Loading model & tokenizer...")
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
        dataset=None,  # we provide queries manually
    )

    env = MatrixGameEnv(opponent_policy="tit_for_tat")

    # Moving averages for PPT plots
    moving_avg_reward = 0.0
    moving_avg_invalid = 0.0
    alpha = 0.1  # smoothing

    num_updates = 50  # PPO updates (you can increase later)

    print("Starting PPO training loop...\n")

    for update_idx in range(num_updates):
        batch_queries = []           # list[str]
        batch_query_tensors = []     # list[torch.LongTensor]
        batch_responses = []         # list[str]
        batch_response_tensors = []  # list[torch.LongTensor]
        batch_rewards = []           # list[torch.Tensor]

        invalid_actions_in_batch = 0

        # One PPO update = multiple 1-step episodes
        for _ in range(ppo_config.batch_size):
            # 1) Reset env and get initial observation
            obs = env.reset()

            # 2) Tokenize observation
            encoded = tokenizer(
                obs,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            # 1D tensor: [seq_len]
            q_tensor = encoded["input_ids"][0].to(DEVICE)

            # 3) Generate ONE new token as the action, using PPOTrainer.generate
            with torch.no_grad():
                # TRL expects list[1D tensor]
                response_tensor = ppo_trainer.generate(
                    [q_tensor],
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]  # shape: [seq_len + 1]

            prompt_len = q_tensor.shape[0]
            new_tokens = response_tensor[prompt_len:]
            action_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            action = parse_action_from_text(action_text)

            # 4) Step the environment
            step_res = env.step(action)
            reward = step_res.reward

            if action not in ("C", "D"):
                invalid_actions_in_batch += 1

            # 5) Record for PPO
            batch_queries.append(obs)
            batch_query_tensors.append(q_tensor)
            batch_responses.append(action_text)
            batch_response_tensors.append(response_tensor)
            batch_rewards.append(
                torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            )

        # -------------------------
        # 3.2. PPO policy update
        # -------------------------
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

        # -------------------------
        # 3.3. Simple metrics
        # -------------------------
        avg_reward = torch.stack(batch_rewards).mean().item()
        invalid_rate = invalid_actions_in_batch / ppo_config.batch_size

        moving_avg_reward = (1 - alpha) * moving_avg_reward + alpha * avg_reward
        moving_avg_invalid = (1 - alpha) * moving_avg_invalid + alpha * invalid_rate

        print(
            f"[Update {update_idx + 1:03d}] "
            f"avg_reward={avg_reward:+.3f} "
            f"ma_reward={moving_avg_reward:+.3f} "
            f"invalid_rate={invalid_rate:.2f} "
            f"ma_invalid={moving_avg_invalid:.2f}"
        )

    # -----------------------------
    # 3.4. Save adapted policy
    # -----------------------------
    output_dir = "ppo_llm_matrix_game"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved adapted model to: {output_dir}")


if __name__ == "__main__":
    main()
