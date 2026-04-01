# env.py
import random

PAYOFF = {
    ("C", "C"): (+2, +2),
    ("C", "D"): (-1, +3),
    ("D", "C"): (+3, -1),
    ("D", "D"): (0, 0),
}

class MatrixGameEnv:
    """
    Simple 1-step Prisoner's Dilemma style game.
    Agent is Player A, opponent is scripted.
    """

    def __init__(self, opponent_policy: str = "tit_for_tat"):
        self.opponent_policy = opponent_policy
        self.history = []

    def reset(self):
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
        Returns a textual description (this is what we feed to the LLM).
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

    def step(self, agent_action: str):
        agent_action = agent_action.upper()
        opp_action = self._opponent_action()

        # default heavy penalty for invalid actions
        if agent_action not in ("C", "D"):
            reward_agent = -2.0
            opp_action = "D"
        else:
            reward_agent, _ = PAYOFF[(agent_action, opp_action)]

        self.history.append((agent_action, opp_action))
        obs = self._observation()
        done = True  # 1-step episode for now

        info = {"opp_action": opp_action}
        return obs, reward_agent, done, info
