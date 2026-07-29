from __future__ import annotations


class RolloutBuffer:
    def __init__(self):
        self.matrix_states = []
        self.scalar_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    @property
    def _fields(self):
        return (
            self.matrix_states,
            self.scalar_states,
            self.actions,
            self.logprobs,
            self.rewards,
            self.dones,
            self.values,
        )

    def clear(self) -> None:
        for field in self._fields:
            field.clear()

    def trim_to_maxlen(self, maxlen: int | None) -> None:
        if maxlen is None:
            return
        if maxlen <= 0:
            self.clear()
            return
        overflow = len(self) - maxlen
        for _ in range(max(0, overflow)):
            for field in self._fields:
                field.pop(0)

    def add(self, matrix_state, scalar_state, action, logprob, reward, done, value, maxlen=None) -> None:
        self.matrix_states.append(matrix_state)
        self.scalar_states.append(scalar_state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.trim_to_maxlen(maxlen)

    def __len__(self) -> int:
        return len(self.rewards)
