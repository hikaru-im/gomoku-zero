import random
import threading
import numpy as np


class ReplayBuffer:
    """Circular replay buffer with on-the-fly D4 symmetry augmentation.

    Stores game trajectories (state, policy, outcome) and samples uniformly
    with random symmetry transforms at sampling time.
    """

    def __init__(self, maxlen=500000):
        self.maxlen = maxlen
        self.buffer = []  # flat list of (state, policy, value)
        self._lock = threading.Lock()

    @property
    def size(self):
        return len(self.buffer)

    def push(self, game_data):
        """Add a full game's data to the buffer.

        Args:
            game_data: list of (state: ndarray(3,15,15), policy: ndarray(225), z: float)
        """
        with self._lock:
            for state, policy, z in game_data:
                self.buffer.append((state.copy(), policy.copy(), z))
            # Trim to maxlen (FIFO)
            if len(self.buffer) > self.maxlen:
                excess = len(self.buffer) - self.maxlen
                self.buffer = self.buffer[excess:]

    def sample(self, batch_size, symmetry=True):
        """Sample a batch with optional random symmetry augmentation.

        Args:
            batch_size: number of samples.
            symmetry: if True, apply random D4 transform to each sample.

        Returns:
            states: (batch, 3, 15, 15) float32
            policies: (batch, 225) float32
            values: (batch,) float32
        """
        with self._lock:
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = np.zeros((len(batch), 3, 15, 15), dtype=np.float32)
        policies = np.zeros((len(batch), 225), dtype=np.float32)
        values = np.zeros(len(batch), dtype=np.float32)

        for i, (s, p, v) in enumerate(batch):
            if symmetry and random.random() < 0.5:
                t = random.randint(0, 7)
                states[i] = _apply_symmetry_state(s, t)
                policies[i] = _apply_symmetry_policy(p, t)
            else:
                states[i] = s
                policies[i] = p
            values[i] = v

        return states, policies, values


# ─── D4 symmetry transforms ───
# 0: identity
# 1: rotate 90 CW
# 2: rotate 180
# 3: rotate 270 CW
# 4: flip horizontal
# 5: flip H + rot 90
# 6: flip H + rot 180
# 7: flip H + rot 270

def _rot90(arr):
    """Rotate 90 degrees CW: (3,15,15) or (15,15)."""
    return np.rot90(arr, k=-1, axes=(-2, -1))

def _flip_h(arr):
    """Flip horizontally: (3,15,15) or (15,15)."""
    return np.flip(arr, axis=-1)

def _apply_symmetry_state(state, t):
    """Apply symmetry transform t to a (3,15,15) state."""
    if t == 0: return state
    s = state.copy()
    if t <= 3:
        for _ in range(t):
            s = _rot90(s)
    else:
        s = _flip_h(s)
        for _ in range(t - 4):
            s = _rot90(s)
    return s

def _apply_symmetry_policy(policy, t):
    """Apply symmetry transform t to a flat (225,) policy."""
    p = policy.reshape(15, 15).copy()
    if t == 0: return policy
    if t <= 3:
        for _ in range(t):
            p = _rot90(p)
    else:
        p = _flip_h(p)
        for _ in range(t - 4):
            p = _rot90(p)
    return p.flatten()
