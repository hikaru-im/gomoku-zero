import math
import numpy as np
import torch


class Node:
    """MCTS tree node with virtual-loss support for parallel simulation."""

    __slots__ = [
        "parent", "children", "prior", "visit_count",
        "value_sum", "virtual_loss", "action",
    ]

    def __init__(self, parent=None, prior=0.0, action=-1):
        self.parent = parent
        self.children = {}  # action (int 0..224) -> Node
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.action = action  # the move (flat index) that led to this node

    @property
    def q_value(self):
        total = self.visit_count + self.virtual_loss
        if total == 0:
            return 0.0
        return (self.value_sum - 3.0 * self.virtual_loss) / total

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_expanded(self):
        return len(self.children) > 0

    def expanded_children(self):
        return self.children.values()


class MCTSBatch:
    """Monte Carlo Tree Search with batch evaluation and virtual loss.

    Usage:
        mcts = MCTSBatch(num_simulations=400, batch_size=16)
        root = Node()
        action_probs = mcts.search(root, board, network)
    """

    def __init__(self, num_simulations=400, batch_size=16,
                 c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, root, board, network, device="cpu", add_noise=True):
        """Run MCTS search and return action probabilities.

        Args:
            root: Root Node (may be pre-expanded for tree reuse).
            board: PyBoard instance (current game state).
            network: ResNetGomoku model in eval mode.
            device: torch device.
            add_noise: Add Dirichlet noise to root priors (self-play).

        Returns:
            np.ndarray of shape (225,) — visit count probabilities.
        """
        if not root.is_expanded:
            self._expand_leaf(root, board, network, device)

        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            # Selection phase — walk down the tree
            node = root
            sim_board = board.copy()
            path = []  # list of nodes traversed

            while node.is_expanded:
                action, node = self._select_child(node)
                x, y = action % 15, action // 15
                sim_board.play_move(x, y, sim_board.current_player)
                path.append(node)

            # Expansion + evaluation
            if sim_board.is_full() or sim_board.check_win(1) or sim_board.check_win(2):
                # Terminal node
                if sim_board.check_win(1):
                    value = 1.0 if sim_board.current_player == 2 else -1.0
                elif sim_board.check_win(2):
                    value = 1.0 if sim_board.current_player == 1 else -1.0
                else:
                    value = 0.0
            else:
                value = self._expand_leaf(node, sim_board, network, device)

            # Backup — walk from leaf up to root's children, then update root
            for visited_node in reversed(path):
                visited_node.virtual_loss -= 1
                visited_node.visit_count += 1
                visited_node.value_sum += value
                value = -value  # flip perspective

            # Root has no virtual_loss; receives the fully-propagated value
            root.visit_count += 1
            root.value_sum += value

        return self._get_visit_probs(root)

    def search_batch(self, roots, boards, network, device="cpu", add_noise=True):
        """Run MCTS on multiple roots in parallel with batched evaluation.

        Args:
            roots: list of N root Nodes.
            boards: list of N PyBoard instances.
            network: ResNetGomoku model.
            device: torch device.
            add_noise: Add Dirichlet noise to each root.

        Returns:
            list of N np.ndarray (225,) — visit count probabilities.
        """
        n = len(roots)
        assert len(boards) == n

        # Initial expansion of all roots (batched)
        unexpanded = []
        for i, (root, board) in enumerate(zip(roots, boards)):
            if not root.is_expanded:
                unexpanded.append(i)

        if unexpanded:
            self._batch_expand(roots, boards, unexpanded, network, device)

        # Apply Dirichlet noise AFTER expansion (needs children to exist)
        if add_noise:
            for root in roots:
                self._add_dirichlet_noise(root)

        for sim in range(self.num_simulations):
            # Collect leaf nodes to evaluate in batch
            leaf_infos = []  # (root_idx, node, board_copy, path)
            for i in range(n):
                node = roots[i]
                sim_board = boards[i].copy()
                path = []

                while node.is_expanded:
                    action, node = self._select_child(node)
                    x, y = action % 15, action // 15
                    sim_board.play_move(x, y, sim_board.current_player)
                    path.append(node)

                # Check terminal
                if sim_board.is_full() or sim_board.check_win(1) or sim_board.check_win(2):
                    if sim_board.check_win(1):
                        value = 1.0 if sim_board.current_player == 2 else -1.0
                    elif sim_board.check_win(2):
                        value = 1.0 if sim_board.current_player == 1 else -1.0
                    else:
                        value = 0.0
                    # Backup — walk from leaf up, then update root
                    for visited_node in reversed(path):
                        visited_node.virtual_loss -= 1
                        visited_node.visit_count += 1
                        visited_node.value_sum += value
                        value = -value

                    roots[i].visit_count += 1
                    roots[i].value_sum += value
                else:
                    leaf_infos.append((i, node, sim_board, path))

            # Batch evaluate all leaves
            if leaf_infos:
                self._batch_expand_generic(leaf_infos, roots, network, device)

        return [self._get_visit_probs(root) for root in roots]

    def _select_child(self, node):
        """Select child with highest PUCT score, applying virtual loss."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_parent_visits = math.sqrt(node.visit_count + node.virtual_loss)

        for action, child in node.children.items():
            if child.visit_count == 0 and child.virtual_loss == 0:
                # Unvisited — prioritize by prior
                if sqrt_parent_visits > 0:
                    u = self.c_puct * child.prior * sqrt_parent_visits
                else:
                    u = child.prior  # first simulation: select by prior directly
            else:
                q = child.q_value
                u = q + self.c_puct * child.prior * sqrt_parent_visits / (
                    1 + child.visit_count + child.virtual_loss
                )
            if u > best_score:
                best_score = u
                best_action = action
                best_child = child

        # Add virtual loss to selected child
        best_child.virtual_loss += 1
        return best_action, best_child

    def _expand_leaf(self, node, board, network, device):
        """Expand a leaf node with network evaluation. Returns the predicted value."""
        state = board.get_state()  # (3, 15, 15)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

        log_probs, value = network.predict(state_tensor)
        log_probs_np = log_probs.cpu().numpy()[0]
        value_np = float(value.cpu().numpy()[0])

        # Get legal move mask
        legal_mask = board.get_legal_moves_mask()  # (15, 15) uint8
        legal_flat = legal_mask.flatten()  # (225,)

        # Create children for legal moves
        for action in range(225):
            if legal_flat[action]:
                prior = math.exp(log_probs_np[action])
                child = Node(parent=node, prior=prior, action=action)
                node.children[action] = child

        return value_np

    def _add_dirichlet_noise(self, root):
        """Add Dirichlet noise to root children priors."""
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                          self.dirichlet_epsilon * noise[i]

    def _batch_expand(self, roots, boards, indices, network, device):
        """Batch-expand multiple unexpanded root nodes."""
        states = []
        for i in indices:
            state = boards[i].get_state()
            states.append(state)

        states_tensor = torch.from_numpy(np.array(states)).to(device)
        log_probs_batch, values_batch = network.predict(states_tensor)
        log_probs_np = log_probs_batch.cpu().numpy()
        values_np = values_batch.cpu().numpy()

        for idx_pos, i in enumerate(indices):
            node = roots[i]
            board = boards[i]
            log_probs = log_probs_np[idx_pos]
            value = values_np[idx_pos]

            legal_mask = board.get_legal_moves_mask().flatten()
            for action in range(225):
                if legal_mask[action]:
                    prior = math.exp(log_probs[action])
                    child = Node(parent=node, prior=prior, action=action)
                    node.children[action] = child

    def _batch_expand_generic(self, leaf_infos, roots, network, device):
        """Batch-expand arbitrary leaf nodes and backup values."""
        states = []
        for (_, node, sim_board, path) in leaf_infos:
            state = sim_board.get_state()
            states.append(state)

        states_tensor = torch.from_numpy(np.array(states)).to(device)
        log_probs_batch, values_batch = network.predict(states_tensor)
        log_probs_np = log_probs_batch.cpu().numpy()
        values_np = values_batch.cpu().numpy()

        for idx_pos, (i, node, sim_board, path) in enumerate(leaf_infos):
            log_probs = log_probs_np[idx_pos]
            value = float(values_np[idx_pos])

            legal_mask = sim_board.get_legal_moves_mask().flatten()
            for action in range(225):
                if legal_mask[action]:
                    prior = math.exp(log_probs[action])
                    child = Node(parent=node, prior=prior, action=action)
                    node.children[action] = child

            # Backup — walk from leaf up, then update root
            v = value
            for visited_node in reversed(path):
                visited_node.virtual_loss -= 1
                visited_node.visit_count += 1
                visited_node.value_sum += v
                v = -v

            roots[i].visit_count += 1
            roots[i].value_sum += v

    def _get_visit_probs(self, root, temperature=1.0):
        """Get action probabilities from visit counts with temperature."""
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

        if temperature == 0:
            # Greedy: put all mass on the most-visited action
            probs = np.zeros(225, dtype=np.float64)
            best = actions[int(np.argmax(visits))]
            probs[best] = 1.0
            return probs

        # Apply temperature
        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        if total == 0:
            return np.zeros(225, dtype=np.float64)

        probs = np.zeros(225, dtype=np.float64)
        for a, v in zip(actions, visits):
            probs[a] = v / total
        return probs
