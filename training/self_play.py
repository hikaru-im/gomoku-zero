import torch
import numpy as np
from core_logic.cboard import PyBoard
from search.mcts_batch import MCTSBatch, Node


def play_one_game(network, device="cpu", num_simulations=400,
                  temperature_threshold=30, resign_threshold=-0.8,
                  resign_count=10, batch_size=16):
    """Play a single self-play game using MCTS.

    Args:
        network: ResNetGomoku in eval mode.
        device: torch device.
        num_simulations: MCTS simulations per move.
        temperature_threshold: moves before switching to greedy.
        resign_threshold: value below which resign counter increments.
        resign_count: consecutive moves below threshold to resign.
        batch_size: unused in single-game mode (kept for API compatibility).

    Returns:
        game_data: list of (state(3,15,15), policy(225), z(float))
        result: 1 if black won, -1 if white won, 0 if draw
    """
    board = PyBoard()
    mcts = MCTSBatch(num_simulations=num_simulations)
    game_data = []
    move_count = 0
    resign_counter = 0
    last_player = 0

    while True:
        state = board.get_state()  # (3, 15, 15)
        player = board.current_player

        # Choose temperature
        temperature = 1.0 if move_count < temperature_threshold else 0.0

        # Run MCTS
        root = Node()
        action_probs = mcts.search(root, board, network, device, add_noise=True)

        # Apply temperature to get visit-based policy
        if temperature > 0:
            # Re-normalize with temperature
            actions = list(root.children.keys())
            visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
            visits = visits ** (1.0 / temperature)
            total = visits.sum()
            if total > 0:
                policy = np.zeros(225, dtype=np.float64)
                for a, v in zip(actions, visits):
                    policy[a] = v / total
            else:
                policy = np.ones(225, dtype=np.float64) / 225.0
            # Sample from policy
            action = np.random.choice(225, p=policy)
        else:
            # Greedy: pick most visited
            action = int(np.argmax(action_probs))
            policy = action_probs  # already visit-prob distribution

        # Make the move
        x, y = action % 15, action // 15
        board.play_move(x, y, player)
        move_count += 1

        game_data.append((state.copy().astype(np.float32),
                          policy.astype(np.float32),
                          player))

        # Check resignation (based on MCTS root value from current player's perspective)
        root_value = root.q_value if root.visit_count > 0 else 0.0
        if player != last_player:
            resign_counter = 0
            last_player = player
        if root_value < resign_threshold:
            resign_counter += 1
        else:
            resign_counter = 0

        if resign_counter >= resign_count:
            # Resign: current player loses
            result = -1 if player == 1 else 1
            break

        # Check game end — player just moved, check if they won
        if board.check_win(player):
            result = 1 if player == 1 else -1
            break
        if board.is_full():
            result = 0
            break

    # Assign outcomes z to each move
    final_data = []
    for state, policy, player in game_data:
        if result == 0:
            z = 0.0
        elif (result == 1 and player == 1) or (result == -1 and player == 2):
            z = 1.0
        else:
            z = -1.0
        final_data.append((state, policy, z))

    return final_data, result


def self_play_worker(network, num_games, device, num_simulations=400,
                     temperature_threshold=30, result_queue=None):
    """Generate multiple self-play games.

    Args:
        network: ResNetGomoku in eval mode.
        num_games: number of games to play.
        device: torch device.
        num_simulations: MCTS simulations per move.
        temperature_threshold: temperature schedule parameter.
        result_queue: optional multiprocessing.Queue to send results.

    Returns:
        list of game_data if result_queue is None, otherwise sends to queue.
    """
    all_games = []
    for _ in range(num_games):
        game_data, result = play_one_game(
            network, device=device,
            num_simulations=num_simulations,
            temperature_threshold=temperature_threshold,
        )
        if result_queue is not None:
            result_queue.put(game_data)
        else:
            all_games.append(game_data)

    if result_queue is None:
        return all_games
    return None
