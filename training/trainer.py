import os
import time
import torch
import torch.optim as optim
import numpy as np

from neural_net.resnet import ResNetGomoku
from training.replay_buffer import ReplayBuffer
from training.self_play import play_one_game


class GomokuTrainer:
    """Orchestrates self-play, training, and evaluation for Gomoku Zero.

    Typical Colab usage:
        trainer = GomokuTrainer(checkpoint_dir="/content/drive/MyDrive/gomoku-zero")
        trainer.run()
    """

    def __init__(
        self,
        checkpoint_dir="./checkpoints",
        num_blocks=10,
        num_filters=128,
        lr=2e-3,
        weight_decay=1e-4,
        momentum=0.9,
        batch_size=512,
        buffer_size=500000,
        self_play_games=100,
        training_steps=1000,
        mcts_simulations=400,
        eval_games=200,
        eval_win_rate=0.55,
        temperature_threshold=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_iterations=1000,
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = device

        # Model
        self.network = ResNetGomoku(num_blocks=num_blocks, num_filters=num_filters).to(device)
        self.network.eval()

        # Store hyperparams for optimizer rebuild
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_iterations = max_iterations

        # Optimizer + scheduler
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_iterations, eta_min=1e-6
        )

        # Training hyperparams
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.self_play_games = self_play_games
        self.mcts_simulations = mcts_simulations
        self.eval_games = eval_games
        self.eval_win_rate = eval_win_rate
        self.temperature_threshold = temperature_threshold

        # Replay buffer
        self.buffer = ReplayBuffer(maxlen=buffer_size)

        # Iteration counter
        self.iteration = 0
        self.best_iteration = 0

    def _save_checkpoint(self, path):
        """Save full trainer state: model, optimizer, scheduler, iteration, buffer."""
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "iteration": self.iteration,
            "num_blocks": self.network.num_blocks,
            "num_filters": self.network.num_filters,
        }, path)
        # Save replay buffer separately (can be large)
        buffer_path = path.replace(".pt", "_buffer.pt")
        self.buffer.save(buffer_path)

    def _rebuild_optimizer(self, scheduler_steps=None):
        """Create fresh optimizer + scheduler (zero momentum), stepped to target position."""
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_iterations, eta_min=1e-6,
        )
        # Works for both PyTorch 1.x (init: last_epoch=-1) and 2.x (init: last_epoch=0).
        if scheduler_steps is None:
            scheduler_steps = self.iteration
        for _ in range(scheduler_steps):
            self.scheduler.step()

    def _load_checkpoint(self, path):
        """Full state restore (for resume). Backward-compatible with old model-only checkpoints."""
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.iteration = ckpt.get("iteration", 0)

        if "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Old-format checkpoint (model weights only): rebuild optimizer + scheduler
            self._rebuild_optimizer()

        # Restore replay buffer if available
        buffer_path = path.replace(".pt", "_buffer.pt")
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)
            print(f"  Restored replay buffer: {self.buffer.size} entries")

    def _load_model_weights(self, path):
        """Restore only model weights (for revert). Keeps optimizer/scheduler/iteration."""
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["model_state_dict"])

    def _latest_checkpoint(self):
        """Find the latest checkpoint for resume. Prefer latest.pt (crash recovery), fall back to best_model.pt."""
        latest = os.path.join(self.checkpoint_dir, "latest.pt")
        if os.path.exists(latest):
            return latest
        best = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best):
            return best
        return None

    def resume(self):
        """Resume from the latest checkpoint with full state restore."""
        ckpt_path = self._latest_checkpoint()
        if ckpt_path:
            self._load_checkpoint(ckpt_path)
            # Checkpoint stores the last completed iteration.
            # Advance by 1 so the loop starts at the next iteration.
            self.iteration += 1
            print(f"Resumed from {ckpt_path}, continuing at iteration {self.iteration}")

    def run(self, max_iterations=1000):
        """Main training loop."""
        print(f"Training on {self.device}")
        print(f"Model: {sum(p.numel() for p in self.network.parameters())} parameters")

        for self.iteration in range(self.iteration, max_iterations):
            t_start = time.time()
            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration}")
            print(f"{'='*60}")

            # Phase 1: Self-play data generation
            print(f"[Self-play] Generating {self.self_play_games} games...")
            self._generate_self_play()

            # Phase 2: Network training
            print(f"[Training] {self.training_steps} steps, buffer size: {self.buffer.size}")
            train_loss, self._last_train_steps = self._train_network()

            # Phase 3: Evaluation
            print(f"[Eval] Playing {self.eval_games} evaluation games...")
            win_rate = self._evaluate()

            elapsed = time.time() - t_start
            print(f"[Iter {self.iteration}] loss={train_loss:.4f} "
                  f"win_rate={win_rate:.3f} time={elapsed:.1f}s")

            # Save training progress (for crash recovery, NOT best model)
            self._save_checkpoint(
                os.path.join(self.checkpoint_dir, "latest.pt")
            )

    def _generate_self_play(self):
        """Run self-play games and store results in replay buffer."""
        self.network.eval()
        total_games = 0
        black_wins = 0
        white_wins = 0
        draws = 0

        for g in range(self.self_play_games):
            try:
                game_data, result = play_one_game(
                    self.network,
                    device=self.device,
                    num_simulations=self.mcts_simulations,
                    temperature_threshold=self.temperature_threshold,
                )
                self.buffer.push(game_data)
                total_games += 1
                if result == 1: black_wins += 1
                elif result == -1: white_wins += 1
                else: draws += 1

                if (g + 1) % 10 == 0:
                    print(f"  Games: {g+1}/{self.self_play_games} "
                          f"(B:{black_wins} W:{white_wins} D:{draws}) "
                          f"buffer: {self.buffer.size}")
            except Exception as e:
                print(f"  Game {g} error: {e}")
                continue

        print(f"  Self-play done: {total_games} games, "
              f"B:{black_wins} W:{white_wins} D:{draws}")

    def _train_network(self):
        """Train the network on replay buffer data."""
        self.network.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        steps_done = 0
        accumulate_steps = 4

        for step in range(self.training_steps):
            if self.buffer.size < self.batch_size:
                print(f"  Buffer too small ({self.buffer.size}), skipping training")
                break

            states, policies, values = self.buffer.sample(self.batch_size)

            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).to(self.device)

            loss, p_loss, v_loss = self.network.loss(states_t, policies_t, values_t)

            # Scale loss by accumulation steps so effective learning rate is correct
            (loss / accumulate_steps).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            # Accumulate gradients over multiple steps for stability
            if (step + 1) % accumulate_steps == 0 or step == self.training_steps - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            steps_done += 1

            if (step + 1) % 200 == 0:
                avg = total_loss / steps_done
                print(f"  Step {step+1}/{self.training_steps} avg_loss={avg:.4f}")

        # Only advance LR schedule if training actually happened
        if steps_done > 0:
            self.scheduler.step()
        self.network.eval()

        avg_loss = total_loss / max(steps_done, 1)
        return avg_loss, steps_done

    def _evaluate(self):
        """Evaluate current network against the best saved network."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if not os.path.exists(best_path):
            # No previous checkpoint — auto-promote current network
            self._save_checkpoint(
                os.path.join(self.checkpoint_dir, "best_model.pt")
            )
            self.best_iteration = self.iteration
            print("  No previous checkpoint — saved as best model")
            return 1.0

        # Load best model weights into a fresh network for evaluation
        best_net = ResNetGomoku(
            num_blocks=self.network.num_blocks,
            num_filters=self.network.num_filters,
        ).to(self.device)
        ckpt = torch.load(best_path, map_location=self.device)
        best_net.load_state_dict(ckpt["model_state_dict"])
        best_net.eval()

        # Play evaluation games (new network vs best network)
        wins = 0
        losses = 0
        draws = 0
        eval_sims = max(100, self.mcts_simulations // 2)

        for g in range(self.eval_games):
            try:
                # Alternate colors to eliminate first-move bias
                swap = (g % 2 == 1)
                result = self._eval_game(
                    self.network if not swap else best_net,
                    best_net if not swap else self.network,
                    num_simulations=eval_sims,
                    new_net_is_black=(not swap),
                )
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1
            except Exception as e:
                draws += 1
                continue

            if (g + 1) % 50 == 0:
                wr = wins / (g + 1) if (g + 1) > 0 else 0
                print(f"  Eval: {g+1}/{self.eval_games} wins={wins} "
                      f"losses={losses} draws={draws} wr={wr:.3f}")

        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0

        # Save current iteration model
        current_path = os.path.join(self.checkpoint_dir, f"model_iter_{self.iteration}.pt")
        self._save_checkpoint(current_path)

        if win_rate >= self.eval_win_rate:
            self._save_checkpoint(
                os.path.join(self.checkpoint_dir, "best_model.pt")
            )
            self.best_iteration = self.iteration
            print(f"  ** New best model! win_rate={win_rate:.3f} **")
        else:
            # Revert: restore model weights + rebuild optimizer (clear stale momentum).
            # Only add +1 scheduler step if training actually happened this iteration.
            self._load_model_weights(os.path.join(self.checkpoint_dir, "best_model.pt"))
            extra = 1 if getattr(self, '_last_train_steps', 0) > 0 else 0
            self._rebuild_optimizer(scheduler_steps=self.iteration + extra)
            print(f"  Did not improve (wr={win_rate:.3f}), reverting to best model")

        return win_rate

    def _eval_game(self, black_net, white_net, num_simulations=200,
                   new_net_is_black=True):
        """Play one evaluation game between new and old networks.

        Returns: 1 if new wins, -1 if old wins, 0 for draw.
        """
        from search.mcts_batch import MCTSBatch, Node
        from core_logic.cboard import PyBoard

        board = PyBoard()
        nets = [black_net, white_net]  # index 0=black, 1=white

        while True:
            player_idx = board.current_player - 1  # 0 or 1
            net = nets[player_idx]
            temperature = 0.0  # evaluation: always greedy

            mcts = MCTSBatch(num_simulations=num_simulations)
            root = Node()
            action_probs = mcts.search(
                root, board, net, self.device, add_noise=False
            )

            # If no legal moves, current player loses
            if len(board.legal_moves()) == 0:
                last_player = 1 if board.current_player == 2 else 2
                winner_is_new = (new_net_is_black and last_player == 1) or \
                                (not new_net_is_black and last_player == 2)
                return 1 if winner_is_new else -1

            action = int(np.argmax(action_probs))

            x, y = action % 15, action // 15
            board.play_move(x, y, board.current_player)

            # Check winner
            last_player = 1 if board.current_player == 2 else 2
            if board.check_win(last_player):
                winner_is_new = (new_net_is_black and last_player == 1) or \
                                (not new_net_is_black and last_player == 2)
                return 1 if winner_is_new else -1

            if board.is_full():
                return 0
