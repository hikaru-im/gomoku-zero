import os
import time
import math
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
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = device

        # Model
        self.network = ResNetGomoku(num_blocks=num_blocks, num_filters=num_filters).to(device)
        self.network.eval()

        # Optimizer
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.lr = lr

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

        # LR scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[100, 200, 300, 400], gamma=0.1
        )

        # Iteration counter
        self.iteration = 0
        self.best_iteration = 0

    def _save_checkpoint(self, network, path):
        network.save_checkpoint(path)

    def _load_checkpoint(self, path):
        return self.network.load_checkpoint(path, self.device)

    def _latest_checkpoint(self):
        """Find the latest best_model checkpoint."""
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(path):
            return path
        return None

    def resume(self):
        """Resume from the latest checkpoint."""
        ckpt_path = self._latest_checkpoint()
        if ckpt_path:
            self._load_checkpoint(ckpt_path)
            print(f"Resumed from {ckpt_path}")
            # Try to load iteration number
            iter_path = os.path.join(self.checkpoint_dir, "iteration.txt")
            if os.path.exists(iter_path):
                with open(iter_path, "r") as f:
                    self.iteration = int(f.read().strip())
                print(f"Resuming from iteration {self.iteration}")

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
            train_loss = self._train_network()

            # Phase 3: Evaluation
            print(f"[Eval] Playing {self.eval_games} evaluation games...")
            win_rate = self._evaluate()

            elapsed = time.time() - t_start
            print(f"[Iter {self.iteration}] loss={train_loss:.4f} "
                  f"win_rate={win_rate:.3f} time={elapsed:.1f}s")

            # Save iteration counter
            with open(os.path.join(self.checkpoint_dir, "iteration.txt"), "w") as f:
                f.write(str(self.iteration + 1))

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

        for step in range(self.training_steps):
            if self.buffer.size < self.batch_size:
                print(f"  Buffer too small ({self.buffer.size}), skipping training")
                break

            states, policies, values = self.buffer.sample(self.batch_size)

            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).to(self.device)

            loss, p_loss, v_loss = self.network.loss(states_t, policies_t, values_t)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            # Accumulate gradients over multiple steps for stability
            if (step + 1) % 4 == 0 or step == self.training_steps - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            steps_done += 1

            if (step + 1) % 200 == 0:
                avg = total_loss / steps_done
                print(f"  Step {step+1}/{self.training_steps} avg_loss={avg:.4f}")

        self.scheduler.step()
        self.network.eval()

        return total_loss / max(steps_done, 1)

    def _evaluate(self):
        """Evaluate current network against the best saved network."""
        best_path = self._latest_checkpoint()
        if best_path is None:
            # No previous checkpoint — auto-promote current network
            self._save_checkpoint(
                self.network,
                os.path.join(self.checkpoint_dir, "best_model.pt")
            )
            self.best_iteration = self.iteration
            print("  No previous checkpoint — saved as best model")
            return 1.0

        # Load best model
        best_net = ResNetGomoku(
            num_blocks=self.network.num_blocks,
            num_filters=self.network.num_filters,
        ).to(self.device)
        best_net.load_checkpoint(best_path, self.device)
        best_net.eval()

        # Play evaluation games (new network vs best network)
        wins = 0
        losses = 0
        draws = 0
        eval_sims = max(100, self.mcts_simulations // 2)

        for g in range(self.eval_games):
            try:
                result = self._eval_game(
                    self.network, best_net,
                    num_simulations=eval_sims,
                )
                if result == 1: wins += 1
                elif result == -1: losses += 1
                else: draws += 1
            except Exception as e:
                draws += 1
                continue

            if (g + 1) % 50 == 0:
                wr = wins / (g + 1) if (g + 1) > 0 else 0
                print(f"  Eval: {g+1}/{self.eval_games} wins={wins} "
                      f"losses={losses} draws={draws} wr={wr:.3f}")

        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0

        # Save current model
        current_path = os.path.join(self.checkpoint_dir, f"model_iter_{self.iteration}.pt")
        self._save_checkpoint(self.network, current_path)

        if win_rate >= self.eval_win_rate:
            self._save_checkpoint(
                self.network,
                os.path.join(self.checkpoint_dir, "best_model.pt")
            )
            self.best_iteration = self.iteration
            print(f"  ** New best model! win_rate={win_rate:.3f} **")
        else:
            # Revert to best model
            self._load_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pt"))
            print(f"  Did not improve (wr={win_rate:.3f}), reverting to best model")

        return win_rate

    def _eval_game(self, new_net, old_net, num_simulations=200):
        """Play one evaluation game between new and old networks.

        new_net plays as black, old_net plays as white.
        Returns: 1 if new wins, -1 if old wins, 0 for draw.
        """
        from search.mcts_batch import MCTSBatch, Node
        from cboard import PyBoard

        board = PyBoard()
        move_count = 0
        nets = [new_net, old_net]  # index 0=black(new), 1=white(old)

        while True:
            player_idx = board.current_player - 1  # 0 or 1
            net = nets[player_idx]
            temperature = 1.0 if move_count < 30 else 0.0

            mcts = MCTSBatch(num_simulations=num_simulations)
            root = Node()
            action_probs = mcts.search(
                root, board, net, self.device, add_noise=False
            )

            if temperature > 0:
                action = np.random.choice(225, p=action_probs)
            else:
                action = int(np.argmax(action_probs))

            x, y = action % 15, action // 15
            board.play_move(x, y, board.current_player)
            move_count += 1

            # Check winner
            last_player = 1 if board.current_player == 2 else 2
            if board.check_win(last_player):
                return 1 if last_player == 1 else -1  # 1=new(black) wins, -1=old(white) wins

            if board.is_full():
                return 0
