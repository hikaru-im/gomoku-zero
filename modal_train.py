"""Modal.com training entry point for Gomoku Zero.

Usage:
    modal run modal_train.py                    # default: 100 iterations on A10G
    modal run modal_train.py --iterations 500   # custom iteration count
    modal run modal_train.py --gpu A100         # use A100 instead of A10G
"""

import modal

app = modal.App("gomoku-zero")

# ─── Persistent volume for checkpoints ───
VOLUME_NAME = "gomoku-zero-checkpoints"
checkpoint_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ─── Custom image: build C++ core at image build time ───
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++", "make")
    .pip_install(
        "torch>=2.0",
        "numpy",
        "cython",
    )
    # Copy project into the image
    .copy_local_dir(".", "/gomoku-zero", exclude=[".git", "__pycache__", "*.pyc", ".claude"])
    # Compile C++/Cython extension
    .run_commands(
        "cd /gomoku-zero && python core_logic/setup.py build_ext --inplace",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/checkpoints": checkpoint_volume},
    timeout=7200,       # 2 hours per call
    enable_output=True,
)
def train(iterations: int = 100, start_iteration: int = 0):
    """Run self-play training for the given number of iterations.

    Checkpoints are saved to /checkpoints (persistent volume).
    """
    import sys
    import os

    sys.path.insert(0, "/gomoku-zero")
    os.chdir("/gomoku-zero")

    from training.trainer import GomokuTrainer

    ckpt_dir = "/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = GomokuTrainer(
        checkpoint_dir=ckpt_dir,
        num_blocks=10,
        num_filters=128,
        lr=2e-3,
        batch_size=512,
        buffer_size=500000,
        self_play_games=50,
        training_steps=500,
        mcts_simulations=200,
        eval_games=100,
        eval_win_rate=0.55,
        temperature_threshold=30,
        device="cuda",
        max_iterations=iterations + start_iteration,
    )

    # Resume from checkpoint on volume
    trainer.resume()

    # Override iteration to start from the correct position
    if start_iteration > 0:
        trainer.iteration = start_iteration

    trainer.run(max_iterations=iterations + start_iteration)

    # Commit volume writes
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    iterations: int = 100,
    gpu: str = "A10G",
    start_iteration: int = 0,
):
    """Launch training on Modal.

    Args:
        iterations: number of training iterations to run.
        gpu: GPU type ("A10G", "A100", "T4", "H100").
        start_iteration: iteration to resume from (0 = auto-detect from checkpoint).
    """
    print(f"Launching Gomoku Zero training: {iterations} iterations on {gpu}")
    print(f"Checkpoints stored in volume: {VOLUME_NAME}")

    train.with_options(gpu=gpu).remote(
        iterations=iterations,
        start_iteration=start_iteration,
    )
