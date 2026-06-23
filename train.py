"""
Headless NEAT training script.
Runs many generations, evaluates every genome against the game,
saves the winning genome + training log to disk for the web dashboard.

Usage:
    python train.py [gens]
    gens = max number of generations (default 100)
"""
import os
import sys
import json
import pickle
import math
import copy
from pathlib import Path

import neat

# optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from game import Game, Bird, SCREEN_H, SCREEN_W, PLAY_H, PIPE_W, PIPE_SPACING



BASE_DIR     = Path(__file__).parent.resolve()
LOG_PATH     = BASE_DIR / "logs" / "training.jsonl"
WINNER_DIR   = BASE_DIR / "logs"
CONFIG_PATH  = BASE_DIR / "config-feedforward.txt"


def export_winner_net(genome, out_path, input_ids=None, output_ids=None):
    """Export a NEAT feed-forward genome in a browser-friendly format."""
    enabled_connections = [cg for cg in genome.connections.values() if cg.enabled]
    if input_ids is None:
        input_ids = sorted({cg.key[0] for cg in enabled_connections if cg.key[0] < 0})
    else:
        input_ids = list(input_ids)

    if output_ids is None:
        source_ids = {cg.key[0] for cg in enabled_connections}
        output_ids = sorted(nid for nid in genome.nodes if nid not in source_ids)
    else:
        output_ids = list(output_ids)

    output_id_set = set(output_ids)
    nodes = [{"id": nid, "type": "input"} for nid in input_ids]
    for nid, node in sorted(genome.nodes.items()):
        nodes.append({
            "id": nid,
            "type": "output" if nid in output_id_set else "hidden",
            "bias": round(node.bias, 6),
            "response": round(getattr(node, "response", 1.0), 6),
            "activation": getattr(node, "activation", "tanh"),
            "aggregation": getattr(node, "aggregation", "sum"),
        })

    connections = []
    for cg in enabled_connections:
        connections.append({
            "in":     cg.key[0],
            "out":    cg.key[1],
            "weight": round(cg.weight, 6),
        })

    payload = {
        "nodes": nodes,
        "connections": connections,
        "fitness": genome.fitness,
        "input_ids": input_ids,
        "output_ids": output_ids,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# Fixed seeds per generation — all genomes face the SAME layouts for fair comparison.
# Different generations get different seeds to prevent memorization.
# Seed = gen * 100 + run_index (run_index = 0..num_runs-1)
_GEN_SEED_OFFSET = 0

def set_gen_seeds(gen):
    global _GEN_SEED_OFFSET
    _GEN_SEED_OFFSET = gen * 100 + 1

def eval_genome(genome, config, num_runs=6):
    """Play a bird; return (avg_fitness, avg_score).
    Fitness = score * 50 + frames_alive * 0.1
    Each genome is tested against num_runs DIFFERENT pipe layouts.
    Uses FIXED seeds per generation (all genomes face the same layouts)
    so the resulting network GENERALIZES instead of memorizing one layout."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_fitness = 0.0
    total_score = 0
    max_frames = 60 * 90

    for run_i in range(num_runs):
        game = Game(seed=_GEN_SEED_OFFSET + run_i)
        fitness = 0.0
        prev_score = 0
        for _ in range(max_frames):
            if not game.bird.alive:
                break
            state = game.bird.get_state(game.next_pipe())
            out = net.activate(state)
            flap = out[0] > 0.5
            game.step(flap)
            fitness += 0.1
            if game.bird.score > prev_score:
                fitness += 50
                prev_score = game.bird.score
        total_fitness += fitness
        total_score += game.bird.score

    return total_fitness / num_runs, total_score / num_runs


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness, _ = eval_genome(genome, config)


class LogReporter(neat.reporting.BaseReporter):
    """Per-generation JSONL log + optional tqdm progress bar."""

    def __init__(self, use_tqdm=False):
        self.best_per_gen = []
        self.alltime_best_genome = None
        self.alltime_best_score = -1
        self.alltime_best_gen = -1
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.use_tqdm = use_tqdm
        self._progress = None

    def start_generation(self, generation):
        self.gen = generation
        set_gen_seeds(generation)  # fixed seeds for fair comparison
        if self._progress:
            self._progress.update(1)

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for _, g in population.items()]
        best_fit = max(fits)
        mean_fit = sum(fits) / len(fits)
        species_count = len(species.species)
        # Replay the best genome to get its actual score
        _, best_score = eval_genome(best_genome, config)

        # Track all-time best genome across ALL generations
        if self.alltime_best_genome is None or best_score > self.alltime_best_score:
            self.alltime_best_genome = copy.deepcopy(best_genome)
            self.alltime_best_score = best_score
            self.alltime_best_gen = self.gen

        record = {
            "gen":          self.gen,
            "best_fitness": round(best_fit, 2),
            "mean_fitness": round(mean_fit, 2),
            "best_score":   best_score,
            "species":      species_count,
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        # When tqdm is active, show stats in the progress bar (no extra lines)
        if self.use_tqdm and self._progress:
            self._progress.set_postfix({
                'best': f'{best_fit:.0f}',
                'mean': f'{mean_fit:.0f}',
                'score': best_score,
                'sp': species_count,
            })
        else:
            # Without tqdm, print one line per generation
            print(f"[gen {self.gen:>3}] best_fit={best_fit:7.2f}  "
                  f"mean_fit={mean_fit:7.2f}  score={best_score:>3}  "
                  f"species={species_count}")


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # optional progress bar for generations
    if tqdm:
        progress = tqdm(total=gens, desc="Training", unit="gen")
    else:
        progress = None

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    input_ids = list(config.genome_config.input_keys)
    output_ids = list(config.genome_config.output_keys)

    p = neat.Population(config)
    # Silent mode when tqdm is active — no StdOutReporter at all
    if not tqdm:
        p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    log_reporter = LogReporter(use_tqdm=bool(tqdm))
    p.add_reporter(log_reporter)

    # Store progress bar in the reporter so it can be updated each generation
    log_reporter._progress = progress

    _ = p.run(eval_genomes, gens)  # discard last-gen winner

    # Close progress bar
    if progress:
        progress.close()

    # Use the all-time best genome instead of the last generation's winner
    winner = log_reporter.alltime_best_genome
    if winner is None:
        print("ERROR: no genome found! Training may have failed.")
        return

    # Run final validation on 10 fresh seeds (not seen during training)
    print("\nRunning final validation (10 unseen seeds)...")
    total_score = 0
    val_seeds = [9999 + i for i in range(10)]
    for seed in val_seeds:
        game = Game(seed=seed)
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        for _ in range(5400):
            if not game.bird.alive:
                break
            state = game.bird.get_state(game.next_pipe())
            out = net.activate(state)
            game.step(out[0] > 0.5)
        total_score += game.bird.score
    avg_final_score = total_score / 10
    print(f"  Validation avg score: {avg_final_score:.1f}  (range: call python train.py for details)")

    # Save winner
    with (WINNER_DIR / "winner.pkl").open("wb") as f:
        pickle.dump(winner, f)

    # Save winner topology for web viz.
    export_winner_net(winner, WINNER_DIR / "winner_net.json", input_ids, output_ids)

    print(f"\n>>> Saved ALL-TIME BEST genome (gen {log_reporter.alltime_best_gen}, "
          f"training score={log_reporter.alltime_best_score:.0f}) to logs/winner.pkl  "
          f"(fitness={winner.fitness:.2f})")


if __name__ == "__main__":
    main()