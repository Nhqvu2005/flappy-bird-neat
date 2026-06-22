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
from pathlib import Path

import neat

from game import Game, Bird, SCREEN_H, SCREEN_W, PLAY_H, PIPE_W, PIPE_SPACING


BASE_DIR     = Path(__file__).parent.resolve()
LOG_PATH     = BASE_DIR / "logs" / "training.jsonl"
WINNER_DIR   = BASE_DIR / "logs"
CONFIG_PATH  = BASE_DIR / "config-feedforward.txt"


def eval_genome(genome, config):
    """Play one bird; return fitness.
    Fitness = score * 50  +  frames_alive * 0.1
    Encourages passing pipes (huge reward) but also surviving long."""
    game = Game(seed=hash(genome.key) & 0xFFFF)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0.0
    max_frames = 60 * 90     # ~90 seconds at 60 fps is one upper bound

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

    return fitness, game.bird.score


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness, _ = eval_genome(genome, config)


class LogReporter(neat.reporting.BaseReporter):
    """Per-generation JSONL log + print to stdout."""

    def __init__(self):
        self.best_per_gen = []
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def start_generation(self, generation):
        self.gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for _, g in population.items()]
        best_fit = max(fits)
        mean_fit = sum(fits) / len(fits)
        species_count = len(species.species)
        # Replay the best genome to get its actual score
        _, best_score = eval_genome(best_genome, config)
        record = {
            "gen":          self.gen,
            "best_fitness": round(best_fit, 2),
            "mean_fitness": round(mean_fit, 2),
            "best_score":   best_score,
            "species":      species_count,
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[gen {self.gen:>3}] best_fit={best_fit:7.2f}  "
              f"mean_fit={mean_fit:7.2f}  score={best_score:>3}  "
              f"species={species_count}")


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        str(CONFIG_PATH),
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(LogReporter())

    winner = p.run(eval_genomes, gens)

    # Save winner
    with (WINNER_DIR / "winner.pkl").open("wb") as f:
        pickle.dump(winner, f)

    # Save winner topology for web viz.
    # neat-python uses negative IDs for inputs, 0 for the first output.
    nodes = []
    for nid in winner.nodes:
        if nid < 0:
            t = "input"
        elif nid == 0:
            t = "output"
        else:
            t = "hidden"
        nodes.append({"id": nid, "type": t})
    connections = []
    for cg in winner.connections.values():
        if cg.enabled:
            connections.append({
                "in":     cg.key[0],
                "out":    cg.key[1],
                "weight": round(cg.weight, 3),
            })
    with (WINNER_DIR / "winner_net.json").open("w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "connections": connections,
                   "fitness": winner.fitness}, f, indent=2)

    print(f"\n>>> Saved winner to logs/winner.pkl  (fitness={winner.fitness:.2f})")


if __name__ == "__main__":
    main()