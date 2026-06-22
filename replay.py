"""
Live training visualization.
Shows the entire population playing simultaneously, plus the
best neural network's topology and a real-time fitness curve.

Press ESC or close window to quit.
"""
import os
import sys
import json
import math
import pickle
from collections import deque
from pathlib import Path

import pygame
import neat

from game import Game, Bird, SCREEN_W, SCREEN_H, GROUND_H, PLAY_H, PIPE_W


BASE_DIR     = Path(__file__).parent.resolve()
LOG_PATH     = BASE_DIR / "logs" / "training.jsonl"
WINNER_PKL   = BASE_DIR / "logs" / "winner.pkl"
CONFIG_PATH  = BASE_DIR / "config-feedforward.txt"

# Colors
SKY        = (135, 206, 235)
GROUND     = (222, 184, 135)
PIPE_GREEN = (60, 180, 60)
PIPE_DARK  = (40, 130, 40)
BIRD       = (255, 220, 0)
BIRD_EYE   = (0, 0, 0)
TEXT       = (20, 20, 20)
GRID       = (200, 200, 200)
BEST_GLOW  = (255, 50, 50)
NN_POS     = (50, 50, 50)
NN_NEG     = (50, 50, 200)


def draw_pipes(surf, game):
    for p in game.pipes:
        x = int(p.x)
        top_h = int(p.top)
        bot_y = int(p.bot)
        pygame.draw.rect(surf, PIPE_GREEN, (x, 0, PIPE_W, top_h))
        pygame.draw.rect(surf, PIPE_DARK, (x, 0, PIPE_W, top_h), 2)
        pygame.draw.rect(surf, PIPE_GREEN, (x, bot_y, PIPE_W, PLAY_H - bot_y))
        pygame.draw.rect(surf, PIPE_DARK, (x, bot_y, PIPE_W, PLAY_H - bot_y), 2)


def draw_bird(surf, bird, color=BIRD, highlight=False):
    x = int(bird.x); y = int(bird.y)
    rect = pygame.Rect(x - 17, y - 12, 34, 24)
    pygame.draw.ellipse(surf, color, rect)
    # Eye
    pygame.draw.circle(surf, BIRD_EYE, (x + 8, y - 4), 3)
    # Beak
    beak = [(x + 14, y - 2), (x + 22, y), (x + 14, y + 4)]
    pygame.draw.polygon(surf, (255, 140, 0), beak)
    if highlight:
        pygame.draw.ellipse(surf, BEST_GLOW, rect, 2)


def draw_ground(surf):
    pygame.draw.rect(surf, GROUND, (0, PLAY_H, SCREEN_W, GROUND_H))
    pygame.draw.line(surf, (0, 0, 0), (0, PLAY_H), (SCREEN_W, PLAY_H), 2)


def draw_neural_net(surf, genome, config, x0, y0, w, h):
    """Draw the network. In neat-python, inputs are negative IDs (-1, -2, ...),
    output starts at 0, hidden nodes have positive IDs."""
    inputs  = [k for k in genome.nodes if k < 0]
    outputs = [k for k in genome.nodes if k == 0]
    hiddens = [k for k in genome.nodes if k > 0]

    def pos(node_id, col, n, i):
        cx = x0 + col * (w // 2)
        if n == 1:
            cy = y0 + h // 2
        else:
            cy = y0 + int((i + 0.5) * h / n)
        return cx, cy

    pos_in  = {nid: pos(nid, 0, len(inputs),  i) for i, nid in enumerate(inputs)}
    pos_out = {nid: pos(nid, 2, len(outputs), i) for i, nid in enumerate(outputs)}
    pos_h   = {nid: pos(nid, 1, max(1, len(hiddens)), i) for i, nid in enumerate(hiddens)}

    pos_of  = {**pos_in, **pos_h, **pos_out}

    # Draw connections
    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        if cg.key[0] not in pos_of or cg.key[1] not in pos_of:
            continue
        a = pos_of[cg.key[0]]; b = pos_of[cg.key[1]]
        col = NN_POS if cg.weight > 0 else NN_NEG
        thick = max(1, min(5, int(abs(cg.weight))))
        pygame.draw.line(surf, col, a, b, thick)

    # Draw nodes
    for nid, (cx, cy) in pos_of.items():
        if nid in inputs:
            pygame.draw.circle(surf, (200, 240, 200), (cx, cy), 11)
        elif nid in outputs:
            pygame.draw.circle(surf, (240, 200, 200), (cx, cy), 11)
        else:
            pygame.draw.circle(surf, (220, 220, 255), (cx, cy), 10)
        pygame.draw.circle(surf, (0, 0, 0), (cx, cy), 11, 2)


def draw_fitness_chart(surf, points, x0, y0, w, h):
    """Sparkline: x=gen, y=best_fitness."""
    pygame.draw.rect(surf, (255, 255, 255), (x0, y0, w, h))
    pygame.draw.rect(surf, (0, 0, 0), (x0, y0, w, h), 1)
    if not points:
        return
    max_y = max(p for _, p in points) or 1.0
    pts = []
    for i, (_, fit) in enumerate(points):
        px = x0 + int(i * (w - 2) / max(1, len(points) - 1)) + 1
        py = y0 + h - 2 - int(fit / max_y * (h - 4))
        pts.append((px, py))
    if len(pts) >= 2:
        pygame.draw.lines(surf, (0, 150, 0), False, pts, 2)


def main():
    if not WINNER_PKL.exists():
        print(f"Missing {WINNER_PKL}. Run train.py first.")
        sys.exit(1)

    pygame.init()
    pygame.display.set_caption("FlappyAI - best bird replay")
    clock = pygame.time.Clock()

    # Load winner
    with WINNER_PKL.open("rb") as f:
        winner = pickle.load(f)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Layout: game (left), NN (top-right), chart (bottom-right)
    PANEL_W = 360
    win = pygame.display.set_mode((SCREEN_W + PANEL_W, SCREEN_H))
    font  = pygame.font.SysFont("consolas", 16)
    title = pygame.font.SysFont("consolas", 18, bold=True)

    game = Game()
    history = deque(maxlen=200)
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                history.append((rec["gen"], rec["best_fitness"]))
            except Exception:
                pass

    alive = True
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); return

        if game.bird.alive:
            state = game.bird.get_state(game.next_pipe())
            out = net.activate(state)
            game.step(out[0] > 0.5)
        else:
            # Auto-reset so the replay keeps going
            pygame.time.wait(700)
            game.reset()

        # === Draw ===
        surf = win
        surf.fill(SKY)
        draw_pipes(surf, game)
        draw_ground(surf)
        draw_bird(surf, game.bird, highlight=True)

        # Panel
        panel_x = SCREEN_W
        pygame.draw.rect(surf, (245, 245, 245), (panel_x, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(surf, (0, 0, 0), (panel_x, 0), (panel_x, SCREEN_H), 2)

        surf.blit(title.render("Best Neural Network", True, TEXT), (panel_x + 10, 10))
        draw_neural_net(surf, winner, config,
                        x0=panel_x + 20, y0=45, w=PANEL_W - 40, h=260)

        surf.blit(title.render("Fitness curve (per gen)", True, TEXT),
                  (panel_x + 10, 320))
        draw_fitness_chart(surf, list(history),
                           x0=panel_x + 20, y0=350, w=PANEL_W - 40, h=160)

        surf.blit(font.render(f"Score: {game.bird.score}", True, TEXT),
                  (panel_x + 10, 520))
        surf.blit(font.render(f"Frames alive: {game.frame}", True, TEXT),
                  (panel_x + 10, 540))
        surf.blit(font.render(f"Trained fitness: {winner.fitness:.1f}", True, TEXT),
                  (panel_x + 10, 560))
        surf.blit(font.render("Press ESC to quit", True, TEXT),
                  (panel_x + 10, 590))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()