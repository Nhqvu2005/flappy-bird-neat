"""
Flappy Bird - core game logic
Decoupled from rendering so it can run headless during NEAT training,
or be drawn to a Pygame surface during visualization.
"""
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- World constants ---
SCREEN_W = 400
SCREEN_H = 600
GROUND_H = 80
PLAY_H = SCREEN_H - GROUND_H

GRAVITY = 0.5
FLAP_VY = -8.0
MAX_VY = 10.0

PIPE_W = 60
PIPE_GAP = 140          # vertical gap between top/bottom pipe
PIPE_SPACING = 200      # horizontal distance between consecutive pipes
PIPE_SPEED = 3.0

BIRD_W = 34
BIRD_H = 24


@dataclass
class Pipe:
    x: float
    gap_y: float        # y-center of the gap

    @property
    def top(self) -> float:
        return self.gap_y - PIPE_GAP / 2

    @property
    def bot(self) -> float:
        return self.gap_y + PIPE_GAP / 2


@dataclass
class Bird:
    x: float = SCREEN_W * 0.3
    y: float = PLAY_H * 0.5
    vy: float = 0.0
    alive: bool = True
    score: int = 0
    fitness: float = 0.0

    def flap(self) -> None:
        self.vy = FLAP_VY

    def update(self) -> None:
        self.vy = min(self.vy + GRAVITY, MAX_VY)
        self.y += self.vy

    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x - BIRD_W / 2, self.y - BIRD_H / 2, BIRD_W, BIRD_H)

    def get_state(self, next_pipe: Pipe) -> List[float]:
        """Inputs for the NEAT network.
        Normalized values so the network doesn't need to learn raw pixel offsets."""
        return [
            self.y / PLAY_H,                              # 1. bird vertical position (0-1)
            (self.vy - FLAP_VY) / (MAX_VY - FLAP_VY),     # 2. velocity (0-1, 0=just flapped)
            (next_pipe.x - self.x) / SCREEN_W,            # 3. horizontal distance to next pipe
            (next_pipe.top) / PLAY_H,                     # 4. top pipe bottom y (0-1)
            (next_pipe.bot) / PLAY_H,                     # 5. bottom pipe top y (0-1)
        ]


class Game:
    """Single-bird game; for population training, instantiate many Birds in one env."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.bird = Bird()
        self.pipes: List[Pipe] = []
        self.frame = 0
        self._spawn_pipe(SCREEN_W + 50)
        self._spawn_pipe(SCREEN_W + 50 + PIPE_SPACING)

    def _spawn_pipe(self, x: float) -> None:
        gap_center = self.rng.uniform(120, PLAY_H - 120)
        self.pipes.append(Pipe(x=x, gap_y=gap_center))

    def next_pipe(self) -> Pipe:
        return min(self.pipes, key=lambda p: p.x - self.bird.x if p.x + PIPE_W >= self.bird.x else 1e9)

    def step(self, do_flap: bool) -> bool:
        """Advance one frame. Returns False if bird died this step."""
        if not self.bird.alive:
            return False

        if do_flap:
            self.bird.flap()

        self.bird.update()

        # Scroll pipes
        for p in self.pipes:
            p.x -= PIPE_SPEED

        # Spawn new pipe when the second-to-last one is far enough left
        if self.pipes[-1].x < SCREEN_W - PIPE_SPACING:
            self._spawn_pipe(SCREEN_W + 50)

        # Remove pipes that went off-screen
        self.pipes = [p for p in self.pipes if p.x + PIPE_W > -10]

        # Score when bird crosses pipe center
        np = self.next_pipe()
        if np.x + PIPE_W / 2 < self.bird.x and not hasattr(np, "_scored"):
            np._scored = True
            self.bird.score += 1

        # Collisions
        if self.bird.y + BIRD_H / 2 >= PLAY_H:        # hit ground
            self.bird.y = PLAY_H - BIRD_H / 2
            self.bird.alive = False
        elif self.bird.y - BIRD_H / 2 <= 0:           # hit ceiling
            self.bird.y = BIRD_H / 2
            self.bird.vy = 0
        else:
            bx, by, bw, bh = self.bird.rect()
            for p in self.pipes:
                if (bx + bw > p.x and bx < p.x + PIPE_W):
                    if by < p.top or by + bh > p.bot:
                        self.bird.alive = False
                        break

        self.frame += 1
        return self.bird.alive

    def reset(self) -> None:
        self.bird = Bird()
        self.pipes = []
        self.frame = 0
        self._spawn_pipe(SCREEN_W + 50)
        self._spawn_pipe(SCREEN_W + 50 + PIPE_SPACING)
