/* Flappy Bird engine in vanilla JS (mirrors Python game.py constants). */
(() => {
  const SCREEN_W = 400, SCREEN_H = 600, GROUND_H = 80, PLAY_H = SCREEN_H - GROUND_H;
  const GRAVITY = 0.5, FLAP_VY = -8.0, MAX_VY = 10.0;
  const PIPE_W = 60, PIPE_GAP = 140, PIPE_SPACING = 200, PIPE_SPEED = 2.0;
  const BIRD_W = 34, BIRD_H = 24;

  function rand(min, max) { return min + Math.random() * (max - min); }

  function nextPipe(pipes, birdX) {
    let best = null, bestDx = Infinity;
    for (const p of pipes) {
      if (p.x + PIPE_W < birdX) continue;
      const dx = p.x - birdX;
      if (dx < bestDx) { bestDx = dx; best = p; }
    }
    return best;
  }

  class Game {
    constructor() {
      this.bird = { x: SCREEN_W * 0.3, y: PLAY_H * 0.5, vy: 0, alive: true, score: 0, lastFlap: -999 };
      this.pipes = [];
      this.spawnPipe(SCREEN_W + 50);
      this.spawnPipe(SCREEN_W + 50 + PIPE_SPACING);
      this.frame = 0;
    }
    spawnPipe(x) {
      const gapCenter = rand(120, PLAY_H - 120);
      this.pipes.push({ x, gapY: gapCenter, scored: false });
    }
    getState() {
      const p = nextPipe(this.pipes, this.bird.x);
      return [
        this.bird.y / PLAY_H,
        (this.bird.vy - FLAP_VY) / (MAX_VY - FLAP_VY),
        (p.x - this.bird.x) / SCREEN_W,
        (p.gapY - PIPE_GAP / 2) / PLAY_H,
        (p.gapY + PIPE_GAP / 2) / PLAY_H,
      ];
    }
    step(doFlap) {
      if (!this.bird.alive) return false;
      // Flap cooldown: prevents rapid re-flapping that shoots bird upward
      if (doFlap && this.frame - this.bird.lastFlap >= 3) {
        this.bird.vy = FLAP_VY;
        this.bird.lastFlap = this.frame;
      }
      this.bird.vy = Math.min(this.bird.vy + GRAVITY, MAX_VY);
      this.bird.y += this.bird.vy;

      for (const p of this.pipes) p.x -= PIPE_SPEED;

      if (this.pipes[this.pipes.length - 1].x < SCREEN_W - PIPE_SPACING) {
        this.spawnPipe(SCREEN_W + 50);
      }
      this.pipes = this.pipes.filter(p => p.x + PIPE_W > -10);

      const p = nextPipe(this.pipes, this.bird.x);
      if (p && p.x + PIPE_W / 2 < this.bird.x && !p.scored) {
        p.scored = true;
        this.bird.score++;
      }

      // Collisions
      if (this.bird.y + BIRD_H / 2 >= PLAY_H) { this.bird.y = PLAY_H - BIRD_H / 2; this.bird.alive = false; }
      else if (this.bird.y - BIRD_H / 2 <= 0) { this.bird.y = BIRD_H / 2; this.bird.vy = 0; }

      if (this.bird.alive) {
        const bx = this.bird.x - BIRD_W / 2, by = this.bird.y - BIRD_H / 2;
        for (const pp of this.pipes) {
          if (bx + BIRD_W > pp.x && bx < pp.x + PIPE_W) {
            if (by < pp.gapY - PIPE_GAP / 2 || by + BIRD_H > pp.gapY + PIPE_GAP / 2) {
              this.bird.alive = false; break;
            }
          }
        }
      }
      this.frame++;
      return this.bird.alive;
    }
    reset() {
      this.bird = { x: SCREEN_W * 0.3, y: PLAY_H * 0.5, vy: 0, alive: true, score: 0 };
      this.pipes = [];
      this.spawnPipe(SCREEN_W + 50);
      this.spawnPipe(SCREEN_W + 50 + PIPE_SPACING);
      this.frame = 0;
    }
  }

  window.Flappy = {
    Game,
    SCREEN_W,
    SCREEN_H,
    GROUND_H,
    PLAY_H,
    PIPE_W,
    PIPE_GAP,
    PIPE_SPEED,
    BIRD_W,
    BIRD_H,
  };
})();