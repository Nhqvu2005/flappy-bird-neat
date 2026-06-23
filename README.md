# FlappyAI — Self-learning Flappy Bird Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![NEAT](https://img.shields.io/badge/NEAT-NeuroEvolution-brightgreen)](https://neat-python.readthedocs.io/)

> A virtual bird that learns to play Flappy Bird through evolution — no training data,
> no labels, no pre-designed neural network. Just **natural selection** over generations.

---

## Demo

<p align="center">
  <img src="docs/demo.gif" alt="Flappy AI Demo" width="300">
  <br>
  <em>AI playing Flappy Bird — avg 22.2 pipes across 10 unseen seeds (best 42)</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Training Guide](#training-guide)
- [Hyperparameter Tuning](#hyperparameter-tuning-for-better-accuracy)
- [Deployment Options](#deployment-options)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [FAQ & Troubleshooting](#faq--troubleshooting)

---

## Overview

This project uses the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm to evolve
a neural network that controls a Flappy Bird. Key highlights:

| Feature | Description |
|---|---|
| **No hand-engineered network** | NEAT automatically adds/removes hidden nodes as needed |
| **Pure-Python game engine** | Headless, runs without GPU |
| **Real-time web dashboard** | Fitness chart, neural network viz, playable AI/manual demo |
| **Pygame replay** | Watch the AI with live topology + historical fitness chart |
| **Fair genome comparison** | Fixed seeds per generation — all genomes face the same layouts |
| **All-time best genome tracking** | Saves the best-ever genome, not just the final generation |

---

## System Architecture

```
                    ┌──────────────────────┐
                    │   config-feedforward.txt │
                    │   (NEAT hyperparams)   │
                    └──────────┬───────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│                     train.py                                  │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────┐  │
│  │Population│ → │eval_genomes()│ → │LogReporter (tracking) │  │
│  │  (400)   │   │  (6 runs/    │   │ + Fitness-based       │  │
│  │          │   │   genome)    │   │   selection            │  │
│  └──────────┘   └──────┬───────┘   └───────────────────────┘  │
│                         │                                       │
│                    ┌────▼───────┐                               │
│                    │ game.py     │   (headless game engine)     │
│                    │ step()      │                               │
│                    │ get_state() │                               │
│                    │ collision   │                               │
│                    └────────────┘                               │
└───────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    logs/             │
                    │  ├─training.jsonl    │  ← per-generation stats
                    │  ├─winner.pkl        │  ← best genome (pickle)
                    │  └─winner_net.json   │  ← best network (JSON)
                    └──────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌──────────┐        ┌───────────┐        ┌──────────┐
   │ server.py │        │ replay.py │        │ train.py │
   │ + web/    │        │ + Pygame  │        │(terminal)│
   │ Browser   │        │ Desktop   │        │ Headless │
   └──────────┘        └───────────┘        └──────────┘
```

---

## How It Works

### Neural Network

Each bird is controlled by a feedforward network with:

**5 inputs** (normalized to [0, 1]):

| # | Signal | Description | Formula |
|---|---|---|---|
| 1 | `bird_y` | Bird's vertical position | `y / PLAY_H` |
| 2 | `velocity` | Velocity (0=just flapped, 1=falling fast) | `(vy - FLAP_VY) / (MAX_VY - FLAP_VY)` |
| 3 | `pipe_dx` | Horizontal distance to next pipe | `(pipe.x - bird.x) / SCREEN_W` |
| 4 | `top_y` | Top pipe's bottom edge | `pipe.top / PLAY_H` |
| 5 | `bot_y` | Bottom pipe's top edge | `pipe.bot / PLAY_H` |

**1 output**: Value in [-1, 1] — bird flaps when output **exceeds 0.5** (continuous flap).

**Hidden nodes**: None initially — NEAT adds hidden nodes via mutation (`node_add_prob=0.3`). Topology evolves naturally.

> **Note**: NEAT's pruning often removes the `pipe_dx` connection, so the bird may not "see" approaching pipes. Consider increasing `conn_add_prob` if this limits performance.

### Flap Mechanism

Two approaches were tested:

1. **Continuous flap** (current): `flap = out > 0.5`, with **FLAP_COOLDOWN=3** frames in game.py.
   - A 3-frame cooldown prevents rapid re-flapping:
   ```
   Frame:   1  2  3  4  5  6  7  8  9
   Output:  0.7 0.8 0.6 0.9 0.8 0.7 0.6 0.8 0.4
   Flap:    yes no  no  yes no  no  no  yes no
   ```
   - This prevents the bird from shooting up like a rocket.
   - Training score: **29.5 avg** (500 gen), validation **22.2 avg**

2. **Rising-edge flap** (experimental): `flap = prev_out <= 0.5 < out[0]`
   - Produced higher peak scores (42 avg) but winners developed extreme negative bias (-7.2), causing the bird to never flap until nearly hitting the ground.
   - Reverted to continuous flap for reliability.

### Fitness Function

```
Fitness = frames_alive × 0.1 + Σ(pipes_passed) × (50 + center_bonus)
```

- **0.1 per frame**: Rewards survival
- **+50 per pipe passed**: Primary reward
- **center_bonus (0–10)**: Bonus for passing through the **middle of the gap**:
  ```
  dist = |bird.y - gap_center|
  centering = max(0, 1 - dist / (PIPE_GAP/2))
  bonus = centering × 10
  ```
  Encourages precise alignment, reducing top/bottom pipe collisions.

### Fixed Seeds Per Generation

```python
_GEN_SEED_OFFSET = gen * 100 + 1
# All genomes face the same 6 layouts within a generation
```

- **Before**: Random seeds per genome → good genomes got hard layouts → unfair comparison → slow convergence.
- **Now**: Fixed layouts per generation → fair comparison → faster evolution.

---

## Training Guide

### Prerequisites

```bash
pip install neat-python tqdm pygame  # pygame optional (for replay)
```

### Basic Training

```bash
python train.py                  # 100 generations (default)
python train.py 200              # 200 generations
python train.py 500              # 500 generations (recommended for best results)
```

Output:
- `logs/training.jsonl` — per-generation log
- `logs/winner.pkl` — all-time best genome (pickle)
- `logs/winner_net.json` — best network topology (JSON)
- Terminal: best_fitness, mean_fitness, best_score, species count each gen

### Parallel Training

The trainer uses **ParallelEvaluator** from neat-python to distribute genome evaluation across all CPU cores:

```bash
python train.py 500    # automatically uses all CPU cores
```

On a 12-core machine, 500 generations completes in ~6 minutes (vs ~30 min single-threaded).

### Viewing Results

**Web dashboard** (recommended):
```bash
python server.py              # http://127.0.0.1:8765/
# Press A for AI mode, Space for manual play
```

**Pygame replay** (desktop):
```bash
python replay.py
# ESC to quit
```

---

## Hyperparameter Tuning (for Better Accuracy)

### 1. NEAT Config (`config-feedforward.txt`)

#### Population & Evolution

| Parameter | Current | Suggestion | Effect |
|---|---|---|---|
| `pop_size` | 400 | ↑ 500–800 | More gene diversity, slower |
| `elitism` | 2 | ↑ 3–5 | Keep more top performers |
| `survival_threshold` | 0.2 | ↑ 0.3–0.4 | Allow more weaker genomes (diversity) |
| `no_fitness_termination` | True | Keep as-is | Don't stop early |

#### Mutation

| Parameter | Value | Suggestion | Effect |
|---|---|---|---|
| `conn_add_prob` | 0.5 | ↑ 0.7 for complexity | Add new connections |
| `conn_delete_prob` | 0.3 | ↑ 0.5 to prune | Remove redundant connections |
| `node_add_prob` | 0.2 | ↑ 0.3–0.5 | Add hidden nodes |
| `node_delete_prob` | 0.2 | Keep or ↓ 0.1 | Remove redundant nodes |
| `weight_mutate_rate` | 0.8 | ↓ 0.5 near convergence | Fine-tune weights |
| `weight_mutate_power` | 0.5 | ↓ 0.3 for fine-tuning | Weight change magnitude |
| `bias_mutate_rate` | 0.7 | Keep or ↓ 0.5 | Fine-tune biases |

#### Species & Stagnation

| Parameter | Value | Suggestion | Effect |
|---|---|---|---|
| `compatibility_threshold` | 3.0 | ↑ 4–5 (fewer species) | Merge similar species |
| `max_stagnation` | 20 | ↑ 30–40 | Give species more recovery time |

### 2. Code Parameters

| Parameter | Location | Value | Suggestion | Effect |
|---|---|---|---|---|
| `num_runs` | `eval_genome()` | 6 | ↑ 10–12 | More thorough, slower |
| `PIPE_GAP` | `game.py` | 140 | ± 20 | Difficulty — smaller = harder |
| `PIPE_SPEED` | `game.py` | 2.0 | ± 0.5 | Speed — faster = harder |
| `FLAP_COOLDOWN` | `game.py` | 3 | ± 1–2 | Control flap rate |
| `max_frames` | `eval_genome()` | 5400 (60×90) | ↑ 7200 | Longer max survival |

### 3. Fitness Tuning

Adjust in `eval_genome()`:

```python
fitness += 0.1         # ↑ 0.2 → reward survival more
fitness += 50          # ↑ 80 → reward pipe-passing more
fitness += centering * 10  # ↑ 20 → tighter centering
```

### 4. Adding Inputs

To improve, add a 6th input (e.g., distance to gap center):

```python
# game.py → Bird.get_state()
(
    self.y / PLAY_H,
    (self.vy - FLAP_VY) / (MAX_VY - FLAP_VY),
    (next_pipe.x - self.x) / SCREEN_W,
    (next_pipe.top) / PLAY_H,
    (next_pipe.bot) / PLAY_H,
    (self.y - next_pipe.gap_y) / PLAY_H,  # input 6: offset from gap center
)
```

Update `num_inputs = 6` in `config-feedforward.txt`.

### 5. Training Strategy

1. **Fast scan**: 100 gens, pop_size=200 → see trends
2. **Main training**: 400–500 gens, pop_size=400–600
3. **If stuck (plateau)**: Set `reset_on_extinction = True`, or increase `conn_add_prob`/`node_add_prob`
4. **If network too complex**: Increase `conn_delete_prob`, decrease `node_add_prob`

---

## Deployment Options

### 1. Local Development (default)

```bash
python train.py 400
python server.py          # http://127.0.0.1:8765/
```

### 2. Server Deployment

Deploy to a VPS/dedicated server:

```bash
# Install
pip install neat-python

# Train (headless)
python train.py 500

# Serve on custom port
python server.py 8080

# Reverse proxy with nginx
# server {
#     listen 80;
#     server_name flappyai.example.com;
#     location / {
#         proxy_pass http://127.0.0.1:8080;
#     }
# }
```

API endpoints:
| Endpoint | Description |
|---|---|
| `/` | Web dashboard (index.html) |
| `/api/log` | Training log (JSON array) |
| `/api/winner` | Winner network topology (JSON) |

### 3. Cloudflare Tunnel (for production)

Use Cloudflare Tunnel for secure public access:

```bash
# Add DNS record
cloudflared tunnel route dns <tunnel-id> flappyai.example.com

# Add ingress to /etc/cloudflared/config.yml
#   - hostname: flappyai.example.com
#     service: http://localhost:8765

# Reload cloudflared
systemctl restart cloudflared
```

### 4. CI/CD Integration (GitHub Actions)

Set up automated daily training:
1. Push code → GitHub
2. Action runs `python train.py 200` on server
3. Winner + logs push back to repo
4. Dashboard updates in real-time

---

## Experimental Results

| Metric | Random seeds | Fixed seeds (v1) | Rising-edge + tuning | **Parallel (500 gen)** | Description |
|---|---|---|---|---|---|
| Best score (avg 6 runs) | 22.33 | 29.00 | 42.00 | **29.50** | Average pipes passed |
| Best generation | gen 168 | gen 158 | gen 329 | **gen 41** | Generation with best score |
| Winner hidden nodes | 1 | 8 | 5 | **2** | Network complexity |
| Winner fitness | ~1100 | ~1821 | ~2974 | **~2052** | Peak fitness |
| Validation (10 seeds) | 3.7 avg | 7.0 avg | 26.1 avg | **22.2 avg** | Unseen layout avg pipes |
| Gens to stabilize >20 pipes | ~120 | ~80 | ~37 | **~33** | Convergence speed |
| Training time (500 gen) | ~30 min | ~30 min | ~30 min | **~6 min** | CPU wall clock |
| Zero-score seeds | 6/10 | 3/10 | 1/10 | **0/10** | Seeds where bird dies at pipe 1 |

### Key Insights

- **Fixed seeds per generation** was the most important improvement — enabling fair
  genome comparison and significantly boosting convergence speed.
- **Center bonus** encourages passing through the gap center, reducing top/bottom collisions.
- **Continuous flap + cooldown** works well when coupled with `pop_size=400` and `num_runs=6`.
- **Parallel evaluator** (12 cores) reduced training time from ~30 min to ~6 min (5× faster).
- **Current limitations**: The winner network often lacks `pipe_dx` connections, meaning the bird doesn't "see" approaching pipes — it reacts based on height and velocity only, not pipe proximity.
- The best score was achieved relatively early (gen 41), with later generations failing to improve further — suggesting the population got stuck in a local optimum.

### Future Improvements

1. **Force `pipe_dx` connections**: Increase `conn_add_prob` to 0.8–0.9 so the network learns to react based on pipe distance, not just height/velocity.

2. **Larger population**: Increase `pop_size` to 600–800 and run for 1000+ generations.

3. **Cycle detection/reset**: Enable `reset_on_extinction = True` to restart species when they stagnate, giving the population a chance to break out of local optima.

4. **6th input — gap distance**: Add `(bird.y - gap_center) / PLAY_H` as a 6th input so the network knows exactly how far from the gap center the bird is.

5. **Dynamic difficulty**: Start with wider `PIPE_GAP` (e.g., 160) and gradually narrow it during training, so the network learns progressively harder layouts.

6. **Rising-edge flap re-evaluation**: The rising-edge approach showed higher peak scores (42 avg) but produced winners with extreme negative bias. A hybrid approach (continuous flap during training, rising-edge during inference) could combine the best of both.

7. **Reward shaping**: Increase `center_bonus` multiplier from 10 to 20, and add a penalty for hitting near pipe edges to encourage tighter centering.

---

## Project Structure

```
FlappyAI/
├── game.py                   # Flappy Bird engine (headless)
├── train.py                  # NEAT training script
├── replay.py                 # Pygame viewer
├── server.py                 # Web server + API
├── record_demo.py            # Headless GIF recorder
├── config-feedforward.txt    # NEAT hyperparameters
├── requirements.txt          # Python dependencies
├── start.bat                 # Windows menu
├── logs/                     # Runtime outputs
│   ├── training.jsonl        #   Per-generation log
│   ├── winner.pkl            #   Best genome (pickle)
│   └── winner_net.json       #   Best network (JSON)
├── docs/                     # Documentation assets
│   └── demo.gif              #   Demo GIF
├── web/                      # Dashboard frontend
│   ├── index.html
│   ├── style.css
│   ├── flappy.js             #   Game engine (JS)
│   └── app.js                #   Chart.js + NN viz + player
├── README.md                 # This file (English)
├── README.vi.md              # Vietnamese version
└── LICENSE                   # MIT
```

---

## FAQ & Troubleshooting

### "Training is too slow"

- Decrease `pop_size` to 200–300
- Decrease `num_runs` to 3–4

### "Bird always dies at the first pipe"

1. Bird overshoots upward → **Hits top pipe**
   - **Fix**: Flap cooldown (already implemented, 3 frames)
   - Or: decrease `FLAP_VY` in game.py to -6

2. Bird drops onto bottom pipe → **Hits bottom pipe**
   - **Fix**: Ensure network has connections to `bot_y` and `velocity` inputs
   - Or: increase `PIPE_GAP` to 150–160

3. Bird doesn't react to approaching pipes
   - **Fix**: Check if `pipe_dx` input is connected in the network (via winner_net.json)
   - If not: increase `conn_add_prob`

### "Web dashboard not showing"

```bash
# Check files exist
ls logs/winner_net.json logs/training.jsonl

# If missing: retrain
python train.py

# If port conflict: use a different port
python server.py 8766
```

### "Want to improve the score"

See [Hyperparameter Tuning](#hyperparameter-tuning-for-better-accuracy) above.

### "Want to see the network topology"

Open http://127.0.0.1:8765/ and view the SVG network visualization on the right.
Or open `logs/winner_net.json` in any text editor.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Nhqvu2005** — [GitHub](https://github.com/Nhqvu2005)

## Credits

- [NEAT-Python](https://neat-python.readthedocs.io/) — NEAT library
- [Kenneth O. Stanley](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) — NEAT algorithm
- [Dong Nguyen](https://en.wikipedia.org/wiki/Flappy_Bird) — Original Flappy Bird
