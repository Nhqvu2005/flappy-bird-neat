# FlappyAI

A self-learning agent that **learns to play Flappy Bird from scratch** using the
**NEAT (NeuroEvolution of Augmenting Topologies)** algorithm — no training data,
no labels, just natural selection across generations.

![demo](https://raw.githubusercontent.com/Nhqvu2005/FlappyAI/main/docs/demo.png)

## What you get

| Feature | How it works |
|---|---|
| **Pure-Python game engine** | `game.py` — decoupled physics + collision; can run headless |
| **NEAT training** | `train.py` — evolves a population of 80 neural networks |
| **Live Pygame replay** | `replay.py` — watch the winner play, with the **NN topology drawn live** |
| **Web dashboard** | `web/` + `server.py` — interactive Chart.js fitness graph, NN visualization, in-browser playable demo with AI toggle |

## Quick start (Windows)

```bat
:: 1. Install Python dependencies (one-time)
start.bat  ->  1

:: 2. Train the AI (100 generations, ~5-15 min on a laptop CPU)
start.bat  ->  2

:: 3. Open the web dashboard
start.bat  ->  4
:: -> http://127.0.0.1:8765/
```

Or on any platform:

```bash
pip install -r requirements.txt
python train.py 50            # train for 50 generations
python server.py              # http://127.0.0.1:8765/
python replay.py              # local Pygame window
```

## How does it learn?

Each bird has a tiny neural network with **5 inputs**:

1. Bird's vertical position (0-1)
2. Bird's vertical velocity (0-1)
3. Horizontal distance to next pipe (0-1)
4. Top pipe bottom y (0-1)
5. Bottom pipe top y (0-1)

And **1 output** (flap if > 0.5). Hidden nodes are *added by NEAT* — the
topology grows over time. The fitness function rewards passing pipes (huge
bonus) plus staying alive per frame.

After ~50 generations, the best bird reliably passes dozens of pipes.

## Files

```
FlappyAI/
├── game.py                # Flappy Bird engine (no rendering)
├── train.py               # Headless NEAT training loop
├── replay.py              # Pygame viewer with live NN + fitness chart
├── server.py              # Tiny HTTP server (serves web/ + JSON APIs)
├── config-feedforward.txt # NEAT hyperparameters
├── requirements.txt
├── start.bat              # Windows menu
├── web/
│   ├── index.html
│   ├── style.css
│   ├── flappy.js          # Game engine in JS (mirrors game.py)
│   └── app.js             # Chart.js + SVG NN + playable demo
└── logs/                  # Created at runtime
    ├── training.jsonl     # One record per generation
    ├── winner.pkl         # Best genome
    └── winner_net.json    # Winner topology for web viz
```

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

For educational/personal use. The NEAT algorithm is from Kenneth O. Stanley;
this project uses the `neat-python` library.