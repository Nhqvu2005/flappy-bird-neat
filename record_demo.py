"""
Record a demo GIF of the AI playing Flappy Bird.
Runs headless, captures frames, outputs a GIF ready for GitHub.

Usage:
    python record_demo.py [output.gif] [duration_seconds]
"""
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from game import Game, Bird, Pipe, SCREEN_H, SCREEN_W, PLAY_H, GROUND_H, PIPE_W, PIPE_GAP, BIRD_W, BIRD_H, PIPE_SPEED


def draw_frame(draw, game, frame_idx):
    """Draw the Flappy Bird game state onto a PIL image."""
    # Sky gradient
    for y in range(SCREEN_H):
        r = int(135 + (220-135) * y / SCREEN_H)
        g = int(206 + (240-206) * y / SCREEN_H)
        b = int(235 + (255-235) * y / SCREEN_H)
        draw.point((0, y), fill=(r, g, b))
        draw.point((SCREEN_W - 1, y), fill=(r, g, b))
    draw.rectangle([0, 0, SCREEN_W, SCREEN_H], fill=None)

    # Draw ground
    draw.rectangle([0, PLAY_H, SCREEN_W, SCREEN_H], fill=(87, 54, 31))
    draw.rectangle([0, PLAY_H, SCREEN_W, PLAY_H + 4], fill=(133, 183, 62))

    # Draw pipes
    for p in game.pipes:
        # Top pipe
        draw.rectangle([p.x, 0, p.x + PIPE_W, p.top], fill=(0, 180, 0))
        draw.rectangle([p.x - 2, 0, p.x, p.top], fill=(0, 210, 0))
        draw.rectangle([p.x + PIPE_W, 0, p.x + PIPE_W + 2, p.top], fill=(0, 140, 0))
        # Bottom pipe
        draw.rectangle([p.x, p.bot, p.x + PIPE_W, PLAY_H], fill=(0, 180, 0))
        draw.rectangle([p.x - 2, p.bot, p.x, PLAY_H], fill=(0, 210, 0))
        draw.rectangle([p.x + PIPE_W, p.bot, p.x + PIPE_W + 2, PLAY_H], fill=(0, 140, 0))

    # Draw bird
    bx = int(game.bird.x - BIRD_W / 2)
    by = int(game.bird.y - BIRD_H / 2)
    # Body
    draw.ellipse([bx, by, bx + BIRD_W, by + BIRD_H], fill=(240, 200, 20), outline=(180, 140, 0))
    # Eye
    draw.ellipse([bx + BIRD_W - 10, by + 4, bx + BIRD_W - 4, by + 10], fill=(255, 255, 255))
    draw.ellipse([bx + BIRD_W - 9, by + 5, bx + BIRD_W - 6, by + 8], fill=(0, 0, 0))
    # Beak
    draw.polygon([(bx + BIRD_W - 2, by + BIRD_H // 2 - 2),
                  (bx + BIRD_W + 6, by + BIRD_H // 2),
                  (bx + BIRD_W - 2, by + BIRD_H // 2 + 2)], fill=(255, 120, 0))

    # Score
    draw.text((10, 10), f"Score: {game.bird.score}", fill=(255, 255, 255), font=None)

    return game.bird.alive


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "docs/demo.gif"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    from train import WINNER_DIR, CONFIG_PATH
    import pickle
    import neat

    # Load winner
    winner_path = WINNER_DIR / "winner.pkl"
    if not winner_path.exists():
        print("ERROR: no winner.pkl found. Run train.py first.")
        return 1

    with winner_path.open("rb") as f:
        winner = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        str(CONFIG_PATH),
    )

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Record frames
    print(f"Recording {duration}s of AI gameplay...")
    game = Game(seed=12345)
    frames = []
    max_frames = int(60 * duration)
    prev_out = 0.0

    for frame in range(max_frames):
        if not game.bird.alive:
            break

        # Create frame
        img = Image.new("RGB", (SCREEN_W, SCREEN_H))
        draw = ImageDraw.Draw(img)
        draw_frame(draw, game, frame)
        frames.append(img)

        # AI step
        state = game.bird.get_state(game.next_pipe())
        out = net.activate(state)
        flap = prev_out <= 0.5 < out[0]
        prev_out = out[0]
        game.step(flap)

        if frame % 120 == 0:
            print(f"  Frame {frame}, score={game.bird.score}")

    print(f"  Final score: {game.bird.score}, frames: {len(frames)}")

    if not frames:
        print("ERROR: no frames recorded!")
        return 1

    # Reduce frames to ~8-10 FPS for smaller GIF
    target_frames = int(duration * 8)  # ~8 fps
    step = max(1, len(frames) // target_frames if len(frames) >= target_frames else 1)
    frames = frames[::step]
    print(f"  Reduced to {len(frames)} frames for GIF")

    # Save GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(2, int(1000 * step / 60)),  # ms per frame
        loop=0,
        optimize=True,
    )
    print(f"  Saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
