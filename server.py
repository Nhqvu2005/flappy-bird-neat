"""
Headless game playable in browser (Flappy Bird in vanilla JS).
Uses the winner genome shipped under logs/winner.json for autopilot mode.

Endpoints:
    GET /                -> index.html
    GET /api/log         -> list of {gen, best_fitness, mean_fitness, best_score, species}
    GET /api/winner      -> {nodes, connections, fitness}
    GET /api/log         -> training.jsonl as JSON array
"""
import json
import http.server
import socketserver
import sys
from pathlib import Path

from game import Game
import neat


BASE_DIR   = Path(__file__).parent.resolve()
WEB_DIR    = BASE_DIR / "web"
LOG_FILE   = BASE_DIR / "logs" / "training.jsonl"
WINNER_PKL = BASE_DIR / "logs" / "winner.pkl"
WINNER_JSON = BASE_DIR / "logs" / "winner_net.json"
CONFIG_PATH = BASE_DIR / "config-feedforward.txt"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/log":
            data = []
            if LOG_FILE.exists():
                for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        pass
            return self._json(data)

        if self.path == "/api/winner":
            # Build winner_net.json from .pkl if missing
            if not WINNER_JSON.exists() and WINNER_PKL.exists():
                self._export_winner()
            if WINNER_JSON.exists():
                return self._json(json.loads(WINNER_JSON.read_text(encoding="utf-8")))
            return self._json({"error": "no winner yet - run train.py first"}, status=404)

        return super().do_GET()

    def _export_winner(self):
        import pickle
        with WINNER_PKL.open("rb") as f:
            winner = pickle.load(f)
        nodes = []
        for nid in winner.nodes:
            if nid < 0:
                t = "input"
            elif nid == 0:
                t = "output"
            else:
                t = "hidden"
            nodes.append({"id": nid, "type": t})
        conns = []
        for cg in winner.connections.values():
            if cg.enabled:
                conns.append({"in": cg.key[0], "out": cg.key[1], "weight": round(cg.weight, 3)})
        WINNER_JSON.write_text(json.dumps(
            {"nodes": nodes, "connections": conns, "fitness": winner.fitness},
            indent=2), encoding="utf-8")

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quieter logging
        sys.stderr.write("[web] " + fmt % args + "\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"FlappyAI dashboard: http://127.0.0.1:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Bye.")


if __name__ == "__main__":
    main()