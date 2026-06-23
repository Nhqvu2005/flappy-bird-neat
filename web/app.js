/* FlappyAI dashboard - ties game + Chart.js + NN visualization together. */

const FLAP_KEYS = new Set(["Space", "ArrowUp"]);

let game = new Flappy.Game();
let mode = "manual";                 // "manual" | "ai"
let bestScore = parseInt(localStorage.getItem("flappyai_best") || "0");
let winnerNet = null;                // loaded from /api/winner
let resetTimer = null;

const canvas = document.getElementById("game");
const ctx    = canvas.getContext("2d");
const scoreEl = document.getElementById("score");
const bestEl  = document.getElementById("best");
const modeEl  = document.getElementById("mode-label");
bestEl.textContent = bestScore;

// ----- Input -----
window.addEventListener("keydown", (e) => {
  if (FLAP_KEYS.has(e.code)) {
    e.preventDefault();
    if (mode === "manual") userFlap();
  }
  if (e.code === "KeyA") { mode = "ai";     modeEl.textContent = "Mode: AI"; }
  if (e.code === "KeyM") { mode = "manual"; modeEl.textContent = "Mode: Manual"; }
});
canvas.addEventListener("mousedown", () => { if (mode === "manual") userFlap(); });
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  if (mode === "manual") userFlap();
}, { passive: false });

function userFlap() { game.step(true); render(performance.now()); }

// ----- Feed-forward inference (same as neat-python tanh) -----
function activate(net, inputs) {
  // Returns output array. net: {nodes:[{id,type,act}], connections:[{in,out,weight}]}
  const values = {};
  const inputIds = net.input_ids || net.nodes
    .filter(n => n.type === "input")
    .map(n => n.id)
    .sort((a, b) => a - b);
  const inputMap = {};
  inputIds.forEach((id, index) => {
    inputMap[id] = inputs[index] ?? 0;
  });

  // Order: inputs first, then by id (topological-ish)
  const sorted = [...net.nodes].sort((a, b) => {
    const oa = a.type === "input" ? 0 : a.type === "hidden" ? 1 : 2;
    const ob = b.type === "input" ? 0 : b.type === "hidden" ? 1 : 2;
    return oa - ob || a.id - b.id;
  });
  for (const n of sorted) {
    if (n.type === "input") {
      values[n.id] = inputMap[n.id] ?? 0;
    } else {
      let s = 0;
      for (const c of net.connections) {
        if (c.out === n.id) s += (values[c.in] ?? 0) * c.weight;
      }
      const bias = n.bias ?? 0;
      const response = n.response ?? 1;
      values[n.id] = Math.tanh(bias + response * s);   // matches neat-python tanh
    }
  }
  const outputIds = net.output_ids || net.nodes
    .filter(n => n.type === "output")
    .map(n => n.id)
    .sort((a, b) => a - b);
  return outputIds.map(id => values[id] ?? 0);
}

// ----- Render -----
function drawPipes() {
  ctx.fillStyle = "#3cb43c";
  ctx.strokeStyle = "#288228";
  ctx.lineWidth = 2;
  for (const p of game.pipes) {
    const top = p.gapY - Flappy.PIPE_GAP / 2;
    const bot = p.gapY + Flappy.PIPE_GAP / 2;
    // Top pipe
    ctx.fillRect(p.x, 0, Flappy.PIPE_W, top);
    ctx.strokeRect(p.x, 0, Flappy.PIPE_W, top);
    // Cap on top pipe
    ctx.fillStyle = "#5ed15e";
    ctx.fillRect(p.x - 3, top - 8, Flappy.PIPE_W + 6, 8);
    ctx.strokeRect(p.x - 3, top - 8, Flappy.PIPE_W + 6, 8);
    ctx.fillStyle = "#3cb43c";
    // Bottom pipe
    ctx.fillRect(p.x, bot, Flappy.PIPE_W, Flappy.PLAY_H - bot);
    ctx.strokeRect(p.x, bot, Flappy.PIPE_W, Flappy.PLAY_H - bot);
    // Cap on bottom pipe
    ctx.fillStyle = "#5ed15e";
    ctx.fillRect(p.x - 3, bot, Flappy.PIPE_W + 6, 8);
    ctx.strokeRect(p.x - 3, bot, Flappy.PIPE_W + 6, 8);
    ctx.fillStyle = "#3cb43c";
  }
}
function drawGround(t) {
  // t: time in ms for animated stripes
  ctx.fillStyle = "#deb887";
  ctx.fillRect(0, Flappy.PLAY_H, Flappy.SCREEN_W, Flappy.GROUND_H);
  ctx.fillStyle = "#c19a6b";
  const stripeY = Flappy.PLAY_H + 18;
  const offset = ((t || 0) * 0.05) % 24;
  for (let x = -24; x < Flappy.SCREEN_W; x += 24) {
    ctx.fillRect(x - offset, stripeY, 12, 4);
  }
  ctx.fillStyle = "#000";
  ctx.fillRect(0, Flappy.PLAY_H, Flappy.SCREEN_W, 2);
}
function drawBird(t, prevY) {
  const x = game.bird.x, y = game.bird.y;
  // Smooth rendered y toward physics y for a less jittery look
  const renderY = prevY == null ? y : prevY + (y - prevY) * 0.6;

  // Tilt based on velocity (clamped), nose-up when climbing, nose-down when falling
  const v = game.bird.vy;
  const tilt = Math.max(-0.5, Math.min(1.2, v * 0.08));

  ctx.save();
  ctx.translate(x, renderY);
  ctx.rotate(tilt);

  // Body
  ctx.fillStyle = "#ffdc00";
  ctx.beginPath(); ctx.ellipse(0, 0, 17, 12, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#b48f00";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Wing flap (subtle two-frame animation)
  const flapPhase = Math.floor(((t || 0) / 90) % 2);
  ctx.fillStyle = "#f1c40f";
  if (flapPhase === 0) {
    ctx.beginPath();
    ctx.ellipse(-4, 4, 8, 4, 0, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.ellipse(-2, 6, 6, 3, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // Eye
  ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.arc(6, -3, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#000";
  ctx.beginPath(); ctx.arc(7.5, -3, 2, 0, Math.PI * 2); ctx.fill();

  // Beak
  ctx.fillStyle = "#ff8c00";
  ctx.beginPath();
  ctx.moveTo(12, -1); ctx.lineTo(22, 1); ctx.lineTo(12, 5); ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "#a85f00";
  ctx.stroke();

  ctx.restore();
}
let _prevBirdY = null;
function render(t) {
  ctx.fillStyle = "#87ceeb";
  ctx.fillRect(0, 0, Flappy.SCREEN_W, Flappy.SCREEN_H);
  drawPipes();
  drawGround(t);
  drawBird(t, _prevBirdY);
  _prevBirdY = _prevBirdY == null ? game.bird.y : _prevBirdY + (game.bird.y - _prevBirdY) * 0.6;
  scoreEl.textContent = game.bird.score;
  if (game.bird.score > bestScore) {
    bestScore = game.bird.score;
    localStorage.setItem("flappyai_best", bestScore);
    bestEl.textContent = bestScore;
  }
}

// ----- Main loop -----
let _prevAIOutput = 0.0;
function loop(t) {
  if (game.bird.alive) {
    if (mode === "ai" && winnerNet) {
      const out = activate(winnerNet, game.getState());
      const flap = _prevAIOutput <= 0.5 && out[0] > 0.5;
      _prevAIOutput = out[0];
      game.step(flap);
    } else if (mode === "manual") {
      game.step(false);   // manual only advances via input; gravity handled in step(false)
    }
  } else {
    // freeze smoothed y on death so the bird stays where it died
    if (_prevBirdY == null) _prevBirdY = game.bird.y;
    if (!resetTimer) {
      resetTimer = setTimeout(() => { game.reset(); _prevBirdY = null; resetTimer = null; }, 700);
    }
  }
  render(t);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ----- Fetch training log + winner net -----
async function loadDashboard() {
  try {
    const [log, win] = await Promise.all([
      fetch("/api/log").then(r => r.json()),
      fetch("/api/winner").then(r => r.json()),
    ]);
    drawFitnessChart(log);
    if (!win.error) {
      winnerNet = win;
      document.getElementById("winner-info").textContent =
        `Fitness: ${win.fitness.toFixed(1)}  •  Nodes: ${win.nodes.length}  •  Active connections: ${win.connections.length}`;
      drawNN(win);
    } else {
      document.getElementById("winner-info").textContent = win.error;
    }
  } catch (e) {
    document.getElementById("caption").textContent = "Could not reach /api/ - run train.py and start server.py first.";
  }
}

function drawFitnessChart(records) {
  if (!records.length) {
    document.getElementById("caption").textContent =
      "No training log yet. Run `python train.py` to generate one.";
    return;
  }
  const labels = records.map(r => r.gen);
  const bestF  = records.map(r => r.best_fitness);
  const meanF  = records.map(r => r.mean_fitness);
  const score  = records.map(r => r.best_score);

  new Chart(document.getElementById("fitnessChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Best fitness",  data: bestF, borderColor: "#22c55e", backgroundColor: "rgba(34,197,94,.15)", tension: .2 },
        { label: "Mean fitness",  data: meanF, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,.10)", tension: .2 },
        { label: "Best score",    data: score, borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,.10)", tension: .2, yAxisID: "y1" },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Generation" } },
        y: { title: { display: true, text: "Fitness" }, beginAtZero: true },
        y1: { title: { display: true, text: "Pipes passed" }, position: "right", grid: { drawOnChartArea: false }, beginAtZero: true },
      },
    },
  });
  document.getElementById("caption").textContent =
    `${records.length} generations logged. Latest: gen ${records[records.length-1].gen}.`;
}

function drawNN(net) {
  const svg = document.getElementById("nn");
  svg.innerHTML = "";
  const W = 700, H = 420;
  const PAD = 40;

  // In neat-python, input nodes have negative IDs, output = 0, hidden = positive.
  const inputs  = net.nodes.filter(n => n.id < 0).sort((a, b) => a.id - b.id);
  const outputs = net.nodes.filter(n => n.id === 0);
  const hiddens = net.nodes.filter(n => n.id > 0);

  // Layout columns
  const colX = [PAD, W/2, W - PAD];
  const posOf = {};
  const setCol = (arr, col, label) => {
    arr.forEach((n, i) => {
      const y = H/2 + (i - (arr.length - 1)/2) * Math.min(70, 300 / Math.max(arr.length, 1));
      posOf[n.id] = { x: colX[col], y, type: n.type, label };
    });
  };
  setCol(inputs,  0, "input");
  setCol(outputs, 2, "output");
  setCol(hiddens, 1, "hidden");

  // Connections
  for (const c of net.connections) {
    const a = posOf[c.in], b = posOf[c.out];
    if (!a || !b) continue;
    const stroke = c.weight > 0 ? "#16a34a" : "#2563eb";
    const w = Math.max(1, Math.min(5, Math.abs(c.weight)));
    svg.innerHTML += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"
                          stroke="${stroke}" stroke-width="${w}" opacity="0.55"/>`;
  }
  // Nodes
  for (const id in posOf) {
    const p = posOf[id];
    const fill = p.type === "input" ? "#dcfce7" : p.type === "output" ? "#fee2e2" : "#dbeafe";
    svg.innerHTML += `<circle cx="${p.x}" cy="${p.y}" r="14" fill="${fill}" stroke="#000" stroke-width="2"/>`;
    if (p.type === "input") {
      const labels = ["bird y", "velocity", "pipe dx", "top y", "bot y"];
      const idx = inputs.findIndex(n => n.id === parseInt(id));
      svg.innerHTML += `<text x="${p.x - 24}" y="${p.y + 5}" text-anchor="end" font-size="13">${labels[idx] || "in"}</text>`;
    }
    if (p.type === "output") {
      svg.innerHTML += `<text x="${p.x + 24}" y="${p.y + 5}" font-size="13">flap</text>`;
    }
  }
}

loadDashboard();