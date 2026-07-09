// UI del asistente SLM. Solo presentación: muestra mensajes humanos.
// No decide objetos, slots ni intents (eso vive en el backend/guardrails).

import { getBackendHealth, getExecutionJob, sendRobotCommand } from "./utils/robotApi.js";

const el = {
  landing: document.getElementById("landing"),
  landingHint: document.getElementById("landing-hint"),
  chat: document.getElementById("chat"),
  back: document.getElementById("back-btn"),
  statusDot: document.getElementById("status-dot"),
  statusLabel: document.getElementById("status-label"),
  emptyHint: document.getElementById("empty-hint"),
  messages: document.getElementById("messages"),
  bottom: document.getElementById("bottom-anchor"),
  input: document.getElementById("input"),
  send: document.getElementById("send-btn"),
};

const state = {
  ready: false,
  backendReachable: true,
  processing: false,
  pollTimer: null,
};

// ---------- Helpers de DOM ----------
function scrollToBottom() {
  el.bottom?.scrollIntoView({ behavior: "smooth" });
}

function clearEmptyHint() {
  if (el.emptyHint) el.emptyHint.style.display = "none";
}

function addUserMessage(text) {
  clearEmptyHint();
  const row = document.createElement("div");
  row.className = "row user msg-enter";
  const bubble = document.createElement("div");
  bubble.className = "bubble-user";
  bubble.textContent = text;
  row.appendChild(bubble);
  el.messages.appendChild(row);
  scrollToBottom();
}

function addAssistantBlock() {
  const row = document.createElement("div");
  row.className = "row msg-enter";
  const block = document.createElement("div");
  block.className = "assistant";
  row.appendChild(block);
  el.messages.appendChild(row);
  scrollToBottom();
  return block;
}

function renderSteps(block, steps) {
  // Limpia pasos previos pero conserva el mensaje final si existe.
  block.querySelectorAll(".step").forEach((n) => n.remove());
  const message = block.querySelector(".assistant-message");
  steps.forEach((step) => {
    const div = document.createElement("div");
    div.className = `step ${step.state || ""}`.trim();
    const dot = document.createElement("span");
    dot.className = "dot";
    const label = document.createElement("span");
    label.textContent = step.label;
    div.appendChild(dot);
    div.appendChild(label);
    if (message) block.insertBefore(div, message);
    else block.appendChild(div);
  });
  scrollToBottom();
}

function renderTimings(block, data) {
  block.querySelectorAll(".timing-panel").forEach((n) => n.remove());
  const timings = data.timings || [];
  if (!timings.length && data.elapsed_s == null) return;
  const panel = document.createElement("div");
  panel.className = "timing-panel";
  const lines = [];
  timings.forEach((t) => {
    const status = t.ok === false ? " ✗" : t.ok === true ? " ✓" : "";
    lines.push(`${t.label}: ${t.duration_s}s${status}`);
  });
  if (data.elapsed_s != null) {
    lines.push(`Total: ${data.elapsed_s}s`);
  }
  panel.textContent = lines.join(" · ");
  block.appendChild(panel);
  scrollToBottom();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollExecutionJob(jobId, block) {
  for (;;) {
    const { httpStatus, data } = await getExecutionJob(jobId);
    if (httpStatus === 404 || !data) break;
    if (data.steps?.length) renderSteps(block, data.steps);
    if (data.status === "running") {
      await sleep(600);
      continue;
    }
    renderSteps(block, data.steps || []);
    renderTimings(block, data);
    renderFinalMessage(block, data);
    return data;
  }
  return null;
}

function renderFinalMessage(block, data) {
  const old = block.querySelector(".assistant-message");
  if (old) old.remove();
  const oldQ = block.querySelector(".clarify-question");
  if (oldQ) oldQ.remove();

  const msg = document.createElement("div");
  msg.className = "assistant-message";
  if (
    data.status === "rejected" ||
    data.status === "error" ||
    data.status === "simulation_unavailable" ||
    data.status === "execution_failed"
  ) {
    msg.classList.add("reject");
  }
  msg.textContent = data.public_message || "";
  block.appendChild(msg);

  if (data.requires_clarification && data.clarification_question) {
    const q = document.createElement("div");
    q.className = "clarify-question";
    q.textContent = data.clarification_question;
    block.appendChild(q);
  }
  scrollToBottom();
}

// Pasos "en curso" mientras esperamos al backend (sin inventar progreso real).
const PENDING_STEPS = [
  { label: "Orden recibida", state: "done" },
  { label: "Interpretando petición", state: "active" },
];

// ---------- Estado de salud / warm-up ----------
function applyHealth(health) {
  state.backendReachable = true;
  state.ready = Boolean(health.ready);

  if (state.ready) {
    el.statusDot.className = "status-dot ready";
    el.statusLabel.textContent = "Asistente listo";
    enableInput(true);
    stopPolling();
  } else if (health.status === "error") {
    el.statusDot.className = "status-dot error";
    el.statusLabel.textContent = "Error al iniciar";
    enableInput(false);
  } else {
    el.statusDot.className = "status-dot warming";
    el.statusLabel.textContent = health.message || "Iniciando asistente...";
    enableInput(false);
  }
}

function applyUnreachable() {
  state.backendReachable = false;
  state.ready = false;
  el.statusDot.className = "status-dot error";
  el.statusLabel.textContent = "Sin conexión";
  enableInput(false);
}

function enableInput(enabled) {
  const on = enabled && !state.processing;
  el.input.disabled = !on;
  el.send.disabled = !on || !el.input.value.trim();
  el.input.placeholder = enabled
    ? "Recoge la caja de galletas del primer cajón..."
    : "Iniciando asistente...";
}

async function pollHealth() {
  try {
    const health = await getBackendHealth();
    applyHealth(health);
  } catch (_e) {
    applyUnreachable();
  }
}

function startPolling() {
  pollHealth();
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(pollHealth, 1500);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

// ---------- Envío de órdenes ----------
async function handleSend() {
  const text = el.input.value.trim();
  if (!text || state.processing || !state.ready) return;

  addUserMessage(text);
  el.input.value = "";
  state.processing = true;
  enableInput(true);

  const block = addAssistantBlock();
  renderSteps(block, PENDING_STEPS);

  try {
    const { httpStatus, data } = await sendRobotCommand(text, false);

    if (httpStatus === 503) {
      renderSteps(block, [{ label: "Asistente no disponible", state: "error" }]);
      renderFinalMessage(block, {
        status: "error",
        public_message: data.message || "Iniciando asistente...",
      });
      startPolling();
      return;
    }

    if (!data || !data.public_message) {
      renderSteps(block, [{ label: "Error de interpretación", state: "error" }]);
      renderFinalMessage(block, {
        status: "error",
        public_message: "No he podido interpretar la orden correctamente.",
      });
      return;
    }

    if (data.status === "running" && data.job_id) {
      renderSteps(block, data.steps || PENDING_STEPS);
      await pollExecutionJob(data.job_id, block);
      return;
    }

    renderSteps(block, data.steps || []);
    renderTimings(block, data);
    renderFinalMessage(block, data);
  } catch (_e) {
    renderSteps(block, [{ label: "Sin conexión", state: "error" }]);
    renderFinalMessage(block, {
      status: "error",
      public_message: "No se ha podido conectar con el backend del asistente.",
    });
    applyUnreachable();
    startPolling();
  } finally {
    state.processing = false;
    enableInput(state.ready);
    el.input.focus();
  }
}

// ---------- Navegación ----------
function enterChat() {
  el.landing.classList.add("hidden");
  el.chat.classList.remove("hidden");
  startPolling();
  el.input.focus();
}

function backToLanding() {
  el.chat.classList.add("hidden");
  el.landing.classList.remove("hidden");
  stopPolling();
}

// ---------- Eventos ----------
el.landing.addEventListener("click", enterChat);
document.addEventListener("keydown", (e) => {
  if (!el.chat.classList.contains("hidden")) return;
  if (e.key === "Enter") enterChat();
});
el.back.addEventListener("click", backToLanding);
el.send.addEventListener("click", handleSend);
el.input.addEventListener("input", () => {
  el.send.disabled = !state.ready || state.processing || !el.input.value.trim();
});
el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});
