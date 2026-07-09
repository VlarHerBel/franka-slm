// Capa API limpia para hablar con el backend HTTP del SLM.
// No contiene lógica robótica: no decide objetos, slots ni intents.
//
// URL del backend:
//  - Por defecto usa el mismo origen ('') porque el backend sirve la UI.
//  - Se puede sobreescribir con window.ROBOT_BACKEND_URL (config.js opcional)
//    o, en un futuro build Vite, con import.meta.env.VITE_ROBOT_BACKEND_URL.

function resolveBackendUrl() {
  if (typeof window !== "undefined" && window.ROBOT_BACKEND_URL) {
    return String(window.ROBOT_BACKEND_URL).replace(/\/$/, "");
  }
  try {
    if (import.meta && import.meta.env && import.meta.env.VITE_ROBOT_BACKEND_URL) {
      return String(import.meta.env.VITE_ROBOT_BACKEND_URL).replace(/\/$/, "");
    }
  } catch (_e) {
    // import.meta no disponible fuera de un bundler: usar mismo origen.
  }
  return "";
}

const BACKEND_URL = resolveBackendUrl();

export async function getBackendHealth() {
  const res = await fetch(`${BACKEND_URL}/api/health`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!res.ok && res.status !== 503) {
    throw new Error(`health_http_${res.status}`);
  }
  return res.json();
}

export async function sendRobotCommand(text, execute = false) {
  const res = await fetch(`${BACKEND_URL}/api/robot/command`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ text, execute: Boolean(execute) }),
  });
  const data = await res.json().catch(() => ({}));
  return { httpStatus: res.status, data };
}

export async function getExecutionJob(jobId) {
  const res = await fetch(`${BACKEND_URL}/api/robot/job/${encodeURIComponent(jobId)}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  const data = await res.json().catch(() => ({}));
  return { httpStatus: res.status, data };
}

export { BACKEND_URL };
