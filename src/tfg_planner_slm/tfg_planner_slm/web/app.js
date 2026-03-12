const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const objectsListEl = document.getElementById("objects-list");
const statusTextEl = document.getElementById("status-text");

let lastSignature = "";

function renderMessages(messages) {
  const signature = JSON.stringify(messages);
  if (signature === lastSignature) {
    return;
  }

  lastSignature = signature;
  messagesEl.innerHTML = "";

  messages.forEach((entry) => {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${entry.role === "user" ? "user" : "assistant"}`;

    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = entry.role === "user" ? "Tú" : "SLM";

    const text = document.createElement("div");
    text.className = "message-text";
    text.textContent = entry.message || "";

    wrapper.appendChild(role);
    wrapper.appendChild(text);

    if (entry.plan_preview) {
      const preview = document.createElement("pre");
      preview.className = "plan-preview";
      preview.textContent = JSON.stringify(entry.plan_preview, null, 2);
      wrapper.appendChild(preview);
    }

    messagesEl.appendChild(wrapper);
  });

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderVisibleObjects(objects) {
  objectsListEl.innerHTML = "";

  if (!objects.length) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = "No hay objetos visibles.";
    objectsListEl.appendChild(emptyItem);
    return;
  }

  objects.forEach((objectName) => {
    const item = document.createElement("li");
    item.textContent = objectName;
    objectsListEl.appendChild(item);
  });
}

function renderStatus(status) {
  statusTextEl.textContent = status?.message || "Sin estado disponible.";
}

async function refreshState() {
  try {
    const response = await fetch("/api/state");
    const data = await response.json();
    renderMessages(data.messages || []);
    renderVisibleObjects(data.visible_objects || []);
    renderStatus(data.executor_status || {});
  } catch (error) {
    statusTextEl.textContent = "No se pudo conectar con el backend del chat.";
  }
}

async function sendMessage(message) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error("No se pudo enviar el mensaje.");
  }
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) {
    return;
  }

  inputEl.disabled = true;
  try {
    await sendMessage(message);
    inputEl.value = "";
    await refreshState();
  } catch (error) {
    statusTextEl.textContent = "Error enviando el mensaje al chat.";
  } finally {
    inputEl.disabled = false;
    inputEl.focus();
  }
});

refreshState();
window.setInterval(refreshState, 1000);
