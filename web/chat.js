const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");

function addMsg(text, cls) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function renderSources(container, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  const lbl = document.createElement("span");
  lbl.className = "lbl";
  lbl.textContent = "Nguồn tham khảo:";
  box.appendChild(lbl);
  sources.forEach((s) => {
    const line = document.createElement("div");
    const label = `[${s.n}] ${s.title || s.source || ""}`;
    if (s.url) {
      const a = document.createElement("a");
      a.href = s.url; a.target = "_blank"; a.rel = "noopener";
      a.textContent = label;
      line.appendChild(a);
    } else {
      line.textContent = label;
    }
    box.appendChild(line);
  });
  container.appendChild(box);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  addMsg(q, "user");
  input.value = "";
  send.disabled = true;

  const typing = addMsg("Đang tra cứu...", "typing");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q }),
    });
    typing.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addMsg("Lỗi: " + (err.detail || res.status), "bot no_info");
      return;
    }
    const data = await res.json();
    const cls = data.kind === "emergency" ? "bot emergency"
              : data.kind === "no_info" ? "bot no_info" : "bot";
    const div = addMsg(data.answer, cls);
    renderSources(div, data.sources);
  } catch (err) {
    typing.remove();
    addMsg("Không kết nối được máy chủ: " + err.message, "bot no_info");
  } finally {
    send.disabled = false;
    input.focus();
  }
});
