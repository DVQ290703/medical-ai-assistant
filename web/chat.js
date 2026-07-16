const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");

// Lịch sử hội thoại (client giữ — server stateless). Gửi kèm mỗi /chat.
// Chỉ giữ vài lượt gần nhất để prompt không phình (khớp MAX_HISTORY_TURNS ở backend).
let history = [];
const MAX_HISTORY = 6;

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

function renderFeedback(container, traceId, query) {
  const box = document.createElement("div");
  box.className = "feedback";
  const ask = document.createElement("span");
  ask.textContent = "Câu trả lời có hữu ích? ";
  box.appendChild(ask);

  ["up", "down"].forEach((rating) => {
    const btn = document.createElement("button");
    btn.className = "fb-btn";
    btn.textContent = rating === "up" ? "👍" : "👎";
    btn.onclick = async () => {
      box.querySelectorAll(".fb-btn").forEach((b) => (b.disabled = true));
      try {
        await fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trace_id: traceId, query, rating }),
        });
        ask.textContent = "Cảm ơn phản hồi của bạn! ";
        btn.classList.add("chosen");
      } catch (_) {
        ask.textContent = "Không gửi được phản hồi. ";
      }
    };
    box.appendChild(btn);
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
      body: JSON.stringify({ message: q, history }),
    });
    typing.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addMsg("Lỗi: " + (err.detail || res.status), "bot no_info");
      return;
    }
    const data = await res.json();
    const cls = data.kind === "emergency" ? "bot emergency"
              : data.kind === "clarify" ? "bot clarify"
              : (data.kind === "no_info" || data.kind === "refuse"
                 || data.kind === "degraded") ? "bot no_info" : "bot";
    const div = addMsg(data.answer, cls);
    renderSources(div, data.sources);
    if (data.kind === "normal") renderFeedback(div, data.trace_id, data.query);

    // Lưu lượt vào lịch sử để lượt sau có ngữ cảnh (kể cả câu hỏi lại [clarify]).
    // KHÔNG lưu emergency (luật cứng, không phải mạch hội thoại thường).
    // Với clarify: gắn lại marker [HỎI LẠI] vào bản LƯU (không hiển thị) để server đếm
    // được số lần đã hỏi -> lưới an toàn không hỏi vô tận.
    if (data.kind !== "emergency") {
      const stored = data.kind === "clarify" ? "[HỎI LẠI] " + data.answer : data.answer;
      history.push({ role: "user", content: q });
      history.push({ role: "assistant", content: stored });
      if (history.length > MAX_HISTORY * 2) history = history.slice(-MAX_HISTORY * 2);
    }
  } catch (err) {
    typing.remove();
    addMsg("Không kết nối được máy chủ: " + err.message, "bot no_info");
  } finally {
    send.disabled = false;
    input.focus();
  }
});
