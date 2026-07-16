const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const convoList = document.getElementById("convo-list");
const sidebar = document.getElementById("sidebar");
const menuToggle = document.getElementById("menu-toggle");

const MAX_HISTORY = 6;

// --- Nhiều hội thoại (trong bộ nhớ phiên — tắt trang là mất) ---
// Mỗi cuộc: { id, title, msgs[], history[] }
//   msgs: các tin để render lại khi chuyển cuộc {role:'user'|'bot', text, cls, sources, fb}
//   history: gửi kèm /chat (server stateless) — lượt user/assistant đã cắt gọn
let convos = [];
let activeId = null;
let seq = 0;

function activeConvo() {
  return convos.find((c) => c.id === activeId);
}

function newConvo() {
  // Nếu đang ở một cuộc TRỐNG (chưa nhắn gì) -> không tạo thêm cuộc trống trùng lặp,
  // chỉ dùng lại cuộc trống hiện tại (giống ChatGPT).
  const cur = activeConvo();
  if (cur && cur.msgs.length === 0) {
    input.focus();
    sidebar.classList.remove("open");
    return cur;
  }
  const c = { id: ++seq, title: "Hội thoại mới", msgs: [], history: [] };
  convos.unshift(c);
  activeId = c.id;
  renderConvoList();
  renderChat();
  input.focus();
  sidebar.classList.remove("open");
  return c;
}

function switchConvo(id) {
  if (id === activeId) { sidebar.classList.remove("open"); return; }
  // Dọn cuộc TRỐNG đang mở khi rời sang cuộc khác (không để cuộc trống đọng lại).
  const cur = activeConvo();
  if (cur && cur.msgs.length === 0) {
    convos = convos.filter((c) => c.id !== cur.id);
  }
  activeId = id;
  renderConvoList();
  renderChat();
  sidebar.classList.remove("open");   // đóng sidebar trên mobile sau khi chọn
}

function renderConvoList() {
  convoList.innerHTML = "";
  convos.forEach((c) => {
    const item = document.createElement("div");
    item.className = "convo-item" + (c.id === activeId ? " active" : "");
    item.textContent = c.title;
    item.title = c.title;
    item.onclick = () => switchConvo(c.id);
    convoList.appendChild(item);
  });
}

// Vẽ lại toàn bộ tin của cuộc đang mở (khi chuyển cuộc / khởi động)
function renderChat() {
  chat.innerHTML = "";
  const c = activeConvo();
  if (!c) return;
  c.msgs.forEach((m) => {
    const div = addMsg(m.text, m.cls);
    if (m.sources) renderSources(div, m.sources);
    if (m.fb) renderFeedback(div, m.fb.traceId, m.fb.query);
  });
}

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

newChatBtnInit();
function newChatBtnInit() {
  document.getElementById("new-chat").onclick = () => newConvo();
  if (menuToggle) menuToggle.onclick = () => sidebar.classList.toggle("open");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  const c = activeConvo() || newConvo();

  // Tin của người dùng
  addMsg(q, "user");
  c.msgs.push({ role: "user", text: q, cls: "user" });
  // Đặt tiêu đề cuộc theo câu hỏi đầu tiên
  if (c.msgs.filter((m) => m.role === "user").length === 1) {
    c.title = q.length > 34 ? q.slice(0, 34) + "…" : q;
    renderConvoList();
  }
  input.value = "";
  send.disabled = true;

  const typing = addMsg("Đang tra cứu...", "typing");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q, history: c.history }),
    });
    typing.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const div = addMsg("Lỗi: " + (err.detail || res.status), "bot no_info");
      c.msgs.push({ role: "bot", text: div.textContent, cls: "bot no_info" });
      return;
    }
    const data = await res.json();
    const cls = data.kind === "emergency" ? "bot emergency"
              : data.kind === "clarify" ? "bot clarify"
              : (data.kind === "no_info" || data.kind === "refuse"
                 || data.kind === "degraded") ? "bot no_info" : "bot";
    const div = addMsg(data.answer, cls);
    renderSources(div, data.sources);
    const showFb = data.kind === "normal";
    if (showFb) renderFeedback(div, data.trace_id, data.query);

    // Lưu tin bot vào cuộc (để render lại khi chuyển cuộc)
    c.msgs.push({
      role: "bot", text: data.answer, cls,
      sources: data.sources,
      fb: showFb ? { traceId: data.trace_id, query: data.query } : null,
    });

    // Lưu lượt vào history của CUỘC NÀY (ngữ cảnh gửi backend lượt sau).
    // KHÔNG lưu emergency (luật cứng). clarify: gắn lại marker để server đếm số lần hỏi.
    if (data.kind !== "emergency") {
      const stored = data.kind === "clarify" ? "[HỎI LẠI] " + data.answer : data.answer;
      c.history.push({ role: "user", content: q });
      c.history.push({ role: "assistant", content: stored });
      if (c.history.length > MAX_HISTORY * 2) c.history = c.history.slice(-MAX_HISTORY * 2);
    }
  } catch (err) {
    typing.remove();
    addMsg("Không kết nối được máy chủ: " + err.message, "bot no_info");
  } finally {
    send.disabled = false;
    input.focus();
  }
});

// Khởi động: tạo sẵn 1 hội thoại trống
newConvo();
