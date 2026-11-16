// ─────────────────────────────────────────────────────────────
// Фронтенд RAG Assistant. Никаких фреймворков: обычный JS.
// Три задачи:
//   1) показать документы, которые уже лежат в базе  → GET /documents
//   2) загрузить новый файл                          → POST /upload
//   3) задать вопрос и получить ответ потоком        → POST /ask/stream
// ─────────────────────────────────────────────────────────────

// Короткие хелперы, чтобы не писать document.getElementById по сто раз
const $ = (id) => document.getElementById(id);
const messages = $("messages");
const docList = $("doc-list");
const docCount = $("doc-count");
const statusBox = $("upload-status");
const questionInput = $("question");
const sendBtn = $("send-btn");
const fileInput = $("file-input");
const dropzone = $("dropzone");

// ─────────────────────────────────────────────────────────────
// 1. Список документов в базе
// GET /documents → { documents: [{ source: "test.txt", chunks: 1 }, ...] }
// ─────────────────────────────────────────────────────────────
async function loadDocuments() {
  try {
    const res = await fetch("/documents");
    const data = await res.json();

    docCount.textContent = data.documents.length;
    docList.innerHTML = "";

    if (data.documents.length === 0) {
      docList.innerHTML = '<li class="text-xs text-slate-500">Пока пусто — загрузите документ</li>';
      return;
    }

    // Создаём элементы через DOM, а не innerHTML с данными:
    // имя файла может содержать спецсимволы — так безопаснее
    for (const doc of data.documents) {
      const li = document.createElement("li");
      li.className = "flex items-center justify-between gap-2 rounded-lg bg-slate-950/50 px-3 py-2";
      li.innerHTML = `
        <span class="truncate text-slate-300" title="${doc.source}">📄 ${doc.source}</span>
        <span class="shrink-0 rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-400">${doc.chunks}</span>
      `;
      docList.appendChild(li);
    }
  } catch (err) {
    docList.innerHTML = '<li class="text-xs text-red-400">Не удалось загрузить список документов</li>';
    console.error(err);
  }
}

// ─────────────────────────────────────────────────────────────
// 2. Загрузка файла
// POST /upload с multipart/form-data → сервер режет файл на чанки
// и добавляет векторы в Chroma, затем возвращает { filename, chunks }
// ─────────────────────────────────────────────────────────────
async function uploadFile(file) {
  if (!file) return;
  const allowed = file.name.endsWith(".pdf") || file.name.endsWith(".txt");
  if (!allowed) {
    showStatus("Поддерживаются только PDF и TXT", "error");
    return;
  }

  showStatus(`Загружаю «${file.name}»…`, "info");

  try {
    const form = new FormData();
    form.append("file", file);           // имя поля должно совпадать с параметром в FastAPI

    const res = await fetch("/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    showStatus(`«${data.filename}» добавлен: ${data.chunks} чанк(ов)`, "success");
    loadDocuments();                     // обновляем левую панель
  } catch (err) {
    showStatus("Ошибка загрузки. Проверьте сервер и формат файла.", "error");
    console.error(err);
  }
}

function showStatus(text, kind) {
  const styles = {
    info: "bg-slate-800 text-slate-300",
    success: "bg-emerald-500/15 text-emerald-300",
    error: "bg-red-500/15 text-red-300",
  };
  statusBox.textContent = text;
  statusBox.className = `mt-3 rounded-lg px-3 py-2 text-xs ${styles[kind]}`;
}

// Клик по зоне → открыть системный выбор файла
fileInput.addEventListener("change", () => {
  uploadFile(fileInput.files[0]);
  fileInput.value = "";                  // чтобы можно было загрузить тот же файл снова
});

// Drag & drop
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("border-accent", "bg-accent/10");
});
dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("border-accent", "bg-accent/10");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("border-accent", "bg-accent/10");
  uploadFile(e.dataTransfer.files[0]);
});

// ─────────────────────────────────────────────────────────────
// 3. Чат со стримингом
// POST /ask/stream отдаёт Server-Sent Events (SSE), каждый — строка:
//   data: {"type":"sources","sources":[...]}
//   data: {"type":"token","text":"..."}      ← приходит по кусочку
//   data: {"type":"done"}
//   data: {"type":"error","message":"..."}
// Читаем поток вручную через fetch + ReadableStream, без библиотек.
// ─────────────────────────────────────────────────────────────

// Добавить пузырь в чат и вернуть его тело (куда можно дописывать текст)
function addBubble(role) {
  const isUser = role === "user";
  const wrap = document.createElement("div");
  wrap.className = isUser ? "flex justify-end" : "flex justify-start";

  const bubble = document.createElement("div");
  bubble.className = isUser
    ? "max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-sm bg-accent px-4 py-3 text-sm leading-relaxed text-white"
    : "max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tl-sm bg-slate-800/80 px-4 py-3 text-sm leading-relaxed";
  wrap.appendChild(bubble);

  // Контейнер под будущие чипы источников
  if (!isUser) {
    const sources = document.createElement("div");
    sources.className = "mt-2 flex flex-wrap gap-1.5";
    wrap.appendChild(sources);
  }

  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

// Чипы «источник · chunk N» под ответом
function renderSources(wrapSources, list) {
  wrapSources.innerHTML = "";
  for (const s of list) {
    const chip = document.createElement("span");
    chip.className = "rounded-full border border-slate-700 bg-slate-900 px-2.5 py-1 text-[11px] text-slate-400";
    chip.textContent = `${s.source} · chunk ${s.chunk}`;
    wrapSources.appendChild(chip);
  }
}

async function askQuestion(question) {
  addBubble("user").textContent = question;

  const answerBubble = addBubble("assistant");
  answerBubble.classList.add("typing-cursor");
  answerBubble.textContent = "";

  const sourcesBox = answerBubble.parentElement.querySelector("div.mt-2");

  sendBtn.disabled = true;

  try {
    const res = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    // Читаем поток кусками. События разделены "\n\n",
    // поэтому хвост сохраняем в buffer до следующего чтения.
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();              // последний кусок может быть неполным

      for (const raw of events) {
        if (!raw.startsWith("data: ")) continue;
        const payload = JSON.parse(raw.slice(6));   // убираем префикс "data: "

        if (payload.type === "sources") {
          renderSources(sourcesBox, payload.sources);
        } else if (payload.type === "token") {
          answerBubble.textContent += payload.text;
          messages.scrollTop = messages.scrollHeight;
        } else if (payload.type === "error") {
          answerBubble.textContent = `Ошибка: ${payload.message}`;
        }
      }
    }
  } catch (err) {
    answerBubble.textContent = "Не удалось получить ответ. Проверьте соединение и ключ API.";
    console.error(err);
  } finally {
    answerBubble.classList.remove("typing-cursor");
    sendBtn.disabled = false;
    messages.scrollTop = messages.scrollHeight;
  }
}

// Отправка формы: Enter — отправить, Shift+Enter — перенос строки
$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  questionInput.style.height = "auto";
  askQuestion(question);
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});

// Автовысота textarea при вводе
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
});

// ── Старт: подгружаем список документов при открытии страницы ──
loadDocuments();
