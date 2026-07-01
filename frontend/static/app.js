// Minimal vanilla-JS frontend for the RAG Knowledge Assistant.
// Talks to the same FastAPI backend that serves this page.

const $ = (id) => document.getElementById(id);
const apiKey = () => $("apikey").value.trim();

function headers() {
  return { "Content-Type": "application/json", "x-api-key": apiKey() };
}

async function refreshStatus() {
  try {
    const res = await fetch("/health");
    const h = await res.json();
    $("pill-llm").textContent = "llm: " + h.providers.llm;
    $("pill-emb").textContent = "emb: " + h.providers.embeddings;
    $("pill-store").textContent = "store: " + h.providers.vector_store;
    $("pill-count").textContent = h.vectors_in_store + " vectors";
  } catch {
    $("pill-count").textContent = "offline";
  }
}

async function ingest() {
  const text = $("doctext").value.trim();
  const source = $("source").value.trim() || "pasted-note.md";
  const out = $("ingest-result");
  if (!text) {
    out.className = "result-line err";
    out.textContent = "Please paste some document text first.";
    return;
  }
  const btn = $("ingest-btn");
  btn.disabled = true;
  out.className = "result-line";
  out.textContent = "Ingesting…";
  try {
    const res = await fetch("/ingest", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ documents: [{ text, source }] }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Ingest failed");
    out.className = "result-line ok";
    out.textContent =
      `✓ Ingested "${source}" — ${data.total_chunks} chunk(s). ` +
      `Store now holds ${data.vectors_in_store} vector(s).`;
    $("doctext").value = "";
    refreshStatus();
  } catch (e) {
    out.className = "result-line err";
    out.textContent = "✗ " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// Turn [1], [2] markers in the answer into highlighted spans with source tooltips.
function renderAnswer(answer, citations) {
  const bySource = {};
  citations.forEach((c) => (bySource[c.marker] = c.source));
  const escaped = answer.replace(/[&<>]/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch])
  );
  return escaped.replace(/\[(\d+)\]/g, (m) => {
    const src = bySource[m] || "";
    return `<span class="marker" title="${src}">${m}</span>`;
  });
}

async function ask() {
  const question = $("question").value.trim();
  const top_k = parseInt($("topk").value, 10) || 3;
  const wrap = $("answer-wrap");
  const err = $("query-error");
  err.className = "error hidden";
  if (question.length < 3) {
    err.className = "error";
    err.textContent = "Question must be at least 3 characters.";
    return;
  }
  const btn = $("ask-btn");
  btn.disabled = true;
  btn.textContent = "Thinking…";
  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ question, top_k }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Query failed");

    $("answer").innerHTML = renderAnswer(data.answer, data.citations);
    $("citations").innerHTML = data.citations
      .map(
        (c) => `
      <div class="citation">
        <div class="cite-head">
          <span class="marker">${c.marker}</span>
          <span class="src">${c.source}</span>
          <span class="score">score ${c.score.toFixed(3)}</span>
        </div>
        <div class="snippet">${c.snippet.replace(/[<>]/g, "")}</div>
      </div>`
      )
      .join("");
    wrap.classList.remove("hidden");
  } catch (e) {
    err.className = "error";
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

$("ingest-btn").addEventListener("click", ingest);
$("ask-btn").addEventListener("click", ask);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
});

refreshStatus();
