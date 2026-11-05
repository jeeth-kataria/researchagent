"use strict";

// All paper/model text enters the document through textContent, never innerHTML.
const $ = (id) => document.getElementById(id);
const roles = {
  investigator: "Paper investigator",
  methods: "Methods analyst",
  reviewer: "Critical reviewer",
  writer: "Report writer",
};
const storageKey = "research-desk.run.v1";
let run = null;
let running = false;
let selectedFile = null;
let selectedEvidence = null;
let view = "report";
let config = { ready: false, model: "openai/gpt-oss-20b", max_upload_mb: 4 };

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}
function notice(message) {
  $("service-message").hidden = !message;
  $("service-message").textContent = message || "";
}
function formError(message) {
  $("form-error").hidden = !message;
  $("form-error").textContent = message || "";
}
function save() {
  if (!run) return;
  try {
    sessionStorage.setItem(storageKey, JSON.stringify(run));
  } catch (_) {
    /* Storage may be disabled/full; the current in-memory result still works. */
  }
}
function setView(name, focus = false) {
  if (!run) {
    $("paper").focus();
    return;
  }
  view = name;
  document.querySelectorAll(".tabs [data-view]").forEach((button) => {
    const active = button.dataset.view === name;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    $(`panel-${button.dataset.view}`).hidden = !active;
  });
  document
    .querySelectorAll(".side-link")
    .forEach((button) =>
      button.classList.toggle("active", button.dataset.view === name),
    );
  if (focus) $(`tab-${name}`).focus();
}
function citation(evidenceId) {
  const entry = run.evidence[evidenceId];
  if (!entry) return element("span", "small muted", "Source unavailable");
  const button = element(
    "button",
    "citation",
    `${evidenceId} · PDF page ${entry.page}`,
  );
  button.type = "button";
  button.setAttribute(
    "aria-label",
    `Read source ${evidenceId} on PDF page ${entry.page}`,
  );
  button.addEventListener("click", () => {
    selectedEvidence = evidenceId;
    renderSource();
    if (matchMedia("(max-width: 650px)").matches)
      $("source-notes").scrollIntoView({ block: "start" });
    $("source-notes").focus({ preventScroll: true });
  });
  return button;
}
function findingNode(finding) {
  const block = element("div", "finding");
  block.append(element("p", "", finding.statement));
  const meta = element("div", "finding-meta");
  meta.append(element("span", "kind-label", finding.kind.replaceAll("_", " ")));
  (finding.evidence_ids || []).forEach((id) => meta.append(citation(id)));
  block.append(meta);
  return block;
}
function section(title, number) {
  const block = element("section", "report-section");
  const heading = element("div", "section-heading");
  heading.append(
    element("span", "section-number", number),
    element("h2", "", title),
  );
  block.append(heading);
  return block;
}
function listSection(title, number, values) {
  const block = section(title, number);
  const list = element("ul", "report-list");
  values.forEach((value) => list.append(element("li", "", value)));
  block.append(list);
  return block;
}
function renderReport() {
  const host = $("report-content");
  host.replaceChildren();
  const state = run.state;
  if (state.status !== "complete") {
    const empty = element("div", "empty-state");
    empty.append(
      element(
        "h2",
        "",
        state.status === "incomplete"
          ? "The investigation paused."
          : "The paper is being examined.",
      ),
    );
    empty.append(
      element(
        "p",
        "",
        state.status === "incomplete"
          ? state.error
          : "Findings and source passages will appear as the team works. The report is written after the review.",
      ),
    );
    if (state.status === "incomplete")
      empty.append(
        element(
          "p",
          "",
          "Available findings are preserved below and in Team notes. No completed report was produced.",
        ),
      );
    host.append(empty);
    for (const role of ["investigator", "methods"]) {
      const output = state[`${role}_findings`];
      if (output?.findings?.length) {
        const block = section(
          `${roles[role]} — available findings`,
          role === "investigator" ? "01" : "02",
        );
        output.findings.forEach((finding) =>
          block.append(findingNode(finding)),
        );
        host.append(block);
      }
    }
    return;
  }
  const output = state.writer_findings;
  ["Answer", "Contributions", "Methods and evidence", "Limitations"].forEach(
    (topic, index) => {
      const block = section(topic, String(index + 1).padStart(2, "0"));
      output.findings
        .filter((finding) => finding.topic === topic)
        .forEach((finding) => block.append(findingNode(finding)));
      host.append(block);
    },
  );
  host.append(
    listSection(
      "Follow-up questions — proposals",
      "05",
      output.follow_up_questions.length
        ? output.follow_up_questions
        : ["No follow-up proposals were produced."],
    ),
  );
  const uncertainties = [
    ...new Set([
      ...(output.uncertainties || []),
      ...(state.review.uncertainties || []),
      ...(state.investigator_findings.uncertainties || []),
      ...(state.methods_findings.uncertainties || []),
    ]),
  ];
  if (uncertainties.length)
    host.append(listSection("Remaining uncertainties", "06", uncertainties));
  if (state.review.issues?.length) {
    const block = section(
      state.review.correction_required
        ? "Unresolved reviewer concerns"
        : "Reviewer concerns",
      "07",
    );
    state.review.issues.forEach((issue) =>
      block.append(findingNode({ ...issue, kind: "reviewer_concern" })),
    );
    host.append(block);
  }
  host.append(
    element(
      "p",
      "report-disclaimer",
      "The report uses only the uploaded paper. All four roles share the same model; their agreement is not scientific verification. The writer’s report has not received another reviewer pass.",
    ),
  );
}
function jsonDetails(title, value) {
  const details = element("details");
  details.append(
    element("summary", "", title),
    element("pre", "", JSON.stringify(value, null, 2)),
  );
  return details;
}
function renderTeam() {
  const host = $("team-content");
  host.replaceChildren();
  for (const role of ["investigator", "methods", "reviewer", "writer"]) {
    const block = element("section", "team-section");
    block.append(element("h2", "", roles[role]));
    const output =
      role === "reviewer" ? run.state.review : run.state[`${role}_findings`];
    if (!output || !Object.keys(output).length)
      block.append(
        element("p", "panel-intro", "No completed findings for this role yet."),
      );
    else if (role !== "reviewer") {
      output.findings.forEach((finding) => block.append(findingNode(finding)));
      if (output.uncertainties?.length)
        block.append(listSection("Uncertainties", "—", output.uncertainties));
    } else {
      block.append(
        element(
          "p",
          "panel-intro",
          output.correction_required
            ? `Correction requested for the ${output.target === "methods" ? "methods analyst" : "investigator"}.`
            : "No correction requested. This is not independent verification.",
        ),
      );
      output.issues.forEach((issue) =>
        block.append(findingNode({ ...issue, kind: "reviewer_concern" })),
      );
      if (output.uncertainties.length)
        block.append(
          listSection("Remaining uncertainties", "—", output.uncertainties),
        );
    }
    host.append(block);
  }
  run.state.reviews.forEach((review, index) =>
    host.append(
      jsonDetails(
        `Reviewer pass ${index + 1} — ${review.correction_required ? "correction requested" : "no correction requested"}`,
        review,
      ),
    ),
  );
  if (run.state.correction_history.length)
    host.append(
      jsonDetails(
        "Correction history — earlier findings",
        run.state.correction_history,
      ),
    );
}
function renderSource() {
  const host = $("selected-source");
  host.replaceChildren();
  const entry = run?.evidence?.[selectedEvidence];
  if (!entry)
    host.append(
      element(
        "p",
        "margin-empty",
        "Choose an evidence reference to read its source here.",
      ),
    );
  else {
    host.append(
      element(
        "p",
        "source-label",
        `${entry.evidence_id} / PDF PAGE ${entry.page}`,
      ),
    );
    host.append(element("blockquote", "source-quote", entry.text));
    const button = element(
      "button",
      "text-button accent",
      "Read the extracted page →",
    );
    button.type = "button";
    button.addEventListener("click", () => {
      setView("pages");
      const page = $(`page-${entry.page}`);
      if (page) page.scrollIntoView({ block: "start" });
    });
    host.append(button);
  }
  const reviewHost = $("margin-review");
  reviewHost.replaceChildren();
  const review = run?.state?.review;
  if (!review || !Object.keys(review).length)
    reviewHost.append(
      element(
        "p",
        "margin-empty",
        "The reviewer’s concerns will appear after the review.",
      ),
    );
  else if (review.issues.length)
    review.issues.forEach((issue) =>
      reviewHost.append(element("p", "", issue.statement)),
    );
  else
    reviewHost.append(
      element(
        "p",
        "margin-empty",
        "No correction requested. Check the cited passages before relying on the findings.",
      ),
    );
}
function renderSources() {
  const query = $("source-filter").value.trim().toLowerCase();
  const host = $("sources-content");
  host.replaceChildren();
  const entries = Object.values(run.evidence).filter((entry) =>
    `${entry.evidence_id} ${entry.page} ${entry.text}`
      .toLowerCase()
      .includes(query),
  );
  for (const entry of entries) {
    const block = element("article", "source-entry");
    block.append(citation(entry.evidence_id), element("p", "", entry.text));
    host.append(block);
  }
  if (!entries.length)
    host.append(element("p", "panel-intro", "No matching extracted passages."));
}
function renderPages() {
  $("pages-content").replaceChildren();
  Object.entries(run.pages).forEach(([number, text]) => {
    const block = element("section", "page-entry");
    block.id = `page-${number}`;
    block.append(
      element("h3", "", `PDF PAGE ${number}`),
      element("p", "", text || "No text was extracted from this page."),
    );
    $("pages-content").append(block);
  });
}
function appendEvent(event, index) {
  const row = element("div", "event-row");
  row.append(
    element("span", "event-index", String(index + 1).padStart(2, "0")),
  );
  const content = element("div");
  let label = event.kind.replaceAll("_", " ");
  if (event.kind === "tool_executed") label = event.tool;
  if (event.kind === "model_request") label = `model request ${event.number}`;
  content.append(
    element(
      "div",
      "event-title",
      `${roles[event.agent] || event.agent} / ${label}`,
    ),
  );
  if (event.kind === "tool_executed") {
    content.append(
      jsonDetails("Arguments and actual tool result", {
        arguments: event.arguments,
        result: event.result,
      }),
    );
  }
  if (event.message) content.append(element("p", "muted", event.message));
  row.append(content);
  $("activity-content").append(row);
}
function updateStatus(event) {
  if (!run) return;
  let message;
  if (run.state.status === "complete")
    message = "Investigation complete · Report not independently verified";
  else if (run.state.status === "incomplete")
    message = "Incomplete investigation · Available findings preserved";
  else if (event) {
    const role = roles[event.agent] || event.agent;
    const actions = {
      visit_started: "starting",
      model_request: "requesting a model response",
      visit_completed: "findings recorded",
      correction_scheduled: `one correction scheduled for ${roles[event.target] || event.target}`,
      tool_executed: `used ${event.tool}`,
      incomplete: "could not finish",
    };
    message = `${role} — ${actions[event.kind] || event.kind.replaceAll("_", " ")}`;
  } else message = "Preparing the investigation…";
  $("status-text").textContent = message;
  $("status-dot").className = `status-dot ${run.state.status}`;
  const calls = run.state.model_calls || 0;
  const tools = run.state.tool_calls || 0;
  $("run-counts").textContent =
    `${calls} requests · ${tools} tools · ${run.state.correction_count || 0}/1 corrections`;
}
function render() {
  $("results").hidden = !run;
  $("export").disabled = !run || running;
  $("new-run").hidden = !run;
  $("new-run").disabled = running;
  $("change-paper").disabled = running;
  if (!run) return;
  $("breadcrumb").textContent = `Investigation / ${run.meta.filename}`;
  $("side-filename").textContent = run.meta.filename;
  $("result-filename").textContent = run.meta.filename;
  $("result-page-count").textContent =
    `/ ${Object.keys(run.pages).length} physical pages`;
  $("result-question").textContent = run.meta.question;
  $("provenance").textContent =
    `PDF: ${run.meta.filename}. SHA-256: ${run.meta.digest}. Model via Groq: ${run.meta.model}. This record belongs to the PDF and question shown above.`;
  renderReport();
  renderTeam();
  renderSource();
  renderSources();
  renderPages();
  updateStatus(run.events[run.events.length - 1]);
  setView(view);
}
function setRunning(value) {
  running = value;
  document.body.classList.toggle("is-running", value);
  [
    "paper",
    "question",
    "example",
    "start",
    "new-run",
    "change-paper",
    "export",
  ].forEach((id) => ($(id).disabled = value));
  if (!value) {
    $("start").disabled = !config.ready;
    $("export").disabled = !run;
  }
}
function selectFile(file) {
  if (running) return;
  formError("");
  if (
    file &&
    (!file.name.toLowerCase().endsWith(".pdf") ||
      file.size > config.max_upload_mb * 1024 * 1024)
  ) {
    selectedFile = null;
    $("paper").value = "";
    formError(`Choose one PDF no larger than ${config.max_upload_mb} MB.`);
  } else selectedFile = file;
  $("file-title").textContent = selectedFile
    ? selectedFile.name
    : "Choose a paper";
  $("file-description").textContent = selectedFile
    ? `${(selectedFile.size / 1024).toFixed(0)} KB · ready to investigate`
    : "Drop a PDF here or browse your files.";
  if (!run)
    $("side-filename").textContent = selectedFile
      ? selectedFile.name
      : "No paper selected";
}
function acceptMessage(message) {
  if (message.type === "extraction") {
    run = {
      meta: message.meta,
      pages: message.pages,
      evidence: message.evidence,
      events: [],
      state: {
        status: "running",
        error: "",
        final_report: "",
        investigator_findings: {},
        methods_findings: {},
        writer_findings: {},
        review: {},
        reviews: [],
        correction_history: [],
        model_calls: 0,
        tool_calls: 0,
        correction_count: 0,
      },
    };
    selectedEvidence = Object.keys(run.evidence)[0] || null;
    view = "report";
    $("activity-content").replaceChildren();
    $("setup-section").hidden = true;
    notice("");
    render();
  } else if (message.type === "event" && run) {
    run.events.push(message.event);
    if (message.event.kind === "model_request")
      run.state.model_calls = message.event.number;
    if (message.event.kind === "tool_executed") run.state.tool_calls += 1;
    appendEvent(message.event, run.events.length - 1);
    updateStatus(message.event);
  } else if (message.type === "state" && run) {
    run.state = message.state;
    render();
  }
  save();
}
async function startInvestigation(event) {
  event.preventDefault();
  if (running) return;
  const question = $("question").value.trim();
  if (!selectedFile || !question) {
    formError("Choose a PDF and enter a question before starting.");
    return;
  }
  if (!config.ready) {
    formError("The server is not ready. Check the service message above.");
    return;
  }
  formError("");
  const form = new FormData();
  form.append("paper", selectedFile, selectedFile.name);
  form.append("question", question);
  setRunning(true);
  notice("Uploading your paper and preparing the investigation…");
  let began = false;
  try {
    const response = await fetch("/api/research", {
      method: "POST",
      body: form,
      credentials: "same-origin",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        error.error || "The server could not start this investigation.",
      );
    }
    if (!response.body)
      throw new Error(
        "Your browser could not receive the investigation stream.",
      );
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { value, done } = await reader.read();
      pending += done
        ? decoder.decode()
        : decoder.decode(value, { stream: true });
      let newline;
      while ((newline = pending.indexOf("\n")) !== -1) {
        const line = pending.slice(0, newline);
        pending = pending.slice(newline + 1);
        if (line.trim()) {
          const message = JSON.parse(line);
          if (message.type === "extraction") began = true;
          acceptMessage(message);
        }
      }
      if (done) break;
    }
    if (pending.trim()) acceptMessage(JSON.parse(pending));
    if (!began || run.state.status === "running")
      throw new Error(
        "The connection closed before the investigation finished. No automatic retry was made.",
      );
  } catch (error) {
    const message =
      error instanceof SyntaxError
        ? "The response could not be read. No automatic retry was made."
        : error.message;
    if (began && run.state.status !== "complete") {
      run.state.status = "incomplete";
      run.state.error = message;
      run.state.final_report = "";
      save();
    } else formError(message);
    if (!began) $("setup-section").hidden = false;
  } finally {
    notice("");
    setRunning(false);
    render();
  }
}
function markdownReport() {
  const s = run.state;
  const parts = [
    "# Paper Research Team — Research Desk",
    `PDF: ${run.meta.filename}`,
    `SHA-256: ${run.meta.digest}`,
    `Model via Groq: ${run.meta.model}`,
    `Status: ${s.status}`,
    `Question: ${run.meta.question}`,
    `Model requests: ${s.model_calls} · Tool executions: ${s.tool_calls}`,
  ];
  parts.push(
    s.status === "complete"
      ? s.final_report
      : `## Incomplete run\n${s.error}\nNo completed report was produced.`,
  );
  for (const [title, value] of [
    ["Investigator findings", s.investigator_findings],
    ["Methods findings", s.methods_findings],
    ["Reviewer history", s.reviews],
    ["Correction history", s.correction_history],
  ]) {
    parts.push(
      `## ${title}\n\n\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\``,
    );
  }
  const referenced = new Set(parts.join("\n").match(/\bE\d{4,}\b/g) || []);
  parts.push("## Referenced source passages");
  [...referenced].sort().forEach((id) => {
    const source = run.evidence[id];
    if (source)
      parts.push(`### ${id} · PDF page ${source.page}\n\n${source.text}`);
  });
  parts.push(
    "Only the uploaded paper was investigated. No other literature was searched and global novelty was not established. Valid citations and matching quotes do not establish semantic support. Agent agreement is not scientific verification.",
  );
  return parts.join("\n\n");
}
$("export").addEventListener("click", () => {
  if (!run || running) return;
  const url = URL.createObjectURL(
    new Blob([markdownReport()], { type: "text/markdown;charset=utf-8" }),
  );
  const link = element("a");
  link.href = url;
  link.download = "paper-research-report.md";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("new-run").addEventListener("click", () => {
  if (running) return;
  run = null;
  selectedEvidence = null;
  selectedFile = null;
  view = "report";
  try {
    sessionStorage.removeItem(storageKey);
  } catch (_) {
    /* In-memory clear still applies. */
  }
  $("paper").value = "";
  $("question").value = "";
  $("question-length").textContent = "0 / 2,000";
  $("setup-section").hidden = false;
  $("breadcrumb").textContent = "Research desk / New investigation";
  $("side-filename").textContent = "No paper selected";
  $("activity-content").replaceChildren();
  $("source-filter").value = "";
  selectFile(null);
  formError("");
  render();
  document
    .querySelectorAll(".side-link")
    .forEach((button) =>
      button.classList.toggle("active", button.dataset.view === "report"),
    );
  $("paper").focus();
});
$("change-paper").addEventListener("click", () => {
  if (running) return;
  $("setup-section").hidden = false;
  $("question").value = run.meta.question;
  $("question").dispatchEvent(new Event("input"));
  $("setup-section").scrollIntoView({ block: "start" });
  $("question").focus({ preventScroll: true });
});
$("paper").addEventListener("change", () =>
  selectFile($("paper").files[0] || null),
);
$("question").addEventListener(
  "input",
  () =>
    ($("question-length").textContent =
      `${$("question").value.length.toLocaleString()} / 2,000`),
);
$("example").addEventListener("click", () => {
  $("question").value =
    "Explain this paper’s contribution, assess its experimental evidence, identify its stated limitations, and suggest follow-up research questions.";
  $("question").dispatchEvent(new Event("input"));
  $("question").focus();
});
$("investigation-form").addEventListener("submit", startInvestigation);
$("source-filter").addEventListener("input", () => {
  if (run) renderSources();
});
document
  .querySelectorAll("[data-view]")
  .forEach((button) =>
    button.addEventListener("click", () => setView(button.dataset.view)),
  );
const tabs = [...document.querySelectorAll(".tabs [role=tab]")];
tabs.forEach((button, index) =>
  button.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft")
      next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      setView(tabs[next].dataset.view, true);
    }
  }),
);
["dragenter", "dragover"].forEach((name) =>
  $("drop-area").addEventListener(name, (event) => {
    event.preventDefault();
    if (!running) $("drop-area").classList.add("drag-over");
  }),
);
["dragleave", "drop"].forEach((name) =>
  $("drop-area").addEventListener(name, (event) => {
    event.preventDefault();
    $("drop-area").classList.remove("drag-over");
  }),
);
$("drop-area").addEventListener("drop", (event) => {
  if (running) return;
  if (event.dataTransfer.files.length !== 1) {
    formError("Choose exactly one PDF.");
    return;
  }
  const file = event.dataTransfer.files[0];
  selectFile(file);
  if (selectedFile) {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    $("paper").files = transfer.files;
  }
});
try {
  const saved = sessionStorage.getItem(storageKey);
  if (saved) {
    const candidate = JSON.parse(saved);
    if (
      candidate?.meta &&
      candidate?.state &&
      candidate?.evidence &&
      candidate?.pages &&
      Array.isArray(candidate.events)
    ) {
      run = candidate;
      if (run.state.status === "running") {
        run.state.status = "incomplete";
        run.state.error =
          "The page was refreshed during execution. Available findings were kept; the investigation was not restarted.";
        run.state.final_report = "";
      }
      selectedEvidence = Object.keys(run.evidence)[0] || null;
      $("setup-section").hidden = true;
      $("activity-content").replaceChildren();
      run.events.forEach(appendEvent);
      render();
      save();
    }
  }
} catch (_) {
  try {
    sessionStorage.removeItem(storageKey);
  } catch (_) {
    /* Storage unavailable. */
  }
}
$("start").disabled = true;
fetch("/api/research", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error();
    return response.json();
  })
  .then((value) => {
    config = value;
    $("start").disabled = running || !config.ready;
    $("file-hint").textContent =
      `English · embedded text · up to ${config.max_upload_mb} MB`;
    if (!config.ready)
      notice(
        "Investigations are not available yet: the server needs its Groq API key.",
      );
  })
  .catch(() =>
    notice(
      "The Python service could not be reached. If you are running locally, start the project with python app.py.",
    ),
  );
