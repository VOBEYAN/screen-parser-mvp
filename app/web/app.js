const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  result: null,
  runs: [],
  selectedNodeId: null,
  selectedLevel: "",
  imageMode: "source",
  boxScope: "detector",
  zoom: 1,
  previewUrl: "",
  parseTimer: null,
  hitNodes: [],
  components: [],
};

const colors = {
  Panel: "#55d4ff",
  Region: "#7c6cff",
  Content: "#91a3b7",
  Title: "#ffb84d",
  Border: "#55a8ff",
  Chart: "#2ed18b",
  Table: "#ff8f55",
  Map: "#b88cff",
  MetricCard: "#ff6b7a",
  Decorate: "#d0d8e2",
  Filter: "#4bd7c8",
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  restoreTheme();
  loadStatus();
  loadComponents();
  renderResultViews();
  loadRuns(false);
});

function bindEvents() {
  const form = $("#parseForm");
  const fileInput = $("#imageInput");
  const dropZone = $("#dropZone");
  const topK = $("#topKInput");

  form.addEventListener("submit", parseCurrentImage);
  topK.addEventListener("input", () => {
    $("#topKValue").textContent = topK.value;
  });

  fileInput.addEventListener("change", () => {
    handleFilePreview(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropZone.addEventListener(name, () => dropZone.classList.remove("dragover"));
  });
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (!file) return;
    fileInput.files = event.dataTransfer.files;
    handleFilePreview(file);
  });

  $$("[data-image-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      state.imageMode = button.dataset.imageMode;
      setActiveButton("[data-image-mode]", button);
      renderImage();
    });
  });

  $$("[data-box-scope]").forEach((button) => {
    button.addEventListener("click", () => {
      state.boxScope = button.dataset.boxScope;
      setActiveButton("[data-box-scope]", button);
      renderImage();
    });
  });

  $$("[data-zoom]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.zoom;
      if (action === "in") state.zoom = Math.min(2.4, state.zoom + 0.15);
      if (action === "out") state.zoom = Math.max(0.45, state.zoom - 0.15);
      if (action === "reset") state.zoom = 1;
      renderImage();
    });
  });

  $$("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      setActiveButton("[data-tab]", button);
      $$(".tab-view").forEach((view) => {
        view.classList.toggle("active", view.dataset.view === button.dataset.tab);
      });
    });
  });

  $("#nodeSearch").addEventListener("input", renderResultViews);
  $("#typeFilter").addEventListener("change", renderResultViews);
  $("#levelFilter").addEventListener("change", () => selectLevel($("#levelFilter").value));
  $("#clearSelectionBtn").addEventListener("click", () => selectNode(null));
  $("#refreshRunsBtn").addEventListener("click", () => loadRuns(false));
  $("#copyJsonBtn").addEventListener("click", copyJson);
  $("#copySchemaBtn").addEventListener("click", copySchemaComponents);
  $("#themeBtn").addEventListener("click", toggleTheme);
  document.addEventListener("click", (event) => {
    if (!event.target.closest("#hitPicker") && !event.target.closest(".bbox-layer")) {
      hideHitPicker();
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideHitPicker();
  });
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "状态读取失败");
    renderStatus(data.status);
  } catch (error) {
    $("#statusList").innerHTML = `<div><dt>状态</dt><dd>${escapeHtml(error.message)}</dd></div>`;
  }
}

async function loadRuns(autoLoad) {
  try {
    const response = await fetch("/api/runs");
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "历史读取失败");
    state.runs = data.runs || [];
    renderRuns();
    if (autoLoad && !state.result && state.runs.length) {
      loadRun(state.runs[0].runId);
    }
  } catch (error) {
    $("#runList").textContent = error.message;
  }
}

async function loadComponents() {
  try {
    const response = await fetch("/api/components");
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "组件列表读取失败");
    state.components = data.components || [];
    renderInspector();
  } catch (error) {
    console.warn(error);
  }
}

async function loadRun(runId) {
  try {
    showToast("正在打开历史结果 " + runId);
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "结果读取失败");
    setResult(data.result);
  } catch (error) {
    showToast(error.message);
  }
}

async function parseCurrentImage(event) {
  event.preventDefault();
  const form = $("#parseForm");
  const file = $("#imageInput").files[0];
  if (!file) {
    showToast("请先选择一张图片");
    return;
  }

  const parseBtn = $("#parseBtn");
  parseBtn.disabled = true;
  parseBtn.textContent = "解析中";
  startProgress();

  try {
    const response = await fetch("/parse", {
      method: "POST",
      body: new FormData(form),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "解析失败");
    setResult(data.result);
    await loadRuns(false);
    showToast("解析完成，已生成证据图和报告");
  } catch (error) {
    showToast(error.message);
  } finally {
    parseBtn.disabled = false;
    parseBtn.textContent = "开始解析";
    stopProgress();
  }
}

function handleFilePreview(file) {
  if (!file) return;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $("#localPreview").src = state.previewUrl;
  $("#fileName").textContent = file.name;
  $("#previewTitle").textContent = file.name;
  $("#previewMeta").textContent = `${formatBytes(file.size)} · ${file.type || "image"}`;
}

function setResult(result) {
  state.result = result;
  state.selectedNodeId = null;
  state.selectedLevel = "";
  $("#levelFilter").value = "";
  state.zoom = 1;
  renderMetrics();
  refreshLevelFilter();
  refreshTypeFilter();
  renderResultViews();
  markActiveRun(result.runId);
}

function renderStatus(status) {
  const llmLabel = status.localQwenEnabled
    ? `本地 LoRA · ${status.localQwenLoaded ? "已加载" : "待加载"}`
    : status.llmEnabled
    ? `已启用 · ${status.llmModel || ""}`
    : status.llmConfigured
      ? "已配置但未启用"
      : "未配置，使用本地规则";
  const rows = [
    ["Detector", status.detector + modelSuffix(status.detectorModel)],
    ["Hierarchy", status.hierarchyMode],
    ["Matcher", status.matcherMode],
    ["VLM", llmLabel],
    ["Catalog", `${status.componentLibraryCount || 0} 组件 / ${status.componentCategoryCount || 0} 类`],
    ["Reference", `${status.visualReferenceCount || 0} 视觉样本`],
  ];
  $("#statusList").innerHTML = rows.map(([label, value]) => `
    <div><dt>${escapeHtml(label)}</dt><dd title="${escapeAttr(value)}">${escapeHtml(value || "-")}</dd></div>
  `).join("");
}

function renderRuns() {
  const target = $("#runList");
  if (!state.runs.length) {
    target.className = "run-list empty";
    target.textContent = "暂无历史结果";
    return;
  }
  target.className = "run-list";
  target.innerHTML = state.runs.map((run) => {
    const summary = run.summary || {};
    return `
      <button type="button" class="run-item" data-run-id="${escapeAttr(run.runId)}">
        <strong>${escapeHtml(run.imageName || run.runId)}</strong>
        <span>${escapeHtml(run.runId)} · ${summary.nodeCount || 0} 节点 · ${summary.overlapCount || 0} 重叠</span>
      </button>
    `;
  }).join("");
  $$(".run-item", target).forEach((item) => {
    item.addEventListener("click", () => loadRun(item.dataset.runId));
  });
  if (state.result) markActiveRun(state.result.runId);
}

function renderMetrics() {
  const summary = state.result?.summary || {};
  const visibleCount = state.result ? getFilteredNodes().length : 0;
  $("#metricNodes").textContent = state.selectedLevel ? visibleCount : (summary.nodeCount || 0);
  $("#metricDetections").textContent = summary.detectionCount || 0;
  $("#metricOverlaps").textContent = summary.overlapCount || 0;
  $("#metricLlmCalls").textContent = (summary.localQwenCallCount || 0) + (summary.llmCallCount || 0);
}

function renderResultViews() {
  renderMetrics();
  renderImage();
  renderComponentSummary();
  renderSchemaCanvas();
  renderComponentsTable();
  renderHierarchy();
  renderOverlaps();
  renderReport();
  renderJson();
  renderInspector();
}

function renderImage() {
  hideHitPicker();
  const stage = $("#imageStage");
  $("#zoomLabel").textContent = Math.round(state.zoom * 100) + "%";
  if (!state.result) {
    stage.className = "image-stage empty";
    stage.innerHTML = `<div class="empty-state"><strong>解析结果会显示在这里</strong><span>上传图片后查看检测框和组件匹配。</span></div>`;
    return;
  }

  const urls = state.result.artifactUrls || {};
  const imageUrl = state.imageMode === "source"
    ? (urls.sourceImage || urls.evidenceImage)
    : (urls.evidenceImage || urls.sourceImage);
  const meta = state.result.imageMeta || {};
  const width = Number(meta.width || 1920);
  const height = Number(meta.height || 1080);
  const nodes = getImageOverlayNodes();

  stage.className = "image-stage";
  stage.innerHTML = `
    <div class="stage-canvas" style="--image-zoom:${state.zoom}">
      <div class="stage-frame">
        <img src="${escapeAttr(imageUrl)}" alt="解析图像" />
        <svg class="bbox-layer" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="检测框">
          ${nodes.map(renderBBox).join("")}
        </svg>
      </div>
    </div>
  `;
  const layer = $(".bbox-layer", stage);
  layer.addEventListener("click", (event) => handleImageHit(event, layer, nodes, width, height));
  layer.addEventListener("mousemove", (event) => updateHoverHit(event, layer, nodes, width, height));
  layer.addEventListener("mouseleave", () => {
    $$(".bbox.hover-hit", stage).forEach((item) => item.classList.remove("hover-hit"));
  });
}

function renderBBox(node) {
  const box = node.bbox || {};
  const x = Number(box.x || 0);
  const y = Number(box.y || 0);
  const w = Math.max(1, Number(box.w || 1));
  const h = Math.max(1, Number(box.h || 1));
  const color = colors[node.type] || "#ffffff";
  const selected = state.selectedNodeId === node.node_id ? " selected" : "";
  const labelY = Math.max(20, y + 22);
  const label = `${node.node_id} ${node.type}`;
  return `
    <rect class="bbox${selected}" data-node-id="${escapeAttr(node.node_id)}"
      x="${x}" y="${y}" width="${w}" height="${h}" stroke="${color}">
      <title>${escapeHtml(label)}</title>
    </rect>
    <text class="bbox-label" x="${x + 8}" y="${labelY}">${escapeHtml(node.node_id)}</text>
  `;
}

function getImageOverlayNodes() {
  if (state.imageMode === "evidence") return [];
  const nodes = getFilteredNodes();
  if (state.boxScope === "detector") {
    return nodes.filter((node) => Boolean(node.detection_id));
  }
  return nodes;
}

function handleImageHit(event, layer, nodes, width, height) {
  event.stopPropagation();
  const point = svgPointFromEvent(event, layer, width, height);
  const hits = nodesAtPoint(nodes, point.x, point.y);
  if (!hits.length) {
    selectNode(null);
    hideHitPicker();
    return;
  }
  if (hits.length === 1) {
    selectNode(hits[0].node_id);
    hideHitPicker();
    return;
  }
  showHitPicker(hits, event.clientX, event.clientY);
}

function updateHoverHit(event, layer, nodes, width, height) {
  const point = svgPointFromEvent(event, layer, width, height);
  const hits = nodesAtPoint(nodes, point.x, point.y);
  const hitIds = new Set(hits.map((node) => node.node_id));
  $$(".bbox", $("#imageStage")).forEach((item) => {
    item.classList.toggle("hover-hit", hitIds.has(item.dataset.nodeId));
  });
}

function svgPointFromEvent(event, layer, width, height) {
  const rect = layer.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * width;
  const y = ((event.clientY - rect.top) / Math.max(rect.height, 1)) * height;
  return { x, y };
}

function nodesAtPoint(nodes, x, y) {
  return nodes
    .filter((node) => {
      const box = node.bbox || {};
      const left = Number(box.x || 0);
      const top = Number(box.y || 0);
      const right = left + Number(box.w || 0);
      const bottom = top + Number(box.h || 0);
      return x >= left && x <= right && y >= top && y <= bottom;
    })
    .sort((a, b) => {
      const areaA = nodeArea(a);
      const areaB = nodeArea(b);
      if (areaA !== areaB) return areaA - areaB;
      return String(a.node_id).localeCompare(String(b.node_id));
    });
}

function nodeArea(node) {
  const box = node.bbox || {};
  return Number(box.w || 0) * Number(box.h || 0);
}

function showHitPicker(nodes, clientX, clientY) {
  state.hitNodes = nodes;
  const picker = $("#hitPicker");
  picker.hidden = false;
  picker.innerHTML = `
    <div class="hit-picker-head">
      <strong>选择命中的组件</strong>
      <span>${nodes.length} 个重叠节点</span>
    </div>
    ${nodes.map((node, index) => {
      const candidate = topCandidate(node);
      const box = node.bbox || {};
      const selected = state.selectedNodeId === node.node_id ? " active" : "";
      return `
        <button class="hit-option${selected}" type="button" data-hit-node="${escapeAttr(node.node_id)}">
          <b>${index + 1}. ${escapeHtml(node.node_id)} · ${escapeHtml(node.type || "-")}</b>
          <span>${escapeHtml(node.component_id || candidate.componentId || candidate.title || "未匹配组件")}</span>
          <small>${Math.round(nodeArea(node))} px2 · ${formatBBox(box)}</small>
        </button>
      `;
    }).join("")}
  `;
  const margin = 12;
  const maxLeft = window.innerWidth - 320 - margin;
  const maxTop = window.innerHeight - Math.min(360, 84 + nodes.length * 60) - margin;
  picker.style.left = `${Math.max(margin, Math.min(clientX + 12, maxLeft))}px`;
  picker.style.top = `${Math.max(margin, Math.min(clientY + 12, maxTop))}px`;
  $$("[data-hit-node]", picker).forEach((button) => {
    button.addEventListener("click", () => {
      selectNode(button.dataset.hitNode);
      hideHitPicker();
    });
  });
}

function hideHitPicker() {
  const picker = $("#hitPicker");
  if (!picker) return;
  picker.hidden = true;
  picker.innerHTML = "";
  state.hitNodes = [];
}

function renderComponentSummary() {
  const target = $("#componentSummary");
  const components = state.result?.summary?.components || [];
  const summary = state.result?.summary || {};
  const mode = summary.contentClassifierMode || "unknown";
  const llmBadge = summary.localQwenEnabled
    ? `<span class="component-chip"><b>Local LoRA</b><span>Qwen3-VL</span><small>${summary.localQwenCallCount || 0} calls</small></span>`
    : summary.llmEnabled
    ? `<span class="component-chip"><b>VLM</b><span>${escapeHtml(summary.llmModel || "model")}</span><small>${summary.llmCallCount || 0} calls</small></span>`
    : `<span class="component-chip"><b>VLM</b><span>未调用大模型</span><small>${escapeHtml(mode)}</small></span>`;
  const ocrBadge = summary.paddleOcrEnabled
    ? `<span class="component-chip"><b>PaddleOCR</b><span>已启用</span><small>${summary.paddleOcrTextCount || 0} texts</small></span>`
    : `<span class="component-chip"><b>PaddleOCR</b><span>未启用</span><small>${escapeHtml(summary.paddleOcrError || "")}</small></span>`;
  if (!components.length) {
    target.innerHTML = llmBadge + ocrBadge + `<span class="component-chip"><span>暂无组件汇总</span></span>`;
    return;
  }
  target.innerHTML = llmBadge + ocrBadge + components.map((item) => `
    <span class="component-chip">
      <b>${escapeHtml(item.componentId)}</b>
      <span>${escapeHtml(item.title || "")}</span>
      <small>${item.count} 次 / ${formatScore(item.avgScore)}</small>
    </span>
  `).join("");
}

function renderSchemaCanvas() {
  const target = $("#schemaCanvas");
  const components = state.result?.aiSchemaComponents || [];
  const openBtn = $("#openSchemaRenderBtn");
  const renderUrl = buildSchemaRenderUrl();
  if (openBtn) openBtn.href = renderUrl || "http://localhost:3020/ai-schema/#/schema-render";
  if (!components.length) {
    target.className = "schema-canvas empty";
    target.innerHTML = `
      <div class="empty-state">
        <strong>${state.result ? "没有可还原的 ai-schema-view 组件" : "暂无还原组件"}</strong>
        <span>识别到图表、表格、地图、标题等节点后，会把参数化组件显示在这里。</span>
      </div>
    `;
    return;
  }
  target.className = "schema-canvas live";
  target.innerHTML = `
    <div class="schema-live-head">
      <strong>真实组件渲染</strong>
      <span>${components.length} 个 ai-schema-view 组件，已注入 option.dataset 等参数</span>
    </div>
    <iframe class="schema-render-frame" src="${escapeAttr(renderUrl)}" title="ai-schema-view 参数化组件渲染"></iframe>
  `;
}

function buildSchemaRenderUrl() {
  if (!state.result?.runId) return "";
  const api = encodeURIComponent("http://127.0.0.1:8765");
  const runId = encodeURIComponent(state.result.runId);
  return `http://localhost:3020/ai-schema/#/schema-render?api=${api}&runId=${runId}`;
}

function renderComponentsTable() {
  const tbody = $("#componentsBody");
  const nodes = getFilteredNodes();
  if (!nodes.length) {
    tbody.innerHTML = `<tr><td colspan="8">暂无匹配节点</td></tr>`;
    return;
  }
  tbody.innerHTML = nodes.map((node) => {
    const candidate = topCandidate(node);
    const classifier = classifierOf(node);
    const box = node.bbox || {};
    const selected = state.selectedNodeId === node.node_id ? " class=\"selected\"" : "";
    return `
      <tr data-node-id="${escapeAttr(node.node_id)}"${selected}>
        <td><code>${escapeHtml(node.node_id)}</code></td>
        <td><span class="pill">${escapeHtml(node.type || "-")}</span></td>
        <td>${escapeHtml(node.component_id || candidate.componentId || "-")}</td>
        <td>${escapeHtml(candidate.title || "-")}</td>
        <td><span class="score">${formatScore(candidate.score)}</span></td>
        <td>${renderClassifierBadge(classifier)}</td>
        <td>${formatPercent(node.confidence || 0)}</td>
        <td>
          <button class="ghost-button compact-correct" type="button" data-correct-node="${escapeAttr(node.node_id)}">纠错</button>
          <small>${formatBBox(box)}</small>
        </td>
      </tr>
    `;
  }).join("");
  $$("tr[data-node-id]", tbody).forEach((row) => {
    row.addEventListener("click", () => selectNode(row.dataset.nodeId));
  });
  $$("[data-correct-node]", tbody).forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectNode(button.dataset.correctNode);
      const inspector = $("#nodeInspector");
      if (inspector) inspector.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
}

function renderHierarchy() {
  const target = $("#hierarchyTree");
  if (!state.result) {
    target.className = "hierarchy-tree empty";
    target.textContent = "暂无层级数据";
    return;
  }
  const nodes = state.result.nodes || [];
  const byParent = new Map();
  const byId = new Map();
  nodes.forEach((node) => {
    byId.set(node.node_id, node);
    const key = node.parent_id || "__root__";
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(node);
  });

  const roots = nodes.filter((node) => node.type === "Screen");
  const startNodes = roots.length ? roots : (byParent.get("__root__") || []);
  target.className = "hierarchy-tree";
  target.innerHTML = renderLayerToolbar(nodes) + startNodes.map((node) => renderTreeNode(node, byParent, new Set())).join("");
  $$("[data-tree-node]", target).forEach((button) => {
    button.addEventListener("click", () => selectNode(button.dataset.treeNode));
  });
  $$("[data-layer-level]", target).forEach((button) => {
    button.addEventListener("click", () => selectLevel(button.dataset.layerLevel));
  });
}

function renderLayerToolbar(nodes) {
  const layers = layerStats(nodes);
  if (!layers.length) return "";
  const allActive = state.selectedLevel ? "" : " active";
  return `
    <div class="layer-toolbar">
      <button type="button" class="layer-chip${allActive}" data-layer-level="">全部层级</button>
      ${layers.map((item) => {
        const active = state.selectedLevel === String(item.level) ? " active" : "";
        return `
          <button type="button" class="layer-chip${active}" data-layer-level="${escapeAttr(item.level)}">
            Level ${escapeHtml(item.level)} <span>${item.count}</span>
          </button>
        `;
      }).join("")}
    </div>
  `;
}

function renderTreeNode(node, byParent, visited) {
  if (!node || visited.has(node.node_id)) return "";
  visited.add(node.node_id);
  const children = byParent.get(node.node_id) || [];
  const candidate = topCandidate(node);
  const selected = state.selectedNodeId === node.node_id ? " active" : "";
  const layerMatch = state.selectedLevel && String(node.level) === state.selectedLevel ? " layer-match" : "";
  const title = node.component_id || candidate.componentId || candidate.title || "";
  return `
    <div class="tree-node">
      <button type="button" class="${selected}${layerMatch}" data-tree-node="${escapeAttr(node.node_id)}">
        <span>${escapeHtml(node.node_id)}</span>
        <b>${escapeHtml(node.type || "-")}</b>
        <em>L${escapeHtml(node.level ?? "-")}</em>
        <small>${escapeHtml(title)}</small>
      </button>
      ${children.map((child) => renderTreeNode(child, byParent, visited)).join("")}
    </div>
  `;
}

function renderOverlaps() {
  const target = $("#overlapPanel");
  const overlaps = state.result?.overlaps || [];
  if (!overlaps.length) {
    target.className = "overlap-panel empty";
    target.textContent = state.result ? "未检测到同级组件异常重叠" : "暂无重叠问题";
    return;
  }
  target.className = "overlap-panel";
  target.innerHTML = overlaps.map((issue) => `
    <div class="overlap-item">
      <strong class="${issue.severity === "error" ? "danger" : "warn"}">${escapeHtml(issue.severity || "warning")}</strong>
      <span>${escapeHtml(issue.source)} 与 ${escapeHtml(issue.target)}</span>
      <small>IoU ${formatScore(issue.iou)} · overlapRatio ${formatScore(issue.overlapRatio)}</small>
      <button class="ghost-button" type="button" data-overlap-node="${escapeAttr(issue.source)}">定位源节点</button>
    </div>
  `).join("");
  $$("[data-overlap-node]", target).forEach((button) => {
    button.addEventListener("click", () => selectNode(button.dataset.overlapNode));
  });
}

function renderReport() {
  const target = $("#reportPanel");
  const urls = state.result?.artifactUrls;
  if (!urls) {
    target.className = "report-panel empty";
    target.textContent = "暂无报告";
    return;
  }
  target.className = "report-panel";
  target.innerHTML = `
    <div class="report-actions">
      <a class="ghost-button" href="${escapeAttr(urls.reportHtml)}" target="_blank" rel="noreferrer">打开 HTML</a>
      <a class="ghost-button" href="${escapeAttr(urls.reportMd)}" target="_blank" rel="noreferrer">打开 Markdown</a>
      <a class="ghost-button" href="${escapeAttr(urls.resultJson)}" target="_blank" rel="noreferrer">打开 JSON</a>
      <a class="ghost-button" href="${escapeAttr(urls.evidenceImage)}" target="_blank" rel="noreferrer">打开证据图</a>
    </div>
    <iframe src="${escapeAttr(urls.reportHtml)}" title="解析报告"></iframe>
  `;
}

function renderJson() {
  $("#jsonOutput").textContent = state.result ? JSON.stringify(state.result, null, 2) : "暂无 JSON";
}

function renderInspector() {
  const target = $("#nodeInspector");
  const node = selectedNode();
  if (!node) {
    if (state.selectedLevel) {
      const nodes = getFilteredNodes();
      target.className = "node-inspector";
      target.innerHTML = `
        <div class="reason-box">
          <strong>已选择 Level ${escapeHtml(state.selectedLevel)}</strong>
          <span>当前层级包含 ${nodes.length} 个可见节点。图像和组件清单已按这一层过滤。</span>
        </div>
        <div class="candidate-list">
          ${nodes.slice(0, 12).map((item) => {
            const candidate = topCandidate(item);
            return `
              <button class="hit-option" type="button" data-inspector-node="${escapeAttr(item.node_id)}">
                <b>${escapeHtml(item.node_id)} · ${escapeHtml(item.type || "-")}</b>
                <span>${escapeHtml(item.component_id || candidate.componentId || candidate.title || "未匹配组件")}</span>
              </button>
            `;
          }).join("")}
        </div>
      `;
      $$("[data-inspector-node]", target).forEach((button) => {
        button.addEventListener("click", () => selectNode(button.dataset.inspectorNode));
      });
      return;
    }
    target.className = "node-inspector empty";
    target.innerHTML = `<strong>未选中节点</strong><span>点击图像上的检测框或表格行查看详情。</span>`;
    return;
  }

  const classifier = classifierOf(node);
  const candidates = (node.candidates || []).slice(0, 8);
  const textEvidence = classifier.textEvidence || classifier.paddleOcrText || classifier.text || "未检测到文字/OCR 未启用";
  const structureEvidence = classifier.structureEvidence || "已按检测类型和布局约束过滤候选";
  target.className = "node-inspector";
  target.innerHTML = `
    <div class="detail-row"><span>Node</span><b>${escapeHtml(node.node_id)}</b></div>
    <div class="detail-row"><span>Type</span><b>${escapeHtml(node.type || "-")}</b></div>
    <div class="detail-row"><span>Parent</span><b>${escapeHtml(node.parent_id || "-")}</b></div>
    <div class="detail-row"><span>Confidence</span><b>${formatPercent(node.confidence || 0)}</b></div>
    <div class="detail-row"><span>BBox</span><code>${formatBBox(node.bbox || {})}</code></div>
    <div class="detail-row"><span>Content</span><b>${escapeHtml(classifier.contentType || "-")}</b></div>
    <div class="detail-row"><span>Mode</span><b>${escapeHtml(classifier.mode || "-")}</b></div>
    <div class="detail-row"><span>PaddleOCR</span><b>${escapeHtml(classifier.paddleOcrText || "-")}</b></div>
    <div class="detail-row"><span>模型读字</span><b>${escapeHtml(classifier.text || "-")}</b></div>
    <div class="detail-row"><span>模型</span><b>${escapeHtml(classifier.llmComponentId || "-")}</b></div>
    ${renderCropEvidence(node)}
    <form class="correction-form" id="correctionForm">
      <strong>纠正为训练样本</strong>
      <select id="correctComponentSelect" required>
        <option value="">选择正确组件</option>
        ${state.components.map((item) => `
          <option value="${escapeAttr(item.componentId)}">${escapeHtml(item.componentId)} · ${escapeHtml(item.title || "")}</option>
        `).join("")}
      </select>
      <input id="correctVisualFormInput" type="text" placeholder="visualForm，可选，如 liquid_vertical_bar" />
      <textarea id="correctNoteInput" rows="2" placeholder="备注，可选"></textarea>
      <button type="submit" class="primary-button">保存纠错样本</button>
    </form>
    <div class="candidate-list">
      <strong>候选组件</strong>
      ${candidates.length ? candidates.map(renderCandidate).join("") : `<span class="reason-box">暂无候选</span>`}
    </div>
    <div class="reason-box">
      <strong>文字证据</strong>
      <span>${escapeHtml(textEvidence)}</span>
    </div>
    <div class="reason-box">
      <strong>结构证据</strong>
      <span>${escapeHtml(structureEvidence)}</span>
    </div>
    <div class="reason-box">
      <strong>视觉证据</strong>
      <span>${escapeHtml(classifier.visualEvidence || "暂无")}</span>
    </div>
    <div class="reason-box">
      <strong>判别解释</strong>
      <span>${escapeHtml(classifier.reason || "暂无解释")}</span>
    </div>
  `;
  const correctionForm = $("#correctionForm");
  if (correctionForm) correctionForm.addEventListener("submit", saveCorrectionSample);
}

function renderCropEvidence(node) {
  const urls = state.result?.artifactUrls || {};
  const imageUrl = urls.sourceImage || urls.evidenceImage || "";
  const meta = state.result?.imageMeta || {};
  const box = node?.bbox || {};
  const imageW = Number(meta.width || 0);
  const imageH = Number(meta.height || 0);
  const x = Number(box.x || 0);
  const y = Number(box.y || 0);
  const w = Math.max(1, Number(box.w || 0));
  const h = Math.max(1, Number(box.h || 0));
  if (!imageUrl || !imageW || !imageH || !w || !h) {
    return `
      <div class="reason-box">
        <strong>图片证据</strong>
        <span>缺少原图或 bbox，无法生成裁剪预览。</span>
      </div>
    `;
  }

  const previewW = 320;
  const previewH = 180;
  const padding = 14;
  const scale = Math.min(3, Math.max(0.12, Math.min((previewW - padding * 2) / w, (previewH - padding * 2) / h)));
  const cropW = w * scale;
  const cropH = h * scale;
  const offsetX = (previewW - cropW) / 2 - x * scale;
  const offsetY = (previewH - cropH) / 2 - y * scale;
  return `
    <div class="reason-box crop-evidence">
      <strong>图片证据</strong>
      <div class="crop-evidence-frame"
        style="--crop-preview-w:${previewW}px;--crop-preview-h:${previewH}px;--crop-image-w:${imageW * scale}px;--crop-image-h:${imageH * scale}px;--crop-offset-x:${offsetX}px;--crop-offset-y:${offsetY}px;">
        <img src="${escapeAttr(imageUrl)}" alt="节点裁剪图" />
      </div>
      <span>${escapeHtml(formatBBox(box))}</span>
    </div>
  `;
}

async function saveCorrectionSample(event) {
  event.preventDefault();
  const node = selectedNode();
  if (!node || !state.result) return;
  const correctComponentId = $("#correctComponentSelect").value;
  if (!correctComponentId) {
    showToast("请选择正确组件");
    return;
  }
  try {
    const response = await fetch("/api/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runId: state.result.runId,
        nodeId: node.node_id,
        correctComponentId,
        visualForm: $("#correctVisualFormInput").value,
        note: $("#correctNoteInput").value,
      }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "保存失败");
    showToast("已保存纠错训练样本");
  } catch (error) {
    showToast(error.message);
  }
}

function renderCandidate(candidate) {
  return `
    <div class="candidate">
      <div>
        <strong>${escapeHtml(candidate.componentId || "-")}</strong>
        <small>${escapeHtml(candidate.title || candidate.categoryName || "")}</small>
      </div>
      <b>${formatScore(candidate.score)}</b>
    </div>
  `;
}

function refreshTypeFilter() {
  const filter = $("#typeFilter");
  const previous = filter.value;
  const counts = state.result?.summary?.typeCounts || {};
  const options = [`<option value="">全部类型</option>`].concat(
    Object.keys(counts).sort().map((type) => `<option value="${escapeAttr(type)}">${escapeHtml(type)} (${counts[type]})</option>`)
  );
  filter.innerHTML = options.join("");
  if (previous && counts[previous]) filter.value = previous;
}

function refreshLevelFilter() {
  const filter = $("#levelFilter");
  const previous = filter.value;
  const layers = layerStats(state.result?.nodes || []);
  const options = [`<option value="">全部层级</option>`].concat(
    layers.map((item) => `<option value="${escapeAttr(item.level)}">Level ${escapeHtml(item.level)} (${item.count})</option>`)
  );
  filter.innerHTML = options.join("");
  if (previous && layers.some((item) => String(item.level) === previous)) filter.value = previous;
}

function layerStats(nodes) {
  const map = new Map();
  (nodes || [])
    .filter((node) => node.type !== "Screen" && node.level !== undefined && node.level !== null)
    .forEach((node) => {
      const level = String(node.level);
      map.set(level, (map.get(level) || 0) + 1);
    });
  return Array.from(map.entries())
    .map(([level, count]) => ({ level, count }))
    .sort((a, b) => Number(a.level) - Number(b.level));
}

function getFilteredNodes() {
  const result = state.result;
  if (!result) return [];
  const search = $("#nodeSearch").value.trim().toLowerCase();
  const type = $("#typeFilter").value;
  const level = state.selectedLevel || $("#levelFilter").value;
  return (result.nodes || [])
    .filter((node) => node.type !== "Screen")
    .filter((node) => !level || String(node.level) === String(level))
    .filter((node) => !type || node.type === type)
    .filter((node) => {
      if (!search) return true;
      const candidate = topCandidate(node);
      const text = [
        node.node_id,
        node.type,
        node.component_id,
        candidate.componentId,
        candidate.title,
        classifierOf(node).contentType,
        classifierOf(node).llmComponentId,
      ].join(" ").toLowerCase();
      return text.includes(search);
    });
}

function selectLevel(level) {
  state.selectedLevel = level ? String(level) : "";
  $("#levelFilter").value = state.selectedLevel;
  state.selectedNodeId = null;
  hideHitPicker();
  renderMetrics();
  renderResultViews();
}

function selectNode(nodeId) {
  if (nodeId && state.selectedLevel) {
    const node = (state.result?.nodes || []).find((item) => item.node_id === nodeId);
    if (node && String(node.level) !== state.selectedLevel) {
      state.selectedLevel = "";
      $("#levelFilter").value = "";
    }
  }
  state.selectedNodeId = nodeId;
  renderMetrics();
  renderImage();
  renderComponentsTable();
  renderHierarchy();
  renderInspector();
}

function selectedNode() {
  if (!state.result || !state.selectedNodeId) return null;
  return (state.result.nodes || []).find((node) => node.node_id === state.selectedNodeId) || null;
}

function topCandidate(node) {
  return Array.isArray(node?.candidates) && node.candidates.length ? node.candidates[0] : {};
}

function classifierOf(node) {
  return node?.features?.contentClassifier || {};
}

function renderClassifierBadge(classifier) {
  const type = classifier.contentType || "-";
  const llm = classifier.llmComponentId ? ` / ${classifier.llmComponentId}` : "";
  return `<span class="pill">${escapeHtml(type + llm)}</span>`;
}

function formatBBox(box) {
  return `x:${round(box.x)} y:${round(box.y)} w:${round(box.w)} h:${round(box.h)}`;
}

function formatScore(value) {
  if (typeof value !== "number") return "-";
  return value.toFixed(3);
}

function formatPercent(value) {
  if (typeof value !== "number") return "0%";
  return Math.round(value * 100) + "%";
}

function percentage(value, total) {
  return Math.max(0, Math.min(100, (Number(value || 0) / Math.max(Number(total || 1), 1)) * 100));
}

function datasetPreview(dataset) {
  if (dataset === undefined || dataset === null) return "空";
  if (typeof dataset === "string") return dataset.slice(0, 18);
  if (Array.isArray(dataset)) return `${dataset.length} 项`;
  if (typeof dataset === "object") return Object.keys(dataset).slice(0, 3).join(", ") || "对象";
  return String(dataset);
}

function round(value) {
  const number = Number(value || 0);
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function modelSuffix(path) {
  if (!path) return "";
  return " · " + String(path).split("/").pop();
}

function setActiveButton(selector, activeButton) {
  $$(selector).forEach((button) => button.classList.toggle("active", button === activeButton));
}

function markActiveRun(runId) {
  $$(".run-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.runId === runId);
  });
}

function startProgress() {
  const band = $("#progressBand");
  const fill = $("#progressFill");
  const steps = $$(".progress-steps span");
  band.hidden = false;
  let tick = 0;
  fill.style.width = "12%";
  steps.forEach((step, index) => step.classList.toggle("active", index === 0));
  state.parseTimer = window.setInterval(() => {
    tick = Math.min(tick + 1, steps.length - 1);
    fill.style.width = `${Math.min(92, 18 + tick * 18)}%`;
    steps.forEach((step, index) => step.classList.toggle("active", index <= tick));
  }, 900);
}

function stopProgress() {
  const fill = $("#progressFill");
  if (state.parseTimer) window.clearInterval(state.parseTimer);
  state.parseTimer = null;
  fill.style.width = "100%";
  window.setTimeout(() => {
    $("#progressBand").hidden = true;
    fill.style.width = "12%";
  }, 500);
}

async function copyJson() {
  if (!state.result) {
    showToast("暂无 JSON 可复制");
    return;
  }
  const text = JSON.stringify(state.result, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    showToast("JSON 已复制");
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
    showToast("JSON 已复制");
  }
}

async function copySchemaComponents() {
  const components = state.result?.aiSchemaComponents || [];
  if (!components.length) {
    showToast("暂无可复制的组件 JSON");
    return;
  }
  const payload = {
    source: "screen-parser-mvp",
    runId: state.result?.runId,
    componentList: components,
  };
  try {
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    showToast("组件 JSON 已复制");
  } catch {
    showToast("复制失败，请从 JSON 面板查看 aiSchemaComponents");
  }
}

function restoreTheme() {
  const theme = localStorage.getItem("screen-parser-theme") || "dark";
  document.documentElement.dataset.theme = theme;
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("screen-parser-theme", next);
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}
