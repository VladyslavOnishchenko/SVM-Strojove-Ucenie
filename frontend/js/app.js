const state = {
  currentDataset: null,
  columnSchema: null,
  columnInfo: null,
  trainingResults: null,
  modelTrained: false,
  hyperparameters: { kernel: 'rbf', C: 1.0, gamma: 'scale', auto_tune: false },
};

// ── Utility ──────────────────────────────────────────────────────────────────

function showSection(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('section-hidden');
  el.classList.add('fade-in');
}

function hideSection(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('section-hidden');
}

function showError(msg) {
  const box = document.getElementById('error-box');
  const text = document.getElementById('error-text');
  text.textContent = msg;
  box.classList.remove('section-hidden');
  box.scrollIntoView({ behavior: 'smooth', block: 'center' });
  setTimeout(() => box.classList.add('section-hidden'), 6000);
}

function showLoading(btn) {
  btn.dataset.origText = btn.textContent;
  btn.classList.add('btn-loading');
  btn.disabled = true;
}

function hideLoading(btn) {
  btn.classList.remove('btn-loading');
  btn.disabled = false;
  if (btn.dataset.origText) btn.textContent = btn.dataset.origText;
}

async function apiFetch(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try { const j = await resp.json(); detail = j.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
}

// ── Sekcia 1: Výber datasetu ──────────────────────────────────────────────────

async function initExamples() {
  try {
    const examples = await apiFetch('/api/datasets/examples');
    const container = document.getElementById('example-buttons');
    container.innerHTML = '';
    const labels = {
      iris: 'Iris (150 riadkov)',
      wine: 'Wine (178 riadkov)',
      bank_marketing: 'Bank Marketing (500 riadkov)',
      heart_disease: 'Heart Disease (300 riadkov)',
    };
    examples.forEach(ds => {
      const btn = document.createElement('button');
      btn.className = 'dataset-btn';
      btn.textContent = labels[ds.name] || ds.name;
      btn.dataset.name = ds.name;
      btn.onclick = () => loadExampleDataset(ds.name);
      container.appendChild(btn);
    });
  } catch (e) {
    showError('Nepodarilo sa načítať zoznam datasetov: ' + e.message);
  }
}

async function loadExampleDataset(name) {
  const btn = [...document.querySelectorAll('.dataset-btn')].find(b => b.dataset.name === name);
  if (btn) showLoading(btn);
  try {
    const data = await apiFetch(`/api/datasets/examples/${name}/load`, { method: 'POST' });
    state.currentDataset = data;
    document.querySelectorAll('.dataset-btn').forEach(b => b.classList.remove('active'));
    if (btn) { hideLoading(btn); btn.classList.add('active'); }
    onDatasetLoaded(data);
  } catch (e) {
    if (btn) hideLoading(btn);
    showError('Chyba pri načítaní datasetu: ' + e.message);
  }
}

async function uploadCSV(file) {
  const btn = document.getElementById('btn-upload');
  showLoading(btn);
  const formData = new FormData();
  formData.append('file', file);
  try {
    const resp = await fetch('/api/datasets/upload', { method: 'POST', body: formData });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { const j = await resp.json(); detail = j.detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const json = await resp.json();
    state.currentDataset = json;
    document.querySelectorAll('.dataset-btn').forEach(b => b.classList.remove('active'));
    hideLoading(btn);
    onDatasetLoaded(json);
  } catch (e) {
    hideLoading(btn);
    showError('Chyba pri nahrávaní súboru: ' + e.message);
  }
}

function onDatasetLoaded(data) {
  document.getElementById('dataset-info').textContent =
    `Načítaný: ${data.dataset_name || 'vlastný'} — ${data.n_rows} riadkov, ${data.n_columns} stĺpcov`;
  hideSection('section-schema');
  hideSection('section-training');
  hideSection('section-results');
  hideSection('section-visualization');
  hideSection('section-prediction');
  state.modelTrained = false;
  updateModelStatus(false);
  loadSchema();
}

// ── Sekcia 2: Typy stĺpcov ────────────────────────────────────────────────────

async function loadSchema() {
  try {
    const schema = await apiFetch('/api/datasets/current/schema');
    state.columnInfo = schema.columns;
    renderSchemaTable(schema.columns);
    showSection('section-schema');
    document.getElementById('section-schema').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    showError('Chyba pri načítaní schémy: ' + e.message);
  }
}

function renderSchemaTable(columns) {
  const tbody = document.getElementById('schema-tbody');
  tbody.innerHTML = '';
  columns.forEach(col => {
    const tr = document.createElement('tr');
    if (col.suggested_type === 'target') tr.classList.add('target-row');
    tr.innerHTML = `
      <td class="col-name">${col.name}</td>
      <td class="col-unique">${col.unique_count}</td>
      <td><span class="type-badge ${col.suggested_type}">${col.suggested_type}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function collectSchema() {
  const schema = {};
  if (state.columnInfo) {
    state.columnInfo.forEach(col => { schema[col.name] = col.suggested_type; });
  }
  return schema;
}

function onSchemaConfirmed() {
  state.columnSchema = collectSchema();
  showSection('section-training');
  document.getElementById('section-training').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Sekcia 3: Trénovanie ──────────────────────────────────────────────────────

function initTrainingControls() {
  document.getElementById('ctrl-kernel').addEventListener('change', e => {
    state.hyperparameters.kernel = e.target.value;
  });
  document.getElementById('ctrl-C').addEventListener('input', e => {
    state.hyperparameters.C = parseFloat(e.target.value) || 1.0;
    document.getElementById('lbl-C').textContent = e.target.value;
  });
  document.getElementById('ctrl-gamma').addEventListener('change', e => {
    state.hyperparameters.gamma = e.target.value;
  });
  document.getElementById('ctrl-autotune').addEventListener('change', e => {
    state.hyperparameters.auto_tune = e.target.checked;
    document.getElementById('manual-params').style.opacity = e.target.checked ? '0.4' : '1';
  });
}

async function trainModel() {
  const btn = document.getElementById('btn-train');
  showLoading(btn);
  hideSection('section-results');
  hideSection('section-visualization');
  hideSection('section-prediction');

  const schema = state.columnSchema || collectSchema();
  const payload = {
    column_schema: schema,
    hyperparameters: { ...state.hyperparameters },
  };

  try {
    const results = await apiFetch('/api/train/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.trainingResults = results;
    state.modelTrained = true;
    hideLoading(btn);
    btn.textContent = 'Trénovať znovu';
    updateModelStatus(true);
    renderResults(results);
    showSection('section-results');
    document.getElementById('section-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    hideLoading(btn);
    showError('Chyba pri trénovaní: ' + e.message);
  }
}

// ── Sekcia 4: Výsledky ────────────────────────────────────────────────────────

function renderResults(r) {
  document.getElementById('res-accuracy').textContent = (r.accuracy * 100).toFixed(1) + ' %';

  document.getElementById('res-time').textContent = r.training_time_seconds != null
    ? r.training_time_seconds.toFixed(2) + ' s' : '—';

  const cvEl = document.getElementById('res-cv');
  if (r.cv_mean_accuracy != null) {
    const std = r.cv_std_accuracy != null ? ` ± ${(r.cv_std_accuracy * 100).toFixed(1)}` : '';
    cvEl.textContent = (r.cv_mean_accuracy * 100).toFixed(1) + std + ' %';
  } else {
    cvEl.textContent = '—';
  }

  renderPerClassMetrics(r);
  renderConfusionMatrix(r);
}

function renderPerClassMetrics(r) {
  const container = document.getElementById('per-class-metrics');
  container.innerHTML = '';
  if (!r.per_class_metrics) return;
  Object.entries(r.per_class_metrics).forEach(([cls, m]) => {
    const card = document.createElement('div');
    card.className = 'per-class-card';
    card.innerHTML = `
      <div class="per-class-title">${cls}</div>
      <div class="grid grid-cols-3 gap-2 text-center">
        <div>
          <div class="per-class-value metric-precision">${(m.precision * 100).toFixed(0)}</div>
          <div class="per-class-label">Precision</div>
        </div>
        <div>
          <div class="per-class-value metric-recall">${(m.recall * 100).toFixed(0)}</div>
          <div class="per-class-label">Recall</div>
        </div>
        <div>
          <div class="per-class-value metric-f1">${(m.f1_score * 100).toFixed(0)}</div>
          <div class="per-class-label">F1</div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderConfusionMatrix(r) {
  const container = document.getElementById('confusion-matrix');
  container.innerHTML = '';
  if (!r.confusion_matrix || !r.classes) return;

  const mat = r.confusion_matrix;
  const cls = r.classes;
  const maxVal = Math.max(...mat.flat());

  const table = document.createElement('table');
  table.className = 'cm-table';

  const thead = document.createElement('thead');
  const hrow = document.createElement('tr');
  hrow.innerHTML = '<th class="py-2 px-3 text-left text-xs" style="color:var(--color-text-muted)">Skut. \\ Pred.</th>' +
    cls.map(c => `<th>${c}</th>`).join('');
  thead.appendChild(hrow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  mat.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<th class="text-right pr-2 text-xs" style="color:var(--color-text-muted)">${cls[i]}</th>` +
      row.map((val, j) => {
        const ratio = maxVal > 0 ? val / maxVal : 0;
        const cellCls = ratio > 0.6 ? 'cm-cell-high' : ratio > 0.2 ? 'cm-cell-mid' : 'cm-cell-low';
        const bold = i === j ? ' font-bold' : '';
        return `<td class="${cellCls}${bold}">${val}</td>`;
      }).join('');
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);

  showSection('section-visualization');
  showSection('section-prediction');
  buildPredictionForm();
}

// ── Sekcia 5: Vizualizácia ────────────────────────────────────────────────────

async function loadVisualization() {
  const btn = document.getElementById('btn-viz');
  showLoading(btn);
  document.getElementById('viz-container').innerHTML =
    '<p class="text-sm text-center py-8" style="color:var(--color-text-muted)">Vypočítavam vizualizáciu...</p>';
  try {
    const data = await apiFetch('/api/model/visualization');
    hideLoading(btn);
    btn.textContent = 'Obnoviť vizualizáciu';
    renderVisualization(data);
  } catch (e) {
    hideLoading(btn);
    showError('Chyba pri generovaní vizualizácie: ' + e.message);
    document.getElementById('viz-container').innerHTML =
      '<p class="text-sm text-center py-8" style="color:var(--color-danger)">Vizualizácia zlyhala.</p>';
  }
}

function renderVisualization(data) {
  const container = document.getElementById('viz-container');
  container.innerHTML = '';

  const classes = data.classes;
  const palette = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'];

  const traces = [];

  if (data.grid && data.grid.predictions) {
    const { x_range, y_range, resolution, predictions } = data.grid;
    const n = classes.length;

    const colorscale = [];
    classes.forEach((_, i) => {
      colorscale.push([i / n, palette[i % palette.length]]);
      colorscale.push([(i + 1) / n - 0.001, palette[i % palette.length]]);
    });
    colorscale[colorscale.length - 1][0] = 1.0;

    const xVals = Array.from({ length: resolution }, (_, i) =>
      x_range[0] + (x_range[1] - x_range[0]) * i / (resolution - 1)
    );
    const yVals = Array.from({ length: resolution }, (_, i) =>
      y_range[0] + (y_range[1] - y_range[0]) * i / (resolution - 1)
    );

    traces.push({
      type: 'heatmap',
      x: xVals,
      y: yVals,
      z: predictions,
      colorscale,
      showscale: false,
      opacity: 0.3,
      hoverinfo: 'skip',
    });
  }

  classes.forEach((cls, i) => {
    const pts = data.points.filter(p => p.class === cls);
    traces.push({
      type: 'scatter',
      mode: 'markers',
      name: cls,
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      marker: {
        color: palette[i % palette.length],
        size: 7,
        line: { width: 0.5, color: '#fff' },
      },
    });
  });

  const layout = {
    title: { text: '2D PCA projekcia + rozhodovacie hranice', font: { size: 14, color: '#94a3b8' } },
    xaxis: { title: 'PC 1', zeroline: false, gridcolor: 'rgba(99,102,241,0.1)', zerolinecolor: 'rgba(99,102,241,0.2)', color: '#94a3b8' },
    yaxis: { title: 'PC 2', zeroline: false, gridcolor: 'rgba(99,102,241,0.1)', zerolinecolor: 'rgba(99,102,241,0.2)', color: '#94a3b8' },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    legend: { orientation: 'h', y: -0.15, font: { color: '#94a3b8' } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(13,13,26,0.8)',
    font: { color: '#94a3b8' },
  };

  Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
}

// ── Sekcia 6: Predikcia ───────────────────────────────────────────────────────

function buildPredictionForm() {
  if (!state.columnSchema) return;
  const form = document.getElementById('prediction-form');
  form.innerHTML = '';

  const colInfoMap = {};
  if (state.columnInfo) {
    state.columnInfo.forEach(c => { colInfoMap[c.name] = c; });
  }

  Object.entries(state.columnSchema)
    .filter(([, t]) => t !== 'target' && t !== 'ignore')
    .forEach(([col, type]) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'flex flex-col';

      const label = document.createElement('label');
      label.className = 'pred-field-label';
      label.textContent = col;

      const colInfo = colInfoMap[col];
      const sampleVal = colInfo && colInfo.sample_values && colInfo.sample_values.length > 0
        ? colInfo.sample_values[0] : null;

      let input;
      if (type === 'numeric') {
        input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.placeholder = '0.0';
        input.className = 'pred-input-field pred-input';
        if (sampleVal !== null) input.value = parseFloat(sampleVal);
      } else {
        // binary and categorical: send actual string values the model was trained on
        input = document.createElement('input');
        input.type = 'text';
        input.placeholder = type === 'binary' ? '0 / 1' : 'text';
        input.className = 'pred-input-field pred-input';
        if (sampleVal !== null) input.value = sampleVal;
      }
      input.dataset.col = col;
      input.dataset.coltype = type;

      wrapper.appendChild(label);
      wrapper.appendChild(input);
      form.appendChild(wrapper);
    });
}

async function predict() {
  const btn = document.getElementById('btn-predict');
  showLoading(btn);
  document.getElementById('prediction-result').classList.add('section-hidden');

  const inputData = {};
  let valid = true;
  document.querySelectorAll('.pred-input').forEach(inp => {
    const col = inp.dataset.col;
    const coltype = inp.dataset.coltype;
    const raw = inp.value.trim();
    if (raw === '') { valid = false; return; }
    if (coltype === 'numeric') {
      const n = parseFloat(raw);
      if (isNaN(n)) { valid = false; return; }
      inputData[col] = n;
    } else {
      // binary and categorical: send the original string the model was trained on
      inputData[col] = raw;
    }
  });

  if (!valid) {
    hideLoading(btn);
    showError('Vyplňte prosím všetky vstupné polia.');
    return;
  }

  try {
    const result = await apiFetch('/api/predict/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input_data: inputData }),
    });
    hideLoading(btn);
    renderPrediction(result);
  } catch (e) {
    hideLoading(btn);
    showError('Chyba pri predikcii: ' + e.message);
  }
}

function renderPrediction(result) {
  const box = document.getElementById('prediction-result');
  document.getElementById('pred-class').textContent = result.predicted_class;

  const probContainer = document.getElementById('pred-probabilities');
  probContainer.innerHTML = '';
  if (result.probabilities) {
    const sorted = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([cls, prob], idx) => {
      const pct = (prob * 100).toFixed(1);
      const isTop = idx === 0;
      const barStyle = isTop
        ? 'background: linear-gradient(90deg, var(--color-primary), var(--color-accent));'
        : 'background: #d1d5db;';
      const row = document.createElement('div');
      row.className = 'flex items-center gap-3';
      row.innerHTML = `
        <span class="pred-prob-label">${cls}</span>
        <div class="pred-prob-track">
          <div class="pred-prob-bar" style="width:${pct}%; ${barStyle}"></div>
        </div>
        <span class="pred-prob-pct">${pct} %</span>
      `;
      probContainer.appendChild(row);
    });
  }
  box.classList.remove('section-hidden');
  box.classList.add('fade-in');
}

// ── Status indikátor ──────────────────────────────────────────────────────────

function updateModelStatus(_trained) {
  // Status pill hidden via CSS — nothing to update.
}

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  await initExamples();
  document.getElementById('ctrl-autotune').checked = false;
  initTrainingControls();

  document.getElementById('upload-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) uploadCSV(file);
  });
  document.getElementById('btn-upload').addEventListener('click', () => {
    document.getElementById('upload-input').click();
  });

  document.getElementById('btn-confirm-schema').addEventListener('click', onSchemaConfirmed);
  document.getElementById('btn-train').addEventListener('click', trainModel);
  document.getElementById('btn-viz').addEventListener('click', loadVisualization);
  document.getElementById('btn-predict').addEventListener('click', predict);

  try {
    const status = await apiFetch('/api/model/status');
    if (status.is_trained) {
      state.modelTrained = true;
      updateModelStatus(true);
    }
  } catch (_) {}
});
