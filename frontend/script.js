const searchInput = document.getElementById('search');
const playerListEl = document.getElementById('playerList');
const selectedChips = document.getElementById('selectedChips');
const compareBtn = document.getElementById('compareBtn');
const errorEl = document.getElementById('error');
const reasoningEl = document.getElementById('reasoning');
const cardA = document.getElementById('cardA');
const cardB = document.getElementById('cardB');
const heroAName = document.getElementById('heroAName');
const heroBName = document.getElementById('heroBName');
const heroAVal = document.getElementById('heroAVal');
const heroBVal = document.getElementById('heroBVal');
const heroDelta = document.getElementById('heroDelta');
const heroDeltaLabel = document.getElementById('heroDeltaLabel');
const heroCardA = document.getElementById('heroCardA');
const heroCardB = document.getElementById('heroCardB');
const versCardA = document.getElementById('versCardA');
const versCardB = document.getElementById('versCardB');
const modelNameEl = document.getElementById('modelName');
const modelMaeEl = document.getElementById('modelMae');
const modelRmseEl = document.getElementById('modelRmse');
const modelR2El = document.getElementById('modelR2');
const modelDetailsEl = document.getElementById('modelDetails');
const insightPills = document.getElementById('insightPills');
const driversChartEl = document.getElementById('driversChart');
const extraComparisons = document.getElementById('extraComparisons');
const similarBtn = document.getElementById('similarBtn');
const similarError = document.getElementById('similarError');
const similarGrid = document.getElementById('similarGrid');
const causalBody = document.getElementById('causalBody');
const networkEl = document.getElementById('network');
const radarEl = document.getElementById('radar3d');
const whatIfContainer = document.getElementById('whatIfContainer');
const rootStyle = getComputedStyle(document.documentElement);
const cssVar = (name) => rootStyle.getPropertyValue(name).trim() || name;
const anomKind = document.getElementById('anomKind');
const anomPos = document.getElementById('anomPos');
const anomLeague = document.getElementById('anomLeague');
const anomAge = document.getElementById('anomAge');
const anomBtn = document.getElementById('anomBtn');
const anomError = document.getElementById('anomError');
const anomList = document.getElementById('anomList');
const anomScatter = document.getElementById('anomScatter');
const riskChartEl = document.getElementById('riskChart');
const compChartEl = document.getElementById('compChart');

let players = [];
let playerMap = new Map();
let selectedIds = [];

function optionLabel(p) {
  return `${p.Name} (OVR ${p.OVR}, ${p.Position})`;
}

async function runSingleAnalysis(id) {
  // Use predict endpoint for single view
  const data = await fetchJson(`/api/predict?ids=${id}`);
  const p = (data.players || [])[0];
  if (!p) throw new Error('Prediction unavailable.');
  heroAName.textContent = p.name;
  heroBName.textContent = '—';
  heroAVal.textContent = `OVR ${p.predicted_ovr.toFixed(2)}`;
  heroBVal.textContent = 'OVR —';
  heroDelta.textContent = '—';
  heroDeltaLabel.textContent = 'Single player view';
  heroCardA.classList.remove('winner', 'loser');
  heroCardB.classList.remove('winner', 'loser');
  reasoningEl.textContent = 'Add another player to see deltas and SHAP drivers.';
  renderInsightPills([]);
  renderDriversChart([], 0);
  renderRadar([], p.name, '');
  renderWhatIfControls([], null, null, null);
  fetchQuantiles(p.id);
  renderComposition([], p.name);
  renderRiskProfile(null); // Clear or showing loading
  extraComparisons.innerHTML = '';
  extraComparisons.innerHTML = '';
  renderVersatilityFromPlayers(p, null);
  // update previews
  const pa = playerMap.get(String(id));
  if (pa && p.card) pa.card = p.card;
  renderPreview();
}

async function runMultiAnalysis(ids) {
  if (ids.length < 2) return;
  const anchor = ids[0];
  const target = ids[1];
  const data = await fetchJson(`/api/compare?player_a_id=${anchor}&player_b_id=${target}`);
  heroAVal.textContent = `OVR ${data.player_a.predicted_ovr.toFixed(2)}`;
  heroBVal.textContent = `OVR ${data.player_b.predicted_ovr.toFixed(2)}`;
  heroAName.textContent = data.player_a.name;
  heroBName.textContent = data.player_b.name;
  const delta = data.ovr_difference;
  heroDelta.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}`;
  heroDelta.style.color = delta >= 0 ? cssVar('--positive') : cssVar('--negative');
  heroDeltaLabel.textContent = delta >= 0 ? 'Anchor leads' : 'Target leads';
  heroCardA.classList.remove('winner', 'loser');
  heroCardB.classList.remove('winner', 'loser');
  if (delta > 0.05) { heroCardA.classList.add('winner'); heroCardB.classList.add('loser'); }
  else if (delta < -0.05) { heroCardB.classList.add('winner'); heroCardA.classList.add('loser'); }
  reasoningEl.textContent = data.natural_language_summary;
  renderInsightPills(data.top_reasons || []);
  renderDriversChart(data.top_reasons || [], delta);
  const pa = playerMap.get(anchor);
  const pb = playerMap.get(target);
  if (data.player_a.card && pa) pa.card = data.player_a.card;
  if (data.player_b.card && pb) pb.card = data.player_b.card;
  renderPreview();
  renderRadar(data.top_reasons, data.player_a.name, data.player_b.name);
  renderWhatIfControls(data.top_reasons, data.player_b.id || target, data.player_b.predicted_ovr, data.player_a.predicted_ovr);
  fetchQuantiles(data.player_b.id || target);
  renderComposition(data.top_reasons, data.player_a.name);
  renderVersatilityFromPlayers(data.player_a, data.player_b);

  // extra comparisons for 3rd/4th against anchor
  extraComparisons.innerHTML = '';
  const others = ids.slice(2);
  for (const oid of others) {
    try {
      const sub = await fetchJson(`/api/compare?player_a_id=${anchor}&player_b_id=${oid}`);
      const card = document.createElement('div');
      card.style.border = `1px solid ${cssVar('--panel-border')}`;
      card.style.background = 'rgba(255,255,255,0.05)';
      card.style.borderRadius = '12px';
      card.style.padding = '10px';
      const gap = sub.ovr_difference;
      card.innerHTML = `
        <div style="font-weight:700;">${sub.player_b.name}</div>
        <div class="muted" style="font-size:12px;">Δ vs anchor: ${gap >= 0 ? '+' : ''}${gap.toFixed(2)}</div>
        <div class="chip-group" style="margin-top:6px; display:flex; gap:6px;">
          <span class="pill">OVR ${sub.player_b.predicted_ovr.toFixed(2)}</span>
          <span class="pill">Pos ${sub.player_b.position || '—'}</span>
        </div>
      `;
      extraComparisons.appendChild(card);
    } catch (err) {
      // skip silently
    }
  }
}

function renderSelectedChips() {
  if (!selectedChips) return;
  selectedChips.innerHTML = '';
  selectedIds.forEach(id => {
    const p = playerMap.get(String(id));
    const chip = document.createElement('span');
    chip.className = 'pill active';
    chip.style.cursor = 'pointer';
    chip.textContent = p ? `${p.Name} · ${p.Position}` : id;
    chip.onclick = () => toggleSelect(id);
    selectedChips.appendChild(chip);
  });
}

function renderPlayerList(list) {
  if (!playerListEl) return;
  playerListEl.innerHTML = '';
  list.forEach(p => {
    const isSelected = selectedIds.includes(String(p.ID));
    const disabled = !isSelected && selectedIds.length >= 4;
    const card = document.createElement('div');
    card.className = `player-option ${isSelected ? 'selected' : ''}`;
    if (disabled) card.style.opacity = '0.5';

    card.onclick = () => {
      if (disabled) return;
      toggleSelect(p.ID);
    };
    card.innerHTML = `
      <div>
        <div style="font-weight:700;">${p.Name}</div>
        <div class="muted small">${p.Position} • OVR ${p.OVR} • ${p.Team || ''}</div>
      </div>
      <div>
        ${isSelected ? '<span style="color:var(--accent-cyan)">✔</span>' : ''}
      </div>
    `;
    playerListEl.appendChild(card);
  });
  renderSelectedChips();
}

function toggleSelect(id) {
  const sid = String(id);
  if (selectedIds.includes(sid)) {
    selectedIds = selectedIds.filter(x => x !== sid);
  } else {
    if (selectedIds.length >= 4) return;
    selectedIds.push(sid);
  }
  renderSelectedChips();
  renderPlayerList(playersFiltered());
  renderPreview();
}

function playersFiltered(term = searchInput?.value || '') {
  const t = term.toLowerCase();
  return players.filter(p => p.Name.toLowerCase().includes(t));
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (e) {
    // leave data null to surface text/error
  }
  if (!res.ok || (data && data.error)) {
    throw new Error((data && data.error) || text || 'Request failed');
  }
  return data || {};
}

function renderModelMeta(data) {
  if (!modelNameEl) return;
  if (!data || data.error) {
    modelNameEl.textContent = 'Model unavailable';
    modelDetailsEl.textContent = data && data.error ? data.error : 'Train a model to populate.';
    modelMaeEl.textContent = '—';
    modelRmseEl.textContent = '—';
    modelR2El.textContent = '—';
    return;
  }
  modelNameEl.textContent = data.best_model || 'Unknown';
  modelMaeEl.textContent = data.mae !== undefined ? Number(data.mae).toFixed(3) : '—';
  modelRmseEl.textContent = data.rmse !== undefined ? Number(data.rmse).toFixed(3) : '—';
  modelR2El.textContent = data.r2 !== undefined ? Number(data.r2).toFixed(3) : '—';
  const trialInfo = data.trials ? `${data.trials} trials` : 'sweep ready';
  modelDetailsEl.textContent = `Auto-selected via boosted sweeps (${trialInfo}).`;
}

async function loadModelMeta() {
  try {
    const data = await fetchJson('/api/model_meta');
    renderModelMeta(data);
  } catch (err) {
    renderModelMeta({ error: err.message });
  }
}

async function loadPlayers(search = '') {
  const params = new URLSearchParams({ limit: 500 });
  if (search) params.append('search', search);
  const data = await fetchJson(`/api/players?${params.toString()}`);
  players = data.players || [];
  playerMap = new Map(players.map(p => [String(p.ID), p]));
  renderPlayerList(players);
  renderPreview();
}

function filterPlayers(term) {
  const filtered = playersFiltered(term);
  renderPlayerList(filtered.length ? filtered : players);
}

function renderPreview() {
  const pa = playerMap.get(selectedIds[0]);
  const pb = playerMap.get(selectedIds[1]);
  const setCard = (elImg, nameEl, valEl, p) => {
    if (!p) { elImg.src = ''; elImg.alt = ''; nameEl.textContent = '—'; valEl.textContent = 'OVR —'; return; }
    elImg.src = p.card || '';
    elImg.alt = p.Name;
    nameEl.textContent = p.Name;
    valEl.textContent = `OVR ${p.OVR ?? '—'}`;
  };
  setCard(cardA, heroAName, heroAVal, pa);
  setCard(cardB, heroBName, heroBVal, pb);
}

async function compare() {
  errorEl.textContent = '';
  const ids = selectedIds;
  if (!ids.length) {
    errorEl.textContent = 'Select at least one player.';
    return;
  }
  compareBtn.disabled = true;
  compareBtn.textContent = 'Analyzing...';
  try {
    if (ids.length === 1) {
      await runSingleAnalysis(ids[0]);
    } else {
      await runMultiAnalysis(ids);
    }
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    compareBtn.disabled = false;
    compareBtn.textContent = 'Analyze Selection';
  }
}

function renderInsightPills(topReasons) {
  if (!insightPills) return;
  insightPills.innerHTML = '';
  if (!topReasons || !topReasons.length) {
    insightPills.innerHTML = '<span class="muted">No drivers yet.</span>';
    return;
  }
  const sorted = [...topReasons].filter(r => !isNaN(r.impact)).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 6);
  sorted.forEach(r => {
    const pill = document.createElement('span');
    const pos = r.impact >= 0;
    pill.className = `pill ${pos ? 'positive' : 'negative'}`;
    const pct = (r.percent_of_delta !== null && r.percent_of_delta !== undefined && !isNaN(r.percent_of_delta)) ? `${r.percent_of_delta.toFixed(0)}%` : '';
    pill.textContent = `${r.feature} ${pos ? '+' : ''}${r.impact.toFixed(2)}${pct ? ` (${pct})` : ''}`;
    insightPills.appendChild(pill);
  });
}

function renderDriversChart(topReasons, delta) {
  if (!driversChartEl || !window.echarts) return;
  const chart = echarts.init(driversChartEl);

  if (!topReasons || !topReasons.length) {
    chart.clear();
    return;
  }

  const features = topReasons.slice(0, 10);
  const names = features.map(r => r.feature);
  const impacts = features.map(r => Number(r.impact) || 0);
  const statsDelta = features.map(r => r.stat_delta);

  chart.setOption({
    animationDuration: 1000,
    animationEasing: 'cubicOut',
    textStyle: { fontFamily: 'Inter' },
    backgroundColor: 'transparent',
    grid: { left: 10, right: 40, top: 10, bottom: 20, containLabel: true },
    xAxis: {
      type: 'value',
      axisLabel: { color: cssVar('--text-secondary'), fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'category',
      data: names,
      axisLabel: { color: cssVar('--text-primary'), fontWeight: 600, fontSize: 12 },
      axisLine: { show: false },
      axisTick: { show: false },
      inverse: true
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(20,20,20,0.95)',
      borderColor: cssVar('--panel-border'),
      textStyle: { color: '#fff' },
      formatter: (params) => {
        const p = params[0];
        const idx = p.dataIndex;
        const stat = statsDelta[idx];
        const pct = (features[idx].percent_of_delta !== undefined && !isNaN(features[idx].percent_of_delta))
          ? `${features[idx].percent_of_delta.toFixed(0)}%`
          : '—';
        return `
          <div style="font-weight:700; margin-bottom:4px;">${p.name}</div>
          <div style="font-size:12px; color:#a1a1aa;">
            Impact: <span style="color:${p.value > 0 ? cssVar('--positive') : cssVar('--negative')}">${p.value > 0 ? '+' : ''}${p.value.toFixed(3)}</span><br/>
            Stat Delta: <span style="color:#fff;">${stat ?? '—'}</span><br/>
            Contrib: <span style="color:#fff;">${pct}</span>
          </div>`;
      }
    },
    series: [{
      type: 'bar',
      data: impacts.map(v => ({
        value: v,
        itemStyle: {
          color: v >= 0
            ? new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: 'rgba(34, 211, 238, 0.6)' }, { offset: 1, color: 'rgba(34, 211, 238, 1)' }])
            : new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: 'rgba(251, 113, 133, 0.6)' }, { offset: 1, color: 'rgba(251, 113, 133, 1)' }]),
          shadowBlur: 10,
          shadowColor: v >= 0 ? 'rgba(34, 211, 238, 0.3)' : 'rgba(251, 113, 133, 0.3)'
        }
      })),
      label: {
        show: true,
        position: v => v.value >= 0 ? 'right' : 'left',
        formatter: ({ value }) => value.toFixed(2),
        color: cssVar('--text-primary'),
        fontWeight: 600,
        distance: 10
      },
      barWidth: 16,
      itemStyle: { borderRadius: 4 },
      showBackground: true,
      backgroundStyle: { color: 'rgba(255,255,255,0.02)', borderRadius: 4 }
    }]
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderVersatilityFromPlayers(playerA, playerB) {
  const renderCard = (el, label, player) => {
    if (!el) return;
    if (!player || !player.versatility) {
      el.innerHTML = `<div class="tiny muted" style="margin-bottom:4px;">${label}</div><div class="muted small">Select a player to see versatility.</div>`;
      return;
    }
    const v = player.versatility;
    const altList = v.alt_positions && v.alt_positions.length ? v.alt_positions.join(', ') : 'None';
    const altGroups = v.alt_groups && v.alt_groups.length ? v.alt_groups.join(', ') : 'None';
    const dist = (v.role_group_distance !== undefined && v.role_group_distance !== null)
      ? Number(v.role_group_distance).toFixed(1)
      : '—';
    el.innerHTML = `
      <div class="tiny muted" style="margin-bottom:4px;">${label}</div>
      <div class="vers-title">${player.name || player.Name || '—'}</div>
      <div class="vers-meta">
        <span class="pill ${v.has_alt_role ? 'positive' : ''}">${v.has_alt_role ? 'Has alt role' : 'Single role'}</span>
        <span class="pill">Alt count: ${v.alt_position_count ?? 0}</span>
        <span class="pill">Group dist: ${dist}</span>
      </div>
      <div class="muted small">Primary group: ${v.primary_group || '—'} • Alt groups: ${altGroups}</div>
      <div class="muted small">Alt positions: ${altList}</div>
    `;
  };
  renderCard(versCardA, 'Anchor', playerA);
  renderCard(versCardB, 'Target', playerB);
}

async function fetchSimilar() {
  similarError.textContent = '';
  similarGrid.innerHTML = '';
  const a = selectedIds[0];
  if (!a) {
    similarError.textContent = 'Pick an anchor player first.';
    return;
  }
  similarBtn.disabled = true;
  similarBtn.textContent = 'Searching...';
  try {
    const data = await fetchJson(`/api/similar?player_id=${a}&k=6`);
    lastSimilar = data.results || [];
    lastPlayerA = playerMap.get(a);
    (data.results || []).forEach(r => {
      const card = document.createElement('div');
      card.style.border = `1px solid ${cssVar('--panel-border')}`;
      card.style.background = 'rgba(255,255,255,0.03)';
      card.style.borderRadius = '12px';
      card.style.padding = '10px';
      card.style.display = 'flex';
      card.style.gap = '12px';
      card.style.alignItems = 'center';

      const img = document.createElement('img');
      img.src = r.card || '';
      img.alt = r.name || '';
      img.style.width = '64px';
      img.style.height = '88px';
      img.style.objectFit = 'contain';
      img.style.borderRadius = '4px';
      img.referrerPolicy = 'no-referrer';

      const info = document.createElement('div');
      info.innerHTML = `
          <div style="font-weight:700; color:var(--text-primary)">${r.name || 'Unknown'}</div>
          <div class="muted small">${r.team || ''} • ${r.league || ''}</div>
          <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:6px;">
            <span class="pill">POS ${r.position || '—'}</span>
            <span class="pill">OVR ${r.ovr ?? '—'}</span>
            <span class="pill active">Sim ${r.similarity !== undefined ? r.similarity.toFixed(3) : '—'}</span>
          </div>
        `;

      card.appendChild(img);
      card.appendChild(info);
      similarGrid.appendChild(card);
    });
    if (!similarGrid.children.length) {
      similarError.textContent = 'No similar players found.';
    }
    renderNetwork(lastSimilar, lastPlayerA);
  } catch (err) {
    similarError.textContent = err.message;
  } finally {
    similarBtn.disabled = false;
    similarBtn.textContent = 'Find Similars';
  }
}

async function fetchAnomalies() {
  if (!anomList || !anomScatter) return;
  anomError.textContent = '';
  anomList.innerHTML = '';
  const params = new URLSearchParams();
  params.append('kind', anomKind.value);
  if (anomPos.value) params.append('position', anomPos.value.toUpperCase());
  if (anomLeague.value) params.append('league', anomLeague.value);
  if (anomAge.value) params.append('max_age', anomAge.value);
  params.append('limit', 20);
  try {
    const data = await fetchJson(`/api/anomalies?${params.toString()}`);
    const results = data.results || [];
    results.forEach(r => {
      const div = document.createElement('div');
      div.className = 'card';
      div.style.padding = '10px';
      div.innerHTML = `
        <div style="font-weight:700; color: white;">${r.name}</div>
        <div class="muted small">${r.league || ''} • ${r.position || ''} • Age ${r.age ?? ''}</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;">
          <span class="pill">OVR ${r.ovr && r.ovr.toFixed ? r.ovr.toFixed(1) : r.ovr}</span>
          <span class="pill">Pred ${r.predicted && r.predicted.toFixed ? r.predicted.toFixed(1) : r.predicted}</span>
          <span class="pill ${r.residual >= 0 ? 'positive' : 'negative'}">Res ${r.residual >= 0 ? '+' : ''}${r.residual && r.residual.toFixed ? r.residual.toFixed(2) : r.residual}</span>
        </div>
      `;
      anomList.appendChild(div);
    });
    if (results.length === 0) {
      anomError.textContent = 'No anomalies found for these filters.';
    }
    renderAnomalyScatter(results);
  } catch (err) {
    anomError.textContent = err.message;
  }
}

function renderAnomalyScatter(data) {
  if (!anomScatter || !window.echarts) return;
  const chart = echarts.init(anomScatter);

  if (!data || !data.length) {
    chart.clear();
    anomScatter.innerHTML = '<div class="muted" style="height:100%; display:flex; align-items:center; justify-content:center;">No data available.</div>';
    return;
  }

  const formattedData = data.map(d => ({
    value: [d.predicted, d.ovr],
    name: d.name,
    residual: d.residual || (d.ovr - d.predicted),
    ...d
  }));

  const minVal = Math.min(...formattedData.map(d => Math.min(d.value[0], d.value[1]))) - 2;
  const maxVal = Math.max(...formattedData.map(d => Math.max(d.value[0], d.value[1]))) + 2;

  chart.setOption({
    animationDuration: 1000,
    textStyle: { fontFamily: 'Inter' },
    backgroundColor: 'transparent',
    grid: { left: 40, right: 30, top: 20, bottom: 40 },
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(20,20,20,0.95)',
      borderColor: '#333',
      textStyle: { color: '#fff' },
      formatter: (p) => {
        const d = p.data;
        const res = d.residual;
        return `
             <div style="font-weight:700; margin-bottom:4px;">${d.name}</div>
             <div class="small" style="color:#cbd5e1;">Predicted: <b style="color:#fff">${d.value[0].toFixed(1)}</b></div>
             <div class="small" style="color:#cbd5e1;">Actual: <b style="color:#fff">${d.value[1].toFixed(1)}</b></div>
             <div class="small" style="margin-top:4px; color:${res >= 0 ? cssVar('--positive') : cssVar('--negative')}">
                Residual: ${res > 0 ? '+' : ''}${res.toFixed(2)}
             </div>
            `;
      }
    },
    xAxis: {
      name: 'Predicted OVR',
      nameLocation: 'middle',
      nameGap: 24,
      min: minVal,
      max: maxVal,
      axisLabel: { color: cssVar('--text-secondary') },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
      axisLine: { lineStyle: { color: '#52525b' } }
    },
    yAxis: {
      name: 'Actual OVR',
      nameLocation: 'middle',
      nameGap: 24,
      min: minVal,
      max: maxVal,
      axisLabel: { color: cssVar('--text-secondary') },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
      axisLine: { lineStyle: { color: '#52525b' } }
    },
    series: [
      {
        type: 'line',
        data: [[minVal, minVal], [maxVal, maxVal]],
        showSymbol: false,
        lineStyle: { width: 1, type: 'dashed', color: 'rgba(255,255,255,0.3)' },
        silent: true,
        z: 1
      },
      {
        type: 'scatter',
        data: formattedData,
        symbolSize: 12,
        itemStyle: {
          color: (params) => {
            const r = params.data.residual;
            // Strong Red for < -2, Neutral for 0, Strong Green for > 2
            if (r > 1) return '#34d399'; // Green
            if (r < -1) return '#f43f5e'; // Red
            return '#94a3b8'; // Slate/Gray
          },
          shadowBlur: 8,
          shadowColor: 'rgba(0,0,0,0.4)',
          borderColor: 'rgba(255,255,255,0.2)',
          borderWidth: 1
        },
        emphasis: {
          focus: 'self',
          scale: true,
          itemStyle: {
            shadowBlur: 20,
            borderColor: '#fff',
            borderWidth: 2
          }
        },
        z: 2
      },
    ],
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderWhatIfControls(topReasons, playerBId, basePredB, basePredA) {
  if (!whatIfContainer) return;
  whatIfContainer.innerHTML = '';
  if (!topReasons || !topReasons.length) {
    whatIfContainer.textContent = 'No drivers available for what-if sliders.';
    return;
  }
  const numericReasons = topReasons.filter(r => !isNaN(r.value_b));
  if (!numericReasons.length) {
    whatIfContainer.textContent = 'Top drivers are non-numeric for what-if.';
    return;
  }
  const header = document.createElement('div');
  header.className = 'muted small';
  header.style.marginBottom = '12px';
  header.textContent = 'Adjust Player B stats (Δ updates predicted OVR and gap vs Player A).';
  whatIfContainer.appendChild(header);

  const sliderWrap = document.createElement('div');
  sliderWrap.className = 'grid';
  sliderWrap.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
  whatIfContainer.appendChild(sliderWrap);

  const state = {};
  numericReasons.slice(0, 8).forEach(r => {
    const block = document.createElement('div');
    block.style.background = 'rgba(255,255,255,0.03)';
    block.style.border = `1px solid ${cssVar('--panel-border')}`;
    block.style.borderRadius = '8px';
    block.style.padding = '12px';

    const label = document.createElement('div');
    label.style.fontWeight = '700';
    label.style.fontSize = '13px';
    label.style.display = 'flex';
    label.style.justifyContent = 'space-between';

    const labelText = document.createElement('span');
    labelText.textContent = r.feature;
    const valSpan = document.createElement('span');
    valSpan.className = 'muted';
    valSpan.textContent = r.value_b;

    label.appendChild(labelText);
    label.appendChild(valSpan);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = Math.max(0, Number(r.value_b) - 15);
    input.max = Math.min(99, Number(r.value_b) + 15);
    input.step = 1;
    input.value = Number(r.value_b);

    input.oninput = () => {
      valSpan.textContent = input.value;
      state[r.feature] = Number(input.value);
      debouncedWhatIfUpdate(playerBId, state, basePredA);
    };
    block.appendChild(label);
    block.appendChild(input);
    sliderWrap.appendChild(block);
    state[r.feature] = Number(r.value_b);
  });

  const resultsRow = document.createElement('div');
  resultsRow.style.marginTop = '16px';
  resultsRow.style.padding = '16px';
  resultsRow.style.background = 'rgba(255,255,255,0.03)';
  resultsRow.style.borderRadius = 'var(--radius-md)';
  resultsRow.id = 'whatIfResults';
  whatIfContainer.appendChild(resultsRow);

  debouncedWhatIfUpdate = debounce((pid, adjustments, baseA) => runWhatIf(pid, adjustments, baseA), 250);
}

function updateWhatIfResults(adjOVR, newDelta) {
  const el = document.getElementById('whatIfResults');
  if (!el) return;
  el.innerHTML = `
    <div style="font-weight:700; font-size: 1.1em; color: var(--accent-cyan);">Adjusted Player B OVR: ${adjOVR.toFixed(2)}</div>
    <div class="muted">New gap (A - B): ${newDelta >= 0 ? '+' : ''}${newDelta.toFixed(2)}</div>
  `;
}

async function runWhatIf(playerId, adjustments, basePredA) {
  try {
    const data = await fetchJson('/api/whatif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_id: playerId, adjustments })
    });
    const newDelta = basePredA - data.adjusted;
    updateWhatIfResults(data.adjusted, newDelta);
  } catch (err) {
    const el = document.getElementById('whatIfResults');
    if (el) el.textContent = err.message;
  }
}

function debounce(fn, wait) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

let debouncedWhatIfUpdate = null;
let lastSimilar = [];
let lastPlayerA = null;

async function fetchQuantiles(playerId) {
  const panel = document.getElementById('quantCard');
  if (!panel) return;
  panel.innerHTML = '<span class="muted">Loading quantiles...</span>';
  try {
    const data = await fetchJson(`/api/quantiles?player_id=${playerId}`);
    const p10 = data.p10;
    const p50 = data.p50;
    const p90 = data.p90;
    const spread = p90 - p10;
    const volPct = p50 ? (spread / p50) * 100 : 0;
    const upper = p90 - p50;
    const lower = p50 - p10;
    const skewLabel =
      upper > lower * 1.2 ? 'Upside skew' :
        lower > upper * 1.2 ? 'Downside skew' : 'Balanced spread';
    const variability =
      spread < 4 ? 'Tight range (stable)' :
        spread < 8 ? 'Moderate variability' : 'High volatility';
    const pos = (v) => {
      const denom = spread || 1e-6;
      const pct = ((v - p10) / denom) * 100;
      return Math.min(100, Math.max(0, pct));
    };
    panel.innerHTML = `
      <div style="display:flex; flex-direction: column; gap: 1rem;">
        <div>
          <div style="font-size:12px; color: var(--text-secondary); margin-bottom:6px; display:flex; justify-content:space-between;">
              <span>Floor (P10)</span><span>Median (P50)</span><span>Ceiling (P90)</span>
          </div>
          <div style="position:relative; height:12px; border-radius:999px; background: rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.1);">
            <div style="position:absolute; left:0; right:0; top:4px; height:4px; background:linear-gradient(90deg, var(--negative), var(--accent-cyan), var(--positive)); opacity:0.3; border-radius:4px;"></div>
            <div style="position:absolute; left:${pos(p10)}%; top:50%; transform:translate(-50%,-50%); width:10px; height:10px; border-radius:50%; background:var(--negative); border:2px solid #fff;" title="P10 ${p10.toFixed(2)}"></div>
            <div style="position:absolute; left:${pos(p50)}%; top:50%; transform:translate(-50%,-50%); width:14px; height:14px; border-radius:50%; background:var(--accent-cyan); border:2px solid #fff; z-index:2;" title="P50 ${p50.toFixed(2)}"></div>
            <div style="position:absolute; left:${pos(p90)}%; top:50%; transform:translate(-50%,-50%); width:10px; height:10px; border-radius:50%; background:var(--positive); border:2px solid #fff;" title="P90 ${p90.toFixed(2)}"></div>
          </div>
          <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:12px; font-weight:700;">
            <span>${p10.toFixed(1)}</span>
            <span>${p50.toFixed(1)}</span>
            <span>${p90.toFixed(1)}</span>
          </div>
        </div>
        <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px;">
          <div style="font-weight:700; margin-bottom:4px; font-size: 0.9em; color: var(--text-primary);">Analysis</div>
          <div class="muted small" style="line-height:1.5;">
            ${variability} • Range: ${spread.toFixed(2)} pts<br/>
            ${skewLabel} • Volatility: ${volPct.toFixed(1)}%
          </div>
        </div>
      </div>
    `;
    const riskChartEl = document.getElementById('riskChart');
    if (riskChartEl) riskChartEl.style.opacity = '1';
    renderRiskProfile(data);
  } catch (err) {
    if (panel) panel.textContent = err.message;
    // Fail silently or show text for riskChartEl
  }
}

function renderRiskProfile(data) {
  const riskChartEl = document.getElementById('riskChart');
  if (!riskChartEl || !window.echarts) return;
  const chart = echarts.init(riskChartEl);
  if (!data) {
    chart.clear();
    riskChartEl.innerHTML = '<div class="muted" style="height:100%; display:flex; align-items:center; justify-content:center;">Run analysis to see risk profile.</div>';
    return;
  }

  const p10 = data.p10;
  const p50 = data.p50;
  const p90 = data.p90;

  // Generate Bell Curve Data (Normal Distribution approximation)
  const mean = p50;
  const stdDev = (p90 - p10) / 3.29; // Approx for 90% CI
  const minX = mean - 4 * stdDev;
  const maxX = mean + 4 * stdDev;
  const points = [];
  for (let x = minX; x <= maxX; x += stdDev / 10) {
    const y = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
    points.push([x, y]);
  }

  chart.setOption({
    textStyle: { fontFamily: 'Inter' },
    backgroundColor: 'transparent',
    grid: { left: 20, right: 20, top: 40, bottom: 20, containLabel: true },
    xAxis: {
      type: 'value',
      min: parseFloat(minX.toFixed(1)),
      max: parseFloat(maxX.toFixed(1)),
      axisLabel: { color: cssVar('--text-secondary') },
      splitLine: { show: false }
    },
    yAxis: { type: 'value', show: false },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(20,20,20,0.9)',
      textStyle: { color: '#fff' },
      formatter: (params) => {
        const x = params[0].value[0];
        return `OVR ${x.toFixed(1)}`;
      }
    },
    series: [
      {
        type: 'line',
        data: points,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 3, color: cssVar('--accent-violet') },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(139, 92, 246, 0.4)' },
            { offset: 1, color: 'rgba(139, 92, 246, 0.05)' }
          ])
        },
        markLine: {
          symbol: 'none',
          label: { position: 'end', color: '#fff', formatter: '{b}' },
          data: [
            { xAxis: p10, lineStyle: { color: cssVar('--negative'), type: 'dashed' }, label: { formatter: 'Floor\n' + p10.toFixed(0) } },
            { xAxis: p50, lineStyle: { color: cssVar('--text-primary'), type: 'solid' }, label: { formatter: 'Pred\n' + p50.toFixed(0) } },
            { xAxis: p90, lineStyle: { color: cssVar('--positive'), type: 'dashed' }, label: { formatter: 'Ceiling\n' + p90.toFixed(0) } }
          ]
        }
      }
    ]
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderComposition(reasons, playerName) {
  const compChartEl = document.getElementById('compChart');
  if (!compChartEl || !window.echarts) return;
  const chart = echarts.init(compChartEl);

  if (!reasons || !reasons.length) {
    chart.clear();
    compChartEl.innerHTML = '<div class="muted" style="height:100%; display:flex; align-items:center; justify-content:center;">No data available.</div>';
    return;
  }

  // Heuristics to categorize attributes
  const cats = {
    'Physical': ['PAC', 'Sprint', 'Accel', 'Strength', 'Stamina', 'Jump', 'Agility', 'Balance', 'React'],
    'Technical': ['DRI', 'Ball', 'Control', 'Pass', 'Cross', 'Curve', 'Free', 'Long', 'Short', 'Vis'],
    'Attacking': ['SHO', 'Fin', 'Att', 'Pos', 'Volley', 'Pen', 'Shot', 'Power'],
    'Defending': ['DEF', 'Inter', 'Head', 'Mark', 'Stand', 'Slid', 'Aggress']
  };

  const scores = { 'Physical': 0, 'Technical': 0, 'Attacking': 0, 'Defending': 0, 'Other': 0 };

  reasons.forEach(r => {
    const feature = r.feature;
    let found = false;
    for (const [cat, keywords] of Object.entries(cats)) {
      if (keywords.some(k => feature.includes(k))) {
        scores[cat] += Math.abs(r.impact); // Use absolute impact to gauge importance
        found = true;
        break;
      }
    }
    if (!found) scores['Other'] += Math.abs(r.impact);
  });

  const data = Object.entries(scores)
    .filter(([_, val]) => val > 0.1)
    .map(([name, value]) => ({ name, value }));

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item', backgroundColor: 'rgba(20,20,20,0.9)', textStyle: { color: '#fff' } },
    legend: { bottom: 0, textStyle: { color: cssVar('--text-secondary') }, icon: 'circle' },
    series: [
      {
        name: 'Rating Composition',
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '45%'],
        itemStyle: {
          borderRadius: 5,
          borderColor: '#18181b',
          borderWidth: 2
        },
        label: { show: false },
        data: data,
        color: [
          cssVar('--accent-cyan'),
          cssVar('--accent-violet'),
          cssVar('--accent-emerald'),
          cssVar('--accent-rose'),
          '#94a3b8'
        ]
      }
    ]
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderNetwork(similarResults, playerA) {
  if (!networkEl || !window.echarts) return;

  // Dispose existing instance to ensure clean state or get existing
  let chart = echarts.getInstanceByDom(networkEl);
  if (!chart) chart = echarts.init(networkEl);

  if (!playerA || !similarResults || !similarResults.length) {
    chart.clear();
    networkEl.innerHTML = '<div class="muted" style="height:100%; display:flex; align-items:center; justify-content:center;">Run a similarity search to see the network.</div>';
    return;
  }

  // Ensure container has height
  if (networkEl.clientHeight === 0) {
    networkEl.style.height = '400px';
  }

  const sims = similarResults.map(s => s.similarity || 0);
  const maxSim = Math.max(...sims);
  const minSim = Math.min(...sims);
  const norm = (v) => (maxSim - minSim) > 1e-6 ? (v - minSim) / (maxSim - minSim) : 0.5;

  const nodes = [{
    id: 'center',
    name: playerA.Name || playerA.name || 'Player A',
    category: 0,
    symbolSize: 60,
    itemStyle: {
      color: new echarts.graphic.RadialGradient(0.5, 0.5, 0.5, [
        { offset: 0, color: '#10b981' },
        { offset: 1, color: '#047857' }
      ]),
      borderColor: '#ecfdf5',
      borderWidth: 3,
      shadowBlur: 20,
      shadowColor: 'rgba(16, 185, 129, 0.5)'
    },
    label: { show: true, color: '#fff', fontWeight: 800, fontSize: 14, textShadowBlur: 4, textShadowColor: '#000' },
  }];

  const edges = [];
  similarResults.forEach((s, idx) => {
    const score = norm(s.similarity || 0);
    const size = 25 + score * 25;
    const color = score > 0.6 ? '#22d3ee' : '#a78bfa'; // Cyan for high match, Violet for lower

    nodes.push({
      id: String(s.id || idx),
      dataId: s.id, // Store real ID for click handler
      name: s.name || s.id,
      category: 1,
      symbolSize: size,
      value: s.similarity || 0,
      itemStyle: {
        color: color,
        borderColor: '#fff',
        borderWidth: 2,
        shadowBlur: 10 + score * 10,
        shadowColor: color
      },
      label: {
        show: true,
        position: 'bottom',
        color: cssVar('--text-secondary'),
        fontSize: 11,
        formatter: (p) => p.name.length > 10 ? p.name.substring(0, 9) + '..' : p.name
      },
      tooltip: {
        formatter: `${s.name}<br/>OVR ${s.ovr || '—'}<br/>Sim ${s.similarity !== undefined ? s.similarity.toFixed(3) : '—'}`
      }
    });

    edges.push({
      source: 'center',
      target: String(s.id || idx),
      lineStyle: {
        width: 1 + score * 4,
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 1, y2: 0,
          colorStops: [{ offset: 0, color: '#10b981' }, { offset: 1, color: color }],
        },
        curveness: 0.15
      },
      value: s.similarity || 0,
    });
  });

  chart.setOption({
    backgroundColor: 'transparent',
    animationDurationUpdate: 1500,
    animationEasingUpdate: 'quinticInOut',
    tooltip: { backgroundColor: 'rgba(20,20,20,0.9)', textStyle: { color: '#fff' }, borderColor: '#333' },
    series: [{
      type: 'graph',
      layout: 'force',
      roam: true,
      draggable: true,
      focusNodeAdjacency: true,
      force: {
        repulsion: 1000,
        gravity: 0.1,
        edgeLength: [120, 220],
        friction: 0.6
      },
      data: nodes,
      edges,
      label: { fontFamily: 'Inter' }
    }],
  });

  // Interaction: Click to compare
  chart.off('click'); // remove old listeners
  chart.on('click', (params) => {
    if (params.dataType === 'node' && params.data.dataId) {
      // Add to selection if not present
      const id = String(params.data.dataId);
      if (!selectedIds.includes(id)) {
        if (selectedIds.length >= 4) selectedIds.pop(); // keep limit
        selectedIds.push(id);
        renderSelectedChips();
        renderPlayerList(playersFiltered());
        compare(); // Auto-run comparison

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  });

  window.addEventListener('resize', () => chart.resize());
}

// Global Resize Handler for details/summary
document.querySelectorAll('details').forEach(d => {
  d.addEventListener('toggle', () => {
    if (d.open) {
      requestAnimationFrame(() => {
        document.querySelectorAll('.details-content div[id]').forEach(div => {
          const chart = echarts.getInstanceByDom(div);
          if (chart) chart.resize();
        });
      });
    }
  });
});

function renderRadar(topReasons, nameA, nameB) {
  if (!radarEl || !window.echarts) return;
  const chart = echarts.init(radarEl);

  if (!topReasons || !topReasons.length) {
    chart.clear();
    radarEl.innerHTML = '<div class="muted" style="height:100%; display:flex; align-items:center; justify-content:center;">No numeric drivers to plot.</div>';
    return;
  }

  const usable = topReasons.filter(r => !isNaN(r.value_a) && !isNaN(r.value_b));
  if (!usable.length) return;

  const features = usable.slice(0, 8).map(r => r.feature);
  const valsA = usable.slice(0, 8).map(r => Number(r.value_a));
  const valsB = usable.slice(0, 8).map(r => Number(r.value_b));
  const maxVal = Math.max(...valsA, ...valsB) || 1;

  // Create indicators with some buffer
  const indicator = features.map(f => ({ name: f, max: maxVal * 1.2 }));

  chart.setOption({
    backgroundColor: 'transparent',
    radar: {
      indicator,
      center: ['50%', '55%'],
      radius: '65%',
      splitNumber: 3,
      splitArea: {
        show: true,
        areaStyle: {
          color: ['rgba(255,255,255,0.01)', 'rgba(255,255,255,0.03)']
        }
      },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
      axisName: {
        color: cssVar('--text-secondary'),
        backgroundColor: 'rgba(0,0,0,0.3)',
        borderRadius: 4,
        padding: [4, 6]
      },
    },
    legend: {
      data: [nameA || 'Player A', nameB || 'Player B'],
      bottom: 10,
      itemGap: 20,
      textStyle: { color: '#e5e7eb', fontSize: 13, fontWeight: 600 },
      selectedMode: 'multiple'
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: valsA,
            name: nameA || 'Player A',
            itemStyle: { color: '#10b981' },
            lineStyle: { width: 3, color: '#10b981', shadowBlur: 10, shadowColor: 'rgba(16, 185, 129, 0.5)' },
            areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(16, 185, 129, 0.4)' }, { offset: 1, color: 'rgba(16, 185, 129, 0.05)' }]) },
            symbol: 'circle',
            symbolSize: 6
          },
          {
            value: valsB,
            name: nameB || 'Player B',
            itemStyle: { color: '#22d3ee' },
            lineStyle: { width: 3, color: '#22d3ee', shadowBlur: 10, shadowColor: 'rgba(34, 211, 238, 0.5)' },
            areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(34, 211, 238, 0.4)' }, { offset: 1, color: 'rgba(34, 211, 238, 0.05)' }]) },
            symbol: 'circle',
            symbolSize: 6
          }
        ]
      },
    ],
    tooltip: { trigger: 'item', backgroundColor: 'rgba(20,20,20,0.9)', textStyle: { color: '#fff' } },
  });
  window.addEventListener('resize', () => chart.resize());
}

searchInput.addEventListener('input', (e) => filterPlayers(e.target.value));
compareBtn.addEventListener('click', compare);
similarBtn.addEventListener('click', fetchSimilar);
anomBtn.addEventListener('click', fetchAnomalies);

renderVersatilityFromPlayers(null, null);
loadModelMeta();
loadPlayers().catch(err => {
  errorEl.textContent = err.message;
});

// Load causal effects on page load
(async function loadCausal() {
  try {
    const data = await fetchJson('/api/causal');
    causalBody.innerHTML = '';
    (data.effects || []).slice(0, 12).forEach(e => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${e.feature}</td>
        <td>${Number(e.per_unit_effect).toFixed(3)}</td>
        <td>${Number(e.per_five_effect).toFixed(3)}</td>
      `;
      causalBody.appendChild(tr);
    });
    if (!causalBody.children.length) {
      causalBody.innerHTML = '<tr><td colspan="3" class="muted">Run scripts/causal_effects.py to populate.</td></tr>';
    }
  } catch (err) {
    causalBody.innerHTML = `<tr><td colspan="3" class="muted">${err.message}</td></tr>`;
  }
})();
