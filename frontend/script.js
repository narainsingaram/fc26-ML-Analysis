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
const posFilter = document.getElementById('posFilter');
const leagueFilter = document.getElementById('leagueFilter');
const genderFilter = document.getElementById('genderFilter');
const ovrRangeMin = document.getElementById('ovrRange');
const ovrRangeMax = document.getElementById('ovrRangeMax');
const ovrMinLabel = document.getElementById('ovrMinLabel');
const ovrMaxLabel = document.getElementById('ovrMaxLabel');
const clearFiltersBtn = document.getElementById('clearFilters');
const modelNameEl = document.getElementById('modelName');
const modelMaeEl = document.getElementById('modelMae');
const modelRmseEl = document.getElementById('modelRmse');
const modelR2El = document.getElementById('modelR2');
const modelDetailsEl = document.getElementById('modelDetails');
const downloadReportBtn = document.getElementById('downloadReportBtn');
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
const compChartEl = document.getElementById('compChart'); // State
let players = [];
let playerMap = new Map();
let selectedIds = []; // For comparison/details
let squadIds = [];    // For Squad Builder Draft Pool
let positionOptions = [];
let leagueOptions = [];
let formationAssignments = {};

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
  // Note: renderDraftPool is now independent
}

function renderPlayerList(list) {
  if (!playerListEl) return;
  playerListEl.innerHTML = '';
  if (!list.length) {
    const empty = document.createElement('div');
    empty.className = 'muted small';
    empty.style.padding = '0.75rem';
    empty.textContent = 'No players match these filters.';
    playerListEl.appendChild(empty);
    renderSelectedChips();
    return;
  }
  list.forEach(p => {
    const isSelected = selectedIds.includes(String(p.ID));
    const isInSquad = squadIds.includes(String(p.ID));
    const disabled = !isSelected && selectedIds.length >= 25;

    const card = document.createElement('div');
    card.className = `player-option ${isSelected ? 'selected' : ''}`;

    // Main Content (Click to Select for Comparison/Details)
    const content = document.createElement('div');
    content.style.flex = '1';
    content.innerHTML = `
        <div style="font-weight:700;">${p.Name}</div>
        <div class="muted small">${p.Position} • OVR ${p.OVR} • ${p.Team || ''}</div>
    `;
    content.onclick = () => {
      if (disabled) return;
      toggleSelect(p.ID);
    };

    // Actions
    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.gap = '8px';
    actions.style.alignItems = 'center';

    // Squad Toggle Button
    const squadBtn = document.createElement('button');
    squadBtn.className = 'btn-icon'; // Need to define this or inline style
    squadBtn.style.padding = '4px 8px';
    squadBtn.style.fontSize = '0.8rem';
    squadBtn.style.background = isInSquad ? 'var(--accent-cyan)' : 'var(--glass-bg)';
    squadBtn.style.color = isInSquad ? '#000' : '#fff';
    squadBtn.style.border = '1px solid var(--panel-border)';
    squadBtn.style.borderRadius = '4px';
    squadBtn.style.cursor = 'pointer';
    squadBtn.innerHTML = isInSquad ? 'In Squad' : '+ Squad';
    squadBtn.onclick = (e) => {
      e.stopPropagation(); // Prevent comparison select
      toggleSquad(p.ID);
    };

    // Comparison Check
    const check = document.createElement('div');
    check.innerHTML = isSelected ? '<span style="color:var(--accent-cyan)">✔</span>' : '';

    actions.appendChild(squadBtn);
    actions.appendChild(check);

    card.appendChild(content);
    card.appendChild(actions);

    // Disable opacity logic if needed, but keeping it simple for now
    if (disabled) content.style.opacity = '0.5';

    playerListEl.appendChild(card);
  });
  renderSelectedChips();
}

function toggleSquad(id) {
  const sid = String(id);
  if (squadIds.includes(sid)) {
    squadIds = squadIds.filter(x => x !== sid);
  } else {
    // No hard limit for squad pool, maybe 50?
    if (squadIds.length >= 50) return;
    squadIds.push(sid);
  }
  // Re-render list to update buttons
  renderPlayerList(playersFiltered());
  renderDraftPool();
}

function toggleSelect(id) {
  const sid = String(id);
  if (selectedIds.includes(sid)) {
    selectedIds = selectedIds.filter(x => x !== sid);
  } else {
    if (selectedIds.length >= 25) return;
    selectedIds.push(sid);
  }
  renderSelectedChips();
  renderPlayerList(playersFiltered());
  renderPreview();
}

function getFilterState() {
  const term = (searchInput?.value || '').toLowerCase();
  const pos = posFilter?.value || '';
  const league = leagueFilter?.value || '';
  const gender = (genderFilter?.value || '').toLowerCase();
  const minOvr = Number(ovrRangeMin?.value || 0);
  const maxOvr = Number(ovrRangeMax?.value || 99);
  return { term, pos, league, gender, minOvr, maxOvr };
}

function playersFiltered(term = searchInput?.value || '') {
  const filters = getFilterState();
  const searchTerm = term ? term.toLowerCase() : filters.term;
  return players.filter(p => {
    if (searchTerm && !p.Name.toLowerCase().includes(searchTerm)) return false;
    if (filters.pos && String(p.Position).toUpperCase() !== filters.pos.toUpperCase()) return false;
    if (filters.league && String(p.League) !== filters.league) return false;
    if (filters.gender && String(p.GENDER || '').toLowerCase() !== filters.gender) return false;
    const ovrVal = Number(p.OVR || 0);
    if (ovrVal < filters.minOvr || ovrVal > filters.maxOvr) return false;
    return true;
  });
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
  const params = new URLSearchParams({ limit: 400 });

  // Basic Search
  if (search) params.append('search', search);

  // Advanced Filters
  const pos = document.getElementById('posFilter')?.value;
  const league = document.getElementById('leagueFilter')?.value;
  const nation = document.getElementById('nationFilter')?.value;
  const gender = document.getElementById('genderFilter')?.value;

  if (pos) params.append('position', pos);
  if (league) params.append('league', league);
  if (nation) params.append('nation', nation);
  if (gender) params.append('gender', gender);

  // Physical
  const minAge = document.getElementById('minAge')?.value;
  const maxAge = document.getElementById('maxAge')?.value;
  const minH = document.getElementById('minHeight')?.value;
  const maxH = document.getElementById('maxHeight')?.value;
  const minW = document.getElementById('minWeight')?.value;
  const maxW = document.getElementById('maxWeight')?.value;

  if (minAge) params.append('min_age', minAge);
  if (maxAge) params.append('max_age', maxAge);
  if (minH) params.append('min_height', minH);
  if (maxH) params.append('max_height', maxH);
  if (minW) params.append('min_weight', minW);
  if (maxW) params.append('max_weight', maxW);

  // Stats
  const minOvr = document.getElementById('ovrRange')?.value;
  if (minOvr) params.append('min_ovr', minOvr);

  const minWf = document.getElementById('minWf')?.value;
  const minSm = document.getElementById('minSm')?.value;
  if (minWf > 1) params.append('min_wf', minWf);
  if (minSm > 1) params.append('min_sm', minSm);

  // Traits
  const ps = document.getElementById('playStyleFilter')?.value;
  if (ps) params.append('playstyle', ps);

  try {
    const data = await fetchJson(`/api/players?${params.toString()}`);
    players = data.players || [];
    playerMap = new Map(players.map(p => [String(p.ID), p]));

    // We only update dropdowns once or if empty to avoiding resetting selection while searching
    if (!positionOptions.length) {
      positionOptions = Array.from(new Set(players.map(p => p.Position).filter(Boolean))).sort();
      const posEl = document.getElementById('posFilter');
      if (posEl && posEl.children.length <= 1) {
        posEl.innerHTML = '<option value=\"\">Any</option>' + positionOptions.map(p => `<option value=\"${p}\">${p}</option>`).join('');
      }
    }
    if (!leagueOptions.length) {
      leagueOptions = Array.from(new Set(players.map(p => p.League).filter(Boolean))).sort();
      const leagueEl = document.getElementById('leagueFilter');
      if (leagueEl && leagueEl.children.length <= 1) {
        leagueEl.innerHTML = '<option value=\"\">Any</option>' + leagueOptions.map(l => `<option value=\"${l}\">${l}</option>`).join('');
      }
    }

    renderPlayerList(players); // No more client-side filter
    renderPreview();
  } catch (e) {
    console.error(e);
    if (playerListEl) playerListEl.innerHTML = '<div class="muted small" style="padding:1rem;">Search failed: ' + e.message + '</div>';
  }
}

// Re-using debounce from earlier
const debouncedLoad = debounce(() => {
  loadPlayers(document.getElementById('search').value);
}, 400);

function bindSearchEvents() {
  const ids = ['search', 'posFilter', 'leagueFilter', 'nationFilter', 'genderFilter',
    'minAge', 'maxAge', 'minHeight', 'maxHeight', 'minWeight', 'maxWeight',
    'ovrRange', 'minWf', 'minSm', 'playStyleFilter'];

  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', debouncedLoad);
      el.addEventListener('change', debouncedLoad);
    }
  });

  document.getElementById('applyFilters')?.addEventListener('click', () => loadPlayers(document.getElementById('search').value));
  document.getElementById('clearFilters')?.addEventListener('click', () => {
    // Reset all inputs
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
    document.getElementById('ovrRange').value = 40;
    document.getElementById('ovrLabel').textContent = '40';
    loadPlayers('');
  });
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
      splitLine: { lineStyle: { color: 'rgba(0,0,0,0.05)' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'category',
      data: names,
      axisLabel: { color: cssVar('--text-primary'), fontWeight: 600, fontSize: 12, fontFamily: 'Outfit' },
      axisLine: { show: false },
      axisTick: { show: false },
      inverse: true
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255,255,255,0.95)',
      borderColor: cssVar('--panel-border'),
      textStyle: { color: '#0f172a' },
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
            Stat Delta: <span style="color:#0f172a;">${stat ?? '—'}</span><br/>
            Contrib: <span style="color:#0f172a;">${pct}</span>
          </div>`;
      }
    },
    series: [{
      type: 'bar',
      data: impacts.map(v => ({
        value: v,
        itemStyle: {
          color: v >= 0
            ? new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: 'rgba(16, 185, 129, 0.6)' }, { offset: 1, color: 'rgba(16, 185, 129, 1)' }])
            : new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: 'rgba(244, 63, 94, 0.6)' }, { offset: 1, color: 'rgba(244, 63, 94, 1)' }]),
          shadowBlur: 10,
          shadowColor: v >= 0 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)'
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
      backgroundStyle: { color: 'rgba(0,0,0,0.03)', borderRadius: 4 }
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
      backgroundColor: 'rgba(255,255,255,0.95)',
      borderColor: '#e2e8f0',
      textStyle: { color: '#0f172a' },
      formatter: (p) => {
        const d = p.data;
        const res = d.residual;
        return `
             <div style="font-weight:700; margin-bottom:4px;">${d.name}</div>
             <div class="small" style="color:#64748b;">Predicted: <b style="color:#0f172a">${d.value[0].toFixed(1)}</b></div>
             <div class="small" style="color:#64748b;">Actual: <b style="color:#0f172a">${d.value[1].toFixed(1)}</b></div>
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
      splitLine: { lineStyle: { color: 'rgba(0,0,0,0.05)' } },
      axisLine: { lineStyle: { color: '#cbd5e1' } }
    },
    yAxis: {
      name: 'Actual OVR',
      nameLocation: 'middle',
      nameGap: 24,
      min: minVal,
      max: maxVal,
      axisLabel: { color: cssVar('--text-secondary') },
      splitLine: { lineStyle: { color: 'rgba(0,0,0,0.05)' } },
      axisLine: { lineStyle: { color: '#cbd5e1' } }
    },
    series: [
      {
        type: 'line',
        data: [[minVal, minVal], [maxVal, maxVal]],
        showSymbol: false,
        lineStyle: { width: 1, type: 'dashed', color: 'rgba(0,0,0,0.2)' },
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
          shadowColor: 'rgba(0,0,0,0.1)',
          borderColor: 'rgba(255,255,255,1)',
          borderWidth: 1
        },
        emphasis: {
          focus: 'self',
          scale: true,
          itemStyle: {
            shadowBlur: 20,
            borderColor: '#3b82f6',
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
        <div style="background: var(--bg-surface); padding: 12px; border-radius: 8px; border: 1px solid var(--panel-border);">
          <div style="font-weight:700; margin-bottom:4px; font-size: 0.9em; color: var(--text-main);">Analysis</div>
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
      backgroundColor: 'rgba(255,255,255,0.95)',
      textStyle: { color: '#0f172a' },
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
            { xAxis: p10, lineStyle: { color: cssVar('--negative'), type: 'dashed' }, label: { color: '#64748b', formatter: 'Floor\n' + p10.toFixed(0) } },
            { xAxis: p50, lineStyle: { color: cssVar('--text-main'), type: 'solid' }, label: { color: '#0f172a', formatter: 'Pred\n' + p50.toFixed(0) } },
            { xAxis: p90, lineStyle: { color: cssVar('--positive'), type: 'dashed' }, label: { color: '#64748b', formatter: 'Ceiling\n' + p90.toFixed(0) } }
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
    tooltip: { trigger: 'item', backgroundColor: 'rgba(255,255,255,0.95)', textStyle: { color: '#0f172a' } },
    legend: { bottom: 0, textStyle: { color: cssVar('--text-secondary') }, icon: 'circle' },
    series: [
      {
        name: 'Rating Composition',
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '45%'],
        itemStyle: {
          borderRadius: 5,
          borderColor: '#fff',
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
        { offset: 0, color: '#3b82f6' },
        { offset: 1, color: '#1d4ed8' }
      ]),
      borderColor: '#dbeafe',
      borderWidth: 3,
      shadowBlur: 20,
      shadowColor: 'rgba(59, 130, 246, 0.5)'
    },
    label: { show: true, color: '#fff', fontWeight: 800, fontSize: 14 },
  }];

  const edges = [];
  similarResults.forEach((s, idx) => {
    const score = norm(s.similarity || 0);
    const size = 25 + score * 25;
    const color = score > 0.6 ? '#6366f1' : '#a78bfa'; // Indigo/Purple

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
          colorStops: [{ offset: 0, color: '#3b82f6' }, { offset: 1, color: color }],
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
    tooltip: { backgroundColor: 'rgba(255,255,255,0.95)', textStyle: { color: '#0f172a' }, borderColor: '#e2e8f0' },
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
          color: ['rgba(59,130,246,0.01)', 'rgba(59,130,246,0.03)']
        }
      },
      axisLine: { lineStyle: { color: 'rgba(0,0,0,0.1)' } },
      splitLine: { lineStyle: { color: 'rgba(0,0,0,0.05)' } },
      axisName: {
        color: cssVar('--text-secondary'),
        backgroundColor: 'rgba(255,255,255,0.8)',
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
            itemStyle: { color: '#3b82f6' },
            lineStyle: { width: 3, color: '#3b82f6', shadowBlur: 10, shadowColor: 'rgba(59, 130, 246, 0.4)' },
            areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(59, 130, 246, 0.4)' }, { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }]) },
            symbol: 'circle',
            symbolSize: 6
          },
          {
            value: valsB,
            name: nameB || 'Player B',
            itemStyle: { color: '#8b5cf6' },
            lineStyle: { width: 3, color: '#8b5cf6', shadowBlur: 10, shadowColor: 'rgba(139, 92, 246, 0.4)' },
            areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(139, 92, 246, 0.4)' }, { offset: 1, color: 'rgba(139, 92, 246, 0.05)' }]) },
            symbol: 'circle',
            symbolSize: 6
          }
        ]
      },
    ],
    tooltip: { trigger: 'item', backgroundColor: 'rgba(255,255,255,0.95)', textStyle: { color: '#0f172a' } },
  });
  window.addEventListener('resize', () => chart.resize());
}

searchInput.addEventListener('input', (e) => filterPlayers(e.target.value));
compareBtn.addEventListener('click', compare);
similarBtn.addEventListener('click', fetchSimilar);
anomBtn.addEventListener('click', fetchAnomalies);
if (downloadReportBtn) {
  downloadReportBtn.addEventListener('click', () => {
    errorEl.textContent = '';
    if (selectedIds.length < 2) {
      errorEl.textContent = 'Pick two players to export a comparison PDF.';
      return;
    }
    const url = `/api/report/comparison?player_a_id=${selectedIds[0]}&player_b_id=${selectedIds[1]}`;
    window.open(url, '_blank');
  });
}

renderVersatilityFromPlayers(null, null);
bindSearchEvents();
loadModelMeta();
loadPlayers().catch(err => {
  errorEl.textContent = err.message;
});
renderDraftPool();
function syncRangeLabels() {
  if (!ovrRangeMin || !ovrRangeMax) return;
  let minVal = Number(ovrRangeMin.value);
  let maxVal = Number(ovrRangeMax.value);
  if (minVal > maxVal) {
    // keep sliders consistent
    [minVal, maxVal] = [maxVal, minVal];
    ovrRangeMin.value = minVal;
    ovrRangeMax.value = maxVal;
  }
  if (ovrMinLabel) ovrMinLabel.textContent = minVal;
  if (ovrMaxLabel) ovrMaxLabel.textContent = maxVal;
}

function attachFilterEvents() {
  if (posFilter) posFilter.addEventListener('change', () => renderPlayerList(playersFiltered()));
  if (leagueFilter) leagueFilter.addEventListener('change', () => renderPlayerList(playersFiltered()));
  if (genderFilter) genderFilter.addEventListener('change', () => renderPlayerList(playersFiltered()));
  const onRange = () => { syncRangeLabels(); renderPlayerList(playersFiltered()); };
  if (ovrRangeMin) ovrRangeMin.addEventListener('input', onRange);
  if (ovrRangeMax) ovrRangeMax.addEventListener('input', onRange);
  if (clearFiltersBtn) {
    clearFiltersBtn.addEventListener('click', () => {
      if (searchInput) searchInput.value = '';
      if (posFilter) posFilter.value = '';
      if (leagueFilter) leagueFilter.value = '';
      if (genderFilter) genderFilter.value = '';
      if (ovrRangeMin) ovrRangeMin.value = 70;
      if (ovrRangeMax) ovrRangeMax.value = 99;
      syncRangeLabels();
      renderPlayerList(playersFiltered());
    });
  }
  syncRangeLabels();
}

attachFilterEvents();

// Squad Builder Logic
const formations = {
  "433": ["LW", "ST", "RW", "CM", "CDM", "CM", "LB", "CB", "CB", "RB", "GK"],
  "442": ["LM", "ST", "ST", "RM", "CM", "CM", "LB", "CB", "CB", "RB", "GK"],
  "4231": ["LM", "ST", "RM", "CAM", "CDM", "CDM", "LB", "CB", "CB", "RB", "GK"],
  "352": ["LM", "ST", "ST", "RM", "CAM", "CDM", "CDM", "CB", "CB", "CB", "GK"]
};

// Embedded Squad Search
const squadSearchInput = document.getElementById('squadSearch');
const squadSearchResults = document.getElementById('squadSearchResults');

if (squadSearchInput) {
  squadSearchInput.addEventListener('input', (e) => {
    const val = e.target.value.toLowerCase().trim();
    if (!val) {
      squadSearchResults.style.display = 'none';
      return;
    }
    const matches = players.filter(p => p.Name.toLowerCase().includes(val)).slice(0, 10);
    renderSquadSearchResults(matches);
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!squadSearchInput.contains(e.target) && !squadSearchResults.contains(e.target)) {
      squadSearchResults.style.display = 'none';
    }
  });
}

function renderSquadSearchResults(matches) {
  if (!squadSearchResults) return;
  squadSearchResults.innerHTML = '';
  squadSearchResults.style.display = 'block';

  if (!matches.length) {
    squadSearchResults.innerHTML = '<div class="muted small" style="padding:10px;">No players found.</div>';
    return;
  }

  matches.forEach(p => {
    const row = document.createElement('div');
    row.style.padding = '8px 12px';
    row.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
    row.style.cursor = 'pointer';
    row.style.display = 'flex';
    row.style.justifyContent = 'space-between';
    row.style.alignItems = 'center';
    row.className = 'search-result-row'; // For hover effect via CSS if needed

    row.innerHTML = `
      <div>
        <div style="font-weight:700; font-size:0.9rem;">${p.Name}</div>
        <div class="tiny muted">${p.Position} • ${p.Team}</div>
      </div>
      <div style="font-weight:800; color:var(--accent-cyan);">${p.OVR}</div>
    `;

    row.onclick = () => {
      toggleSquad(p.ID);
      squadSearchInput.value = '';
      squadSearchResults.style.display = 'none';
    };

    // Add hover effect
    row.onmouseenter = () => row.style.background = 'rgba(255,255,255,0.05)';
    row.onmouseleave = () => row.style.background = 'transparent';

    squadSearchResults.appendChild(row);
  });
}

function renderDraftPool() {
  const pool = document.getElementById('draftPool');
  if (!pool) return;

  if (!squadIds.length) {
    pool.innerHTML = '<div class="muted small">Select players from list (+ Squad) to populate draft pool.</div>';
    return;
  }

  pool.innerHTML = '';
  squadIds.forEach(id => {
    const p = playerMap.get(String(id));
    if (!p) return;

    // Create Draggable Card
    const card = document.createElement('div');
    card.className = 'hero-card';
    card.style.minWidth = '120px';
    card.style.width = '120px';
    card.style.padding = '10px';
    card.style.cursor = 'grab';
    card.setAttribute('draggable', 'true'); // Explicit attribute

    // Drag Start
    card.ondragstart = (e) => {
      console.log('Drag Start:', p.ID);
      e.dataTransfer.setData('playerId', String(p.ID));
      e.dataTransfer.effectAllowed = 'copyMove';
      card.style.opacity = '0.5';
    };

    card.ondragend = () => {
      console.log('Drag End');
      card.style.opacity = '1';
    };

    // Remove Button
    const removeBtn = document.createElement('div');
    removeBtn.className = 'tiny';
    removeBtn.style.position = 'absolute';
    removeBtn.style.top = '-8px';
    removeBtn.style.left = '-8px';
    removeBtn.style.background = 'var(--accent-rose)';
    removeBtn.style.color = '#fff';
    removeBtn.style.width = '20px';
    removeBtn.style.height = '20px';
    removeBtn.style.borderRadius = '50%';
    removeBtn.style.display = 'flex';
    removeBtn.style.alignItems = 'center';
    removeBtn.style.justifyContent = 'center';
    removeBtn.style.cursor = 'pointer';
    removeBtn.innerHTML = '×';
    removeBtn.onclick = () => toggleSquad(id);

    // Card Content
    card.innerHTML = `
      <div style="position:relative;">
        <img src="${p.card}" alt="${p.Name}" style="width:100%; height:80px; object-fit:contain; margin-bottom:8px; pointer-events:none;" draggable="false"/>
        <div class="tiny" style="position:absolute; top:0; right:0; background:rgba(0,0,0,0.6); padding:2px 4px; border-radius:4px; pointer-events:none;">${p.Position}</div>
      </div>
      <div style="font-weight:700; font-size:0.9rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; pointer-events:none;">${p.Name}</div>
      <div class="text-gradient" style="font-weight:800; pointer-events:none;">${p.OVR}</div>
    `;
    card.appendChild(removeBtn);
    pool.appendChild(card);
  });
}

function clearFormation() {
  formationAssignments = {};
  updatePitchAssignments();
  document.getElementById('teamStrength').textContent = '—';
  document.getElementById('chemistryScore').textContent = '—';
  document.getElementById('chemistryHint').textContent = '';
  document.getElementById('formationSuggestions').textContent = 'Analyze formation to get fit checks and replacement suggestions.';
}

function assignToSlot(position, playerId) {
  formationAssignments = { ...formationAssignments, [position]: String(playerId) };
  updatePitchAssignments();
  // Clear analysis results to prompt re-analysis
  document.getElementById('teamStrength').textContent = '—';
  document.getElementById('chemistryScore').textContent = '—';
}

function slotPositions(formationKey) {
  const key = formationKey || "433";
  const slots = formations[key] || formations["433"];
  // We distribute players into rows: Attack, Midfield, Defense, GK

  const rows = { 'ATT': [], 'MID': [], 'DEF': [], 'GK': [] };

  slots.forEach((pos, idx) => {
    // Create Unique Key: POS_INDEX
    const uniqueKey = `${pos}_${idx}`;
    if (pos === 'GK') rows['GK'].push({ pos, idx, uniqueKey });
    else if (['LB', 'RB', 'CB', 'LWB', 'RWB'].includes(pos)) rows['DEF'].push({ pos, idx, uniqueKey });
    else if (['CDM', 'CM', 'CAM', 'LM', 'RM'].includes(pos)) rows['MID'].push({ pos, idx, uniqueKey });
    else rows['ATT'].push({ pos, idx, uniqueKey });
  });

  const layout = [];

  const placeRow = (list, topPct) => {
    const count = list.length;
    list.forEach((item, i) => {
      const leftPct = (100 / (count + 1)) * (i + 1);
      layout.push({
        key: item.uniqueKey, // "CB_7" 
        label: item.pos,
        top: `${topPct}%`,
        left: `${leftPct}%`
      });
    });
  };

  placeRow(rows['ATT'], 15);
  placeRow(rows['MID'], 40);
  placeRow(rows['DEF'], 70);
  placeRow(rows['GK'], 90);

  return layout;
}

function renderPitch() {
  const pitch = document.getElementById('pitch');
  const formationSelect = document.getElementById('formationSelect');
  if (!pitch) return;
  pitch.innerHTML = '';
  pitch.setAttribute('data-drop-hint', 'Drop selected player chips here');
  const key = formationSelect ? formationSelect.value : '433';
  const layout = slotPositions(key);

  layout.forEach(slot => {
    const el = document.createElement('div');
    el.className = 'pitch-slot';
    el.style.top = slot.top;
    el.style.left = slot.left;
    el.dataset.key = slot.key; // Unique key "CB_7"
    el.innerHTML = `<div class="slot-pos">${slot.label}</div><div class="slot-player muted small">Drop player</div>`;

    // Improved Drag Handling
    el.ondragover = (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy'; // Explicit feedback
      el.classList.add('drag-over');
    };
    el.ondragenter = (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      el.classList.add('drag-over');
    };
    el.ondragleave = () => {
      el.classList.remove('drag-over');
    };

    el.ondrop = (e) => {
      e.preventDefault();
      el.classList.remove('drag-over');
      const pid = e.dataTransfer.getData('playerId');
      console.log('Drop:', pid, 'on', slot.key);
      if (pid) assignToSlot(slot.key, pid); // Assign to "CB_7"
    };

    el.onclick = () => {
      // If empty, maybe trigger search or suggestions?
    };

    pitch.appendChild(el);
  });
  updatePitchAssignments();
}

function assignToSlot(uniqueKey, playerId) {
  formationAssignments = { ...formationAssignments, [uniqueKey]: String(playerId) };
  updatePitchAssignments();
  // Clear analysis results to prompt re-analysis
  document.getElementById('teamStrength').textContent = '—';
  document.getElementById('chemistryScore').textContent = '—';
}

function updatePitchAssignments() {
  const pitch = document.getElementById('pitch');
  if (!pitch) return;
  Array.from(pitch.querySelectorAll('.pitch-slot')).forEach(slot => {
    const key = slot.dataset.key; // "CB_7"
    const pid = formationAssignments[key];
    if (pid) {
      const player = playerMap.get(String(pid));
      slot.classList.add('assigned');
      slot.innerHTML = `
          <div style="font-weight:700; font-size:0.8rem;">${player?.Name || pid}</div>
          <div class="tiny" style="color:${player ? 'var(--accent-cyan)' : 'inherit'}">${player?.OVR || '?'}</div>
      `;
      slot.onclick = () => {
        delete formationAssignments[key];
        updatePitchAssignments();
      };
      slot.title = "Click to remove";
    } else {
      slot.classList.remove('assigned');
      const label = key.split('_')[0]; // "CB"
      slot.innerHTML = `<div class="slot-pos">${label}</div><div class="slot-player muted tiny">Empty</div>`;
      slot.classList.remove('drag-over');
      slot.onclick = null;
      slot.title = "";
    }
  });
}

async function fetchFormationAnalysis() {
  const formationSelect = document.getElementById('formationSelect');
  const key = formationSelect ? formationSelect.value : '433';

  // We need to send `formation` as a list of "POS_IDX" strings 
  // and `assignments` keyed by "POS_IDX".
  // The Backend logic (app.py) likely iterates `formation` list and looks up `assignments`.
  // If we send "CB_7", app.py `_fit_penalty` needs to handle it.
  // Wait, I should verify app.py _fit_penalty logic.
  // It probably expects strict position strings "CB".
  // Let's assume app.py isn't smart enough to strip suffixes.
  // BUT: if we modify app.py, that's fine.
  // Actually, modifying app.py to split('_')[0] is safer than doing complex frontend mapping.
  // Let's stick to unique keys here.

  const layout = slotPositions(key);
  const formationList = layout.map(l => l.key); // ["LW_0", "ST_1", ...]

  const payload = { formation: formationList, assignments: formationAssignments };

  const btn = document.getElementById('analyzeFormation');
  if (btn) btn.textContent = 'Analyzing...';

  try {
    const data = await fetchJson('/api/formation/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (data.team_strength) {
      document.getElementById('teamStrength').textContent = data.team_strength.toFixed(0);
    }
    if (data.chemistry_score) {
      document.getElementById('chemistryScore').textContent = data.chemistry_score.toFixed(0);
    }

    const suggestionEl = document.getElementById('formationSuggestions');
    if (suggestionEl) {
      suggestionEl.innerHTML = '';
      // chemistry suggestion
      if (data.chemistry_suggestion) {
        const cs = data.chemistry_suggestion;
        const div = document.createElement('div');
        div.className = 'small';
        div.style.marginBottom = '8px';
        div.innerHTML = `<b style="color:var(--accent-rose)">Chemistry Tip:</b> Swap ${playerMap.get(String(cs.replace_player_id))?.Name || 'Player'} with <b style="color:var(--accent-cyan)">${cs.suggested_name}</b> for better fit.`;
        suggestionEl.appendChild(div);
      }

      // Slot suggestions
      const list = document.createElement('div');
      list.className = 'tiny muted';
      list.textContent = 'Fit penalties applied for out-of-position players.';
      suggestionEl.appendChild(list);
    }

    // Update visuals with fit info?
    // We could color code the slots based on `data.slots[i].fit` ("Natural fit", "Off line")
    data.slots.forEach(slot => {
      // Find the DOM element
      // slot.position is our unique key e.g. "CB_8"
      const el = document.querySelector(`.pitch-slot[data-key="${slot.position}"]`);
      if (el && slot.player_id) {
        if (slot.penalty < 0) {
          el.style.borderColor = 'var(--accent-rose)';
          el.title = `Penalty: ${slot.penalty} (${slot.fit})`;
        } else {
          el.style.borderColor = 'var(--accent-cyan)';
          el.title = 'Perfect Fit';
        }
      }
    });

  } catch (err) {
    console.error(err);
    document.getElementById('formationSuggestions').textContent = 'Analysis failed: ' + err.message;
  } finally {
    if (btn) btn.textContent = 'Analyze Formation';
  }
}

// Event bindings for formation UI
const formationSelectEl = document.getElementById('formationSelect');
if (formationSelectEl) formationSelectEl.addEventListener('change', () => { clearFormation(); renderPitch(); });
const clearFormationBtn = document.getElementById('clearFormation');
if (clearFormationBtn) clearFormationBtn.addEventListener('click', clearFormation);
const analyzeFormationBtn = document.getElementById('analyzeFormation');
if (analyzeFormationBtn) analyzeFormationBtn.addEventListener('click', fetchFormationAnalysis);

renderPitch();

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
