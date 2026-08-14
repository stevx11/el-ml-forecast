/* EuroLeague AI — client.
 *
 * Vanilla ES modules, no framework and no build step. The whole app is three
 * files served by the API container: nothing to install, nothing to keep in
 * sync with the backend, and no toolchain that can rot between sessions.
 *
 * The design principle throughout: never show a number without showing how much
 * to trust it. A projection appears with its range, its threshold probabilities
 * and its reasons, because a bare "14.2 points" invites far more confidence
 * than this or any model has earned.
 */

const API = "/api/v1";

const TARGETS = [
  { key: "MINUTES", short: "MIN", label: "Minutes", decimals: 1 },
  { key: "POINTS", short: "PTS", label: "Points", decimals: 1 },
  { key: "REBOUNDS", short: "REB", label: "Rebounds", decimals: 1 },
  { key: "ASSISTS", short: "AST", label: "Assists", decimals: 1 },
  { key: "THREE_POINTERS_MADE", short: "3PM", label: "Three-pointers", decimals: 1 },
];

const state = {
  view: "fixtures",
  season: null,
  windowDays: 14,
  fixtures: [],
  selectedGameId: null,
  projections: null,
  squadFilter: "all",
  sort: { key: "MINUTES", dir: "desc" },
  status: null,
};

/* ------------------------------------------------------------------ utils */

/** Escape text before it reaches innerHTML. Our own API is the only source
 *  here, but player names come from an upstream feed and "trusted source" is
 *  how injection bugs are always justified. */
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

async function api(path) {
  const response = await fetch(API + path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

const fmt = (value, decimals = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? "–"
    : Number(value).toFixed(decimals);

const pct = (value) =>
  value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`;

function dayKey(iso) {
  return new Date(iso).toLocaleDateString(undefined, {
    weekday: "long", day: "numeric", month: "long",
  });
}

function timeLabel(iso) {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function relative(iso) {
  const delta = new Date(iso) - Date.now();
  const days = Math.round(delta / 86400000);
  if (Math.abs(days) >= 1) {
    const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
    return rtf.format(days, "day");
  }
  const hours = Math.round(delta / 3600000);
  return new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(hours, "hour");
}

/** Confidence is relative, not absolute — see distribution.py. The bands are
 *  deliberately coarse so nobody reads 0.53 as a precise quantity. */
function confidenceBand(value) {
  if (value === null || value === undefined) return { cls: "mid", label: "—" };
  if (value >= 0.55) return { cls: "high", label: "High" };
  if (value >= 0.45) return { cls: "mid", label: "Typical" };
  return { cls: "low", label: "Low" };
}

/* --------------------------------------------------------------- fixtures */

async function loadStatus() {
  try {
    state.status = await api("/board/status");
  } catch {
    state.status = null;
  }
  const pill = $("#status-pill");
  if (!state.status) {
    pill.textContent = "API unavailable";
    pill.className = "pill warn";
    return;
  }
  const { active_models: models, predictions } = state.status;
  if (!models.length) {
    pill.textContent = "No trained models";
    pill.className = "pill warn";
    pill.title = "Run: euroleague-ai ml train";
  } else {
    pill.textContent = `${models.length} models · ${predictions.toLocaleString()} projections`;
    pill.className = "pill ok";
    pill.title = models.map((m) => `${m.target}: ${m.model}`).join("\n");
  }
}

async function loadSeasons() {
  const select = $("#season-select");
  try {
    const { seasons } = await api("/board/seasons");
    if (!seasons.length) {
      select.innerHTML = `<option>No data</option>`;
      return;
    }
    select.innerHTML = seasons
      .map((s) => `<option value="${esc(s.code)}">${esc(s.code)} — ${s.games} games</option>`)
      .join("");
    state.season = state.season && seasons.some((s) => s.code === state.season)
      ? state.season
      : seasons[0].code;
    select.value = state.season;
  } catch {
    select.innerHTML = `<option>Unavailable</option>`;
  }
}

async function loadFixtures() {
  const list = $("#fixture-list");
  list.innerHTML = `<div style="padding:16px">${
    Array.from({ length: 6 }, () => `<div class="skeleton skeleton-row"></div>`).join("")
  }</div>`;

  try {
    const data = await api(
      `/board/fixtures?days=${state.windowDays}` +
        (state.season ? `&season=${encodeURIComponent(state.season)}` : "")
    );
    state.fixtures = data.fixtures;
  } catch (error) {
    list.innerHTML = `<div class="empty" style="margin:16px">
      <div class="empty-icon">⚠</div>
      <h3>Could not reach the API</h3>
      <p>${esc(error.message)}</p></div>`;
    return;
  }

  renderFixtures();

  // Land on something useful: the soonest fixture that actually has projections,
  // otherwise the soonest fixture. Opening on an empty game teaches nothing.
  if (!state.selectedGameId && state.fixtures.length) {
    const preferred = state.fixtures.find((f) => f.predictions > 0) || state.fixtures[0];
    selectGame(preferred.game_id);
  }
}

function renderFixtures() {
  const list = $("#fixture-list");
  if (!state.fixtures.length) {
    list.innerHTML = `<div class="empty" style="margin:16px">
      <div class="empty-icon">📅</div>
      <h3>No fixtures in this window</h3>
      <p>Try a longer window, or ingest the season:</p>
      <code>euroleague-ai ingest all --season ${esc(state.season || "E2026")}</code>
    </div>`;
    return;
  }

  let html = "";
  let currentDay = null;
  for (const fixture of state.fixtures) {
    const day = dayKey(fixture.tip_off_utc);
    if (day !== currentDay) {
      currentDay = day;
      html += `<div class="day-label">${esc(day)}</div>`;
    }

    const finished = fixture.status === "FINISHED";
    const chip = finished
      ? `<span class="chip final">Final ${fixture.home_score ?? ""}–${fixture.away_score ?? ""}</span>`
      : fixture.predictions > 0
        ? `<span class="chip ready">Projected</span>`
        : `<span class="chip none">Not projected</span>`;

    html += `
      <button class="fixture" role="listitem" data-game="${esc(fixture.game_id)}"
              aria-current="${fixture.game_id === state.selectedGameId}">
        <div class="fixture-teams">
          <span>${esc(fixture.home_team)}</span>
          <span class="vs">vs</span>
          <span>${esc(fixture.away_team)}</span>
        </div>
        <div class="fixture-meta">
          <span class="faint">${esc(timeLabel(fixture.tip_off_utc))}</span>
          ${fixture.round ? `<span class="faint">· R${fixture.round}</span>` : ""}
          ${chip}
        </div>
      </button>`;
  }
  list.innerHTML = html;

  $$(".fixture", list).forEach((el) =>
    el.addEventListener("click", () => selectGame(el.dataset.game))
  );
}

/* ------------------------------------------------------------ projections */

async function selectGame(gameId) {
  state.selectedGameId = gameId;
  state.squadFilter = "all";
  renderFixtures();

  const panel = $("#game-panel");
  panel.innerHTML = `<div class="skeleton skeleton-row" style="height:60px;margin-bottom:16px"></div>` +
    Array.from({ length: 8 }, () => `<div class="skeleton skeleton-row"></div>`).join("");

  try {
    state.projections = await api(`/games/${gameId}/predictions`);
  } catch (error) {
    panel.innerHTML = `<div class="empty"><div class="empty-icon">⚠</div>
      <h3>Could not load projections</h3><p>${esc(error.message)}</p></div>`;
    return;
  }
  renderGame();
}

function renderGame() {
  const panel = $("#game-panel");
  const data = state.projections;
  const fixture = state.fixtures.find((f) => f.game_id === state.selectedGameId);
  if (!data) return;

  const home = data.home_team || fixture?.home_team || "Home";
  const away = data.away_team || fixture?.away_team || "Away";

  let html = `
    <div class="game-head">
      <div>
        <div class="matchup">
          <span>${esc(home)}</span><span class="at">vs</span><span>${esc(away)}</span>
        </div>
        <p class="muted" style="margin-top:6px">
          Tip-off ${esc(new Date(data.tip_off_utc).toLocaleString())} · ${esc(relative(data.tip_off_utc))}
        </p>
      </div>`;

  if (data.players.length) {
    const asOf = data.players[0].prediction_timestamp;
    html += `<div class="asof">
      <strong>Known as of ${esc(new Date(asOf).toLocaleString())}</strong><br />
      Built only from information available at that moment — no post-game data.
    </div>`;
  }
  html += `</div>`;

  if (!data.players.length) {
    const finished = fixture?.status === "FINISHED";
    panel.innerHTML = html + `<div class="empty" style="margin-top:24px">
      <div class="empty-icon">${finished ? "🏁" : "○"}</div>
      <h3>${finished ? "This game was never projected" : "No projections for this fixture yet"}</h3>
      <p>${finished
        ? "Predictions are written before tip-off. A finished game with none was played before the model was serving."
        : "Run the pipeline to project every fixture in the window:"}</p>
      ${finished ? "" : `<code>euroleague-ai ml predict --days ${state.windowDays}</code>`}
    </div>`;
    return;
  }

  // Squad filter. Team codes come from the feature snapshot, so the split is
  // exactly the one inference used rather than a fresh guess.
  const codes = [...new Set(data.players.map((p) => p.team_code).filter(Boolean))];
  if (codes.length > 1) {
    html += `<div class="squad-tabs">
      <button class="squad-tab" data-squad="all" aria-pressed="${state.squadFilter === "all"}">Both squads</button>
      ${codes.map((code) => `<button class="squad-tab" data-squad="${esc(code)}"
          aria-pressed="${state.squadFilter === code}">${esc(code)}</button>`).join("")}
    </div>`;
  }

  const players = data.players.filter(
    (p) => state.squadFilter === "all" || p.team_code === state.squadFilter
  );

  const { key, dir } = state.sort;
  players.sort((a, b) => {
    const av = a.targets?.[key] ?? -1;
    const bv = b.targets?.[key] ?? -1;
    return dir === "desc" ? bv - av : av - bv;
  });

  html += `<div class="table-wrap"><table>
    <thead><tr>
      <th>Player</th>
      ${TARGETS.map((t) => `<th class="sortable" data-sort="${t.key}"
          aria-sort="${key === t.key ? (dir === "desc" ? "descending" : "ascending") : "none"}"
          title="${esc(t.label)}">${t.short}</th>`).join("")}
      <th title="Probability of scoring 15 or more">P(15+)</th>
      <th title="How tight this projection is relative to a typical one">Conf</th>
    </tr></thead><tbody>`;

  for (const player of players) {
    const points = player.projections?.POINTS;
    const p15 = points?.threshold_probabilities?.["15"];
    const confidence = points?.confidence ?? player.projections?.MINUTES?.confidence;
    const band = confidenceBand(confidence);

    html += `<tr data-player="${esc(player.player_id)}">
      <td><div class="player-cell">
        ${player.team_code ? `<span class="team-tag">${esc(player.team_code)}</span>` : ""}
        <span class="player-name" title="${esc(player.player_name)}">${esc(player.player_name)}</span>
      </div></td>
      ${TARGETS.map((t) => {
        const value = player.targets?.[t.key];
        const lead = t.key === "MINUTES" || t.key === "POINTS" ? " lead" : "";
        return `<td class="num${lead}">${fmt(value, t.decimals)}</td>`;
      }).join("")}
      <td><div class="prob">
        <span class="prob-track"><span class="prob-fill" style="width:${Math.round((p15 ?? 0) * 100)}%"></span></span>
        <span class="prob-value num">${pct(p15)}</span>
      </div></td>
      <td><span class="conf" title="${esc(band.label)}${confidence != null ? ` (${confidence.toFixed(2)})` : ""}">
        <span class="conf-dot ${band.cls}"></span>${esc(band.label)}
      </span></td>
    </tr>`;
  }
  html += `</tbody></table></div>
    <p class="faint" style="margin-top:10px">
      Select a player for the full distribution, threshold probabilities and the reasons behind the projection.
      Confidence is relative to a typical projection, not an absolute probability.
    </p>`;

  panel.innerHTML = html;

  $$(".squad-tab", panel).forEach((el) =>
    el.addEventListener("click", () => { state.squadFilter = el.dataset.squad; renderGame(); })
  );
  $$("th.sortable", panel).forEach((el) =>
    el.addEventListener("click", () => {
      const next = el.dataset.sort;
      state.sort = state.sort.key === next
        ? { key: next, dir: state.sort.dir === "desc" ? "asc" : "desc" }
        : { key: next, dir: "desc" };
      renderGame();
    })
  );
  $$("tbody tr", panel).forEach((el) =>
    el.addEventListener("click", () => openDrawer(el.dataset.player))
  );
}

/* ----------------------------------------------------------------- drawer */

function openDrawer(playerId) {
  const player = state.projections?.players.find((p) => p.player_id === playerId);
  if (!player) return;

  $("#drawer-title").textContent = player.player_name;
  $("#drawer-sub").textContent =
    `${player.team_code || ""} · projected ${new Date(player.prediction_timestamp).toLocaleString()}`;

  let html = "";
  for (const target of TARGETS) {
    const projection = player.projections?.[target.key];
    if (!projection) continue;

    html += `<div class="target-block">
      <div class="target-head">
        <span class="target-name">${esc(target.label)}</span>
        <span class="target-value num">${fmt(projection.value, target.decimals)}</span>
      </div>`;

    const q = projection.quantiles;
    if (q) {
      // The 10-90 band with the interquartile range darker inside it, median
      // marked. Showing the spread beside the number is the whole point: the
      // median alone hides everything interesting about a streaky shooter.
      const lo = q["0.1"] ?? 0;
      const hi = q["0.9"] ?? 0;
      const q1 = q["0.25"] ?? lo;
      const q3 = q["0.75"] ?? hi;
      const median = q["0.5"] ?? projection.value;
      const span = Math.max(hi - lo, 0.01);
      const pos = (v) => `${Math.max(0, Math.min(100, ((v - lo) / span) * 100))}%`;

      html += `<div class="dist">
        <div class="dist-bar">
          <div class="dist-range" style="left:0;right:0"></div>
          <div class="dist-inner" style="left:${pos(q1)};width:${
            Math.max(0, Math.min(100, ((q3 - q1) / span) * 100))
          }%"></div>
          <div class="dist-median" style="left:${pos(median)}"></div>
        </div>
        <div class="dist-scale">
          <span class="num">${fmt(lo, target.decimals)}</span>
          <span>80% of outcomes</span>
          <span class="num">${fmt(hi, target.decimals)}</span>
        </div>
      </div>`;
    }

    const thresholds = projection.threshold_probabilities;
    if (thresholds && Object.keys(thresholds).length) {
      html += `<div class="thresholds">`;
      for (const [bar, probability] of Object.entries(thresholds).sort(
        (a, b) => Number(a[0]) - Number(b[0])
      )) {
        html += `<div class="threshold-row">
          <span class="threshold-label num">${esc(bar)}+</span>
          <span class="prob-track" style="width:100%">
            <span class="prob-fill" style="width:${Math.round(probability * 100)}%"></span>
          </span>
          <span class="num" style="text-align:right">${pct(probability)}</span>
        </div>`;
      }
      html += `</div>`;
    }

    if (projection.drivers?.length) {
      html += `<div class="section-title" style="margin:16px 0 8px">Why</div><div class="drivers">`;
      for (const driver of projection.drivers) {
        const positive = driver.direction === "+";
        html += `<div class="driver ${positive ? "pos" : "neg"}">
          <span class="driver-sign">${positive ? "+" : "−"}</span>
          <span>${esc(driver.label)}</span>
          <span class="driver-impact num">${driver.impact > 0 ? "+" : ""}${fmt(driver.impact, 2)}</span>
        </div>`;
      }
      html += `</div>`;
    }

    if (projection.confidence != null) {
      const band = confidenceBand(projection.confidence);
      html += `<p class="faint" style="margin-top:10px">
        Confidence ${band.label.toLowerCase()} (${projection.confidence.toFixed(2)}) ·
        spread ±${fmt(projection.predicted_std, 1)}
      </p>`;
    }
    html += `</div>`;
  }

  if (!html) {
    html = `<div class="empty"><div class="empty-icon">○</div>
      <h3>No detail stored</h3>
      <p>This projection came from a model trained before distributions and
      explanations existed. Retrain and predict again to populate it.</p>
      <code>euroleague-ai ml train &amp;&amp; euroleague-ai ml predict</code></div>`;
  } else {
    html += `<p class="faint" style="margin-top:14px">
      Reasons describe what moved <em>the model</em> for these inputs. They are not
      claims about cause.</p>`;
  }

  $("#drawer-body").innerHTML = html;
  $("#drawer").hidden = false;
  $("#drawer-scrim").hidden = false;
  $("#drawer-close").focus();
}

function closeDrawer() {
  $("#drawer").hidden = true;
  $("#drawer-scrim").hidden = true;
}

/* --------------------------------------------------------------- accuracy */

async function loadAccuracy() {
  const panel = $("#accuracy-panel");
  panel.innerHTML = Array.from({ length: 4 }, () =>
    `<div class="skeleton skeleton-row" style="height:70px;margin-bottom:10px"></div>`
  ).join("");

  let report;
  try {
    report = await api(
      `/board/accuracy${state.season ? `?season=${encodeURIComponent(state.season)}` : ""}`
    );
  } catch (error) {
    panel.innerHTML = `<div class="empty"><div class="empty-icon">⚠</div>
      <h3>Could not load accuracy</h3><p>${esc(error.message)}</p></div>`;
    return;
  }

  const targets = report.targets || {};
  if (!Object.keys(targets).length) {
    panel.innerHTML = `
      <h2 style="margin-bottom:6px">Realised accuracy</h2>
      <p class="muted" style="margin-bottom:20px">How the published projections actually did.</p>
      <div class="empty">
        <div class="empty-icon">📊</div>
        <h3>Nothing scored yet</h3>
        <p>Accuracy appears once fixtures have been played and their box scores ingested.
        This is the only measurement not computed from the same pipeline as the model,
        which is what makes it worth more than the training report.</p>
        <code>euroleague-ai ml score</code>
      </div>`;
    return;
  }

  let html = `
    <h2 style="margin-bottom:6px">Realised accuracy</h2>
    <p class="muted" style="margin-bottom:18px">
      Measured against played fixtures — independent of the training pipeline.
    </p>
    <div class="note">
      <strong>Non-appearances are excluded.</strong> A player who never took the floor scores
      as zero and is counted separately; blaming a points model for a rotation decision would
      make it look worse than it is, and hiding those projections would make it look better.
    </div>
    <div class="cards">`;

  for (const [name, block] of Object.entries(targets)) {
    const covered = block.interval?.covered_80;
    html += `<div class="card">
      <div class="card-label">${esc(name.replace(/_/g, " "))}</div>
      <div class="card-value num">${fmt(block.mae, 3)}</div>
      <div class="card-note">MAE over ${block.scored.toLocaleString()} scored ·
        bias ${block.bias > 0 ? "+" : ""}${fmt(block.bias, 2)}</div>
      ${covered != null ? `<div class="card-note">80% interval held ${pct(covered)}</div>` : ""}
    </div>`;
  }
  html += `</div>`;

  const bands = report.thresholds || [];
  if (bands.length) {
    html += `<div class="section-title">Threshold calibration</div>
      <p class="muted" style="margin:-4px 0 12px">
        Of everything we called 30%, did about 30% happen? This needs no model — just counting.
      </p>
      <div class="table-wrap"><table>
        <thead><tr><th>Stated</th><th>Predictions</th><th>Realised</th><th>Drift</th></tr></thead>
        <tbody>`;
    for (const band of bands) {
      const drift = band.realised - band.expected;
      const colour = Math.abs(drift) > 0.1 ? "var(--warn)" : "var(--text-dim)";
      html += `<tr style="cursor:default">
        <td>${esc(band.band)}</td>
        <td class="num">${band.predictions.toLocaleString()}</td>
        <td class="num">${pct(band.realised)}</td>
        <td class="num" style="color:${colour}">${drift > 0 ? "+" : ""}${pct(drift)}</td>
      </tr>`;
    }
    html += `</tbody></table></div>`;
  }

  if (report.unscored_games) {
    html += `<div class="note" style="margin-top:16px;border-left-color:var(--warn)">
      <strong>${report.unscored_games} finished game(s) not yet scored.</strong>
      Run <code style="all:unset;font-family:var(--mono);color:var(--accent)">euroleague-ai ml score</code>.
    </div>`;
  }

  panel.innerHTML = html;
}

/* ------------------------------------------------------------------- wire */

function switchView(view) {
  state.view = view;
  $$(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.view === view)));
  $("#view-fixtures").hidden = view !== "fixtures";
  $("#view-accuracy").hidden = view !== "accuracy";
  if (view === "accuracy") loadAccuracy();
}

function wire() {
  $$(".tab").forEach((tab) =>
    tab.addEventListener("click", () => switchView(tab.dataset.view))
  );
  $("#season-select").addEventListener("change", (event) => {
    state.season = event.target.value;
    state.selectedGameId = null;
    loadFixtures();
    if (state.view === "accuracy") loadAccuracy();
  });
  $("#window-select").addEventListener("change", (event) => {
    state.windowDays = Number(event.target.value);
    state.selectedGameId = null;
    loadFixtures();
  });
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#drawer-scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });
}

async function boot() {
  wire();
  await loadStatus();
  await loadSeasons();
  await loadFixtures();
}

boot();
