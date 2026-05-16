/* shared.js — common state and helpers for every xG Dashboard page */

const API = "";

// ── Global state ───────────────────────────────────────────────────────────────
let _playerCache = [];
let _smAllShots = [];
let _smTypeFilter = "all";
let _smExclPen = false;
let _smSelectedPlayer = "";
let sortStates = {};

// ── API helper ─────────────────────────────────────────────────────────────────
async function get(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

// ── Utilities ──────────────────────────────────────────────────────────────────
function shortLeague(name) {
  return (
    {
      "Premier League": "EPL",
      "La Liga": "LAL",
      Bundesliga: "BUN",
      "Serie A": "SA",
      "Ligue 1": "L1",
    }[name] || name
  );
}

function esc(s) {
  return (s || "").replace(/'/g, "\\'");
}

function sortTable(tableId, col) {
  const tbl = document.getElementById(tableId);
  if (!tbl) return;
  const body = tbl.querySelector("tbody");
  const rows = Array.from(body.querySelectorAll("tr"));
  const key = `${tableId}_${col}`;
  sortStates[key] = !sortStates[key];
  const asc = sortStates[key];
  rows.sort((a, b) => {
    const av = a.cells[col]?.textContent.trim().replace(/[+%]/g, "") || "";
    const bv = b.cells[col]?.textContent.trim().replace(/[+%]/g, "") || "";
    const an = parseFloat(av),
      bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach((r) => body.appendChild(r));
}

// ── Filter selects ─────────────────────────────────────────────────────────────
function getFilters() {
  return {
    league: document.getElementById("fLeague")?.value || "",
    season: document.getElementById("fSeason")?.value || "",
    team: document.getElementById("fTeam")?.value || "",
  };
}

async function loadMeta() {
  try {
    const [leagues, seasons] = await Promise.all([
      get("/api/leagues"),
      get("/api/seasons"),
    ]);
    const lSel = document.getElementById("fLeague");
    if (lSel)
      leagues.forEach((l) =>
        lSel.insertAdjacentHTML(
          "beforeend",
          `<option value="${l}">${l}</option>`,
        ),
      );
    const sSel = document.getElementById("fSeason");
    if (sSel)
      seasons.forEach((s) =>
        sSel.insertAdjacentHTML(
          "beforeend",
          `<option value="${s}">${s}/${parseInt(s) + 1}</option>`,
        ),
      );
  } catch (e) {}
}

async function loadTeamFilter() {
  const { league, season } = getFilters();
  const sel = document.getElementById("fTeam");
  if (!sel) return;
  const cur = sel.value;
  sel.innerHTML = '<option value="">All Teams</option>';
  try {
    let url = "/api/teams?min_shots=0";
    if (league) url += `&league=${encodeURIComponent(league)}`;
    if (season) url += `&season=${encodeURIComponent(season)}`;
    const teams = await get(url);
    teams.forEach((t) =>
      sel.insertAdjacentHTML(
        "beforeend",
        `<option value="${t.team}" ${t.team === cur ? "selected" : ""}>${t.team}</option>`,
      ),
    );
  } catch (e) {}
}

async function checkStatus() {
  try {
    const d = await get("/api/status");
    const ts =
      d.last_updated !== "never"
        ? new Date(d.last_updated).toLocaleString()
        : "never";
    const el = document.getElementById("lastUpdated");
    if (el) el.textContent = ts;
  } catch (e) {}
}

async function triggerRefresh() {
  try {
    await get("/api/refresh");
  } catch (e) {}
}

// ── Sidebar nav: mark current page active ─────────────────────────────────────
function initNav() {
  const page = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav-item").forEach((a) => {
    const href = a.getAttribute("href") || "";
    if (href && page.startsWith(href.replace(".html", ""))) {
      a.classList.add("active");
    }
  });
}

// ── Player cache (used on players + shotmap pages) ────────────────────────────
async function ensurePlayerCache() {
  if (_playerCache.length) return;
  const { league, season } = getFilters();
  let url = "/api/players?min_shots=1&limit=2000";
  if (league) url += `&league=${encodeURIComponent(league)}`;
  if (season) url += `&season=${encodeURIComponent(season)}`;
  _playerCache = await get(url);
}

function filterPlayers(query) {
  if (!query || query.length < 2) return [];
  const q = query.toLowerCase();
  return _playerCache
    .filter((p) => p.player && p.player.toLowerCase().includes(q))
    .slice(0, 12);
}

// ── Shared init (runs on every page) ──────────────────────────────────────────
async function sharedInit() {
  initNav();
  await Promise.all([loadMeta(), checkStatus()]);
  document.getElementById("fLeague") &&
    (document.getElementById("fLeague").value = "");
  await loadTeamFilter();
}
