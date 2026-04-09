/**
 * Researcher Finder — static frontend
 * Loads config.json (optional embedApiUrl for production / CORS-safe embeddings).
 */
(function () {
  const LS_KEY = "jpal_researcher_finder_openai_key";

  let config = {
    embedApiUrl: "",
    indexUrl: "",
    pageSize: 30,
    relevanceFloor: 0.12,
    relevanceMargin: 0.1,
  };

  let indexData = null;
  /** @type {{ allResults: Array<{r: object, score: number, semantic: number, boost: number, matches: object}>, page: number } | null} */
  let searchResultsState = null;

  function assetUrl(relPath) {
    try {
      const baseEl = document.querySelector("base[href]");
      const base = baseEl && baseEl.href ? baseEl.href : window.location.href;
      return new URL(relPath, base).href;
    } catch {
      return relPath;
    }
  }

  function indexUrl() {
    const u = (config.indexUrl || "").trim();
    return u || assetUrl("profiles_index.json");
  }

  const STOP = new Set([
    "a", "an", "the", "in", "on", "at", "of", "for", "to", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "by", "from", "that", "this", "it",
    "its", "i", "about", "into", "research", "study", "studies", "experiment",
    "experiments", "evidence", "impact", "effect", "effects", "using", "use",
  ]);

  const BOOST_WEIGHTS = {
    "Specific Country Interest": 0.12,
    "Research Interests (open text)": 0.10,
    "Website & publications (keyword index)": 0.10,
    institution: 0.08,
    Sectors: 0.06,
    "Regional Office Affiliation": 0.06,
    Languages: 0.05,
    offices: 0.05,
    "Regional interest": 0.04,
    "Web Bio": 0.04,
    "Publication Notes": 0.03,
    Initiatives: 0.03,
    initiatives: 0.02,
    "Researcher Type": 0.02,
    "Sector/Initiative interest": 0.02,
    "Related Initiative(s)": 0.01,
  };
  const BOOST_CAP = 0.4;

  const queryInput = document.getElementById("queryInput");
  const searchForm = document.getElementById("searchForm");
  const searchBtn = document.getElementById("searchBtn");
  const statusBar = document.getElementById("statusBar");
  const resultsContainer = document.getElementById("resultsContainer");
  const resultsMeta = document.getElementById("resultsMeta");
  const deployBanner = document.getElementById("deployBanner");
  const settingsBtn = document.getElementById("settingsBtn");
  const modalOverlay = document.getElementById("modalOverlay");
  const modalCancel = document.getElementById("modalCancel");
  const modalSave = document.getElementById("modalSave");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const modalKeySection = document.getElementById("modalKeySection");
  const modalProxyNote = document.getElementById("modalProxyNote");
  const filterCountry = document.getElementById("filterCountry");
  const filterOffice = document.getElementById("filterOffice");
  const filterLanguage = document.getElementById("filterLanguage");
  const filterName = document.getElementById("filterName");
  const filterUniversity = document.getElementById("filterUniversity");
  const filterSector = document.getElementById("filterSector");
  const filterType = document.getElementById("filterType");
  const resultsPager = document.getElementById("resultsPager");

  function getApiKey() {
    return localStorage.getItem(LS_KEY) || "";
  }
  function setApiKey(k) {
    localStorage.setItem(LS_KEY, k.trim());
  }

  function usesProxy() {
    return Boolean((config.embedApiUrl || "").trim());
  }

  function setStatus(msg, isError) {
    statusBar.textContent = msg;
    statusBar.className = "status-bar" + (isError ? " error" : "");
  }
  function setStatusHTML(html) {
    statusBar.innerHTML = html;
    statusBar.className = "status-bar";
  }

  async function loadConfig() {
    try {
      const r = await fetch(assetUrl("config.json"), { cache: "no-store" });
      if (r.ok) {
        const j = await r.json();
        if (typeof j.embedApiUrl === "string") config.embedApiUrl = j.embedApiUrl;
        if (typeof j.indexUrl === "string") config.indexUrl = j.indexUrl;
        if (typeof j.pageSize === "number" && j.pageSize > 0) config.pageSize = Math.min(100, j.pageSize);
        if (typeof j.relevanceFloor === "number" && j.relevanceFloor >= 0) config.relevanceFloor = j.relevanceFloor;
        if (typeof j.relevanceMargin === "number" && j.relevanceMargin >= 0) config.relevanceMargin = j.relevanceMargin;
        if (typeof j.topN === "number" && j.topN > 0) config.pageSize = Math.min(100, j.topN);
      }
    } catch {
      /* optional file */
    }
    if (usesProxy()) {
      deployBanner.hidden = false;
      deployBanner.textContent =
        "Embedding requests use the configured API endpoint (no personal OpenAI key required).";
      if (modalKeySection) modalKeySection.hidden = true;
      if (modalProxyNote) modalProxyNote.hidden = false;
    }
  }

  async function loadIndex() {
    if (indexData) return indexData;
    setStatusHTML('<span class="spinner"></span> Loading researcher index…');
    const url = indexUrl();
    const resp = await fetch(url, { cache: "no-cache" });
    if (!resp.ok) {
      throw new Error(`Could not load index (${resp.status}). Check paths and config.json indexUrl.`);
    }
    indexData = await resp.json();
    setStatus("");
    return indexData;
  }

  async function embedQuery(text, model, apiKey) {
    const payload = JSON.stringify({ model, input: text });
    const proxy = (config.embedApiUrl || "").trim();
    if (proxy) {
      const resp = await fetch(proxy, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error?.message || err.error || `Embedding proxy error ${resp.status}`);
      }
      const data = await resp.json();
      if (!data.data?.[0]?.embedding) {
        throw new Error("Invalid response from embedding proxy.");
      }
      return data.data[0].embedding;
    }

    if (!apiKey) {
      throw new Error(
        "No embedding method: add embedApiUrl in config.json (see deploy/cloudflare-embed-worker.js) " +
          "or set an OpenAI API key under Settings."
      );
    }

    const resp = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: payload,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const msg = err.error?.message || `OpenAI error ${resp.status}`;
      if (resp.status === 0 || msg.toLowerCase().includes("fetch")) {
        throw new Error(
          "Network error calling OpenAI. Browsers often block this (CORS). " +
            "Deploy the Cloudflare worker in deploy/ and set embedApiUrl in config.json."
        );
      }
      throw new Error(msg);
    }
    const data = await resp.json();
    return data.data[0].embedding;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  function norm(v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
  }
  function cosine(a, b) {
    const na = norm(a);
    const nb = norm(b);
    return na && nb ? dot(a, b) / (na * nb) : 0;
  }

  function extractTerms(query) {
    const m = query.toLowerCase().match(/[a-zA-Z]{3,}/g);
    const words = m ? m.filter((w) => !STOP.has(w)) : [];
    // “RCT” rarely appears in prose; match “randomized” / “randomised” too.
    if (words.includes("rct")) {
      if (!words.includes("randomized")) words.push("randomized");
      if (!words.includes("randomised")) words.push("randomised");
    }
    return words;
  }

  function keywordBoost(researcher, terms) {
    if (!terms.length) return { boost: 0, matches: {} };
    const fields = researcher.key_fields || {};
    let boost = 0;
    const matches = {};
    for (const [field, weight] of Object.entries(BOOST_WEIGHTS)) {
      const text = (fields[field] || "").toLowerCase();
      if (!text) continue;
      const hits = terms.filter((t) => text.includes(t));
      if (hits.length) {
        boost += weight * hits.length;
        matches[field] = hits;
      }
    }
    return { boost: Math.min(boost, BOOST_CAP), matches };
  }

  function normalizeNameTokens(s) {
    return String(s || "")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  /** CSV / semicolon: any segment may match (OR). */
  function orListMatches(haystackLower, filterCsv) {
    if (!filterCsv || !String(filterCsv).trim()) return true;
    const parts = String(filterCsv)
      .split(/[,;|]/)
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);
    if (!parts.length) return true;
    return parts.some((p) => haystackLower.includes(p));
  }

  function nameFilterMatches(researcherName, filterStr) {
    if (!filterStr || !String(filterStr).trim()) return true;
    const n = normalizeNameTokens(researcherName);
    const tokens = normalizeNameTokens(filterStr)
      .split(/\s+/)
      .filter((t) => t.length > 0);
    if (!tokens.length) return true;
    return tokens.every((t) => n.includes(t));
  }

  function readManualFilters() {
    const g = (el) => (el && el.value ? el.value.trim() : "");
    return {
      country: g(filterCountry),
      office: g(filterOffice),
      language: g(filterLanguage),
      name: g(filterName),
      university: g(filterUniversity),
      sector: g(filterSector),
      type: g(filterType),
    };
  }

  /**
   * Strip key:value and key:"value" from query; return topic text + parsed filters.
   */
  function parseInlineFilters(raw) {
    const filters = {
      country: "",
      office: "",
      language: "",
      name: "",
      university: "",
      sector: "",
      type: "",
    };
    let topic = String(raw || "");
    const re =
      /(?:^|\s)(country|nation|office|lang|language|name|university|school|institution|sector|type)\s*:\s*("([^"]+)"|[^\s]+)/gi;
    let m;
    const map = {
      country: "country",
      nation: "country",
      office: "office",
      lang: "language",
      language: "language",
      name: "name",
      university: "university",
      school: "university",
      institution: "university",
      sector: "sector",
      type: "type",
    };
    while ((m = re.exec(raw)) !== null) {
      const key = map[m[1].toLowerCase()];
      const val = (m[3] != null ? m[3] : m[4] || "").trim();
      if (key && val) filters[key] = val;
      topic = topic.replace(m[0], " ");
    }
    topic = topic.replace(/\s+/g, " ").trim();
    return { topic, filters };
  }

  /** Manual filter inputs override the same key from inline query syntax. */
  function mergeFilters(manual, parsed) {
    return {
      country: manual.country ? manual.country : parsed.country || "",
      office: manual.office ? manual.office : parsed.office || "",
      language: manual.language ? manual.language : parsed.language || "",
      name: manual.name ? manual.name : parsed.name || "",
      university: manual.university ? manual.university : parsed.university || "",
      sector: manual.sector ? manual.sector : parsed.sector || "",
      type: manual.type ? manual.type : parsed.type || "",
    };
  }

  function anyFilterActive(f) {
    return Boolean(f.country || f.office || f.language || f.name || f.university || f.sector || f.type);
  }

  /**
   * Known school tokens → phrases / tokens searched in affiliation haystack.
   * Fixes OpenAlex using full names without a standalone "mit" token, etc.
   */
  const INSTITUTION_QUERY_ALIASES = {
    mit: [
      "mit",
      "massachusetts institute of technology",
      "massachusetts institute",
      "m.i.t",
    ],
    massachusetts: [
      "massachusetts institute of technology",
      "mit",
      "m.i.t",
    ],
    harvard: ["harvard university", "harvard school", "harvard college", " harvard ", "harvard business school"],
    stanford: ["stanford university", " stanford "],
    princeton: ["princeton university", " princeton "],
    yale: ["yale university", " yale "],
    columbia: ["columbia university", "columbia business school"],
    berkeley: [
      "university of california, berkeley",
      "uc berkeley",
      " ucb ",
      "berkeley ",
    ],
    uchicago: ["university of chicago", " uchicago ", "u of chicago"],
    chicago: ["university of chicago", " uchicago ", "u of chicago", "chicago booth", "booth school"],
    duke: ["duke university", " duke "],
    penn: ["university of pennsylvania", " upenn ", "upenn", " wharton"],
    upenn: ["university of pennsylvania", " upenn ", "upenn", " wharton"],
    northwestern: ["northwestern university", " northwestern "],
    cornell: ["cornell university", " cornell "],
    brown: ["brown university", " brown university"],
    dartmouth: ["dartmouth college", "dartmouth university"],
    vanderbilt: ["vanderbilt university", " vanderbilt "],
    rice: ["rice university", " rice university"],
    georgetown: ["georgetown university", " georgetown "],
    cmu: ["carnegie mellon", "carnegie-mellon"],
    carnegiemellon: ["carnegie mellon", "carnegie-mellon"],
    notre: ["university of notre dame", "notre dame"],
    notredame: ["university of notre dame", "notre dame"],
    ucla: ["university of california, los angeles", "ucla"],
    ucsd: ["university of california, san diego", "uc san diego", "ucsd"],
    michigan: ["university of michigan", " u of michigan"],
    wisconsin: ["university of wisconsin", " uw-madison", "uw madison"],
    virginia: ["university of virginia", " uva ", " u.v.a."],
    unc: ["university of north carolina", " unc chapel hill", "unc-chapel hill"],
    jhu: ["johns hopkins", "john hopkins"],
    hopkins: ["johns hopkins", "john hopkins"],
    oxford: ["university of oxford", " oxford university"],
    cambridge: ["university of cambridge", " cambridge university"],
    lse: ["london school of economics", "l.s.e."],
    nber: ["national bureau of economic research", " nber "],
    iza: ["iza institute", " institute of labor economics"],
    nyu: ["new york university", " nyu ", "nyu "],
    bostoncollege: ["boston college", " boston college"],
    bostonuniversity: ["boston university", " boston university"],
  };

  function haystackMatchesInstitutionAliases(hay, hayWords, aliases) {
    for (const raw of aliases) {
      const al = String(raw).toLowerCase().trim();
      if (!al) continue;
      if (!/\s/.test(al) && al.length <= 8) {
        const tok = al.replace(/[^a-z0-9]/g, "");
        if (tok && hayWords.includes(tok)) return true;
      }
      const spaced = al.replace(/[^a-z0-9]+/g, " ").trim();
      if (spaced && hay.includes(spaced)) return true;
    }
    return false;
  }

  /**
   * Match affiliation text: whole-token for unknown short words, alias expansion for known schools.
   * querySpec: canonical key (e.g. "mit") or free-text ("notre dame", "london school").
   */
  function institutionQueryMatch(r, querySpec) {
    const hay = buildInstitutionHaystack(r);
    const hayWords = hay.split(/[^a-z0-9]+/).filter(Boolean);
    const raw = String(querySpec || "").trim().toLowerCase().replace(/\s+/g, " ");
    if (!raw) return false;

    if (INSTITUTION_QUERY_ALIASES[raw]) {
      return haystackMatchesInstitutionAliases(hay, hayWords, INSTITUTION_QUERY_ALIASES[raw]);
    }

    const qTokens = raw.split(/\s+/).filter(Boolean);
    if (qTokens.length === 1) {
      const w = qTokens[0].replace(/[^a-z0-9]/g, "");
      if (w.length < 2) return false;
      if (INSTITUTION_QUERY_ALIASES[w]) {
        return haystackMatchesInstitutionAliases(hay, hayWords, INSTITUTION_QUERY_ALIASES[w]);
      }
      if (hayWords.includes(w)) return true;
      if (w.length >= 8) return hay.includes(w);
      return false;
    }
    return hay.includes(raw);
  }

  /** Two-word prefixes that map to one institution key (avoid rest = second word of name). */
  const INSTITUTION_TWO_WORD_PREFIX = {
    "notre dame": "notre",
    "new york": "nyu",
    "boston college": "bostoncollege",
    "boston university": "bostonuniversity",
  };

  /** First word(s) are a known institution key → restrict pool, rest is embedding/topic. */
  function extractLeadingInstitutionKeyword(topic) {
    const words = String(topic || "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    if (!words.length) return null;
    if (words.length >= 2) {
      const pairRaw = `${words[0]} ${words[1]}`.toLowerCase();
      const pair = pairRaw.replace(/[^a-z\s]/g, "").replace(/\s+/g, " ").trim();
      const key2 = INSTITUTION_TWO_WORD_PREFIX[pair];
      if (key2 && INSTITUTION_QUERY_ALIASES[key2]) {
        return { key: key2, rest: words.slice(2).join(" ").trim() };
      }
    }
    const first = words[0].replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
    if (!first || !(first in INSTITUTION_QUERY_ALIASES)) return null;
    return { key: first, rest: words.slice(1).join(" ").trim() };
  }

  /** Short query that is plausibly a school / org name (not a research sentence). */
  function looksLikeShortInstitutionQuery(topic) {
    const t = String(topic || "").trim();
    if (t.length < 2 || t.length > 56) return false;
    if (/[0-9@]/.test(t)) return false;
    const words = t.split(/\s+/).filter(Boolean);
    if (words.length < 1 || words.length > 4) return false;
    const generic = new Set(["university", "college", "school", "institute", "department", "faculty"]);
    if (words.length === 1 && generic.has(words[0].toLowerCase())) return false;
    const first = words[0].replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
    if (first in INSTITUTION_QUERY_ALIASES) return true;
    if (
      /\b(rct|randomized|randomised|study|studies|impact|evidence|evaluation|trial|transfer|transfers)\b/i.test(
        t
      )
    ) {
      return false;
    }
    if (/\b(poverty|health|climate|education|economics|policy)\b/i.test(t) && words.length > 1) return false;
    return true;
  }

  /** Heuristic: query is probably a person name (avoids embedding for "First Last"). */
  function looksLikePersonNameQuery(s) {
    const words = String(s || "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    if (words.length < 2 || words.length > 4) return false;
    if (/[0-9:.@/]/.test(s)) return false;
    const small = new Set(["the", "and", "for", "in", "on", "of", "or", "to"]);
    for (const w of words) {
      if (w.length < 2 || small.has(w.toLowerCase())) return false;
      if (/^[a-z]/.test(w)) return false;
    }
    return true;
  }

  function buildCountryHaystack(r) {
    const kf = r.key_fields || {};
    return `${kf["Specific Country Interest"] || ""} ${kf["Regional interest"] || ""}`.toLowerCase();
  }

  function buildOfficeHaystack(r) {
    const kf = r.key_fields || {};
    return `${kf.offices || ""} ${kf["Regional Office Affiliation"] || ""}`.toLowerCase();
  }

  function buildLanguageHaystack(r) {
    const kf = r.key_fields || {};
    return `${kf.Languages || ""} ${kf["Web Bio"] || ""} ${kf["Research Interests (open text)"] || ""} ${
      kf["Website & publications (keyword index)"] || ""
    }`.toLowerCase();
  }

  /** Tight haystack: only the actual institutional affiliation. Used by institutionQueryMatch. */
  function buildInstitutionHaystack(r) {
    const kf = r.key_fields || {};
    return `${r.institution || ""} ${kf.institution || ""}`.toLowerCase();
  }

  /** Broader haystack for the manual university filter — includes bio but NOT the full keyword blob. */
  function buildUniversityHaystack(r) {
    const kf = r.key_fields || {};
    return `${r.institution || ""} ${kf.institution || ""} ${kf["Web Bio"] || ""} ${
      kf["Research Interests (open text)"] || ""
    }`.toLowerCase();
  }

  function buildSectorHaystack(r) {
    const kf = r.key_fields || {};
    return `${kf.Sectors || ""} ${kf["Sector/Initiative interest"] || ""} ${
      kf["Research Interests (open text)"] || ""
    }`.toLowerCase();
  }

  function buildTypeHaystack(r) {
    const kf = r.key_fields || {};
    return (kf["Researcher Type"] || "").toLowerCase();
  }

  function researcherPassesFilters(r, f) {
    if (!orListMatches(buildCountryHaystack(r), f.country)) return false;
    if (!orListMatches(buildOfficeHaystack(r), f.office)) return false;
    if (!orListMatches(buildLanguageHaystack(r), f.language)) return false;
    if (!nameFilterMatches(r.name || r.slug, f.name)) return false;
    if (!orListMatches(buildUniversityHaystack(r), f.university)) return false;
    if (!orListMatches(buildSectorHaystack(r), f.sector)) return false;
    if (!orListMatches(buildTypeHaystack(r), f.type)) return false;
    return true;
  }

  /**
   * Adaptive result selection using median + natural-gap detection.
   * Broad topics keep a wide set; narrow queries with a clear leader cut off early.
   */
  function selectRelevantScores(sortedDesc) {
    if (!sortedDesc.length) return [];
    const n = sortedDesc.length;
    const best = sortedDesc[0].score;
    const floor = config.relevanceFloor;

    const median = sortedDesc[Math.floor(n / 2)].score;
    const midpoint = (best + median) / 2;
    const gap = best - median;

    let cutoff;
    if (gap < 0.04) {
      cutoff = Math.max(floor, median - 0.02);
    } else {
      cutoff = Math.max(floor, midpoint);
    }

    let out = sortedDesc.filter((x) => x.score >= cutoff);

    for (let i = 1; i < out.length; i++) {
      const drop = out[i - 1].score - out[i].score;
      if (drop > gap * 0.6 && drop > 0.03 && i >= 3) {
        out = out.slice(0, i);
        break;
      }
    }

    if (out.length === 0) {
      out = sortedDesc.filter((x) => x.score >= Math.max(0.05, best - 0.22));
    }
    if (out.length === 0) out = sortedDesc.slice(0, 1);
    return out;
  }

  function renderResultsPage() {
    if (!searchResultsState) return;
    const { allResults, page } = searchResultsState;
    const pageSize = config.pageSize;
    const start = (page - 1) * pageSize;
    const slice = allResults.slice(start, start + pageSize);
    resultsContainer.innerHTML = "";
    slice.forEach((item, i) => renderCard(item, start + i + 1));
    renderPager();
  }

  function renderPager() {
    if (!resultsPager || !searchResultsState) return;
    const { allResults, page } = searchResultsState;
    const pageSize = config.pageSize;
    const total = allResults.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    if (totalPages <= 1) {
      resultsPager.hidden = true;
      resultsPager.innerHTML = "";
      return;
    }
    resultsPager.hidden = false;
    const mkBtn = (label, p, cur, disabled) =>
      `<button type="button" class="pager-btn${cur ? " is-current" : ""}" data-page="${p}"${
        disabled ? " disabled" : ""
      }>${label}</button>`;
    const parts = [];
    parts.push(mkBtn("Prev", page - 1, false, page <= 1));
    const windowSize = 5;
    let from = Math.max(1, page - Math.floor(windowSize / 2));
    let to = Math.min(totalPages, from + windowSize - 1);
    if (to - from < windowSize - 1) from = Math.max(1, to - windowSize + 1);
    if (from > 1) {
      parts.push(mkBtn("1", 1, page === 1, false));
      if (from > 2) parts.push('<span class="pager-ellipsis">…</span>');
    }
    for (let p = from; p <= to; p++) {
      parts.push(mkBtn(String(p), p, p === page, false));
    }
    if (to < totalPages) {
      if (to < totalPages - 1) parts.push('<span class="pager-ellipsis">…</span>');
      parts.push(mkBtn(String(totalPages), totalPages, page === totalPages, false));
    }
    parts.push(mkBtn("Next", page + 1, false, page >= totalPages));
    resultsPager.innerHTML = `<div class="pager-inner">${parts.join("")}</div>`;
  }

  if (resultsPager) {
    resultsPager.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-page]");
      if (!btn || !searchResultsState) return;
      const p = parseInt(btn.getAttribute("data-page"), 10);
      if (Number.isNaN(p) || p < 1) return;
      const totalPages = Math.max(
        1,
        Math.ceil(searchResultsState.allResults.length / config.pageSize)
      );
      if (p > totalPages) return;
      searchResultsState.page = p;
      renderResultsPage();
      const main = document.getElementById("main");
      if (main) main.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  async function doSearch(e) {
    if (e) e.preventDefault();
    const rawQuery = queryInput.value.trim();
    const manual = readManualFilters();
    const parsed = parseInlineFilters(rawQuery);
    const filters = mergeFilters(manual, parsed.filters);
    const topic = parsed.topic;

    if (!topic && !anyFilterActive(filters)) {
      setStatus("Enter a topic, a name, or at least one filter (country, office, language, or university).", true);
      return;
    }

    const apiKey = getApiKey();

    searchBtn.disabled = true;
    resultsContainer.innerHTML = "";
    resultsMeta.textContent = "";
    if (resultsPager) {
      resultsPager.hidden = true;
      resultsPager.innerHTML = "";
    }
    searchResultsState = null;

    try {
      const idx = await loadIndex();
      const model = idx.model || "text-embedding-3-small";
      let pool = idx.researchers.filter((r) => researcherPassesFilters(r, filters));
      let embedTopic = topic;
      let needsEmbedding = Boolean(topic && topic.trim());
      if (topic && !anyFilterActive(filters) && looksLikePersonNameQuery(topic)) {
        const nameHits = idx.researchers.filter((r) =>
          nameFilterMatches(r.name || r.slug, topic)
        );
        if (nameHits.length > 0) {
          pool = nameHits;
          needsEmbedding = false;
          embedTopic = "";
        }
      }

      if (topic && needsEmbedding) {
        const lead = extractLeadingInstitutionKeyword(topic);
        if (lead) {
          const uniHits = pool.filter((r) => institutionQueryMatch(r, lead.key));
          if (uniHits.length > 0) {
            pool = uniHits;
            embedTopic = lead.rest;
            needsEmbedding = Boolean(embedTopic && embedTopic.trim());
          }
        } else if (looksLikeShortInstitutionQuery(topic)) {
          const uniHits = pool.filter((r) => institutionQueryMatch(r, topic));
          if (uniHits.length > 0) {
            pool = uniHits;
            needsEmbedding = false;
            embedTopic = "";
          }
        }
      }

      if (needsEmbedding && !usesProxy() && !apiKey) {
        setStatus("Add an OpenAI API key under Settings, or configure embedApiUrl in config.json.", true);
        openModal();
        searchBtn.disabled = false;
        return;
      }

      if (pool.length === 0) {
        setStatus("No researchers match your filters. Try broadening country, office, or name.", false);
        resultsMeta.textContent = "0 matches";
        searchBtn.disabled = false;
        return;
      }

      let queryEmb = null;
      if (needsEmbedding) {
        setStatusHTML('<span class="spinner"></span> Embedding query…');
        queryEmb = await embedQuery(embedTopic, model, apiKey);
      }

      setStatusHTML(
        `<span class="spinner"></span> Scoring ${pool.length.toLocaleString()} researchers…`
      );

      const terms = extractTerms(embedTopic || topic);
      const scored = pool.map((r) => {
        let semantic = 0;
        if (queryEmb) {
          semantic = cosine(queryEmb, r.embedding);
          const nar = r.embedding_narrative;
          if (nar && nar.length === r.embedding.length) {
            semantic = Math.max(semantic, cosine(queryEmb, nar));
          }
        }
        const { boost, matches } = keywordBoost(r, terms);
        return { r, score: semantic + boost, semantic, boost, matches };
      });

      scored.sort((a, b) => b.score - a.score);

      let chosen;
      if (needsEmbedding) {
        chosen = selectRelevantScores(scored);
      } else {
        chosen = scored.slice().sort((a, b) => {
          const an = (a.r.name || a.r.slug || "").toLowerCase();
          const bn = (b.r.name || b.r.slug || "").toLowerCase();
          return an.localeCompare(bn);
        });
      }

      searchResultsState = { allResults: chosen, page: 1 };
      const labelParts = [];
      if (topic) labelParts.push(`“${topic}”`);
      if (filters.country) labelParts.push(`country: ${filters.country}`);
      if (filters.office) labelParts.push(`office: ${filters.office}`);
      if (filters.language) labelParts.push(`language: ${filters.language}`);
      if (filters.name) labelParts.push(`name: ${filters.name}`);
      if (filters.university) labelParts.push(`university: ${filters.university}`);
      if (filters.sector) labelParts.push(`sector: ${filters.sector}`);
      if (filters.type) labelParts.push(`type: ${filters.type}`);
      const label = labelParts.length ? labelParts.join(" · ") : "filters";

      setStatus(
        `Showing ${chosen.length.toLocaleString()} researcher${chosen.length === 1 ? "" : "s"} who match (${pool.length.toLocaleString()} after filters, of ${idx.researchers.length.toLocaleString()} total).`
      );
      resultsMeta.textContent = `${chosen.length} match${chosen.length === 1 ? "" : "es"} for ${label}`;

      renderResultsPage();
    } catch (err) {
      setStatus(err.message || String(err), true);
      console.error(err);
      resultsContainer.innerHTML = `<div class="placeholder"><p>${escHtml(err.message || String(err))}</p></div>`;
    } finally {
      searchBtn.disabled = false;
    }
  }

  function renderCard({ r, score, boost, matches }, rank) {
    const card = document.createElement("div");
    card.className = "card";

    const sf = r.key_fields || {};
    const name = r.name || r.slug;
    const site = ((r.website_url || "") + "").trim();
    const personal = ((r.personal_page_url || "") + "").trim();
    const nameHref = site || personal;
    const nameHtml = nameHref
      ? `<a class="card-name" href="${escAttr(
          nameHref
        )}" target="_blank" rel="noopener noreferrer">${escHtml(name)}</a>`
      : `<span class="card-name">${escHtml(name)}</span>`;

    let typeHtml = "";
    const rt = ((sf["Researcher Type"] || "") + "").trim();
    if (rt) {
      const types = rt
        .split(";")
        .map((s) => s.trim())
        .filter(Boolean);
      typeHtml = `<div class="card-types" role="list">${types
        .map((t) => {
          const invited = t.toLowerCase().includes("invited");
          const cls = invited
            ? "card-type-badge card-type-badge--invited"
            : "card-type-badge card-type-badge--affiliate";
          return `<span class="${cls}" role="listitem">${escHtml(t)}</span>`;
        })
        .join("")}</div>`;
    }

    const institution = r.institution
      ? `<div class="card-institution" title="Affiliation from publication index">${escHtml(
          r.institution
        )}</div>`
      : "";

    let metaLines = "";
    if (sf.offices) {
      metaLines += `<div class="card-detail-line"><span class="card-detail-label">J-PAL offices</span> ${escHtml(
        sf.offices
      )}</div>`;
    }
    if (sf.initiatives) {
      metaLines += `<div class="card-detail-line"><span class="card-detail-label">Initiative roster</span> ${escHtml(
        truncate(sf.initiatives, 200)
      )}</div>`;
    }

    const usedHrefs = new Set();
    if (nameHref) usedHrefs.add(nameHref);
    const linkPieces = [];
    const pushLink = (href, label) => {
      const h = ((href || "") + "").trim();
      if (!h || usedHrefs.has(h)) return;
      usedHrefs.add(h);
      linkPieces.push(
        `<a class="card-link" href="${escAttr(h)}" target="_blank" rel="noopener noreferrer">${escHtml(
          label
        )}</a>`
      );
    };
    pushLink(r.cv_url, "CV");
    pushLink(r.web_bio_link_url, "Web bio");
    if (site && personal && site !== personal) {
      if (nameHref === site) pushLink(personal, "Profile page");
      else pushLink(site, "Website");
    }
    const linksHtml =
      linkPieces.length > 0 ? `<div class="card-links">${linkPieces.join(" · ")}</div>` : "";

    const boostHtml =
      boost > 0.001 ? `<div class="score-boost">+${boost.toFixed(3)} keyword</div>` : "";

    let tagsHtml = "";
    if (sf["Sectors"]) {
      sf["Sectors"]
        .split(";")
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, 4)
        .forEach((s) => {
          tagsHtml += `<span class="tag tag-sector">${escHtml(s)}</span>`;
        });
    }
    if (sf["Specific Country Interest"]) {
      sf["Specific Country Interest"]
        .split(";")
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, 3)
        .forEach((s) => {
          tagsHtml += `<span class="tag tag-country">${escHtml(s)}</span>`;
        });
    }
    if (sf["Regional Office Affiliation"]) {
      sf["Regional Office Affiliation"]
        .split(";")
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, 2)
        .forEach((s) => {
          tagsHtml += `<span class="tag tag-region">${escHtml(s)}</span>`;
        });
    }
    if (Object.keys(matches).length) {
      const allHits = [...new Set(Object.values(matches).flat())];
      tagsHtml += `<span class="tag tag-boost">matched: ${escHtml(allHits.join(", "))}</span>`;
    }

    const bioText = sf["Research Interests (open text)"] || sf["Web Bio"] || "";
    const bioHtml = bioText
      ? `<div class="card-bio">${escHtml(truncate(bioText, 220))}</div>`
      : "";

    let evidenceHtml = "";
    const matchEntries = Object.entries(matches);
    if (matchEntries.length) {
      evidenceHtml += `<div class="keyword-match"><strong>Keyword matches</strong><br/>`;
      matchEntries.forEach(([field, tms]) => {
        evidenceHtml += `<em>${escHtml(field)}:</em> “${escHtml(tms.join('", "'))}” — <em>${escHtml(
          truncate(sf[field] || "", 180)
        )}</em><br/>`;
      });
      evidenceHtml += `</div>`;
    }
    function addEvidence(label, text) {
      const t = ((text || "") + "").trim();
      if (!t) return;
      evidenceHtml += `<div class="evidence-item"><div class="evidence-label">${escHtml(
        label
      )}</div><div class="evidence-text">${escHtml(truncate(t, 400))}</div></div>`;
    }
    addEvidence("Initiatives", sf["Initiatives"]);
    addEvidence("Regional interest", sf["Regional interest"]);
    addEvidence("Sector / initiative interest", sf["Sector/Initiative interest"]);
    addEvidence("Publication notes", sf["Publication Notes"]);
    addEvidence("Languages", sf.Languages);
    if (sf.offices) addEvidence("J-PAL offices", sf.offices);
    if (sf.initiatives) addEvidence("Initiative roster", sf.initiatives);

    const hasEvidence = evidenceHtml.length > 0;

    card.innerHTML = `
      <div class="card-header">
        <div class="rank-badge">${rank}</div>
        <div class="card-main">
          ${nameHtml}
          ${typeHtml}
          ${institution}
          ${metaLines}
          ${linksHtml}
        </div>
        <div class="card-score">
          <div class="score-value">score ${score.toFixed(3)}</div>
          ${boostHtml}
        </div>
      </div>
      ${tagsHtml ? `<div class="tags">${tagsHtml}</div>` : ""}
      ${bioHtml}
      ${
        hasEvidence
          ? `
      <div class="evidence-toggle">
        <button type="button" class="toggle-btn">Show details ▾</button>
      </div>
      <div class="evidence-panel">${evidenceHtml}</div>`
          : ""
      }
    `;

    if (hasEvidence) {
      const btn = card.querySelector(".toggle-btn");
      const panel = card.querySelector(".evidence-panel");
      btn.addEventListener("click", () => {
        const open = panel.classList.toggle("open");
        btn.textContent = open ? "Hide details ▴" : "Show details ▾";
      });
    }

    resultsContainer.appendChild(card);
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function escAttr(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;")
      .replace(/</g, "&lt;");
  }
  function truncate(s, n) {
    s = s.trim();
    if (s.length <= n) return s;
    const cut = s.slice(0, n);
    const dot = cut.lastIndexOf(". ");
    return (dot > n * 0.5 ? cut.slice(0, dot + 1) : cut) + "…";
  }

  function openModal() {
    apiKeyInput.value = getApiKey();
    modalOverlay.classList.add("open");
    if (!usesProxy()) apiKeyInput.focus();
  }

  settingsBtn.addEventListener("click", openModal);
  modalCancel.addEventListener("click", () => modalOverlay.classList.remove("open"));
  modalSave.addEventListener("click", () => {
    setApiKey(apiKeyInput.value);
    modalOverlay.classList.remove("open");
    setStatus("Settings saved.", false);
  });
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) modalOverlay.classList.remove("open");
  });

  searchForm.addEventListener("submit", doSearch);

  function showPlaceholder() {
    resultsContainer.innerHTML = `
      <div class="placeholder" id="placeholder">
        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
          <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
        </svg>
        <p>Enter a topic, a researcher name (e.g. <em>Esther Duflo</em>), or use the refine panel.
        Inline filters: <code>country:</code> <code>office:</code> <code>language:</code> <code>university:</code> <code>sector:</code> <code>type:</code>.
        All matching researchers are shown, ranked and paginated.</p>
      </div>`;
  }

  async function init() {
    showPlaceholder();
    await loadConfig();
    setTimeout(() => {
      loadIndex().catch((err) => {
        setStatus("Could not preload index: " + err.message, true);
      });
    }, 300);
  }

  init();
})();
