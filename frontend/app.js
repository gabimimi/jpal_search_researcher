/**
 * Researcher Finder — static frontend
 * Loads config.json (optional embedApiUrl for production / CORS-safe embeddings).
 */
(function () {
  const LS_KEY = "jpal_researcher_finder_openai_key";

  let config = {
    embedApiUrl: "",
    indexUrl: "",
    topN: 15,
  };

  let indexData = null;

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
    "Specific Country Interest": 0.15,
    "Regional Office Affiliation": 0.08,
    "Research Interests (open text)": 0.06,
    Sectors: 0.05,
    offices: 0.05,
    "Regional interest": 0.05,
    "Sector/Initiative interest": 0.05,
    Initiatives: 0.04,
    "Researcher Type": 0.04,
    initiatives: 0.04,
    "Publication Notes": 0.03,
    "Web Bio": 0.03,
    "Related Initiative(s)": 0.03,
    "Website & publications (keyword index)": 0.06,
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
        if (typeof j.topN === "number" && j.topN > 0) config.topN = Math.min(50, j.topN);
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
    const resp = await fetch(url);
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

  async function doSearch(e) {
    if (e) e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    const apiKey = getApiKey();
    if (!usesProxy() && !apiKey) {
      setStatus("Add an OpenAI API key under Settings, or configure embedApiUrl in config.json.", true);
      openModal();
      return;
    }

    searchBtn.disabled = true;
    resultsContainer.innerHTML = "";
    resultsMeta.textContent = "";

    try {
      const idx = await loadIndex();
      const model = idx.model || "text-embedding-3-small";
      const n = config.topN;

      setStatusHTML('<span class="spinner"></span> Embedding query…');
      const queryEmb = await embedQuery(query, model, apiKey);

      setStatusHTML(
        `<span class="spinner"></span> Ranking ${idx.researchers.length.toLocaleString()} researchers…`
      );

      const terms = extractTerms(query);
      // Rank by max(cos q·full, cos q·narrative); narrative = website+profile+CV only (no papers).
      const scored = idx.researchers.map((r) => {
        let semantic = cosine(queryEmb, r.embedding);
        const nar = r.embedding_narrative;
        if (nar && nar.length === r.embedding.length) {
          semantic = Math.max(semantic, cosine(queryEmb, nar));
        }
        const { boost, matches } = keywordBoost(r, terms);
        return { r, score: semantic + boost, semantic, boost, matches };
      });

      scored.sort((a, b) => b.score - a.score);
      const top = scored.slice(0, n);

      setStatus(
        `Ranked ${idx.researchers.length.toLocaleString()} researchers. Showing top ${top.length}.`
      );
      resultsMeta.textContent = `Top ${top.length} for “${query}”`;

      top.forEach((item, i) => renderCard(item, i + 1));
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
        <p>Enter a topic, country, sector, or method (for example <em>cash transfers</em> or <em>education RCTs</em>). Results rank J-PAL affiliates and related researchers by semantic similarity to profile and publication text.</p>
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
