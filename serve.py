"""
Local dev server: serves the frontend and proxies embedding and synthesis
calls to OpenAI so the browser never needs an API key.

Usage:
    python3 serve.py
Then open http://localhost:8000
"""
import json
import os
import sys
from pathlib import Path

# Load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set in .env or environment.", file=sys.stderr)
    sys.exit(1)

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI()

OPENAI_EMBED_URL  = "https://api.openai.com/v1/embeddings"
OPENAI_CHAT_URL   = "https://api.openai.com/v1/chat/completions"
SYNTH_MODEL       = "gpt-4o-mini"
SYNTH_MAX_RESULTS = 20   # cap to keep latency and cost low


@app.post("/embed")
async def embed(request: Request):
    body = await request.body()
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            OPENAI_EMBED_URL,
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


class SynthesizeResult(BaseModel):
    slug: str
    name: str
    institution: str = ""
    research_interests: str = ""
    sectors: str = ""
    initiatives: str = ""
    web_bio: str = ""
    field_matches: str = ""      # e.g. "Specific Country Interest: paraguay"
    website_snippet: str = ""    # short excerpt from website/publications blob around the match

class SynthesizeRequest(BaseModel):
    query: str
    results: list[SynthesizeResult]


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    results = req.results[:SYNTH_MAX_RESULTS]
    if not results:
        return JSONResponse(content={})

    summaries = []
    for r in results:
        lines = [f"- {r.name} ({r.institution})"]
        if r.research_interests:
            lines.append(f"  Research interests: {r.research_interests[:400]}")
        if r.sectors:
            lines.append(f"  Sectors: {r.sectors}")
        if r.initiatives:
            lines.append(f"  Initiatives: {r.initiatives[:200]}")
        if r.web_bio:
            lines.append(f"  Bio: {r.web_bio[:200]}")
        if r.field_matches:
            lines.append(f"  Matched fields: {r.field_matches}")
        if r.website_snippet:
            lines.append(f"  Website/publications excerpt: \"{r.website_snippet[:300]}\"")
        summaries.append("\n".join(lines))

    name_to_slug = {r.name: r.slug for r in results}
    names = [r.name for r in results]

    prompt = (
        "You are helping J-PAL staff find researchers in their network.\n\n"
        f'Search query: "{req.query}"\n\n'
        "The following researchers were retrieved as relevant to the query. "
        "Each entry may include matched fields (the structured profile fields that contained query terms) "
        "and a website/publications excerpt (raw text from their site or papers that contained the query term).\n\n"
        + "\n".join(summaries)
        + "\n\nFor each researcher, write 1 sentence explaining why they were matched. "
        "Base your explanation on the actual evidence above — matched fields and the website excerpt. "
        "If the match is from an excerpt, quote or paraphrase what specifically it shows. "
        "Do not invent reasons. Do not use filler phrases.\n\n"
        "Return a JSON object whose keys are exactly the researcher names listed above "
        "and whose values are their one-sentence explanations.\n"
        f"Names: {json.dumps(names)}"
    )

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            OPENAI_CHAT_URL,
            json={
                "model": SYNTH_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.15,
                "max_tokens": max(400, len(results) * 120),
                "response_format": {"type": "json_object"},
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )

    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    explanations_by_name: dict = json.loads(raw)

    # Map names → slugs for the frontend
    by_slug = {
        name_to_slug[name]: explanation
        for name, explanation in explanations_by_name.items()
        if name in name_to_slug
    }
    return JSONResponse(content=by_slug)


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    print("Open http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
