#!/usr/bin/env python3
"""
remoting.work AI Visibility Checker -- FastAPI backend.

Serves the single-page frontend at "/" and exposes one endpoint:

    POST /api/report   {"url": "example.com", "samples": 3}

which runs the combined readability + AI visibility analysis from full_report.py
and returns everything as JSON for the frontend to render.

Run with:
    python3 -m uvicorn api:app --reload --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from full_report import generate_full_report

app = FastAPI(
    title="remoting.work AI Visibility Checker",
    description="Checks whether AI systems can read, mention, and recommend a website.",
)

STATIC_DIR = Path(__file__).parent / "static"


class ReportRequest(BaseModel):
    url: str = Field(..., min_length=3, description="Website to analyze, e.g. 'remoting.work'")
    # 3 samples keeps a web request around ~40-60s; the CLI default (5) is slower.
    samples: int = Field(3, ge=1, le=10)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/report")
async def create_report(request: ReportRequest):
    try:
        report = await generate_full_report(request.url.strip(), samples=request.samples)
    except Exception as exc:  # surface analysis failures as a clean API error
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}")

    # If the site itself was unreachable (DNS failure, connection refused, typo'd
    # domain), every downstream number is meaningless -- the mention tracker would run
    # on fallback values and can even report false positives. Fail loudly instead.
    profile = report["visibility"].get("detected_profile", {})
    site_unreachable = (
        report["readability"]["robots_txt"] == "Fetch Error"
        and not profile.get("auto_detection_succeeded")
        and str(profile.get("auto_detection_error", "")).startswith("Could not fetch website")
    )
    if site_unreachable:
        raise HTTPException(
            status_code=400,
            detail=f"We couldn't reach '{request.url.strip()}'. Check the address and try again.",
        )

    # Internal CLI-only payloads: _audit_report carries the full page HTML (huge) and
    # _robots_result duplicates what's already in report["readability"].
    report.pop("_audit_report", None)
    report.pop("_robots_result", None)
    return report
