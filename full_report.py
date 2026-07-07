#!/usr/bin/env python3
"""
Full AI SEO Report
Combines the two standalone reports into one run:
  1. LLM Readability check (from llm_readability.py) on top -- can AI read the site,
     score, per-bot robots.txt access, and robots-level recommendations.
  2. AI Visibility report (from ai_visibility.py) below it -- detected brand profile,
     live AI mention/recommendation tracking, and the automated recommendations
     checklist.

The technical audit and robots.txt fetch are shared between both sections, so nothing
is fetched twice. All network work (site audit, robots.txt, LLM queries) runs
concurrently.

Usage: python3 full_report.py <url> [--samples N] [--brand-name X] [--topics ...] [--category X]
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional

from auditor import audit_website
from llm_readability import fetch_robots_txt, build_readability_summary, print_readability_report
from ai_visibility import (
    DEFAULT_SAMPLES,
    track_ai_visibility_from_domain,
    build_recommendations,
    print_visibility_report,
)


async def generate_full_report(
    domain: str,
    samples: int = DEFAULT_SAMPLES,
    brand_name: Optional[str] = None,
    topics: Optional[List[str]] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the combined readability + AI visibility analysis and returns everything as
    one structured dict (used by the API; the CLI below prints the same data).
    All network work (site audit, robots.txt, LLM queries) runs concurrently.
    """
    url = domain if domain.startswith(("http://", "https://")) else f"https://{domain}"

    audit_report, robots_result, visibility = await asyncio.gather(
        audit_website(url),
        fetch_robots_txt(url),
        track_ai_visibility_from_domain(
            domain,
            samples=samples,
            brand_name_override=brand_name,
            topics_override=topics,
            category_override=category,
        ),
    )

    readability = build_readability_summary(audit_report, robots_result)
    recommendations = build_recommendations(visibility, audit_report)

    return {
        "domain": domain,
        "readability": readability,
        "visibility": visibility,
        "technical": {
            "score": audit_report.get("readability_score"),
            "verdict": audit_report.get("verdict"),
            "key_issues": audit_report.get("key_issues", []),
        },
        "recommendations": recommendations,
        # Kept for the CLI printer; stripped by the API before returning JSON.
        "_audit_report": audit_report,
        "_robots_result": robots_result,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined LLM readability + AI visibility report for a website."
    )
    parser.add_argument("domain", type=str, help="The website to analyze, e.g. 'remoting.work'")
    parser.add_argument("--brand-name", type=str, default=None, help="Override the auto-detected brand name")
    parser.add_argument(
        "--topics", nargs="+", default=None,
        help="Override the auto-detected informational topics to test mention rate against"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Override the auto-detected business category, e.g. 'remote job platforms'"
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()

    print(f"Analyzing {args.domain} (readability + AI visibility)...")

    report = await generate_full_report(
        args.domain,
        samples=args.samples,
        brand_name=args.brand_name,
        topics=args.topics,
        category=args.category,
    )

    # --- Section 1: LLM Readability (llm_readability.py output) ---
    print()
    print("=" * 60)
    print("LLM READABILITY CHECK")
    print("=" * 60)
    print_readability_report(report["_audit_report"], report["_robots_result"])

    # --- Section 2: AI Visibility (ai_visibility.py output) ---
    # show_technical stays on: it adds the audit's key issues (anti-bot signals,
    # structure problems), which section 1 doesn't list.
    print()
    print_visibility_report(report["visibility"], report["_audit_report"], report["recommendations"])


if __name__ == "__main__":
    asyncio.run(main())
