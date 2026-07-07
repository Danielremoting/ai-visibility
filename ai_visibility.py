#!/usr/bin/env python3
"""
AI Visibility Tracker: Mentions & Recommendations
Queries a live chat-completion model with realistic prompts and measures how often a
brand is mentioned in AI answers, and how often it's recommended for buying-intent queries.

Uses the OpenAI SDK pointed at Gemini's OpenAI-compatible endpoint by default (free tier
available). Swapping to real OpenAI later requires no code changes -- just set
AI_VISIBILITY_API_KEY / AI_VISIBILITY_BASE_URL / AI_VISIBILITY_MODEL in the environment.

NOTE: This does not cover true "citation" tracking (AI pulling from your page content via
live web retrieval / RAG). That requires a search-grounded API mode (e.g. Perplexity API,
Gemini grounding, or an OpenAI web-search tool) and is a planned follow-up.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

from analyzer import (
    _normalize_domain,
    _domain_matches_text,
    BROWSER_HEADERS,
    evaluate_information_density,
    generate_recommendations,
)
from auditor import audit_website

load_dotenv()

# Gemini's OpenAI-compatible endpoint -- swap AI_VISIBILITY_BASE_URL env var to point
# this at real OpenAI (or any other OpenAI-compatible provider) later, with zero code changes.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_SAMPLES = 5

# AI queries are pure network I/O (waiting on the model's response), so we fire several at
# once instead of one-at-a-time -- this is the single biggest speedup available in this
# module. Kept modest (rather than e.g. 10+) to avoid tripping free-tier rate limits, which
# would just show up as retries/backoff and eat the time savings. Override via env var.
DEFAULT_MAX_WORKERS = int(os.getenv("AI_VISIBILITY_MAX_WORKERS", "4"))

# Informational, non-buying-intent questions. Used to measure raw "mention rate".
MENTION_PROMPT_TEMPLATES = [
    "What should I know about {topic}?",
    "Can you explain {topic} to me?",
    "I'm researching {topic} -- what are the key things to consider?",
]

# Buying-intent, "give me options" questions. Used to measure "recommendation rate".
RECOMMENDATION_PROMPT_TEMPLATES = [
    "What are the top 5 {category}?",
    "Can you recommend the best {category} to use?",
    "Which {category} would you suggest for someone just getting started?",
]


def _resolve_client_config() -> Dict[str, Optional[str]]:
    """
    Reads provider config from the environment. Falls back to Gemini's free-tier,
    OpenAI-compatible endpoint if nothing is configured.
    """
    api_key = (
        os.getenv("AI_VISIBILITY_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    base_url_env = os.getenv("AI_VISIBILITY_BASE_URL")
    if base_url_env is None:
        base_url = GEMINI_OPENAI_BASE_URL
    elif base_url_env.strip().lower() in ("", "openai", "none"):
        base_url = None  # None tells the OpenAI SDK to use its own default endpoint
    else:
        base_url = base_url_env

    model = os.getenv("AI_VISIBILITY_MODEL", DEFAULT_MODEL)

    return {"api_key": api_key, "base_url": base_url, "model": model}


def _get_client() -> AsyncOpenAI:
    config = _resolve_client_config()
    if not config["api_key"]:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY (or AI_VISIBILITY_API_KEY) in your "
            "environment or .env file. Get a free Gemini key at https://aistudio.google.com/apikey"
        )
    client_kwargs: Dict[str, Any] = {"api_key": config["api_key"]}
    if config["base_url"]:
        client_kwargs["base_url"] = config["base_url"]
    return AsyncOpenAI(**client_kwargs)


async def query_ai_model(prompt: str, model: Optional[str] = None, retries: int = 2) -> Dict[str, Any]:
    """
    Sends a single prompt to the configured chat model and returns the plain-text response.
    Retries briefly on transient rate-limit/connection errors. Uses the async OpenAI client
    so many of these can be awaited concurrently (via asyncio.gather) instead of blocking
    one at a time.
    """
    config = _resolve_client_config()
    resolved_model = model or config["model"]

    try:
        client = _get_client()
    except RuntimeError as exc:
        return {"success": False, "text": None, "error": str(exc)}

    last_error = None
    async with client:
        for attempt in range(retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=resolved_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30,
                )
                text = response.choices[0].message.content or ""
                return {"success": True, "text": text, "error": None}
            except RateLimitError as exc:
                last_error = f"Rate-limited: {exc}"
                await asyncio.sleep(3 * (attempt + 1))
            except APIConnectionError as exc:
                last_error = f"Connection error: {exc}"
                await asyncio.sleep(2)
            except APIError as exc:
                last_error = f"API error: {exc}"
                break
            except Exception as exc:
                last_error = f"Unexpected error: {exc}"
                break

    return {"success": False, "text": None, "error": last_error}


def _extract_list_items(text: str) -> List[str]:
    """
    Extracts individual items from a numbered or bulleted list in a text response.
    Falls back to splitting by line if no clear list markers are found.
    """
    items = re.findall(r"(?:^|\n)\s*(?:\d+[.\):]|[-*\u2022])\s*(.+)", text)
    items = [item.strip() for item in items if item.strip()]
    if items:
        return items
    return [line.strip() for line in text.split("\n") if line.strip()]


def _find_brand_position(brand_name: str, domain: str, items: List[str]) -> Optional[int]:
    for idx, item in enumerate(items):
        if _domain_matches_text(domain, brand_name, item):
            return idx + 1
    return None


def _fallback_profile(domain: str, title: str = "") -> Dict[str, Any]:
    """
    Cheap, no-AI-call fallback used when we can't fetch the site or the AI extraction fails.
    """
    clean = _normalize_domain(domain)
    base = clean.split(".")[0].replace("-", " ").replace("_", " ")
    brand_name = title.split("|")[0].split("-")[0].strip() if title else ""
    return {
        "brand_name": brand_name or base.title(),
        "business_category": None,
        "topics": [base],
    }


async def analyze_website_for_profile(domain: str) -> Dict[str, Any]:
    """
    Fetches the site's homepage and asks the configured AI model to extract a brand
    profile (brand name, business category, and representative topics) from the page
    content -- so the tracker only needs a URL as input instead of manually-typed
    brand/topic/category arguments.
    """
    url = domain if domain.startswith(("http://", "https://")) else f"https://{domain}"

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=BROWSER_HEADERS, timeout=12)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"success": False, "error": f"Could not fetch website: {exc}", "fallback": _fallback_profile(domain)}

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.extract()

    title = soup.title.get_text(strip=True) if soup.title else ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_tag.get("content", "").strip() if meta_tag else ""
    visible_text = re.sub(r"\s+", " ", soup.get_text()).strip()[:3000]

    extraction_prompt = (
        "You are preparing an AI-visibility audit for a company website. Based ONLY on the "
        "homepage content below, respond with ONLY valid JSON (no markdown, no commentary) in "
        "exactly this shape:\n"
        '{"brand_name": "...", "business_category": "...", "topics": ["...", "...", "..."]}\n\n'
        "- brand_name: the actual company/product name (not the domain).\n"
        "- business_category: a short plural noun phrase describing what they offer, written so it "
        "fits naturally into the sentence 'What are the top 5 ___?' (e.g. 'remote job platforms', "
        "'AI meal planning apps'). Use null if you cannot tell.\n"
        "- topics: 3 to 5 short, non-branded informational topics/questions a potential customer in "
        "this niche might ask an AI chatbot.\n\n"
        f"Page title: {title}\n"
        f"Meta description: {meta_description}\n"
        f"Page content: {visible_text}"
    )

    result = await query_ai_model(extraction_prompt)
    fallback = _fallback_profile(domain, title)
    if not result["success"]:
        return {"success": False, "error": result["error"], "fallback": fallback}

    cleaned = re.sub(r"^```(?:json)?|```$", "", result["text"].strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return {"success": False, "error": "Could not parse AI response as JSON", "fallback": fallback}

    topics = parsed.get("topics") or []
    return {
        "success": True,
        "brand_name": parsed.get("brand_name") or fallback["brand_name"],
        "business_category": parsed.get("business_category") or None,
        "topics": [t for t in topics if t][:5] or fallback["topics"],
    }


async def _run_bounded(semaphore: asyncio.Semaphore, prompt: str, model: Optional[str]) -> Dict[str, Any]:
    """Runs one AI query, capped by `semaphore` so a large prompt batch doesn't fire
    dozens of requests simultaneously and immediately trip free-tier rate limits."""
    async with semaphore:
        return await query_ai_model(prompt, model=model)


async def track_ai_mentions(
    brand_name: str,
    domain: str,
    topics: List[str],
    samples: int = DEFAULT_SAMPLES,
    model: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[str, Any]:
    """
    For each topic, asks a mention-style prompt `samples` times (since LLM answers are
    non-deterministic) and measures how often the brand/domain is mentioned anywhere
    in the AI's answer. All topic/sample prompts are fired concurrently (they're
    independent network calls) instead of one at a time.
    """
    clean_domain = _normalize_domain(domain)
    total_runs = 0
    total_hits = 0
    failed_runs = 0
    last_error: Optional[str] = None

    prompt_tasks = [
        (topic, MENTION_PROMPT_TEMPLATES[i % len(MENTION_PROMPT_TEMPLATES)].format(topic=topic))
        for topic in topics
        for i in range(samples)
    ]
    topic_results: Dict[str, Any] = {
        topic: {"runs_completed": 0, "mention_hits": 0, "samples": []} for topic in topics
    }

    semaphore = asyncio.Semaphore(max(1, max_workers))
    results = await asyncio.gather(*(_run_bounded(semaphore, prompt, model) for _, prompt in prompt_tasks))

    for (topic, prompt), result in zip(prompt_tasks, results):
        # Empty responses are treated as failures too: counting them as a valid
        # "no mention" run would silently deflate the mention rate.
        if not result["success"] or not (result["text"] or "").strip():
            failed_runs += 1
            last_error = result["error"] or "Model returned an empty response"
            continue

        mentioned = _domain_matches_text(clean_domain, brand_name, result["text"])
        bucket = topic_results[topic]
        bucket["runs_completed"] += 1
        if mentioned:
            bucket["mention_hits"] += 1
        bucket["samples"].append({
            "prompt": prompt,
            "mentioned": mentioned,
            "response_excerpt": result["text"][:280],
        })

    for topic, bucket in topic_results.items():
        runs = bucket["runs_completed"]
        hits = bucket["mention_hits"]
        bucket["mention_rate_percent"] = round((hits / runs * 100), 2) if runs > 0 else None
        total_runs += runs
        total_hits += hits

    overall_rate = round((total_hits / total_runs * 100), 2) if total_runs > 0 else None

    if overall_rate is None:
        verdict = "Unable to Measure (all AI queries failed -- check API key/quota)"
    elif overall_rate >= 50:
        verdict = "Strong Brand Mention Presence in AI Answers"
    elif overall_rate >= 15:
        verdict = "Moderate Brand Mention Presence in AI Answers"
    else:
        verdict = "Weak/No Brand Mentions Detected in AI Answers"

    return {
        "target_brand": brand_name,
        "target_domain": clean_domain,
        "model_used": model or _resolve_client_config()["model"],
        "samples_per_topic": samples,
        "topics_checked": len(topics),
        "total_runs_completed": total_runs,
        "mention_occurrences": total_hits,
        "mention_rate_percent": overall_rate,
        "had_errors": failed_runs > 0,
        "failed_runs": failed_runs,
        "last_error": last_error,
        "topic_details": topic_results,
        "verdict": verdict,
    }


async def track_ai_recommendations(
    brand_name: str,
    domain: str,
    business_category: str,
    samples: int = DEFAULT_SAMPLES,
    model: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[str, Any]:
    """
    Asks buying-intent "top N" / "best of" style prompts `samples` times and measures
    how often the brand is included as a recommended option, and at what list position.
    Prompts are fired concurrently instead of one at a time.
    """
    clean_domain = _normalize_domain(domain)
    prompt_results = []
    runs = 0
    hits = 0
    positions: List[int] = []
    failed_runs = 0
    last_error: Optional[str] = None

    prompts = [
        RECOMMENDATION_PROMPT_TEMPLATES[i % len(RECOMMENDATION_PROMPT_TEMPLATES)].format(category=business_category)
        for i in range(samples)
    ]

    semaphore = asyncio.Semaphore(max(1, max_workers))
    results = await asyncio.gather(*(_run_bounded(semaphore, prompt, model) for prompt in prompts))

    for prompt, result in zip(prompts, results):
        if not result["success"] or not (result["text"] or "").strip():
            failed_runs += 1
            last_error = result["error"] or "Model returned an empty response"
            continue

        runs += 1
        items = _extract_list_items(result["text"])
        position = _find_brand_position(brand_name, clean_domain, items) if items else None
        recommended = position is not None or _domain_matches_text(clean_domain, brand_name, result["text"])

        if recommended:
            hits += 1
        if position:
            positions.append(position)

        prompt_results.append({
            "prompt": prompt,
            "recommended": recommended,
            "position": position,
            "response_excerpt": result["text"][:280],
        })

    recommendation_rate = round((hits / runs * 100), 2) if runs > 0 else None
    avg_position = round(sum(positions) / len(positions), 1) if positions else None

    if recommendation_rate is None:
        verdict = "Unable to Measure (all AI queries failed -- check API key/quota)"
    elif recommendation_rate >= 50:
        verdict = "Frequently Recommended by AI for This Category"
    elif recommendation_rate >= 15:
        verdict = "Occasionally Recommended by AI for This Category"
    else:
        verdict = "Rarely/Never Recommended by AI for This Category"

    return {
        "target_brand": brand_name,
        "target_domain": clean_domain,
        "business_category": business_category,
        "model_used": model or _resolve_client_config()["model"],
        "samples_requested": samples,
        "runs_completed": runs,
        "recommendation_occurrences": hits,
        "recommendation_rate_percent": recommendation_rate,
        "average_position_when_recommended": avg_position,
        "had_errors": failed_runs > 0,
        "failed_runs": failed_runs,
        "last_error": last_error,
        "prompt_details": prompt_results,
        "verdict": verdict,
    }


async def track_ai_visibility(
    brand_name: str,
    domain: str,
    topics: List[str],
    business_category: Optional[str] = None,
    samples: int = DEFAULT_SAMPLES,
    model: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[str, Any]:
    """
    Convenience wrapper that runs both mention and recommendation tracking together.
    Recommendation tracking is skipped if no business_category is provided. When both
    run, they're fired concurrently since they're independent of each other.
    """
    if business_category:
        mention_data, recommendation_data = await asyncio.gather(
            track_ai_mentions(brand_name, domain, topics, samples=samples, model=model, max_workers=max_workers),
            track_ai_recommendations(
                brand_name, domain, business_category, samples=samples, model=model, max_workers=max_workers
            ),
        )
    else:
        mention_data = await track_ai_mentions(
            brand_name, domain, topics, samples=samples, model=model, max_workers=max_workers
        )
        recommendation_data = None

    return {
        "target_brand": brand_name,
        "target_domain": _normalize_domain(domain),
        "mention": mention_data,
        "recommendation": recommendation_data,
    }


async def track_ai_visibility_from_domain(
    domain: str,
    samples: int = DEFAULT_SAMPLES,
    model: Optional[str] = None,
    brand_name_override: Optional[str] = None,
    topics_override: Optional[List[str]] = None,
    category_override: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[str, Any]:
    """
    Fully automated entry point: give it just a domain, and it detects the brand name,
    business category, and relevant topics by reading the homepage (and asking the AI
    model to summarize it), then runs mention + recommendation tracking automatically.
    Any of the *_override params can be passed to skip auto-detection for that field.
    """
    profile = await analyze_website_for_profile(domain)
    detected = profile if profile["success"] else profile.get("fallback", _fallback_profile(domain))

    brand_name = brand_name_override or detected["brand_name"]
    business_category = category_override or detected.get("business_category")
    topics = topics_override or detected.get("topics") or [brand_name]

    result = await track_ai_visibility(
        brand_name, domain, topics, business_category=business_category, samples=samples, model=model,
        max_workers=max_workers,
    )
    result["detected_profile"] = {
        "brand_name": brand_name,
        "business_category": business_category,
        "topics": topics,
        "auto_detection_succeeded": profile["success"],
        "auto_detection_error": profile.get("error"),
    }
    return result


def build_recommendations(result: Dict[str, Any], audit_report: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Computes information density from the audit's already-fetched HTML (no extra network
    calls) and feeds the technical audit + AI visibility results into the shared
    recommendation engine. Used by both this module's CLI and full_report.py.
    Off-page authority isn't checked here (separate, slower lookup), so that part of the
    engine is skipped by passing an empty dict.
    """
    density_data = {"density_score": 0}
    html_content = audit_report.get("html_content")
    if html_content and audit_report.get("structure_analysis"):
        clean_text = re.sub(r"\s+", " ", BeautifulSoup(html_content, "html.parser").get_text()).strip()
        density_data = evaluate_information_density(html_content, clean_text)
    return generate_recommendations(audit_report, density_data, {}, ai_visibility_data=result)


def print_visibility_report(
    result: Dict[str, Any],
    audit_report: Dict[str, Any],
    recommendations: List[Dict[str, str]],
    show_technical: bool = True,
) -> None:
    """
    Prints the human-readable AI visibility report (profile, mention tracking,
    recommendation tracking, technical readability, automated recommendations).
    `show_technical=False` lets callers that already printed their own readability
    section (e.g. full_report.py) skip the duplicate [Technical Readability] block.
    """
    profile = result["detected_profile"]
    print("=" * 60)
    print(f"AI VISIBILITY REPORT FOR: {result['target_brand']} ({result['target_domain']})")
    print("=" * 60)
    if not profile["auto_detection_succeeded"]:
        print(f"(Note: AI auto-detection failed [{profile['auto_detection_error']}] -- used fallback/override values)")
    print(f"Detected brand name:      {profile['brand_name']}")
    print(f"Detected category:        {profile['business_category'] or 'N/A'}")
    print(f"Detected topics:          {profile['topics']}")

    m = result["mention"]
    print(f"\n[Mention Tracking] Model: {m['model_used']}")
    print(f"  Verdict: {m['verdict']}")
    print(f"  Mention Rate: {m['mention_rate_percent']}% ({m['mention_occurrences']}/{m['total_runs_completed']} runs)")
    if m.get("failed_runs"):
        print(f"  WARNING: {m['failed_runs']} AI queries failed and were excluded from the rate.")
        print(f"           Last error: {m.get('last_error')}")
    for topic, details in m["topic_details"].items():
        print(f"    - \"{topic}\": {details['mention_rate_percent']}% ({details['mention_hits']}/{details['runs_completed']})")

    r = result["recommendation"]
    if r:
        print(f"\n[Recommendation Tracking] Category: {r['business_category']}")
        print(f"  Verdict: {r['verdict']}")
        print(f"  Recommendation Rate: {r['recommendation_rate_percent']}% ({r['recommendation_occurrences']}/{r['runs_completed']} runs)")
        if r.get("failed_runs"):
            print(f"  WARNING: {r['failed_runs']} AI queries failed and were excluded from the rate.")
            print(f"           Last error: {r.get('last_error')}")
        if r["average_position_when_recommended"]:
            print(f"  Average Position When Recommended: #{r['average_position_when_recommended']}")
    else:
        print("\n[Recommendation Tracking] Skipped (pass --category to enable)")

    if show_technical:
        print(f"\n[Technical Readability] Score: {audit_report.get('readability_score')}/100")
        print(f"  Verdict: {audit_report.get('verdict')}")
        if audit_report.get("key_issues"):
            print("  Key Issues:")
            for issue in audit_report["key_issues"]:
                print(f"    - {issue}")

    print(f"\n[Automated Recommendations] ({len(recommendations)} Actions):")
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            print(f"  [{idx}] [{rec['category']}] (Priority: {rec['priority']})")
            print(f"      Issue:  {rec['issue']}")
            action_items = rec.get("action_items") or [rec["action"]]
            if len(action_items) > 1:
                print("      Action Items:")
                for item in action_items:
                    print(f"        - {item}")
            else:
                print(f"      Action: {rec['action']}")
            print(f"      Impact: {rec['impact']}")
    else:
        print("  None! This site has strong technical readability and AI visibility.")
    print()


async def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="AI Mention & Recommendation Visibility Tracker. "
        "Give it just a website and it auto-detects the brand name, business category, "
        "and relevant topics from the homepage -- no manual input needed."
    )
    parser.add_argument("domain", type=str, help="The company website to analyze, e.g. 'remoting.work'")
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.json:
        print(f"Analyzing {args.domain} to auto-detect brand profile...")

    url = args.domain if args.domain.startswith(("http://", "https://")) else f"https://{args.domain}"

    # AI mention/recommendation tracking and the technical crawlability audit hit
    # completely independent endpoints (LLM API vs. the target website itself), so run
    # them concurrently instead of waiting for one to finish before starting the other.
    result, audit_report = await asyncio.gather(
        track_ai_visibility_from_domain(
            args.domain,
            samples=args.samples,
            brand_name_override=args.brand_name,
            topics_override=args.topics,
            category_override=args.category,
        ),
        audit_website(url),
    )

    recommendations = build_recommendations(result, audit_report)

    result["technical_readability"] = {
        "readability_score": audit_report.get("readability_score"),
        "verdict": audit_report.get("verdict"),
    }
    result["recommendations"] = recommendations

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_visibility_report(result, audit_report, recommendations)


if __name__ == "__main__":
    asyncio.run(_main())
