#!/usr/bin/env python3
"""
LLM Readability Checker
Determines whether a website's content can currently be read and processed by LLMs.
Usage: python llm_readability.py <url>
"""

import sys
import asyncio
import urllib.parse
import urllib.robotparser
import httpx
from auditor import audit_website, BROWSER_USER_AGENT


# Bots to display, in display order
BOTS_TO_DISPLAY = [
    "Googlebot",
    "GPTBot",
    "PerplexityBot",
    "ClaudeBot",
    "Bingbot",
]


async def fetch_robots_txt(target_url: str):
    """
    Fetch and parse the raw robots.txt for the target URL.

    Returns (bot_statuses, path, reason, detail):
      - bot_statuses: dict of bot -> status string, only populated when reason == "ok"
      - reason: "ok" (parsed normally), "not_found" (404 -> implicitly allowed),
        "blocked" (some other non-200 status, e.g. a Cloudflare 403 challenge page sitting
        in front of /robots.txt itself), or "error" (connection/timeout failure)
      - detail: the HTTP status code when reason == "blocked", otherwise None

    Distinguishing "blocked" from "not_found"/"error" matters: a 403 on /robots.txt means
    the site is actively walling off crawlers, which is a very different (and worse) signal
    than the file simply not existing.
    """
    if "://" not in target_url:
        target_url = "https://" + target_url

    parsed = urllib.parse.urlparse(target_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    path = parsed.path or "/"

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                robots_url,
                headers={"User-Agent": BROWSER_USER_AGENT},
                timeout=10,
            )
    except httpx.RequestError as e:
        print(f"[debug] robots.txt fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None, path, "error", None

    if resp.status_code == 404:
        return None, path, "not_found", None

    if resp.status_code != 200:
        return None, path, "blocked", resp.status_code

    raw_text = resp.text

    rp = urllib.robotparser.RobotFileParser()
    rp.parse(raw_text.splitlines())

    statuses = {}
    for bot in BOTS_TO_DISPLAY:
        allowed = rp.can_fetch(bot, path)
        if allowed:
            statuses[bot] = "Allowed by robots.txt"
        else:
            statuses[bot] = "Blocked by robots.txt"

    return statuses, path, "ok", None


def build_readability_summary(report, robots_result) -> dict:
    """
    Builds a structured summary (dict) of the readability verdict, per-bot access
    list, and robots-level recommendations from an already-fetched audit report +
    fetch_robots_txt() result. Reused by the CLI printer below and by the API
    (app.py) so both always agree.
    """
    bot_statuses, _, reason, detail = robots_result
    score = report.get("readability_score", 0)

    bots = {}
    blocked_bots = []
    if reason == "ok":
        for bot in BOTS_TO_DISPLAY:
            status = bot_statuses.get(bot, "Unknown")
            bots[bot] = status
            if "Blocked" in status:
                blocked_bots.append(bot)
        robots_status = "Found"
    elif reason == "not_found":
        bots = {bot: "Implicitly Allowed" for bot in BOTS_TO_DISPLAY}
        robots_status = "Not found"
    elif reason == "blocked":
        bots = {bot: f"Unknown (robots.txt itself returned HTTP {detail})" for bot in BOTS_TO_DISPLAY}
        robots_status = f"Blocked (HTTP {detail} -- likely an anti-bot/Cloudflare challenge)"
    else:
        bots = {bot: "Unknown (Fetch Error)" for bot in BOTS_TO_DISPLAY}
        robots_status = "Fetch Error"

    # Actionable advice: only present when something is actually wrong. A fully open
    # site (reason == "ok" with no blocked_bots) has nothing to fix here.
    anti_bot_signals = report.get("bot_fetch_check", {}).get("anti_bot_signals", [])
    recs = []
    if reason == "blocked":
        recs.append(
            f"/robots.txt itself returned HTTP {detail}, likely a Cloudflare/anti-bot challenge "
            "blocking crawlers before they can even read your rules. Whitelist AI/search crawler "
            "user-agents at the firewall/CDN level so robots.txt loads normally for them."
        )
    elif reason == "error":
        recs.append(
            "robots.txt could not be reached at all (connection/timeout error). Confirm the domain "
            "resolves and /robots.txt returns a normal response for a plain GET request."
        )
    elif blocked_bots:
        recs.append(
            f"Update robots.txt to explicitly Allow: {', '.join(blocked_bots)} -- these bots are "
            "currently disallowed, so they cannot crawl or cite your content."
        )
    if anti_bot_signals:
        recs.append(
            "Anti-bot protection was detected on the live page fetch (e.g. Cloudflare/CAPTCHA). "
            "Configure your firewall/CDN to allow verified AI crawler user-agents through, otherwise "
            "even an Allow rule in robots.txt won't matter."
        )

    return {
        "score": score,
        "can_read": score >= 70,
        "verdict": "Site can be read by AI" if score >= 70 else "Site cannot be reliably read by AI",
        "bots": bots,
        "robots_txt": robots_status,
        "recommendations": recs,
    }


def print_readability_report(report, robots_result) -> None:
    """Prints the readability section (verdict, bot access, recommendations)."""
    summary = build_readability_summary(report, robots_result)

    print(summary["verdict"])
    print(f"{summary['score']}%")

    print()
    print("Bot Access:")
    for bot, status in summary["bots"].items():
        print(f"{bot}: {status}")
    print(f"robots.txt: {summary['robots_txt']}")

    if summary["recommendations"]:
        print()
        print("Recommendations:")
        for rec in summary["recommendations"]:
            print(f"- {rec}")


async def check_llm_readability(url: str) -> None:
    if "://" not in url:
        url = "https://" + url

    # audit_website() and fetch_robots_txt() hit the same host independently -- run them
    # concurrently rather than waiting for the full audit before starting the second fetch.
    report, robots_result = await asyncio.gather(
        audit_website(url),
        fetch_robots_txt(url),
    )
    print_readability_report(report, robots_result)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python llm_readability.py <url>", file=sys.stderr)
        sys.exit(1)

    target_url = sys.argv[1]
    asyncio.run(check_llm_readability(target_url))


if __name__ == "__main__":
    main()
