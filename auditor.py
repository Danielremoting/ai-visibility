#!/usr/bin/env python3
"""
LLM Readability & Accessibility Auditor
A tool to evaluate how accessible and readable a website is to AI bots and LLM scrapers.
Focuses EXCLUSIVELY on real-time citations and Answer Engine/Generative Search (AEO/GEO) visibility.
"""

import re
import json
import asyncio
import urllib.parse
import urllib.robotparser
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any, List

# AI Search & RAG Agents (Responsible for Live Citations & Answers)
AI_USER_AGENTS = {
    "ChatGPT-User": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; ChatGPT-User/1.0; +http://www.openai.com/gptbot)",
    "Claude-Web": "Mozilla/5.0 (compatible; Claude-Web/1.0; +claudebot@anthropic.com)",
    "PerplexityBot": "Mozilla/5.0 (compatible; PerplexityBot/1.0; +https://www.perplexity.ai/perplexitybot)"
}

# Standard browser agent for comparison
BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


async def check_robots_txt(target_url: str) -> Dict[str, Any]:
    """
    Fetches and parses the website's robots.txt.
    Checks permissions for real-time AI search user-agents.
    """
    parsed_url = urllib.parse.urlparse(target_url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    
    result = {
        "url": robots_url,
        "status_code": None,
        "found": False,
        "agents": {},
        "raw_content": None,
        "warnings": []
    }
    
    try:
        # Use browser user agent to fetch robots.txt to prevent blocks
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                robots_url,
                headers={"User-Agent": BROWSER_USER_AGENT},
                timeout=10
            )
        result["status_code"] = response.status_code
        
        if response.status_code == 200:
            result["found"] = True
            raw_text = response.text
            result["raw_content"] = raw_text
            
            # Setup robots.txt parser
            rp = urllib.robotparser.RobotFileParser()
            lines = raw_text.splitlines()
            rp.parse(lines)
            
            # Check permissions for each AI search agent
            for agent_name in AI_USER_AGENTS.keys():
                allowed_root = rp.can_fetch(agent_name, "/")
                allowed_path = rp.can_fetch(agent_name, parsed_url.path or "/")
                
                result["agents"][agent_name] = {
                    "allowed_on_root": allowed_root,
                    "allowed_on_path": allowed_path,
                    "status": "Allowed" if allowed_path else "Blocked"
                }
                
                # Check for explicit textual blocks (heuristic backup)
                agent_lower = agent_name.lower()
                ua_pattern = rf"user-agent:\s*{re.escape(agent_lower)}"
                if re.search(ua_pattern, raw_text.lower()):
                    result["agents"][agent_name]["explicitly_mentioned"] = True
                else:
                    result["agents"][agent_name]["explicitly_mentioned"] = False
                    
        elif response.status_code == 404:
            result["found"] = False
            result["warnings"].append("robots.txt not found (404). Access is implicitly allowed to all bots.")
            for agent_name in AI_USER_AGENTS.keys():
                result["agents"][agent_name] = {
                    "allowed_on_root": True,
                    "allowed_on_path": True,
                    "status": "Implicitly Allowed",
                    "explicitly_mentioned": False
                }
        else:
            result["found"] = False
            result["warnings"].append(f"robots.txt returned status code {response.status_code}.")
            for agent_name in AI_USER_AGENTS.keys():
                result["agents"][agent_name] = {
                    "allowed_on_root": False,
                    "allowed_on_path": False,
                    "status": f"Restricted (Status {response.status_code})",
                    "explicitly_mentioned": False
                }
                
    except httpx.RequestError as e:
        result["found"] = False
        result["warnings"].append(f"Failed to fetch robots.txt: {str(e)}")
        for agent_name in AI_USER_AGENTS.keys():
            result["agents"][agent_name] = {
                "allowed_on_root": False,
                "allowed_on_path": False,
                "status": "Unknown (Fetch Error)",
                "explicitly_mentioned": False
            }
            
    return result


def detect_anti_bot_signatures(response: httpx.Response) -> List[str]:
    """
    Checks response status, headers, and body for indicators of anti-bot systems 
    (Cloudflare, CAPTCHAs, paywalls, etc.).
    """
    signatures = []
    status = response.status_code
    headers = {k.lower(): v.lower() for k, v in response.headers.items()}
    body = response.text.lower()
    
    # 1. HTTP status codes indicating blocking
    if status == 403:
        signatures.append("403 Forbidden: Access denied by web server")
    elif status == 429:
        signatures.append("429 Too Many Requests: Rate limiting active")
    elif status == 503:
        signatures.append("503 Service Unavailable: Could be server protection or overload")
        
    # 2. Cloudflare detection
    if "cloudflare" in headers.get("server", ""):
        if "cf-ray" in headers:
            signatures.append("Cloudflare: Active protection detected (cf-ray header present)")
        if "captcha" in body or "cf-challenge" in body or "challenge-platform" in body:
            signatures.append("Cloudflare: Security challenge/CAPTCHA detected in page body")
            
    # 3. Captcha and dynamic challenge indicators
    if "captcha" in body and "recaptcha" in body:
        signatures.append("CAPTCHA: Google reCAPTCHA script detected")
    elif "captcha" in body and not "cloudflare" in body:
        signatures.append("CAPTCHA: General CAPTCHA keywords found in content")
        
    # 4. Javascript enforcement warnings
    js_warnings = [
        "please enable javascript",
        "you need to enable javascript to run this app",
        "javascript is required",
        "enable javascript and cookies"
    ]
    for warning in js_warnings:
        if warning in body:
            signatures.append("JavaScript Block: Site requires JS (blocks direct HTML parsers)")
            break
            
    # 5. Paywall / Registration wall indicators
    paywall_keywords = [
        "subscriber-only",
        "subscribe to read the full story",
        "create an account to continue reading",
        "premium article",
        "please log in to read"
    ]
    for kw in paywall_keywords:
        if kw in body:
            signatures.append("Paywall: High probability of paywall or log-in wall")
            break
            
    return signatures


def analyze_html_structure(html_content: str) -> Dict[str, Any]:
    """
    Parses the HTML to assess heading hierarchies, semantic HTML layout, 
    metadata structure, and schema tags.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Page Title and Metadata
    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else None
    
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc.get("content", "").strip() if meta_desc else None
    
    # Robots tag inspection (meta robots)
    meta_robots = soup.find("meta", attrs={"name": "robots"})
    robots_meta = meta_robots.get("content", "").strip().lower() if meta_robots else None
    
    # Specific meta instructions for AI
    ai_meta_tags = []
    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        if "ai" in name or "bot" in name or "crawler" in name:
            ai_meta_tags.append({
                "name": meta.get("name"),
                "content": meta.get("content")
            })
            
    # 2. Semantic HTML Check
    semantic_elements = {
        "main": len(soup.find_all("main")),
        "article": len(soup.find_all("article")),
        "section": len(soup.find_all("section")),
        "header": len(soup.find_all("header")),
        "footer": len(soup.find_all("footer")),
        "nav": len(soup.find_all("nav")),
        "aside": len(soup.find_all("aside"))
    }
    
    has_semantic_structure = (
        semantic_elements["main"] > 0 or 
        semantic_elements["article"] > 0 or 
        semantic_elements["section"] > 0
    )
    
    # 3. Heading Hierarchy Analysis
    headings = {}
    total_headings = 0
    heading_hierarchy = []
    
    for i in range(1, 7):
        tag_name = f"h{i}"
        tags = soup.find_all(tag_name)
        count = len(tags)
        headings[tag_name] = count
        total_headings += count
        
        for t in tags:
            heading_hierarchy.append({
                "level": i,
                "text": t.get_text().strip()
            })
            
    # Heading hierarchy validation issues
    hierarchy_warnings = []
    if headings["h1"] == 0:
        hierarchy_warnings.append("Missing h1 tag: The page lacks a main title heading.")
    elif headings["h1"] > 1:
        hierarchy_warnings.append(f"Multiple h1 tags ({headings['h1']}): Can confuse LLM topic structure.")
        
    has_h2 = headings["h2"] > 0
    has_h3 = headings["h3"] > 0
    if not has_h2 and has_h3:
        hierarchy_warnings.append("Broken hierarchy: Page contains h3 headings but no h2 headings.")
        
    # 4. Structured Data / Schema Markup Detection
    json_ld_schemas = []
    json_ld_tags = soup.find_all("script", type="application/ld+json")
    
    for tag in json_ld_tags:
        try:
            schema_data = json.loads(tag.string or "")
            if isinstance(schema_data, dict):
                json_ld_schemas.append(schema_data.get("@type", "Unknown"))
            elif isinstance(schema_data, list):
                for item in schema_data:
                    if isinstance(item, dict):
                        json_ld_schemas.append(item.get("@type", "Unknown"))
        except (json.JSONDecodeError, TypeError):
            json_ld_schemas.append("Malformed JSON-LD")
            
    # 5. Lists & Tables
    lists_and_tables = {
        "ul_count": len(soup.find_all("ul")),
        "ol_count": len(soup.find_all("ol")),
        "table_count": len(soup.find_all("table")),
        "blockquote_count": len(soup.find_all("blockquote"))
    }
    
    # 6. Text density and script load
    raw_html_len = len(html_content)
    
    clean_soup = BeautifulSoup(html_content, "html.parser")
    for script in clean_soup(["script", "style", "noscript", "svg", "iframe"]):
        script.extract()
        
    page_text = clean_soup.get_text()
    page_text = re.sub(r'\s+', ' ', page_text).strip()
    text_len = len(page_text)
    
    text_to_html_ratio = (text_len / raw_html_len * 100) if raw_html_len > 0 else 0
    
    all_tags = len(soup.find_all())
    script_tags = len(soup.find_all("script"))
    
    js_density_ratio = (script_tags / all_tags * 100) if all_tags > 0 else 0
    
    return {
        "meta": {
            "title": title,
            "description": description,
            "robots_meta": robots_meta,
            "ai_meta_tags": ai_meta_tags
        },
        "semantic_layout": {
            "counts": semantic_elements,
            "has_semantic_structure": has_semantic_structure,
            "lists_and_tables": lists_and_tables
        },
        "headings": {
            "counts": headings,
            "total": total_headings,
            "warnings": hierarchy_warnings,
            "sample_hierarchy": heading_hierarchy[:15]
        },
        "schema": {
            "json_ld_found": len(json_ld_tags) > 0,
            "json_ld_count": len(json_ld_tags),
            "detected_types": json_ld_schemas
        },
        "page_metrics": {
            "html_size_bytes": raw_html_len,
            "clean_text_chars": text_len,
            "text_to_html_ratio_percent": round(text_to_html_ratio, 2),
            "script_tags_count": script_tags,
            "js_density_percent": round(js_density_ratio, 2)
        }
    }


async def _fetch(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """Fetches a single URL, returning either the response or the exception -- kept as a
    plain result dict (instead of raising) so `asyncio.gather` callers don't need
    `return_exceptions=True` juggling for a single well-understood failure mode."""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, timeout=12)
        return {"response": response, "error": None}
    except httpx.RequestError as e:
        return {"response": None, "error": str(e)}


async def audit_website(url: str) -> Dict[str, Any]:
    """
    Performs a comprehensive RAG & Search Citation visibility audit on a given URL.
    Fetches the page with standard browser headers and AI Search Bot headers (ChatGPT-User),
    checks robots.txt permissions for search, and scores accessibility.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        
    report = {
        "url": url,
        "robots_analysis": {},
        "bot_fetch_check": {
            "success": False,
            "status_code": None,
            "anti_bot_signals": [],
            "error": None
        },
        "browser_fetch_check": {
            "success": False,
            "status_code": None,
            "anti_bot_signals": [],
            "error": None
        },
        "structure_analysis": {},
        "html_content": "",
        "readability_score": 0,
        "verdict": "Unknown",
        "key_issues": []
    }
    
    # 1. Robots.txt audit, 2. AI-bot-simulated fetch (ChatGPT-User), and 3. standard-browser
    # fetch are all independent network calls to the same host -- run them all concurrently
    # instead of one after another.
    bot_headers = {
        "User-Agent": AI_USER_AGENTS["ChatGPT-User"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    browser_headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }

    robots_analysis, bot_result, browser_result = await asyncio.gather(
        check_robots_txt(url),
        _fetch(url, bot_headers),
        _fetch(url, browser_headers),
    )
    report["robots_analysis"] = robots_analysis

    if bot_result["error"] is not None:
        report["bot_fetch_check"]["error"] = bot_result["error"]
        report["key_issues"].append(f"AI search bot simulator failed to connect: {bot_result['error']}")
    else:
        bot_response = bot_result["response"]
        report["bot_fetch_check"]["status_code"] = bot_response.status_code
        report["bot_fetch_check"]["anti_bot_signals"] = detect_anti_bot_signatures(bot_response)
        if bot_response.status_code == 200:
            report["bot_fetch_check"]["success"] = True
        else:
            report["key_issues"].append(
                f"AI search bot simulator (ChatGPT-User) received status code {bot_response.status_code}. The crawler might be blocked by web server configuration."
            )

    html_content = ""
    if browser_result["error"] is not None:
        report["browser_fetch_check"]["error"] = browser_result["error"]
        report["key_issues"].append(f"Standard browser request failed: {browser_result['error']}")
    else:
        browser_response = browser_result["response"]
        report["browser_fetch_check"]["status_code"] = browser_response.status_code
        report["browser_fetch_check"]["anti_bot_signals"] = detect_anti_bot_signatures(browser_response)
        if browser_response.status_code == 200:
            report["browser_fetch_check"]["success"] = True
            html_content = browser_response.text
            report["html_content"] = html_content
        else:
            report["key_issues"].append(f"Standard browser request returned error status: {browser_response.status_code}")

    # 4. Check for Scraper Cloaking or Anti-Bot Blocks
    if report["browser_fetch_check"]["success"] and not report["bot_fetch_check"]["success"]:
        report["key_issues"].append("Cloaking/Blocking: The server successfully served pages to browsers, but returned errors/blocked request for ChatGPT-User.")
    
    if len(report["bot_fetch_check"]["anti_bot_signals"]) > 0:
        for signal in report["bot_fetch_check"]["anti_bot_signals"]:
            report["key_issues"].append(f"Scraper block signal: {signal}")
            
    # 5. Run HTML Structure Analysis if page content was fetched successfully
    if html_content:
        report["structure_analysis"] = analyze_html_structure(html_content)
        
        # Check meta robots rules
        robots_meta = report["structure_analysis"]["meta"]["robots_meta"]
        if robots_meta:
            if "noindex" in robots_meta:
                report["key_issues"].append("Meta robots directive 'noindex' blocks search crawlers from indexing this page.")
            if "noai" in robots_meta or "nocrawl" in robots_meta:
                report["key_issues"].append("Meta robots includes AI restrictions ('noai' or 'nocrawl').")
                
        # Structure findings
        headings_warnings = report["structure_analysis"]["headings"]["warnings"]
        for w in headings_warnings:
            report["key_issues"].append(f"Heading Issue: {w}")
            
        if not report["structure_analysis"]["semantic_layout"]["has_semantic_structure"]:
            report["key_issues"].append("Semantic layout deficiency: Page lacks main, article, or section containers. This makes extracting clear copy harder for LLMs.")
            
        if not report["structure_analysis"]["schema"]["json_ld_found"]:
            report["key_issues"].append("Missing structured data: No JSON-LD schema found. Structured schemas are critical for Answer Engine (AEO) accuracy.")
            
        metrics = report["structure_analysis"]["page_metrics"]
        if metrics["text_to_html_ratio_percent"] < 8.0:
            report["key_issues"].append(
                f"Low Text-to-HTML ratio ({metrics['text_to_html_ratio_percent']}%): Page might be JS-heavy, dynamic, or mostly boilerplates/styles."
            )
        if metrics["clean_text_chars"] < 500:
            report["key_issues"].append(f"Thin content warning: Page contains very little readable text ({metrics['clean_text_chars']} characters).")
            
    # 6. Calculate Heuristic Search Visibility Score (0 to 100)
    score = 100
    
    # Robots.txt check ONLY for Search & RAG bots. Anything other than an explicit
    # "Allowed"/"Implicitly Allowed" counts against the score -- this also catches cases
    # like "Restricted (Status 403)" (e.g. a Cloudflare challenge on /robots.txt itself),
    # which is just as bad for AI crawlers as an explicit Disallow rule, even though it's
    # not literally the string "Blocked".
    robots_agents = report["robots_analysis"].get("agents", {})
    allowed_statuses = {"Allowed", "Implicitly Allowed"}
    if robots_agents.get("ChatGPT-User", {}).get("status") not in allowed_statuses:
        score -= 30
    if robots_agents.get("Claude-Web", {}).get("status") not in allowed_statuses:
        score -= 20
    if robots_agents.get("PerplexityBot", {}).get("status") not in allowed_statuses:
        score -= 20
        
    # Connection failure / Anti-bot blocks for ChatGPT-User
    if not report["bot_fetch_check"]["success"] or len(report["bot_fetch_check"]["anti_bot_signals"]) > 0:
        score -= 30

    # If even a plain browser request can't get the page, nothing was actually verified --
    # this is a total, site-wide block (e.g. a Cloudflare wall in front of everything), not
    # merely an AI-specific issue, and should tank the score rather than being ignored just
    # because there's no HTML left to run structure checks against.
    if not report["browser_fetch_check"]["success"]:
        score -= 40
        
    # HTML structure health check
    if html_content:
        sa = report["structure_analysis"]
        if len(sa["headings"]["warnings"]) > 0:
            score -= 10
        if not sa["semantic_layout"]["has_semantic_structure"]:
            score -= 10
        if not sa["schema"]["json_ld_found"]:
            score -= 10
        if sa["page_metrics"]["clean_text_chars"] < 500:
            score -= 10
        elif sa["page_metrics"]["text_to_html_ratio_percent"] < 8.0:
            score -= 5
            
    report["readability_score"] = max(0, score)
    
    # 7. Render Verdict
    if report["readability_score"] >= 80:
        report["verdict"] = "Excellent (Highly Readable & Crawlable by AI Search Engines)"
    elif report["readability_score"] >= 55:
        report["verdict"] = "Moderate (Crawlable, but lacks optimal schema or HTML layout structure)"
    elif report["readability_score"] >= 30:
        report["verdict"] = "Poor (AI Search crawlers blocked in robots.txt or experiencing crawl issues)"
    else:
        report["verdict"] = "Critical (Completely blocked to AI Search bots / anti-bot protection triggered)"
        
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM RAG & Search Citation Auditor")
    parser.add_argument("url", type=str, help="The target website URL to audit")
    parser.add_argument("--json", action="store_true", help="Output raw JSON analysis data")
    
    args = parser.parse_args()
    
    print(f"Auditing website for RAG Search Visibility: {args.url}...\n")
    audit_data = asyncio.run(audit_website(args.url))
    
    if args.json:
        print(json.dumps(audit_data, indent=2))
    else:
        print("=" * 60)
        print(f"AEO/GEO AUDIT REPORT FOR: {audit_data['url']}")
        print(f"VERDICT: {audit_data['verdict']}")
        print(f"AEO / GEO VISIBILITY SCORE: {audit_data['readability_score']}/100")
        print("=" * 60)
        
        print("\n[Robots.txt AI Search Permissions]")
        print(f"  - URL: {audit_data['robots_analysis']['url']}")
        print(f"  - Found: {audit_data['robots_analysis']['found']}")
        
        print("\n  --> Real-Time RAG / Search Bots (Citations & Answers):")
        for bot in ["ChatGPT-User", "Claude-Web", "PerplexityBot"]:
            details = audit_data['robots_analysis'].get('agents', {}).get(bot, {"status": "Unknown"})
            status = details.get("status", "Unknown") if isinstance(details, dict) else details
            print(f"      * {bot:18}: {status}")
            
        print("\n[AI Search Crawler Connection Check]")
        print(f"  - Request Success: {audit_data['bot_fetch_check']['success']}")
        print(f"  - Simulated Agent: ChatGPT-User")
        print(f"  - Status Code: {audit_data['bot_fetch_check']['status_code']}")
        if audit_data['bot_fetch_check']['anti_bot_signals']:
            print("  - Bot signals flagged:")
            for signal in audit_data['bot_fetch_check']['anti_bot_signals']:
                print(f"    * {signal}")
                
        if audit_data.get("structure_analysis"):
            sa = audit_data["structure_analysis"]
            print("\n[HTML Structure Analysis]")
            print(f"  - Title: {sa['meta']['title']}")
            print(f"  - Meta Description: {sa['meta']['description']}")
            print(f"  - Meta Robots: {sa['meta']['robots_meta']}")
            print(f"  - Has Semantic Layout: {sa['semantic_layout']['has_semantic_structure']}")
            print(f"  - JSON-LD Structured Schema: {'Yes' if sa['schema']['json_ld_found'] else 'No'} ({sa['schema']['json_ld_count']} tags)")
            if sa['schema']['detected_types']:
                print(f"    * Detected types: {', '.join(set(sa['schema']['detected_types']))}")
                
            print(f"\n[Page Content Metrics]")
            print(f"  - Raw HTML size: {sa['page_metrics']['html_size_bytes']} bytes")
            print(f"  - Clean Text length: {sa['page_metrics']['clean_text_chars']} characters")
            print(f"  - Text-to-HTML Ratio: {sa['page_metrics']['text_to_html_ratio_percent']}%")
            print(f"  - JS Tag Density: {sa['page_metrics']['js_density_percent']}%")
            
        if audit_data["key_issues"]:
            print("\n[Key Issues Identified]")
            for issue in audit_data["key_issues"]:
                print(f"  [!] {issue}")
        else:
            print("\n[Key Issues Identified]")
            print("  None! Excellent RAG citation visibility.")
        print("-" * 60)
