#!/usr/bin/env python3
"""
AI visibility Scoring and Insights Generator
Provides tools to evaluate page information density, off-page authority mentions,
and generates strategic checklists.
"""

import re
import urllib.parse
import asyncio
import random
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Tuple, Optional

# DuckDuckGo HTML search headers
SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

BROWSER_HEADERS = {
    "User-Agent": SEARCH_HEADERS["User-Agent"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _normalize_domain(domain: str) -> str:
    return domain.lower().replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]


def _domain_matches_text(domain: str, brand_name: str, text: str) -> bool:
    text_lower = text.lower()
    if domain.lower() in text_lower:
        return True
    brand_lower = brand_name.lower().strip()
    if len(brand_lower) >= 4 and brand_lower in text_lower:
        return True
    return False


def _linkedin_company_slugs(domain: str) -> List[str]:
    clean = _normalize_domain(domain)
    base = clean.split(".")[0]
    return list(dict.fromkeys([
        clean.replace(".", "-"),
        f"{base.replace('_', '-')}-work",
        base.replace("_", "-"),
    ]))


def _platform_presence_status(signal_count: int, verified: bool = False) -> str:
    if verified or signal_count >= 5:
        return "Verified Presence"
    if signal_count >= 2:
        return "High Presence"
    if signal_count >= 1:
        return "Low Presence"
    return "No Mentions Detected"


async def _fetch_linkedin_slug(client: httpx.AsyncClient, slug: str, clean_domain: str) -> Optional[Dict[str, Any]]:
    """Fetches one candidate LinkedIn company-page slug. Returns match details or None."""
    url = f"https://www.linkedin.com/company/{slug}"
    try:
        response = await client.get(url, headers=BROWSER_HEADERS, timeout=12)
    except httpx.RequestError:
        return None

    if response.status_code != 200:
        return None

    page_text = response.text.lower()
    if clean_domain not in page_text:
        return None

    followers = None
    follower_match = re.search(r"([\d,]+)\s+followers", response.text, re.IGNORECASE)
    if follower_match:
        followers = int(follower_match.group(1).replace(",", ""))

    return {"url": str(response.url), "followers": followers}


async def _probe_linkedin_presence(domain: str, brand_name: str) -> Dict[str, Any]:
    clean_domain = _normalize_domain(domain)
    slugs = _linkedin_company_slugs(domain)

    # All candidate slugs are independent lookups -- fetch them concurrently, then prefer
    # the first match in the original priority order (same as the old early-break loop).
    async with httpx.AsyncClient(follow_redirects=True) as client:
        slug_results = await asyncio.gather(*(_fetch_linkedin_slug(client, slug, clean_domain) for slug in slugs))

    matched_urls: List[str] = []
    followers = None
    verified = False
    for match in slug_results:
        if match is not None:
            matched_urls.append(match["url"])
            followers = match["followers"]
            verified = True
            break

    signal_count = 1 if verified else 0
    if followers and followers >= 1000:
        signal_count = max(signal_count, 5)
    elif followers and followers >= 100:
        signal_count = max(signal_count, 3)

    return {
        "method": "direct",
        "lookup_status": "OK" if verified else "Not Found",
        "estimated_mention_count": signal_count,
        "mention_links_sample": matched_urls[:5],
        "followers": followers,
        "status": _platform_presence_status(signal_count, verified=verified),
        "details": f"Company page verified ({followers:,} followers)" if followers else (
            "Company page verified" if verified else "No LinkedIn company page found for this domain"
        ),
    }


async def _probe_medium_presence(domain: str, brand_name: str) -> Dict[str, Any]:
    clean_domain = _normalize_domain(domain)
    query = clean_domain
    matched_urls: List[str] = []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://medium.com/search/posts",
                params={"q": query},
                headers=BROWSER_HEADERS,
                timeout=12,
            )
    except httpx.RequestError as exc:
        return {
            "method": "direct",
            "lookup_status": "Connection-Error",
            "estimated_mention_count": 0,
            "mention_links_sample": [],
            "status": "Lookup Failed",
            "details": str(exc),
        }

    if response.status_code != 200:
        return {
            "method": "direct",
            "lookup_status": f"HTTP-Error-{response.status_code}",
            "estimated_mention_count": 0,
            "mention_links_sample": [],
            "status": "Lookup Failed",
            "details": f"Medium search returned HTTP {response.status_code}",
        }

    page_lower = response.text.lower()
    mention_hits = page_lower.count(clean_domain)
    soup = BeautifulSoup(response.text, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "medium.com" not in href:
            continue
        anchor_text = anchor.get_text(" ", strip=True)
        if _domain_matches_text(clean_domain, brand_name, f"{href} {anchor_text}"):
            if href.startswith("/"):
                href = f"https://medium.com{href}"
            matched_urls.append(href)

    matched_urls = list(dict.fromkeys(matched_urls))
    signal_count = max(len(matched_urls), min(mention_hits // 3, 10)) if mention_hits else len(matched_urls)

    return {
        "method": "direct",
        "lookup_status": "OK",
        "estimated_mention_count": signal_count,
        "mention_links_sample": matched_urls[:5],
        "status": _platform_presence_status(signal_count, verified=signal_count >= 5),
        "details": f"{mention_hits} domain mentions detected in Medium search results",
    }


async def _search_reddit_query(client: httpx.AsyncClient, query: str, clean_domain: str, brand_name: str) -> List[str]:
    try:
        response = await client.get(
            "https://old.reddit.com/search",
            params={"q": query, "sort": "relevance", "t": "all"},
            headers=BROWSER_HEADERS,
            timeout=12,
        )
    except httpx.RequestError:
        return []

    if response.status_code != 200:
        return []

    found: List[str] = []
    soup = BeautifulSoup(response.text, "html.parser")
    for result in soup.select("div.search-result"):
        title_link = result.select_one("a.search-title")
        if not title_link:
            continue
        title = title_link.get_text(" ", strip=True)
        href = title_link.get("href", "")
        snippet = result.get_text(" ", strip=True)
        if _domain_matches_text(clean_domain, brand_name, f"{title} {href} {snippet}"):
            found.append(href if href.startswith("http") else f"https://old.reddit.com{href}")
    return found


async def _probe_reddit_presence(domain: str, brand_name: str) -> Dict[str, Any]:
    clean_domain = _normalize_domain(domain)
    queries = [f'"{clean_domain}"', clean_domain, brand_name]

    # Run all query variants concurrently and merge whatever each one finds, instead of
    # trying them one at a time and stopping at the first that returns a hit.
    async with httpx.AsyncClient(follow_redirects=True) as client:
        results_per_query = await asyncio.gather(
            *(_search_reddit_query(client, query, clean_domain, brand_name) for query in queries)
        )

    matched_urls: List[str] = [url for found in results_per_query for url in found]
    matched_urls = list(dict.fromkeys(matched_urls))
    signal_count = len(matched_urls)

    return {
        "method": "direct",
        "lookup_status": "OK",
        "estimated_mention_count": signal_count,
        "mention_links_sample": matched_urls[:5],
        "status": _platform_presence_status(signal_count),
        "details": f"{signal_count} Reddit posts/comments mention the brand or domain",
    }


async def _search_wikipedia_term(client: httpx.AsyncClient, search_term: str, clean_domain: str) -> List[str]:
    try:
        response = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_term,
                "format": "json",
            },
            headers=BROWSER_HEADERS,
            timeout=12,
        )
    except httpx.RequestError:
        return []

    if response.status_code != 200:
        return []

    found: List[str] = []
    for item in response.json().get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if clean_domain in title.lower() or clean_domain in snippet.lower():
            page_title = title.replace(" ", "_")
            found.append(f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title)}")
    return found


async def _probe_wikipedia_presence(domain: str, brand_name: str) -> Dict[str, Any]:
    clean_domain = _normalize_domain(domain)
    search_terms = [clean_domain, brand_name]

    # Run both search terms concurrently and merge whatever each one finds.
    async with httpx.AsyncClient() as client:
        results_per_term = await asyncio.gather(
            *(_search_wikipedia_term(client, term, clean_domain) for term in search_terms)
        )

    matched_urls: List[str] = [url for found in results_per_term for url in found]
    matched_urls = list(dict.fromkeys(matched_urls))
    signal_count = len(matched_urls)

    return {
        "method": "direct",
        "lookup_status": "OK",
        "estimated_mention_count": signal_count,
        "mention_links_sample": matched_urls[:5],
        "status": _platform_presence_status(signal_count, verified=signal_count > 0),
        "details": (
            f"{signal_count} Wikipedia article(s) reference this domain"
            if signal_count else "No Wikipedia article specifically references this domain"
        ),
    }


def _merge_platform_result(direct_result: Dict[str, Any], search_urls: List[str], search_status: str, query: str) -> Dict[str, Any]:
    merged_urls = list(dict.fromkeys(direct_result.get("mention_links_sample", []) + search_urls))
    direct_count = direct_result.get("estimated_mention_count", 0)
    search_count = len(search_urls) if search_status == "OK" else 0
    signal_count = max(direct_count, search_count, len(merged_urls))

    lookup_status = direct_result.get("lookup_status", "Unknown")
    if search_status == "Rate-Limited" and lookup_status in {"OK", "Not Found"}:
        lookup_note = "Direct check succeeded; supplemental search was rate-limited"
    elif search_status not in {"OK", "Rate-Limited"} and lookup_status == "OK":
        lookup_note = f"Direct check succeeded; supplemental search failed ({search_status})"
    elif search_status == "OK":
        lookup_note = "Direct check plus supplemental search"
    else:
        lookup_note = direct_result.get("details", lookup_status)

    result = {
        "query": query,
        "method": direct_result.get("method", "direct"),
        "lookup_status": lookup_status if lookup_status != "Not Found" or search_status == "OK" else "Not Found",
        "estimated_mention_count": signal_count,
        "mention_links_sample": merged_urls[:5],
        "status": _platform_presence_status(signal_count, verified=direct_result.get("followers") is not None or direct_count >= 5),
        "details": lookup_note,
    }
    if direct_result.get("followers") is not None:
        result["followers"] = direct_result["followers"]
    return result


async def search_duckduckgo(query: str) -> Tuple[List[str], str]:
    """
    Queries DuckDuckGo HTML search and extracts the ranking result URLs.
    Sleeps ~2s (with jitter) before each query to respect rate limits and checks for
    bot-block pages. Jitter matters because callers may fire several of these
    concurrently (e.g. one per platform/keyword) -- without it, concurrent calls would
    all wake up and hit DuckDuckGo in the same instant, which looks more bot-like.
    """
    # Sleep to prevent triggering search engine rate limits (non-blocking, so other
    # concurrently-running coroutines keep making progress during this wait).
    await asyncio.sleep(2.0 + random.uniform(0, 1.0))
    
    url = "https://html.duckduckgo.com/html/"
    payload = {"q": query}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload, headers=SEARCH_HEADERS, timeout=12)
        
        # Detect anomaly/captcha blocks (status 202 is typical for DDG challenge pages)
        if response.status_code == 202 or "anomaly.js" in response.text or "confirm this search was made by a human" in response.text:
            return [], "Rate-Limited"
            
        if response.status_code != 200:
            return [], f"HTTP-Error-{response.status_code}"
            
        soup = BeautifulSoup(response.text, "html.parser")
        extracted_urls = []

        def _extract_result_url(href: str) -> Optional[str]:
            if not href:
                return None
            parsed_href = urllib.parse.urlparse(href)
            query_params = urllib.parse.parse_qs(parsed_href.query)
            if "uddg" in query_params:
                return query_params["uddg"][0]
            clean_href = href.strip()
            if clean_href.startswith("http"):
                return clean_href
            return None

        # Parse both legacy and current DuckDuckGo HTML result anchors
        for anchor in soup.find_all("a", class_=["result__url", "result__a"]):
            resolved = _extract_result_url(anchor.get("href"))
            if resolved:
                extracted_urls.append(resolved)

        extracted_urls = list(dict.fromkeys(extracted_urls))
        return extracted_urls, "OK"
    except Exception:
        return [], "Connection-Error"


def evaluate_information_density(html_content: str, clean_text: str) -> Dict[str, Any]:
    """
    Computes text density metrics to identify LLM-friendly facts and layouts.
    Measures the frequency of numbers, statistics, structure tags, and checks for "fluff".
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Count Words
    words = clean_text.split()
    word_count = len(words)
    
    # 2. Detect Fact Indicators (Numbers, Dates, Percentages)
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', clean_text)
    percentages = re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b', clean_text, re.IGNORECASE)
    years = re.findall(r'\b(?:19|20)\d{2}\b', clean_text)
    
    fact_indicators_count = len(numbers) + len(percentages) + len(years)
    fact_ratio = (fact_indicators_count / word_count * 100) if word_count > 0 else 0
    
    # 3. Entity Indicators (Capitalized noun sequences, brand markers, etc.)
    sentences = re.split(r'[.!?]+', clean_text)
    entity_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words_in_sent = sentence.split()
        if len(words_in_sent) < 2:
            continue
        for w in words_in_sent[1:]:
            if w and w[0].isupper() and w.isalpha():
                entity_count += 1
                
    entity_ratio = (entity_count / word_count * 100) if word_count > 0 else 0
    
    # 4. Structural Formatting Counts
    bold_tags = len(soup.find_all(["strong", "b"]))
    list_items = len(soup.find_all("li"))
    tables = len(soup.find_all("table"))
    headers = len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))
    
    structural_weight = (bold_tags * 2) + (list_items * 1) + (tables * 10) + (headers * 3)
    structure_score = min(100, max(0, structural_weight))
    
    # Compute Density score
    density_score = 0
    if word_count > 0:
        fact_comp = min(30, (fact_ratio / 10) * 30)
        entity_comp = min(30, (entity_ratio / 15) * 30)
        struct_comp = min(40, (structure_score / 50) * 40)
        density_score = round(fact_comp + entity_comp + struct_comp, 1)
        
    verdict = "Low (Highly conversational or thin)"
    if density_score >= 75:
        verdict = "High (Extremely factual, structured, and informative)"
    elif density_score >= 40:
        verdict = "Moderate (Balanced text with adequate facts and headers)"
        
    return {
        "word_count": word_count,
        "metrics": {
            "fact_indicators_found": fact_indicators_count,
            "fact_density_percent": round(fact_ratio, 2),
            "estimated_entities_found": entity_count,
            "entity_density_percent": round(entity_ratio, 2),
            "structural_elements": {
                "bold_or_strong": bold_tags,
                "list_items": list_items,
                "tables": tables,
                "headings": headers
            }
        },
        "density_score": density_score,
        "verdict": verdict
    }


async def _check_platform_presence(platform: str, probe, clean_domain: str, brand_name: str, query: str) -> Tuple[Dict[str, Any], str]:
    """Runs one platform's direct probe and its supplemental DuckDuckGo search
    concurrently (they're independent of each other), so all platforms combined only
    take as long as the single slowest platform+search pair."""
    direct_result, (found_urls, search_status) = await asyncio.gather(
        probe(clean_domain, brand_name),
        search_duckduckgo(query),
    )
    return _merge_platform_result(direct_result, found_urls, search_status, query), search_status


async def check_offpage_presence(brand_name: str, domain: str) -> Dict[str, Any]:
    """
    Estimates brand/domain footprint across high-authority platforms heavily crawled by LLMs:
    Reddit, Medium, LinkedIn, and Wikipedia.

    Uses direct platform probes first (reliable), then supplements with DuckDuckGo when available.
    All platforms are checked concurrently since each is an independent network round-trip.
    """
    clean_domain = _normalize_domain(domain)
    # Prefer the domain over generic brand names like "Remoting" that create false positives
    brand_query = f'"{clean_domain}" OR "{brand_name}"'

    platform_probes = {
        "LinkedIn": _probe_linkedin_presence,
        "Medium": _probe_medium_presence,
        "Reddit": _probe_reddit_presence,
        "Wikipedia": _probe_wikipedia_presence,
    }
    platform_queries = {
        "Reddit": f"{brand_query} site:reddit.com",
        "Medium": f"{brand_query} site:medium.com",
        "LinkedIn": f"{brand_query} site:linkedin.com",
        "Wikipedia": f"{brand_query} site:wikipedia.org",
    }

    platforms_in_order = list(platform_probes.keys())
    platform_check_results = await asyncio.gather(*(
        _check_platform_presence(platform, platform_probes[platform], clean_domain, brand_name, platform_queries[platform])
        for platform in platforms_in_order
    ))

    results = {}
    platform_scores = {}
    supplemental_rate_limited = False

    for platform, (platform_result, search_status) in zip(platforms_in_order, platform_check_results):
        if search_status == "Rate-Limited":
            supplemental_rate_limited = True

        results[platform] = platform_result

        if platform_result["status"] == "Verified Presence":
            platform_scores[platform] = 30
        elif platform_result["status"] == "High Presence":
            platform_scores[platform] = 20
        elif platform_result["status"] == "Low Presence":
            platform_scores[platform] = 10
        else:
            platform_scores[platform] = 0

    offpage_score = min(100, sum(platform_scores.values()))

    if supplemental_rate_limited:
        verdict = (
            "Strong Brand Footprint (direct checks; supplemental search partially rate-limited)"
            if offpage_score >= 60 else
            "Moderate Footprint (direct checks; supplemental search partially rate-limited)"
            if offpage_score >= 20 else
            "Weak/No Off-page Presence (direct checks; supplemental search partially rate-limited)"
        )
    else:
        verdict = (
            "Strong Brand Footprint" if offpage_score >= 60 else
            "Moderate Footprint" if offpage_score >= 20 else
            "Weak/No Off-page Presence"
        )

    return {
        "target_brand": brand_name,
        "target_domain": clean_domain,
        "platforms": results,
        "offpage_footprint_score": offpage_score,
        "rate_limited": supplemental_rate_limited,
        "verdict": verdict,
    }


def _weak_offpage_platforms(offpage_data: Dict[str, Any]) -> List[str]:
    """Returns platform names where brand footprint is thin-to-nonexistent, so
    recommendations can call out specific channels instead of a generic list."""
    weak_statuses = {"No Mentions Detected", "Low Presence"}
    platforms = offpage_data.get("platforms", {})
    return [name for name, data in platforms.items() if data.get("status") in weak_statuses]


def generate_recommendations(audit_data: Dict[str, Any], density_data: Dict[str, Any], offpage_data: Dict[str, Any], ai_visibility_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    Translates technical auditor issues and visibility scores into a strategic roadmap
    of client recommendations.
    """
    recommendations = []
    
    # 1. Technical/Access Recommendations
    robots = audit_data.get("robots_analysis", {})
    agents = robots.get("agents", {})
    allowed_statuses = {"Allowed", "Implicitly Allowed"}
    # Checked regardless of `robots.get("found")` -- a robots.txt that itself returns a
    # non-200 status (e.g. blocked by Cloudflare/anti-bot rules, like a 403 challenge page)
    # is just as much of a real-time-crawler problem as an explicit Disallow rule, and
    # should still surface here instead of being silently skipped.
    blocked_search_bots = [
        bot for bot in ["ChatGPT-User", "Claude-Web", "PerplexityBot"]
        if agents.get(bot, {}).get("status") not in allowed_statuses
    ]
    if blocked_search_bots:
        if robots.get("found"):
            issue = f"Robots.txt blocks real-time RAG scrapers: {', '.join(blocked_search_bots)}"
            action = f"Update robots.txt to Allow these user-agents: {', '.join(blocked_search_bots)}"
        else:
            issue = (
                f"robots.txt could not be verified as accessible (status: {robots.get('status_code')}), "
                f"so these real-time RAG scrapers can't confirm they're permitted: {', '.join(blocked_search_bots)}"
            )
            action = (
                "Ensure /robots.txt itself returns a normal 200 response to crawlers (check for a "
                "Cloudflare/anti-bot challenge sitting in front of it), then explicitly Allow these user-agents."
            )
        recommendations.append({
            "category": "Technical Accessibility",
            "priority": "Critical",
            "issue": issue,
            "action": action,
            "impact": "Enables ChatGPT Search and Claude to read and cite your website in real-time answers."
        })
        
    # anti-bot blockages
    if not audit_data.get("bot_fetch_check", {}).get("success") or audit_data.get("bot_fetch_check", {}).get("anti_bot_signals"):
        recommendations.append({
            "category": "Technical Accessibility",
            "priority": "Critical",
            "issue": "AI crawlers trigger anti-bot blocks (e.g. Cloudflare / CAPTCHA firewall filters).",
            "action": "Configure your firewall/CDN (like Cloudflare bypass rules) to whitelist or allow verified AI search crawlers.",
            "impact": "Prevents AI search engines from throwing connection errors and skipping your site."
        })
        
    # 2. AEO & Structure Recommendations
    sa = audit_data.get("structure_analysis", {})
    if sa:
        if not sa.get("schema", {}).get("json_ld_found"):
            recommendations.append({
                "category": "AEO / Schema",
                "priority": "High",
                "issue": "Lacks JSON-LD structured schemas.",
                "action": "Implement JSON-LD structured data. For business landing pages, add Organization schema. For content, add Article or FAQPage schema.",
                "impact": "Helps Answer Engines parse exact entity data, products, and contact info, boosting answer accuracy."
            })
            
        headings = sa.get("headings", {})
        if headings.get("counts", {}).get("h1", 0) == 0:
            recommendations.append({
                "category": "On-Page Formatting",
                "priority": "High",
                "issue": "Main page is missing an <h1> tag.",
                "action": "Ensure there is exactly one clean <h1> tag at the top of the content stating the page's core topic.",
                "impact": "Establishes a clear topic index for LLM document parsing."
            })
        elif headings.get("counts", {}).get("h1", 0) > 1:
            recommendations.append({
                "category": "On-Page Formatting",
                "priority": "Medium",
                "issue": "Multiple <h1> headings found.",
                "action": "Consolidate the main page headings so only the primary page title is an <h1>, converting other headings to <h2>.",
                "impact": "Eliminates structure parsing ambiguity for LLM semantic engines."
            })
            
        if not sa.get("semantic_layout", {}).get("has_semantic_structure"):
            recommendations.append({
                "category": "On-Page Formatting",
                "priority": "Medium",
                "issue": "Page does not use semantic HTML5 layouts (<article>, <section>, <main>).",
                "action": "Refactor HTML markup to wrap page components in semantic containers like <main>, <article>, and <section>.",
                "impact": "Allows scraper algorithms to separate main page body copy from header/footer boilerplate navigation."
            })
            
        metrics = sa.get("page_metrics", {})
        if metrics.get("clean_text_chars", 0) < 300 or metrics.get("text_to_html_ratio_percent", 0.0) < 5.0:
            recommendations.append({
                "category": "Content Accessibility",
                "priority": "High",
                "issue": "Extremely low text-to-HTML ratio / Client-side JS rendering wrapper detected.",
                "action": "Implement Server-Side Rendering (SSR), static generation (SSG), or pre-rendering (e.g. Next.js, Gatsby, or prerender.io).",
                "impact": "Ensures AI bots fetching raw HTML can extract your actual copy, instead of downloading an empty loading spinner."
            })
            
    # 3. Content Density Recommendations
    if density_data.get("density_score", 0) < 50:
        recommendations.append({
            "category": "Content & Density",
            "priority": "High",
            "issue": f"Low Information Density (Density Score: {density_data['density_score']}/100).",
            "action": "Rewrite content to state facts, data points, statistics, and definitions directly. Avoid vague marketing fluff. Format lists into clean bulleted layouts.",
            "impact": "Provides concrete data points that RAG semantic algorithms value and select as quotes for AI answers."
        })
        
    # 4. Off-page presence recommendations
    offpage_score = offpage_data.get("offpage_footprint_score")
    if offpage_score is not None and offpage_score < 30:
        weak_platforms = _weak_offpage_platforms(offpage_data)
        action_items = []
        if "Reddit" in weak_platforms:
            action_items.append(
                "Post content on Reddit: engage weekly in 2-3 niche subreddits relevant to your "
                "category, answering real questions with genuine value (not just links/self-promo)."
            )
        if "Medium" in weak_platforms:
            action_items.append(
                "Publish at least one in-depth article per month on Medium, cross-posted from your own blog."
            )
        if "LinkedIn" in weak_platforms:
            action_items.append(
                "Post weekly company updates, case studies, or industry insights on your LinkedIn "
                "company page to build a followed, verifiable presence."
            )
        if "Wikipedia" in weak_platforms:
            action_items.append(
                "If your brand meets notability guidelines, get a well-sourced Wikipedia article "
                "created or expanded referencing your domain."
            )
        if not action_items:
            action_items.append(
                "Actively distribute content and brand references on Reddit (niche subreddits), "
                "Medium publications, and LinkedIn updates."
            )
        recommendations.append({
            "category": "Off-Page Authority",
            "priority": "High",
            "issue": f"Low brand mentions on authority sources (Off-page Score: {offpage_score}/100).",
            "action": "Actively distribute content and brand references on Reddit (niche subreddits), Medium publications, and LinkedIn updates.",
            "action_items": action_items,
            "impact": "Creates citations in high-authority datasets that AI models crawl daily for real-time reference retrieval."
        })
        
    # 5. Live AI mention/recommendation recommendations
    if ai_visibility_data:
        mention_data = ai_visibility_data.get("mention") or {}
        mention_rate = mention_data.get("mention_rate_percent")
        if mention_rate is not None and mention_rate < 20.0:
            recommendations.append({
                "category": "AI Chat Visibility",
                "priority": "High",
                "issue": f"Low brand mention rate in live AI chatbot answers ({mention_rate}%).",
                "action": "Publish more original, quotable content and build a citable brand presence across high-authority platforms so AI models associate your brand with these topics.",
                "action_items": [
                    "Start a weekly blog publishing original data, guides, and definitions around your "
                    "core topics -- AI models draw on fresh, quotable web content when forming answers.",
                    "Post content on Reddit in relevant niche subreddits; Reddit threads are heavily used "
                    "as real-time retrieval sources by ChatGPT Search and Perplexity.",
                    "Write and share articles on Medium and LinkedIn to build additional indexed, citable "
                    "brand mentions across the web.",
                ],
                "impact": "Increases the odds your brand is named when users ask AI chatbots informational questions in your niche."
            })

        recommendation_data = ai_visibility_data.get("recommendation")
        if recommendation_data:
            recommendation_rate = recommendation_data.get("recommendation_rate_percent")
            if recommendation_rate is not None and recommendation_rate < 20.0:
                recommendations.append({
                    "category": "AI Chat Visibility",
                    "priority": "Medium",
                    "issue": f"Rarely recommended by AI chatbots for buying-intent queries ({recommendation_rate}%).",
                    "action": "Build up third-party reviews/discussions and clear comparison/pricing content, since AI models lean on these signals for 'best of' recommendations.",
                    "action_items": [
                        "Post content on Reddit (niche subreddits, comparison threads) -- AI chat "
                        "assistants frequently surface Reddit opinions when asked for recommendations.",
                        "Write and share comparison/review-style articles on Medium and LinkedIn "
                        "positioning your brand against alternatives.",
                        "Encourage genuine customer reviews on third-party sites (G2, Capterra, niche "
                        "forums) relevant to your category.",
                    ],
                    "impact": "Improves the likelihood of appearing when users ask AI chatbots for top picks or recommendations in your category."
                })

    # Ensure every recommendation exposes a consistent, itemized checklist -- callers/CLIs can
    # always render `action_items` without special-casing which ones were given a richer list above.
    for rec in recommendations:
        rec.setdefault("action_items", [rec["action"]])

    return recommendations
