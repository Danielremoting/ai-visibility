#!/usr/bin/env python3
"""
Comprehensive Test Runner for the AI SEO Visibility Tracker Suite.
Audits and analyzes Technical Crawl Access, Information Density, Brand Footprint,
and Live AI Mentions/Recommendations.
"""

import sys
import asyncio
from auditor import audit_website
from analyzer import (
    evaluate_information_density,
    check_offpage_presence,
    generate_recommendations
)
from ai_visibility import track_ai_visibility

async def main():
    # If a URL is passed as a command line argument, use it; otherwise, default to remoting.work
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://remoting.work"
    
    # Custom keywords and brand selection to showcase relevance
    url_lower = target_url.lower()
    if "meallens" in url_lower:
        brand_name = "MealLensAI"
        keywords = [
            "diabetic meal planning AI",
            "meal plans for chronic conditions",
            "diet planner hypertension"
        ]
        business_category = "AI meal planning apps for chronic conditions"
    elif "remoting" in url_lower:
        brand_name = "Remoting"
        keywords = [
            "hire remote developers",
            "remote work platform",
            "best remote job board"
        ]
        business_category = "remote job platforms"
    else:
        brand_name = "Example Brand"
        keywords = [
            "remote work tools",
            "AI search visibility optimization",
            "generative engine optimization"
        ]
        business_category = "AI search visibility tools"
        
    print("=" * 70)
    print(f"RUNNING COMPLETE SUITE AUDIT ON: {target_url}")
    print(f"TARGET BRAND: {brand_name}")
    print(f"KEYWORDS RETRIEVAL TARGETS: {keywords}")
    print("=" * 70)
    
    try:
        # Steps 1, 3, and 4 don't depend on each other's results (each hits different
        # sites/APIs), so run them all concurrently via asyncio.gather instead of waiting
        # for each to finish in turn. Step 4 (live AI calls) is normally the slowest, so
        # overlapping it with the scraping-heavy steps below saves whatever time step 4
        # takes on its own.
        report, offpage_data, ai_visibility_data = await asyncio.gather(
            audit_website(target_url),                                            # Step 1
            check_offpage_presence(brand_name, target_url),                       # Step 3
            track_ai_visibility(                                                  # Step 4
                brand_name, target_url, keywords, business_category=business_category, samples=5
            ),
        )

        # Step 2: Information Density Analysis (pure local parsing, no network calls,
        # so it just runs on the main thread once step 1's HTML is available).
        density_data = {
            "word_count": 0,
            "density_score": 0,
            "verdict": "Could not parse (Failed connection)",
            "metrics": {
                "fact_indicators_found": 0,
                "fact_density_percent": 0.0,
                "estimated_entities_found": 0,
                "entity_density_percent": 0.0,
                "structural_elements": {"bold_or_strong": 0, "list_items": 0, "tables": 0, "headings": 0}
            }
        }
        html_content = report.get("html_content")
        clean_text = ""
        
        if html_content and report.get("structure_analysis"):
            # Extract clean text from metrics
            sa = report["structure_analysis"]
            
            # Reconstruct clean text from soup
            from bs4 import BeautifulSoup
            import re
            soup = BeautifulSoup(html_content, "html.parser")
            for script in soup(["script", "style", "noscript", "svg", "iframe"]):
                script.extract()
            clean_text = re.sub(r'\s+', ' ', soup.get_text()).strip()
            
            density_data = evaluate_information_density(html_content, clean_text)

        # Step 5: Recommendation Engine
        recommendations = generate_recommendations(report, density_data, offpage_data, ai_visibility_data)
        
        # Calculate overall unified visibility readiness score (combination of crawl, density, and offpage)
        # Combined score: 50% crawler/technical, 25% density, 25% off-page authority
        # Handle None values from rate-limited search results
        offpage_score = offpage_data.get("offpage_footprint_score")
        
        if offpage_score is not None:
            overall_score = round(
                (report["readability_score"] * 0.5) + 
                (density_data["density_score"] * 0.25) + 
                (offpage_score * 0.25)
            )
            overall_label = f"{overall_score}/100"
        else:
            # Partial score: only use available components
            partial = report["readability_score"] * 0.5 + density_data["density_score"] * 0.25
            overall_score = round(partial)
            overall_label = f"{overall_score}/100 (Partial — some lookups were rate-limited)"
        
        # --- PRINT THE DETAILED INSIGHTS ---
        
        print(f"\n[SUMMARY VERDICT]")
        print(f"  - Technical Crawl Access Verdict: {report['verdict']}")
        print(f"  - Technical Crawl Score:          {report['readability_score']}/100")
        print(f"  - Content Information Density:    {density_data['verdict']}")
        print(f"  - Content Density Score:          {density_data['density_score']}/100")
        print(f"  - Brand Off-page Presence:        {offpage_data['verdict']}")
        offpage_display = f"{offpage_score}/100" if offpage_score is not None else "N/A (Rate-Limited)"
        print(f"  - Brand Off-page Presence Score:  {offpage_display}")
        print("-" * 70)
        print(f"  ==> UNIFIED AI READINESS SCORE:   {overall_label}")
        print("-" * 70)
        
        print("\n1. Robots.txt Real-time Scraper Access:")
        for bot, details in report['robots_analysis'].get('agents', {}).items():
            print(f"   - {bot:15}: {details['status']} (Explicitly mentioned: {details.get('explicitly_mentioned', False)})")
            
        print(f"\n2. Scraper Fetch Checks (ChatGPT-User):")
        print(f"   - Standard Browser HTTP Status: {report['browser_fetch_check']['status_code']}")
        print(f"   - AI Search Bot HTTP Status:    {report['bot_fetch_check']['status_code']}")
        
        if report['bot_fetch_check']['anti_bot_signals']:
            print("   - Anti-bot signals flagged:")
            for signal in report['bot_fetch_check']['anti_bot_signals']:
                print(f"     * {signal}")
                
        if report.get("structure_analysis"):
            sa = report["structure_analysis"]
            print(f"\n3. Page Format & Semantic Layout Quality:")
            print(f"   - Schema (JSON-LD) found: {sa['schema']['json_ld_found']} (Total: {sa['schema']['json_ld_count']})")
            if sa['schema']['detected_types']:
                print(f"     * Types: {list(set(sa['schema']['detected_types']))}")
            print(f"   - Main Title (h1) count:  {sa['headings']['counts']['h1']}")
            print(f"   - Heading structure list: { {k: v for k, v in sa['headings']['counts'].items() if v > 0} }")
            print(f"   - HTML layout tags count: {sa['semantic_layout']['counts']}")
            
        print(f"\n4. Content Density & Fact Extraction:")
        print(f"   - Total Clean Word Count: {density_data['word_count']} words")
        print(f"   - Numeric facts found:    {density_data['metrics']['fact_indicators_found']} ({density_data['metrics']['fact_density_percent']}% density)")
        print(f"   - Estimated entities:     {density_data['metrics']['estimated_entities_found']} ({density_data['metrics']['entity_density_percent']}% density)")
        print(f"   - Structural weight:      Bold tags: {density_data['metrics']['structural_elements']['bold_or_strong']}, List items: {density_data['metrics']['structural_elements']['list_items']}")
        
        print(f"\n5. Off-page Brand Mention Footprint:")
        for platform, data in offpage_data['platforms'].items():
            detail = data.get('details', '')
            followers = data.get('followers')
            follower_note = f", {followers:,} LinkedIn followers" if followers else ""
            print(f"   - {platform:10}: {data['status']} ({data['estimated_mention_count']} signals{follower_note})")
            print(f"               {detail}")
            if data.get('mention_links_sample'):
                for link in data['mention_links_sample'][:2]:
                    print(f"               -> {link}")
            
        print(f"\n6. Live AI Mention & Recommendation Tracking:")
        mention_data = ai_visibility_data["mention"]
        print(f"   - Model Used:          {mention_data['model_used']}")
        print(f"   - Mention Verdict:     {mention_data['verdict']}")
        if mention_data['mention_rate_percent'] is not None:
            print(f"   - Mention Rate:        {mention_data['mention_rate_percent']}% ({mention_data['mention_occurrences']}/{mention_data['total_runs_completed']} runs)")
        if mention_data.get('had_errors'):
            print(f"   - Note: some AI queries failed (check API key/quota). Set GEMINI_API_KEY in your .env file.")

        recommendation_data = ai_visibility_data.get("recommendation")
        if recommendation_data:
            print(f"   - Recommendation Verdict: {recommendation_data['verdict']}")
            if recommendation_data['recommendation_rate_percent'] is not None:
                print(f"   - Recommendation Rate: {recommendation_data['recommendation_rate_percent']}% ({recommendation_data['recommendation_occurrences']}/{recommendation_data['runs_completed']} runs)")
                if recommendation_data['average_position_when_recommended']:
                    print(f"   - Avg Position When Recommended: #{recommendation_data['average_position_when_recommended']}")

        print(f"\n7. Automated Strategic Recommendations ({len(recommendations)} Actions):")
        if recommendations:
            for idx, rec in enumerate(recommendations, 1):
                print(f"   [{idx}] [{rec['category']}] (Priority: {rec['priority']})")
                print(f"       Issue:  {rec['issue']}")
                action_items = rec.get("action_items") or [rec["action"]]
                if len(action_items) > 1:
                    print(f"       Action Items:")
                    for item in action_items:
                        print(f"         - {item}")
                else:
                    print(f"       Action: {rec['action']}")
                print(f"       Impact: {rec['impact']}")
        else:
            print("   None! The website has exemplary visibility metrics.")
            
        print(f"\n" + "=" * 70)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error executing complete audit suite: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
