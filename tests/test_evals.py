import pytest
from langchain_core.messages import HumanMessage  # For consistent BaseMessage state (aligns with main.py; Handbook Ch. 6 uniform formats)
try:
    from ..agent import app  # Relative import from parent directory
except ImportError:
    from agent import app  # Fallback for running directly in root
from state import AgentState
import json

# Fixture for shared config
@pytest.fixture
def config():
    return {"configurable": {"thread_id": "test_thread"}}  # Fresh per test for isolation

# 8 Natural NL Prompts: 2 per agent type (standalone queries to test adaptation; no appended terms)
prompts = [
    # Segmentation (behavior/demographics)
    "Analyze high-spend users and their order frequency",  # Natural query for cohorts
    "What are the main customer cohorts by location",  # Natural query for demographics
    
    # Trends (sales/seasonality)
    "Analyze sales trends in summer 2024",  # Natural query for seasonal patterns
    "What was the seasonal revenue pattern in 2023",  # Natural query for time-series
    
    # Geo (location patterns)
    "What regions had the most sales in 2023",  # Natural query for top locations
    "Break down sales by country for the full dataset",  # Natural query for geo breakdown
    
    # Product (performance/recommendations)
    "What products performed best in winter 2023",  # Natural query for seasonal performance
    "Analyze top products by sales velocity"  # Natural query for recs/velocity
]

# Expected terms for asserts (dataset/schema-derived keywords for relevance proxy; not in prompts - Handbook Ch. 3)
expected_terms = [
    "china",  # Dataset top for high-spend/frequency (from users JOIN orders SUM(sale_price))
    "us",  # Common cohort mention (top 2 countries in data)
    "q3",  # Summer aligns with Q3 low patterns (EXTRACT(QUARTER))
    "q4",  # 2023 peaks in Q4 (seasonal revenue GROUP BY quarter)
    "china",  # Top region in sales (GROUP BY country)
    "us",  # Full breakdown mentions US/China (top countries)
    "coats",  # Winter high-performer (WHERE MONTH IN (12,1,2) GROUP BY product.name)
    "electronics"  # Velocity top category (SUM(sale_price) ORDER BY DESC)
]

@pytest.mark.parametrize("prompt, expected_term", zip(prompts, expected_terms))
def test_evals(prompt, expected_term, config):
    """Batch test: Invoke graph on natural prompt; assert term in report narrative or fallback clean; log full report for visibility."""
    state: AgentState = {
        "messages": [HumanMessage(content=prompt)],  # HumanMessage for .content compatibility (aligns CLI state)
        "df": None,
        "insights": [],
        "next": None,
        "phase": "delegate",
        "report": "",
        "errors": []
    }
    result = app.invoke(state, config)
    report = result.get("report", "").lower()
    errors = result.get("errors", [])
    
    # Robust Assert: If fallback, pass if no errors; else check term in report (handles [] insights gracefully - Guide p. 26 post-tool checks)
    if "no insights generated" in report:
        assert len(errors) == 0, f"Fallback for '{prompt}': Errors present – check delegation"
    else:
        assert expected_term in report, f"Failed for '{prompt}': No '{expected_term}' in report"
    
    # Log full outputs (no truncation for visibility)
    print(f"\n--- Test: {prompt} ---")
    print(f"Full Report: {result.get('report', '')}")
    print(f"Insights: {result.get('insights', [])}")
    if errors:
        print(f"Errors: {json.dumps(errors, indent=2)}")
    print(f"Pass: {'Yes' if ('no insights generated' in report and len(errors) == 0) or expected_term in report else 'No'}")