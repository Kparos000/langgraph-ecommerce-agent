import pytest
try:
    from ..agent import app  # Relative import from parent directory
except ImportError:
    from agent import app  # Fallback for running directly in root
from state import AgentState

# Fixture for shared config
@pytest.fixture
def config():
    return {"configurable": {"thread_id": "1"}}

# Test SegmentationAgent: Customer behavior/demographics
def test_segmentation(config):
    state: AgentState = {
        "messages": [("human", "Segment high-spend users by country")],
        "df": None,
        "insights": [],
        "next": None,
        "phase": "delegate",
        "report": ""
    }
    result = app.invoke(state, config)
    report = result["report"].lower()
    # Assert key insight: Expect country-based segmentation (e.g., "high-value" or specific country like "china")
    assert "high-value" in report or "china" in report, "Segmentation report missing expected insights"

# Test TrendsAgent: Sales trends/seasonality
def test_trends(config):
    state: AgentState = {
        "messages": [("human", "What were the sales trends in 2023?")],
        "df": None,
        "insights": [],
        "next": None,
        "phase": "delegate",
        "report": ""
    }
    result = app.invoke(state, config)
    report = result["report"].lower()
    # Assert key insight: Expect "Q4" or revenue figures (e.g., "peak")
    assert "q4" in report or "peak" in report, "Trends report missing Q4 or peak insights"

# Test GeoAgent: Geographic sales patterns
def test_geo(config):
    state: AgentState = {
        "messages": [("human", "What region has the most sales in 2023?")],
        "df": None,
        "insights": [],
        "next": None,
        "phase": "delegate",
        "report": ""
    }
    result = app.invoke(state, config)
    report = result["report"].lower()
    # Assert key insight: Expect "china" and "$611,205" from prior test
    assert "china" in report and "$611,205" in report, "Geo report missing China or $611,205"

# Test ProductAgent: Product performance/recommendations
def test_product(config):
    state: AgentState = {
        "messages": [("human", "What products performed best in winter of 2023?")],
        "df": None,
        "insights": [],
        "next": None,
        "phase": "delegate",
        "report": ""
    }
    result = app.invoke(state, config)
    report = result["report"].lower()
    # Assert key insight: Expect winter-specific terms (e.g., "coats", "winter", or revenue)
    assert any(term in report for term in ["coats", "winter", "$"]), "Product report missing winter or product insights"