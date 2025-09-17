from typing_extensions import TypedDict
import pandas as pd
from typing import List, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # Chat history
    df: pd.DataFrame  # Query results
    insights: List[dict]  # e.g., [{"agent": "Trends", "text": "Q3 peak"}]
    next: str  # Router key (e.g., "trends")
    phase: str  # "delegate" or "done" to control loop
    report: str  # Final Markdown
    errors: List[dict]  # [{"type": "fallback", "agent": "Trends", "error": "str(e)", "retries": 3}] – Track for evals/synthesis