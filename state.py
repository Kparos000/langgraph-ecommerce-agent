from typing_extensions import TypedDict
import pandas as pd
from typing import List, Optional

class AgentState(TypedDict):
    messages: List[dict]  # List of {"role": str, "content": str} for prompt and responses
    df: Optional[pd.DataFrame]  # Query results
    insights: List[dict]  # List of {"agent": str, "text": str} for sub-agent outputs
    next: str  # Next node (e.g., "trends", "synthesis")
    report: str  # Final Markdown report