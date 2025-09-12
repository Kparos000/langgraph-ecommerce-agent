from langchain_core.tools import tool
from google.cloud import bigquery
import pandas as pd
from typing import Dict

@tool
def execute_query(sql: str, client: bigquery.Client) -> str:
    """Execute SQL on BigQuery and return Markdown table."""
    try:
        # Guardrail: Block non-SELECT queries
        if not sql.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries allowed."
        # Ensure dataset prefix
        if "bigquery-public-data.thelook_ecommerce" not in sql:
            sql = sql.replace("FROM `", "FROM `bigquery-public-data.thelook_ecommerce.")
        df = client.query(sql).to_dataframe()
        if df.empty:
            return "No results returned."
        return df.head(1000).to_markdown(index=False)  # Per PRD: LIMIT 1000
    except Exception as e:
        return f"Query error: {str(e)}"

def create_handoff_tool(agent_name: str):
    """Create handoff tool for routing to sub-agent."""
    @tool
    def handoff(state: Dict) -> Dict:
        """Route to specified sub-agent."""
        return {"goto": f"{agent_name}_node", "update": {"messages": state["messages"] + [{"role": "tool", "content": f"Handoff to {agent_name}"}]}}
    handoff.__name__ = f"handoff_to_{agent_name}"
    return handoff

# Instantiate handoff tools for each sub-agent
handoff_segmentation = create_handoff_tool("segmentation")
handoff_trends = create_handoff_tool("trends")
handoff_geo = create_handoff_tool("geo")
handoff_product = create_handoff_tool("product")