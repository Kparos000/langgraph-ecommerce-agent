from langchain_core.tools import tool
from google.cloud import bigquery
import pandas as pd
from langgraph.types import Command
from config import get_bq_client
import re  # For SQL guardrail regex

# SQL Guardrail Regex 
SQL_DML_BLOCKLIST = re.compile(r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\b', re.IGNORECASE)

def get_execute_query_tool():
    @tool
    def wrapped_execute_query(sql: str) -> str:
        """Execute SQL on BigQuery, return DF Markdown summary."""
        # Guardrail check (Operate within defined rails; halt if violated)
        if SQL_DML_BLOCKLIST.search(sql):
            return "Error: Invalid SQL detected (no DML allowed). Regenerate query."
        
        # Retry wrapper (Exponential backoff for transient errors like quota)
        for attempt in range(3):
            try:
                client = get_bq_client()
                query_job = client.query(sql)
                df = query_job.to_dataframe()
                if df.empty:
                    return "Warning: Empty results. Consider broadening query."
                return f"Shape: {df.shape}\n{df.head().to_markdown()}"
            except Exception as e:
                if attempt == 2:  # Last attempt
                    return f"Error after 3 retries: {str(e)}"
                # Reflection: Append retry message to state (but since tool, return for LLM correction)
                continue  # Retry silently
        return "Error: Max retries exceeded."  # Fallback if loop exits early
    return wrapped_execute_query

def create_handoff_tool(agent_name: str):
    @tool
    def handoff(state: dict) -> Command:
        """Hand off the current state to the specified agent for further processing."""
        return Command(goto=agent_name, update={"messages": state["messages"] + [{"role": "tool", "content": f"Handed off to {agent_name}"}]})
    handoff.__name__ = f"handoff_to_{agent_name}"
    return handoff