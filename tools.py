from langchain_core.tools import tool
from google.cloud import bigquery
import pandas as pd
from langgraph.types import Command
from config import get_bq_client

def get_execute_query_tool():
    @tool
    def wrapped_execute_query(sql: str) -> str:
        """Execute SQL on BigQuery, return DF Markdown summary."""
        try:
            client = get_bq_client()
            query_job = client.query(sql)
            df = query_job.to_dataframe()
            return f"Shape: {df.shape}\n{df.head().to_markdown()}"
        except Exception as e:
            return f"Error: {str(e)}"
    return wrapped_execute_query

def create_handoff_tool(agent_name: str):
    @tool
    def handoff(state: dict) -> Command:
        """Hand off the current state to the specified agent for further processing."""
        return Command(goto=agent_name, update={"messages": state["messages"] + [{"role": "tool", "content": f"Handed off to {agent_name}"}]})
    handoff.__name__ = f"handoff_to_{agent_name}"
    return handoff