from langchain_core.tools import tool
from google.cloud import bigquery
import pandas as pd
from langgraph.types import Command
from config import get_bq_client
import re  # For SQL guardrail regex
import logging  # New: For tool traces

logger = logging.getLogger(__name__)  # New: Logger instance

# SQL Guardrail Regex 
SQL_DML_BLOCKLIST = re.compile(r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\b', re.IGNORECASE)

def get_execute_query_tool():
    @tool
    def wrapped_execute_query(sql: str) -> str:
        """Execute SQL on BigQuery, return DF Markdown summary."""
        # New: Log SQL pre-execution
        logger.info(f"Executing SQL (attempt): {sql[:100]}...")
        
        # Guardrail check (Operate within defined rails; halt if violated)
        if SQL_DML_BLOCKLIST.search(sql):
            logger.warning("Invalid SQL detected (DML blocklisted).")  # New: Log violation
            return "Error: Invalid SQL detected (no DML allowed). Regenerate query."
        
        # Retry wrapper (Exponential backoff for transient errors like quota)
        for attempt in range(3):
            try:
                client = get_bq_client()
                query_job = client.query(sql)
                df = query_job.to_dataframe()
                if df.empty:
                    logger.warning("Empty results from query.")  # New: Log empty
                    return "Warning: Empty results. Consider broadening query."
                # New: Log success
                logger.info(f"Query success: df.shape={df.shape}")
                return f"Shape: {df.shape}\n{df.head().to_markdown()}"
            except Exception as e:
                logger.warning(f"Query error (attempt {attempt+1}): {str(e)}")  # New: Log retry/error
                if attempt == 2:  # Last attempt
                    return f"Error after 3 retries: {str(e)}"
                # Reflection: Append retry message to state (but since tool, return for LLM correction)
                continue  # Retry silently
        logger.error("Max retries exceeded for query.")  # New: Log final failure
        return "Error: Max retries exceeded."  # Fallback if loop exits early
    return wrapped_execute_query

def create_handoff_tool(agent_name: str):
    @tool
    def handoff(state: dict) -> Command:
        """Hand off the current state to the specified agent for further processing."""
        logger.info(f"Handing off to {agent_name}")  # New: Log handoff
        return Command(goto=agent_name, update={"messages": state["messages"] + [{"role": "tool", "content": f"Handed off to {agent_name}"}]})
    handoff.__name__ = f"handoff_to_{agent_name}"
    return handoff