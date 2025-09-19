import re
import json
from langchain_core.tools import tool
from config import get_bq_client

@tool
def query_database(sql: str) -> str:
    """Execute SQL query on BigQuery and return results as JSON."""
    client = get_bq_client()
    try:
        df = client.query(sql).to_dataframe()
        return df.to_json(orient="records")
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def validator(sql: str) -> str:
    """Validate SQL query for basic structure and schema compliance."""
    if not re.match(r"^\s*SELECT", sql, re.IGNORECASE):
        return "Invalid: Must start with SELECT."
    # Basic table check (expand with schema later if needed)
    if "JOIN" in sql.upper() and "ON" not in sql.upper():
        return "Invalid: JOIN missing ON clause."
    return "Valid"

@tool
def generate_final_answer(answer: str) -> str:
    """Generate the final answer (placeholder for synthesis)."""
    return answer