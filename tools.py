import re
import json
import time
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
    """Validate SQL query for structure, schema compliance, and safety."""
    # Must start with SELECT
    if not re.match(r"^\s*SELECT", sql, re.IGNORECASE):
        return "Invalid: Must start with SELECT."

    # Must include correct dataset
    if "bigquery-public-data.thelook_ecommerce" not in sql:
        return "Invalid: Query must reference bigquery-public-data.thelook_ecommerce tables."

    # Disallow dangerous commands
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER"]
    if any(word in sql.upper() for word in forbidden):
        return "Invalid: Query contains forbidden operation."

    # Basic JOIN sanity
    if "JOIN" in sql.upper() and "ON" not in sql.upper():
        return "Invalid: JOIN missing ON clause."

    return "Valid"

@tool
def generate_final_answer(answer: str) -> str:
    """Generate the final answer (placeholder for synthesis)."""
    return answer
