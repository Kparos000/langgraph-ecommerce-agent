import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import bigquery
import json

# Load .env variables (GEMINI_API_KEY and GOOGLE_APPLICATION_CREDENTIALS)
load_dotenv()

def get_llm():
    """Initialize Gemini 1.5 Flash LLM for SQL gen and synthesis (text mode for React; enforce JSON via prompts - Handbook Ch. 4)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env - check file and key from AI Studio.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key
        # Removed: response_mime_type="application/json" (conflicts with React text prompts; Guide p. 9: Align config with instructions)
    )

def get_bq_client():
    """Initialize BigQuery client using service account JSON."""
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env - check path to JSON key.")
    return bigquery.Client.from_service_account_json(key_path)

def get_context(client):
    """Fetch dynamic context: date range, countries (from users table)."""
    try:
        # Date range from order_items
        date_query = """
        SELECT MIN(created_at) as min_date, MAX(created_at) as max_date
        FROM `bigquery-public-data.thelook_ecommerce.order_items`
        """
        date_result = client.query(date_query).result()
        min_date, max_date = next(date_result)
        date_span = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

        # Countries from users
        countries_query = """
        SELECT DISTINCT country
        FROM `bigquery-public-data.thelook_ecommerce.users`
        WHERE country IS NOT NULL
        ORDER BY country
        """
        countries_result = client.query(countries_query).result()
        countries = [row.country for row in countries_result]

        # Seasons/age groups (static but modular)
        seasons = {
            "Spring": "Mar-May (Q1 partial)",
            "Summer": "Jun-Aug (Q2)",
            "Autumn": "Sep-Nov (Q3)",
            "Winter": "Dec-Feb (Q4 partial)"
        }
        age_groups = {
            "<18": "Children and teens (0–17 years)",
            "18–24": "Young adults (including late teens and college-aged)",
            "25–34": "Millennials and young professionals",
            "35–54": "Gen X and older Millennials (mature adults)",
            ">54": "Seniors and retirees"
        }
        # Regions: Filtered to dataset countries only (no hallucinations; map based on standard geo groups)
        regions = {
            "North America": ["United States"],
            "South America": ["Brasil", "Colombia"],
            "EMEA": ["Austria", "Belgium", "France", "Germany", "Poland", "Spain", "United Kingdom"],
            "Asia Pacific": ["Australia", "Japan", "South Korea"],
            "China": ["China"]  # Tracked separately per PRD
        }

        return {
            "date_span": date_span,
            "countries": countries,
            "seasons": seasons,
            "age_groups": age_groups,
            "regions": regions
        }
    except Exception as e:
        raise ValueError(f"Context fetch error: {e}")

# Startup test: Validate auth and print metadata (row counts, date ranges, schema)
if __name__ == "__main__":
    try:
        client = get_bq_client()
        dataset = "bigquery-public-data.thelook_ecommerce"
        tables = ["orders", "order_items", "products", "users"]
        print("BigQuery auth success! Fetching dataset metadata...\n")
        for table in tables:
            table_ref = f"{dataset}.{table}"
            # Get table metadata
            table_obj = client.get_table(table_ref)
            
            # Row count
            row_count = table_obj.num_rows
            
            # Schema (field names and types)
            schema = [(field.name, field.field_type) for field in table_obj.schema]
            
            # Date range (if applicable; check for timestamp fields)
            date_fields = [f.name for f in table_obj.schema if f.field_type in ["TIMESTAMP", "DATE"]]
            date_range = "No date fields"
            if date_fields:
                # Assume first date/timestamp field for range (common in e-commerce)
                date_field = date_fields[0]
                query = f"""
                SELECT MIN({date_field}) as min_date, MAX({date_field}) as max_date
                FROM `{table_ref}`
                """
                result = client.query(query).result()
                min_date, max_date = next(result)
                date_range = f"{min_date} to {max_date}"
            # Print formatted metadata
            print(f"Table: {table}")
            print(f"Row Count: {row_count}")
            print(f"Date Range: {date_range}")
            print("Schema:")
            for field_name, field_type in schema:
                print(f"  - {field_name}: {field_type}")
            print()
        # Test context
        context = get_context(client)
        print("Context JSON:", json.dumps(context, indent=2))
        print("Dataset metadata retrieved successfully.")
    except Exception as e:
        print(f"Metadata retrieval error: {e}")
        print("Troubleshoot: Verify .env paths, enable BigQuery API, check billing, confirm dataset access.")