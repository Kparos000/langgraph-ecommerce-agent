import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import bigquery

# Load .env variables (GEMINI_API_KEY and GOOGLE_APPLICATION_CREDENTIALS)
load_dotenv()

def get_llm():
    """Initialize Gemini 1.5 Flash LLM for SQL gen and synthesis."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env - check file and key from AI Studio.")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

def get_bq_client():
    """Initialize BigQuery client using service account JSON."""
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env - check path to JSON key.")
    return bigquery.Client.from_service_account_json(key_path)

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

        print("Dataset metadata retrieved successfully.")
    except Exception as e:
        print(f"Metadata retrieval error: {e}")
        print("Troubleshoot: Verify .env paths, enable BigQuery API, check billing, confirm dataset access.")