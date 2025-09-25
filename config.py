import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import bigquery
import google.auth
from google.auth.exceptions import DefaultCredentialsError
import json

# Load .env robustly
load_dotenv(find_dotenv(usecwd=True), override=True)

def tracing_status() -> dict:
    """Quick check that LangSmith tracing env vars are loaded."""
    tracing = str(os.getenv("LANGCHAIN_TRACING_V2", "")).lower() in ("1", "true", "yes")
    project = os.getenv("LANGCHAIN_PROJECT")
    has_api_key = bool(os.getenv("LANGCHAIN_API_KEY"))
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    creds_exists = bool(creds_path) and os.path.isfile(creds_path)
    return {
        "tracing": tracing,
        "project": project,
        "has_api_key": has_api_key,
        "google_application_credentials": creds_path,
        "google_application_credentials_exists": creds_exists,
    }

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env - check file and key from AI Studio.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key
    )

def get_bq_client():
    """
    Initialize BigQuery client.
    Priority:
    1) If GOOGLE_APPLICATION_CREDENTIALS points to an existing file, use it.
    2) Otherwise, clear that env var and fall back to Application Default Credentials (ADC).
    """
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # If a path is set but doesn't exist, remove it so google.auth.default() won't error out.
    if key_path and not os.path.isfile(key_path):
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        key_path = None

    if key_path:
        return bigquery.Client.from_service_account_json(key_path)

    # Fall back to ADC (gcloud auth application-default login)
    try:
        credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return bigquery.Client(credentials=credentials, project=os.getenv("GOOGLE_CLOUD_PROJECT", project))
    except DefaultCredentialsError as e:
        raise ValueError(
            "No valid credentials found. Either set GOOGLE_APPLICATION_CREDENTIALS to a valid JSON key file "
            "or run `gcloud auth application-default login`. "
            f"Detail: {e}"
        )

def get_schema(client: bigquery.Client) -> dict:
    dataset_id = "bigquery-public-data.thelook_ecommerce"
    tables = client.list_tables(dataset_id)
    schema = {}
    for table in tables:
        table_ref = client.get_table(f"{dataset_id}.{table.table_id}")
        schema[table.table_id] = [field.to_api_repr() for field in table_ref.schema]
    return schema

def get_context(client):
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
        regions = {
            "North America": ["United States"],
            "South America": ["Brasil", "Colombia"],
            "EMEA": ["Austria", "Belgium", "France", "Germany", "Poland", "Spain", "United Kingdom"],
            "Asia Pacific": ["Australia", "Japan", "South Korea"],
            "China": ["China"]
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

if __name__ == "__main__":
    try:
        print("Tracing status:", json.dumps(tracing_status(), indent=2))
        client = get_bq_client()
        dataset = "bigquery-public-data.thelook_ecommerce"
        tables = ["orders", "order_items", "products", "users"]
        print("BigQuery auth success! Fetching dataset metadata...\n")
        for table in tables:
            table_ref = f"{dataset}.{table}"
            table_obj = client.get_table(table_ref)
            row_count = table_obj.num_rows
            schema = [(field.name, field.field_type) for field in table_obj.schema]
            date_fields = [f.name for f in table_obj.schema if f.field_type in ["TIMESTAMP", "DATE"]]
            date_range = "No date fields"
            if date_fields:
                date_field = date_fields[0]
                query = f"""
                SELECT MIN({date_field}) as min_date, MAX({date_field}) as max_date
                FROM `{table_ref}`
                """
                result = client.query(query).result()
                min_date, max_date = next(result)
                date_range = f"{min_date} to {max_date}"
            print(f"Table: {table}")
            print(f"Row Count: {row_count}")
            print(f"Date Range: {date_range}")
            print("Schema:")
            for field_name, field_type in schema:
                print(f"  - {field_name}: {field_type}")
            print()
        schema = get_schema(client)
        print("Full Schema JSON:", json.dumps(schema, indent=2))
        context = get_context(client)
        print("Context JSON:", json.dumps(context, indent=2))
        print("Dataset metadata retrieved successfully.")
    except Exception as e:
        print(f"Metadata retrieval error: {e}")
        print("Troubleshoot: Verify .env paths, enable BigQuery API, check billing, confirm dataset access.")
