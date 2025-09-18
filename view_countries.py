# view_countries.py
import logging
from config import get_bq_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Get BigQuery client
    client = get_bq_client()
    logger.info("BigQuery client initialized successfully")

    # Query to get distinct countries with LIMIT 1000 as per PRD
    query = """
    SELECT DISTINCT country
    FROM `bigquery-public-data.thelook_ecommerce.users`
    ORDER BY country ASC
    LIMIT 1000
    """
    logger.info("Executing query: %s", query)
    query_job = client.query(query)
    results = query_job.result()  # Wait for the query to complete

    # Extract and print countries
    countries = [row.country for row in results]
    logger.info("Retrieved %d unique countries", len(countries))
    print("Unique Countries in the dataset:")
    for country in countries:
        print(country)

except Exception as e:
    logger.error("Error querying countries: %s", str(e))
    print(f"Error: {e}")