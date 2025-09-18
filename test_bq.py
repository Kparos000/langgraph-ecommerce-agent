# Save as test_bq.py and run `python test_bq.py` from C:\Users\kparo\langgraph-ecommerce-agent>
from config import get_bq_client
try:
    client = get_bq_client()
    query_job = client.query("SELECT 1 FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 1")
    results = list(query_job.result())
    print("Success! Results:", results)
except Exception as e:
    print("Error:", e)