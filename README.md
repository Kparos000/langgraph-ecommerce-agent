# Data Analysis LangGraph Agent 
 
Custom AI agent for e-commerce insights on BigQuery `bigquery-public-data.thelook_ecommerce`. 
 
## Setup 
1. `pip install -r requirements.txt` 
2. Copy `.env.example` to `.env`; add keys (Gemini from AI Studio; BigQuery service account via gcloud). 
3. `python -c "from config import get_bq_client; get_bq_client^().query^('SELECT 1 FROM \`bigquery-public-data.thelook_ecommerce.orders\` LIMIT 1^').result^()"` (test auth). 
 
## Run 
`python main.py` 
 
## Samples 
- "winter recs"  Seasonal product insights. 
- ... (TBD) 
 
## Architecture 
[Mermaid diagram TBD] 
 
## Evals 
`pytest tests/` 
