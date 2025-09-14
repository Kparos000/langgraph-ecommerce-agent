# Data Analysis LangGraph Agent

Custom AI agent for e-commerce insights on BigQuery `bigquery-public-data.thelook_ecommerce`.

## Setup
1. `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env`; add keys (Gemini from AI Studio; BigQuery service account via gcloud).
3. Test auth: `python -c "from config import get_bq_client; get_bq_client().query('SELECT 1 FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 1').result()"`.

## Run
`python main.py`

## Samples
- Segmentation: "Segment high-spend users by country" → Insights like "High-value: >$500, 20% users in US".
- Trends: "What were the sales trends in 2023?" → Insights like "Q4: $3.55M (highest, 27% > Q2)".
- Geo: "What region has the most sales in 2023?" → Insights like "China: $611,205 (34%)".
- Product: "What products performed best in winter of 2023?" → Insights like "Coats: $300k sales in Q4".

## Architecture
Mermaid diagram (copy to mermaid.live for PNG export as `diagram.png`):


**Explanation of Choices**:
- **LLM**: Gemini 1.5 Flash for fast SQL gen and synthesis.
- **Data Handling**: BigQuery client with test query on startup; Pandas for df.to_markdown() summaries.
- **Errors**: LLM retry max 3 (Section 3.2); fallback in invoke_sub_agent.
- **Flow**: Manager classifies/handoffs (OpenAI Guide Pages 17-19); subs ReAct loop (single tool call); synthesis aggregates.

## Evals
`pytest tests/` (validates 4 prompts, 80% insight match).

## Flow Summary
Prompt → Manager (classify) → Sub(s) (SQL → Tool → Insights) → Synthesis (report) → CLI/file. Total calls: 4-6 LLM, 1-2 queries.