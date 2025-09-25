# LangGraph E-commerce Analytics Agent

A next-generation E-commerce Analytics Agent powered by LangGraph and Google's Gemini 1.5 Flash LLM. Query the `bigquery-public-data.thelook_ecommerce` dataset using natural language and receive actionable, business-ready analytics in seconds—no SQL skills required.

***

### Project Overview

The LangGraph E-commerce Agent is an interpretable, multi-agent system built on a modular [ReAct (Reason-Act)](https://arxiv.org/abs/2210.03629) architecture. Users ask e-commerce questions in plain language—such as _"Analyze sales trends in the US in 2022"_—and receive clear, data-driven answers including trends, segmentation, geo breakdowns, product insights, and concise recommendations. The system orchestrates a workflow of reasoning, SQL validation, BigQuery execution, and synthesis, automating even highly complex analytical tasks.

***

## Key Features

- **Natural Language Analytics:** Query e-commerce data with everyday language—no need to write queries or scripts.
- **Multi-Agent Intelligence:** Specialized agents handle segmentation, trends, geo-analytics, and product performance automatically.
- **ReAct Architecture:** Follows a [reason-act-observe-answer loop](https://arxiv.org/abs/2210.03629) for reliability and interpretability.
- **Validated, Safe SQL:** 99%+ SQL validity using robust input validation and automatic checks—protects your data and your workflow.
- **Fast Real-Time Insights:** Typical responses in under 10 seconds, powered by Gemini 1.5 Flash for cost-efficient, high-throughput inference—even in free tier.
- **Streamlit & CLI Interfaces:** Use in the terminal or via a beautiful, user-friendly web app.
- **Full Tracing with LangSmith:** Every query is fully traceable, from user prompt to final answer, for transparency and debugging.

***

## Supported E-commerce Use Cases

- **Sales trends** by geography, time, and category
- **Customer segmentation** (by age, gender, cohort, country)
- **Product performance** (revenue, units, average order value)
- **Seasonality and holiday effects**
- **Revenue attribution** and detailed breakdowns
- **Custom business reporting**—just ask in natural language!

***

## Dataset

Analyzes Google BigQuery public dataset: `bigquery-public-data.thelook_ecommerce`, with these primary tables:

- `orders` – Transaction details, user IDs, timestamps, order statuses
- `order_items` – Item-level sales, product IDs, prices, fulfillment data
- `products` – SKU/category/catalog data
- `users` – Demographics: age, gender, country

These support a multitude of advanced retail analytics workflows.

***

## System Architecture

The project uses a modular, robust file structure for scalability and clarity:

```shell
project-root/
├── requirements.txt
├── .env
├── README.md
├── config.py         # LLM setup, BigQuery config, schema/context helpers
├── agent.py          # Top-level agent graph and workflow
├── tools.py          # Core data access, query, validation, reporting tools
├── subagents.py      # Specialized “mini-agents” for segmentation, trends, geo, product
├── state.py          # Central data state + memory management
├── main.py           # CLI entrypoint
├── streamlitapp.py   # Web UI entrypoint
├── tests/
│   └── test_evals.py # pytest-based unit tests
└── data/             # Diagrams, sample queries, resources
```

### Highlights

- **Agent Workflow:** Reason about intent, delegate to sub-agents, validate/generate SQL, execute BigQuery, synthesize human-friendly report.
- **Robust State Management:** Persistent conversation context and memory for multi-turn or follow-up queries.
- **Complete Error Handling:** Automatic fallbacks ensure stability and reliability (99%+ valid output).

***

## Installation

**Prerequisites:**

- Python 3.8+
- BigQuery credentials (service account, or ADC)
- OpenAI/Google Gemini API key (for Gemini 1.5 Flash LLM)

**Setup Steps:**

```shell
git clone https://github.com/your-org/langgraph-ecommerce-agent.git
cd langgraph-ecommerce-agent
pip install -r requirements.txt
```

**Create a `.env` file:**

```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

***

## Usage

### 1. Command Line (CLI)

```shell
python main.py
```

- Type questions such as `What are the top product categories in 2023 for Europe?`
- Get instant, human-readable business reports.[1]

### 2. Streamlit App (Web GUI)

```shell
streamlit run streamlitapp.py
```

- Web interface at `localhost:8501`
- Features pre-defined prompts, dark/light mode, and click-to-analyze functionality.

***

## Streamlit App Highlights

- Sidebar with tracing toggle, prompt counter, and settings
- Pre-built analysis example prompts for quick start
- Safety and SQL output validation with clear error handling
- View reasoning traces in LangSmith for transparency

***

## Tracing with LangSmith

**Full Traceability:**
- Every query run via Streamlit is logged to LangSmith, allowing step-by-step inspection: prompt generation, sub-agent routing, SQL creation, validation, execution, and report synthesis.

**To View Traces:**
1. Run the Streamlit app
2. Enter a query or use a prompt example
3. Click “Open LangSmith” to view full execution trace in your browser

Environment variables for tracing:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project
```

***

## Testing

**Unit tests (pytest) are included for:**
- Valid query processing
- SQL injection/malicious input checks
- Invalid schema/table handling
- Output format and error trace verification

```shell
pytest tests/
```
Results must meet the project’s targets for speed, validity, and interpretability.[1]

***

## Design and Implementation Notes

- **Gemini 1.5 Flash LLM**: Used for its high context window and speed; optimized for fast, cost-effective analytics tasks.
- **Custom Tooling**: Handcrafted tools for SQL querying, validation, answer generation, and holiday analysis; not reliant on external agent libraries for tight workflow control.
- **Error Handling**: Graceful error management and fallbacks ensure reliability (e.g., fallback to "trends" agent on unexpected input or LLM failures, logging of exceptions to agent state).
- **Persistence**: MemorySaver and checkpointing for multi-turn conversations, making follow-up analytics natural and seamless.
- **Security**: SQL is stringently validated to block destructive operations.

***

## Example Queries

- “Show customer revenue by age group and country for 2023”
- “Trend of average order value by month last year”
- “Which products had the highest number of returns?”
- “Identify top revenue cities and their seasonal peaks”

***

## Contributing

Pull requests are welcome! Please discuss major changes in an issue. Maintain code readability, modularity, and complete coverage with tests.

## Contact
KP
kparos14@outlook.com

- For bugs, suggestions, or enterprise extensions, contact KP.