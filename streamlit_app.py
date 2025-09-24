# streamlit_app.py
import os
import json
import time
from pathlib import Path
import streamlit as st

from langchain_core.messages import HumanMessage
from config import get_bq_client, get_schema, get_context, get_llm
from agent import compiled_graph  # uses your existing compiled graph

# ---------- Helpers ----------
COUNTER_FILE = Path("query_counter.txt")

def _load_counter() -> int:
    try:
        return int(COUNTER_FILE.read_text().strip())
    except Exception:
        return 0

def _save_counter(val: int) -> None:
    try:
        COUNTER_FILE.write_text(str(val))
    except Exception:
        pass

def _increment_counter() -> int:
    val = _load_counter() + 1
    _save_counter(val)
    return val

def tracing_status_dict():
    return {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"),
        "LANGCHAIN_API_KEY_set": bool(os.getenv("LANGCHAIN_API_KEY")),
    }

@st.cache_resource(show_spinner=False)
def _bootstrap_context():
    # Create clients and cache schema/context once per app session
    client = get_bq_client()
    schema = json.dumps(get_schema(client))
    context = get_context(client)
    return client, schema, context

def _run_graph(user_query: str, schema: str, context: dict):
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "remaining_steps": "",
        "memory": "",
        "schema": schema,
        "context": context
    }
    config = {"configurable": {"thread_id": "streamlit"}}
    return compiled_graph.invoke(initial_state, config=config)

# ---------- Page Config & Simple Theme Toggle ----------
st.set_page_config(
    page_title="E-commerce Analytics Agent (LangGraph + BigQuery)",
    page_icon="📈",
    layout="wide",
)

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

col_theme, col_trace = st.columns([1, 2])
with col_theme:
    mode = st.toggle("Light / Dark", value=(st.session_state["theme_mode"] == "Dark"))
    st.session_state["theme_mode"] = "Dark" if mode else "Light"

# Simple CSS to soften UI based on toggle (Streamlit doesn't support runtime theme swap)
if st.session_state["theme_mode"] == "Dark":
    st.markdown(
        """
        <style>
            .main { background-color: #0e1117; color: #e0e0e0; }
            .stButton>button { background-color: #1f6feb !important; color: white !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- Header & Intro ----------
st.title("E-commerce Analytics Agent (LangGraph + BigQuery)")
st.caption(
    "This app lets you ask natural-language questions about the "
    "`bigquery-public-data.thelook_ecommerce` dataset. We use the `orders`, `order_items`, "
    "`products`, and `users` tables. The agent (LangGraph + ReAct) plans, validates SQL, queries BigQuery, "
    "and synthesizes a business-ready report. Click a prompt example or enter your own."
)

# LangSmith tracing status (inline)
with col_trace:
    ts = tracing_status_dict()
    ok = ts["LANGCHAIN_TRACING_V2"] in ("true", "True", "1") and ts["LANGCHAIN_API_KEY_set"]
    st.write(
        f"**Tracing enabled:** {'✅' if ok else '❌'} "
        f"(Project: `{os.getenv('LANGCHAIN_PROJECT','-')}`)"
    )
    if ok:
        st.caption("Traces stream to your LangSmith project in real time. Open LangSmith to inspect runs.")

st.divider()

# ---------- Load BQ schema/context once ----------
with st.spinner("Initializing BigQuery context…"):
    client, schema_json, context_dict = _bootstrap_context()

# ---------- Prompt Examples ----------
st.subheader("Prompt Examples")
example_queries = [
    "Analyze sales trends in US in 2022",
    "Analyze sales in China in 2022. Highlight top selling categories and key cities.",
    "Identify top 5 product categories by revenue in 2023",
    "Analyze sales in Australia in 2021 under $50",
    "Give me the top cities by revenue in Japan in 2022",
]

ex_cols = st.columns(len(example_queries))
example_clicked = None
for i, q in enumerate(example_queries):
    # Show the beginning of the prompt on the button (no hover)
    label = q if len(q) <= 38 else (q[:35] + "…")
    if ex_cols[i].button(label, key=f"ex_{i}", help=q):
        example_clicked = q  # trigger auto-run

st.divider()

# ---------- Query Input & Run ----------
# Header adjustment per your request: remove "Your Question"; keep just the input
query = st.text_area(
    "Ask me anything about the dataset",
    placeholder="e.g., Analyze sales trends in US in 2022",
    height=100,
)

# We use a single primary action
run_clicked = st.button("Ask me", type="primary")

# If an example was clicked, auto-run immediately (no second click)
effective_query = example_clicked or (query.strip() if run_clicked else "")

# ---------- Results Area ----------
result_container = st.container()
if effective_query:
    st.info("The agents are working on your prompt…")
    try:
        t0 = time.time()
        # Run the graph
        result = _run_graph(effective_query, schema_json, context_dict)
        elapsed = time.time() - t0

        # Increment global counter AFTER a successful run
        total = _increment_counter()

        # Display output
        with result_container:
            st.success(f"Completed in {elapsed:.2f}s · Total queries: {total}")
            st.markdown("**Final Report:**")
            st.markdown(result["messages"][-1].content)

            # Small expandable debug (optional)
            with st.expander("See sample tool outputs (debug)"):
                # Try to show a few tool messages if present
                tool_snips = []
                for m in result["messages"]:
                    if hasattr(m, "tool_call_id") and m.tool_call_id:
                        tid = m.tool_call_id
                        if tid.endswith("_sql") or tid.endswith("_query"):
                            tool_snips.append(f"**{tid}**\n\n```\n{m.content}\n```")
                if tool_snips:
                    st.markdown("\n\n".join(tool_snips))
                else:
                    st.write("No tool snippets available for this run.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Footer line with lightweight guidance
st.caption(
    "Tip: For country-specific queries, the agent automatically joins `users.country` and filters by year "
    "(using `orders.created_at`)."
)
