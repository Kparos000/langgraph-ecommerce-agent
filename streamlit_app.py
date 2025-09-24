import os
import json
from pathlib import Path
from typing import Optional

import streamlit as st
from langchain_core.messages import HumanMessage
from config import get_bq_client, get_schema, get_context
from agent import compiled_graph

# ---------- Page & basic styles ----------
st.set_page_config(
    page_title="TheLook E-commerce Analyst",
    page_icon="📊",
    layout="wide"
)

# Simple dark/light toggle using CSS (Streamlit does not hot-swap themes at runtime)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def apply_theme(dark: bool):
    if dark:
        st.markdown(
            """
            <style>
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #0E1117 !important;
                color: #FAFAFA !important;
            }
            .stTextInput textarea, .stTextArea textarea, .stTextInput input, .stTextArea, .stButton>button {
                color: #FAFAFA !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<style>/* reset custom dark css if any */</style>",
            unsafe_allow_html=True
        )

apply_theme(st.session_state.dark_mode)

# ---------- Utilities ----------
QUERY_COUNT_PATH = Path("query_count.json")

def get_query_count() -> int:
    if QUERY_COUNT_PATH.exists():
        try:
            return int(json.loads(QUERY_COUNT_PATH.read_text()).get("count", 0))
        except Exception:
            return 0
    return 0

def increment_query_count() -> int:
    count = get_query_count() + 1
    QUERY_COUNT_PATH.write_text(json.dumps({"count": count}))
    return count

@st.cache_resource(show_spinner=False)
def bootstrap_context():
    # One-time BigQuery context bootstrap
    client = get_bq_client()
    schema = json.dumps(get_schema(client))
    context = get_context(client)
    return client, schema, context

def run_agent(user_query: str, schema: str, context: dict):
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "remaining_steps": "",
        "memory": "",
        "schema": schema,
        "context": context
    }
    config = {"configurable": {"thread_id": "ui"}}
    result = compiled_graph.invoke(initial_state, config=config)
    return result

def tracing_enabled_text() -> str:
    enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "")
    # We can’t know the specific run URL here; link to the project home.
    link = "https://smith.langchain.com/"  # users can navigate by project name
    if enabled:
        return f"**Tracing enabled:** True · Project: `{project}` · [Open LangSmith]({link})"
    return "**Tracing enabled:** False"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.toggle("Dark mode", value=st.session_state.dark_mode, key="dark_mode", on_change=lambda: apply_theme(st.session_state.dark_mode))

    st.markdown("### 🔎 Tracing")
    st.caption(tracing_enabled_text())

    st.markdown("---")
    st.markdown("### 📈 Total Queries")
    total = get_query_count()
    query_counter_placeholder = st.metric("All-time executions", total)

# ---------- Header ----------
st.markdown("## Ask me anything about **bigquery-public-data.thelook_ecommerce** dataset")

st.caption(
    "This app lets you query the public **TheLook e-commerce** dataset on BigQuery using natural language. "
    "We focus on the `orders`, `order_items`, `products`, and `users` tables. "
    "Type a question (e.g., *Analyze sales trends in US in 2022*), or click one of the prompt examples below."
)

# ---------- Prompt Examples (visible, one-click runs) ----------
st.markdown("#### Prompt Examples")

examples = [
    "Analyze sales trends in US in 2022",
    "Analyze sales in China in 2022. Highlight top categories under $50.",
    "Identify top 5 product categories by revenue in 2023",
    "Give me the top 10 cities by revenue in 2022 in the United States",
    "Analyze sales trends in Japan in 2021 and recommend two actions"
]

cols = st.columns(5)
example_clicked: Optional[str] = None
for i, (col, text) in enumerate(zip(cols, examples), start=1):
    with col:
        # Full text on the button (no hover)
        if st.button(text, key=f"example_btn_{i}", use_container_width=True):
            example_clicked = text

# ---------- Main input + run ----------
st.markdown("#### Your question")

default_q = example_clicked or st.session_state.get("last_query", "")
query = st.text_area(
    "Enter your query",
    value=default_q,
    key="query_input",
    height=90,
    label_visibility="collapsed",
    placeholder="e.g., Analyze sales trends in US in 2022"
)

left, right = st.columns([1, 5])
with left:
    # Blue primary action
    ask_btn = st.button("Ask me", type="primary", use_container_width=True)

# If user clicked an example, execute immediately (no extra click needed)
trigger_run = example_clicked is not None or ask_btn

# ---------- Run + Output ----------
if trigger_run:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        # Remember last query
        st.session_state["last_query"] = query.strip()

        # Show a friendly spinner message
        with st.spinner("The agents are working on your prompt…"):
            client, schema, context = bootstrap_context()
            try:
                result = run_agent(query.strip(), schema, context)

                # Increment global counter
                new_total = increment_query_count()
                query_counter_placeholder.metric("All-time executions", new_total)

                # Print a small trace of what was executed (debug info)
                # Safety: only show last 5 tool messages
                from langchain_core.messages import ToolMessage
                tool_snips = []
                for m in result.get("messages", [])[-15:]:
                    if isinstance(m, ToolMessage):
                        tid = m.tool_call_id or ""
                        if any(tid.endswith(sfx) for sfx in ["_sql", "_query"]) or tid in ["manual_metrics", "manual_summary"]:
                            # Keep brief
                            content_preview = str(m.content)
                            if isinstance(content_preview, str) and len(content_preview) > 500:
                                content_preview = content_preview[:500] + "…"
                            tool_snips.append(f"- `{tid}`")

                # Separator
                st.markdown("---")
                st.markdown("#### Final Report")
                st.markdown(result["messages"][-1].content)

                # Optional debug trace (collapsed)
                with st.expander("Show recent tool trace (debug)"):
                    if tool_snips:
                        st.markdown("\n".join(tool_snips))
                    else:
                        st.markdown("_No recent tool activity recorded._")

            except Exception as e:
                st.error(f"Execution error: {e}")
