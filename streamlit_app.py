import json
import uuid
from typing import List, Optional

import streamlit as st

from langchain_core.messages import BaseMessage, ToolMessage
from agent import compiled_graph  # uses your current agent.py graph
from config import get_bq_client, get_schema, get_context, tracing_status
from langchain_core.messages import HumanMessage

# ---------- Page / Session Setup ----------
st.set_page_config(page_title="E-commerce LangGraph Agent", page_icon="🛒", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {query, report, delegated, sample_rows, executed_sql}

# ---------- Helpers ----------
def _first_tool_rows(messages: List[BaseMessage]) -> Optional[List[dict]]:
    """
    Try to find the first tool result (JSON array of rows) to preview.
    """
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id and m.tool_call_id.endswith("_query"):
            try:
                data = json.loads(m.content)
                if isinstance(data, list):
                    return data[:5]
            except Exception:
                continue
    return None

def _last_executed_sql(messages: List[BaseMessage]) -> Optional[str]:
    """
    If the sub-agent appended executed SQL (via manual_*_sql ToolMessage), show it.
    """
    last_sql = None
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id and m.tool_call_id.endswith("_sql"):
            last_sql = m.content
    return last_sql

def _delegated_to(memory: str) -> Optional[str]:
    """
    We append manager reasoning to state['memory'] in agent.py.
    Try to extract the last chosen sub-agent from that text.
    """
    if not memory:
        return None
    # naive scan; your manager writes 'Final Answer: {"sub_agent": "..."}'
    lower = memory.lower()
    marker = 'final answer:'
    if marker in lower:
        try:
            frag = memory[lower.rfind(marker) + len(marker):].strip()
            j = json.loads(frag)
            return j.get("sub_agent")
        except Exception:
            return None
    return None

def run_agent_once(user_query: str):
    """
    Execute the compiled graph for a single query; return structured outputs for the UI.
    """
    # Build schema/context once per process; memoize via st.cache_data to save latency
    @st.cache_data(show_spinner=False)
    def _bootstrap():
        client = get_bq_client()
        schema = json.dumps(get_schema(client))
        context = get_context(client)
        return schema, context

    schema, context = _bootstrap()

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "remaining_steps": "",
        "memory": "",
        "schema": schema,
        "context": context,
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    result = compiled_graph.invoke(initial_state, config=config)

    # Pull final report (LLM response at the end)
    final_report = result["messages"][-1].content if result.get("messages") else ""

    # Try to detect which sub-agent was delegated (from memory trail)
    delegated = _delegated_to(result.get("memory", "")) or result.get("remaining_steps", "")

    # Sample rows + last executed SQL (if present)
    sample_rows = _first_tool_rows(result.get("messages", [])) or []
    executed_sql = _last_executed_sql(result.get("messages", []))

    return {
        "report": final_report,
        "delegated": delegated,
        "sample_rows": sample_rows,
        "executed_sql": executed_sql,
        "raw_state": result,
    }


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings & Status")

# LangSmith status
try:
    tstatus = tracing_status()
    st.sidebar.markdown("**LangSmith Tracing**")
    st.sidebar.write(
        f"- Tracing enabled: **{tstatus['tracing']}**\n"
        f"- Project: **{tstatus['project']}**\n"
        f"- API key set: **{tstatus['has_api_key']}**"
    )
except Exception:
    st.sidebar.warning("LangSmith tracing status unavailable.")

# Query counter + thread id
st.sidebar.markdown("---")
st.sidebar.metric("Queries this session", st.session_state.query_count)
st.sidebar.caption(f"Thread: `{st.session_state.thread_id}`")

# Prompt templates
st.sidebar.markdown("---")
st.sidebar.subheader("Prompt templates")
TEMPLATES = [
    "Analyze sales trends in US in 2022",
    "Analyze sales in China in 2024. Highlight top selling products under $50 and explain what drives sales.",
    "Identify top 10 product categories by revenue in 2023",
    "Analyze geographic performance in EMEA in 2022",
    "Give me the top 10 cities by revenue in United States for 2022",
]
for i, tpl in enumerate(TEMPLATES, start=1):
    if st.sidebar.button(f"Template {i}", use_container_width=True):
        st.session_state["prefill"] = tpl

# ---------- Main UI ----------
st.title("🛍️ E-commerce LangGraph ReAct Agent")

prefill = st.session_state.get("prefill", "")
query = st.text_area("Enter your question", value=prefill, height=90, placeholder="e.g., Analyze sales trends in US in 2022")
col_run, col_clear = st.columns([1, 1], vertical_alignment="center")

with col_run:
    run_clicked = st.button("Run", type="primary")
with col_clear:
    if st.button("Clear"):
        st.session_state.history.clear()
        st.session_state.query_count = 0
        st.session_state.prefill = ""

if run_clicked:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Running agent…"):
            try:
                result = run_agent_once(query.strip())
                st.session_state.history.insert(0, {
                    "query": query.strip(),
                    "report": result["report"],
                    "delegated": result["delegated"],
                    "sample_rows": result["sample_rows"],
                    "executed_sql": result["executed_sql"],
                })
                st.session_state.query_count += 1
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- Results ----------
if st.session_state.history:
    latest = st.session_state.history[0]
    st.subheader("Answer")
    st.markdown(latest["report"])

    meta_cols = st.columns(2)
    with meta_cols[0]:
        st.caption(f"Delegated sub-agent: **{latest.get('delegated') or 'n/a'}**")
    with meta_cols[1]:
        st.caption(f"Queries this session: **{st.session_state.query_count}**")

    with st.expander("Preview: sample rows (first tool response)"):
        if latest["sample_rows"]:
            st.code(json.dumps(latest["sample_rows"], indent=2))
        else:
            st.write("No sample rows detected.")

    with st.expander("Preview: last executed SQL"):
        if latest["executed_sql"]:
            st.code(latest["executed_sql"], language="sql")
        else:
            st.write("No executed SQL detected.")

    # History below
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.subheader("History")
        for idx, h in enumerate(st.session_state.history[1:], start=1):
            with st.expander(f"{idx}. {h['query'][:80]}"):
                st.markdown(h["report"])
                st.caption(f"Delegated: **{h.get('delegated') or 'n/a'}**")
                if h["sample_rows"]:
                    st.markdown("**Sample rows:**")
                    st.code(json.dumps(h["sample_rows"], indent=2))
                if h["executed_sql"]:
                    st.markdown("**Executed SQL:**")
                    st.code(h["executed_sql"], language="sql")
else:
    st.info("Enter a question or click a template to get started.")
