from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from config import get_llm
from tools import query_database, validator, generate_final_answer
from typing import Dict, List, Annotated
from operator import add
from state import AgentState
from langgraph.checkpoint.memory import MemorySaver
import re, json

class SubAgentState(Dict):
    messages: Annotated[List[BaseMessage], add]
    memory: str
    schema: str
    context: Dict

tools = [query_database, validator, generate_final_answer]

def get_sub_agent_graph(role: str, specialty: str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a {role} Agent specializing in {specialty}.
You MUST use the available tools to answer queries. Do not ask the user for more information.
The only dataset is `bigquery-public-data.thelook_ecommerce`.

Tables available with schema:
{{schema}}

Context (date ranges, countries, seasons, age groups, regions):
{{context}}

Rules:
- ALWAYS call `validator` then `query_database` before `generate_final_answer`.
- If the query includes a country or region, you MUST join `users` and filter using users.country.
  Example: JOIN `bigquery-public-data.thelook_ecommerce.users` u ON o.user_id = u.id AND u.country='United States'.
- For time-based queries, always use created_at (YEAR, MONTH, QUARTER) for filtering.
- Never output plain text without tool calls.
- If query is invalid (e.g., out-of-range year or missing country), output Final Answer explaining why.

Output format (ReAct):
Thought: reasoning
Action: tool_name
Action Input: tool_input
Observation: tool_output
... repeat until done
Final Answer: report"""),
        ("human", "Question: {input}\n\nMemory: {memory}")
    ])

    bound_llm = llm.bind_tools(tools)
    chain = prompt | bound_llm

    def agent_node(state: SubAgentState):
        question = state["messages"][-1].content
        args = {
            "input": question,
            "memory": state["memory"],
            "schema": state["schema"],
            "context": state["context"],
        }
        first = chain.invoke(args)
        outputs = [first]

        no_tools = not getattr(first, "tool_calls", None)
        no_final = "final answer:" not in (first.content or "").lower()
        if no_tools and no_final:
            followup = (
                f"{question}\n\n"
                "Follow-up: Use tools now. First call validator with a valid BigQuery SQL built from the schema; "
                "then call query_database with that SQL; only then produce Final Answer. "
                "Do not ask me for SQL."
            )
            second = chain.invoke({
                "input": followup,
                "memory": state["memory"],
                "schema": state["schema"],
                "context": state["context"],
            })
            outputs.append(second)

        return {"messages": outputs}

    def tool_node(state: SubAgentState):
        messages = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_func = next(t for t in tools if t.name == tool_call["name"])
            output = tool_func.invoke(tool_call["args"])
            messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
        return {"messages": messages}

    def should_continue(state: SubAgentState):
        if state["messages"][-1].tool_calls:
            return "tool"
        return END

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.add_edge("__start__", "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool", "agent")
    return graph.compile(checkpointer=MemorySaver())

segmentation_graph = get_sub_agent_graph("Segmentation", "customer segments by demographics/RFM")

def segmentation_node(state: AgentState):
    sub_state = {
        "messages": state["messages"],
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    }
    result = segmentation_graph.invoke(sub_state)
    state["messages"] = result["messages"]
    state["needs_retry"] = False
    return state

trends_graph = get_sub_agent_graph("Trends", "sales trends/seasonality/growth")

def _extract_year_and_country(question: str, context: Dict) -> Dict:
    text = question.lower()
    year = None
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        year = int(m.group(1))

    country = None
    if " usa " in f" {text} " or " u.s. " in f" {text} " or " us " in f" {text} ":
        country = "United States"
    if not country and isinstance(context, dict) and "countries" in context and isinstance(context["countries"], list):
        for c in context["countries"]:
            if c and c.lower() in text:
                country = c
                break
    return {"year": year, "country": country}

def _parse_under_threshold(question: str) -> int | None:
    m = re.search(r"under\s*\$?\s*(\d+)", question.lower())
    if m:
        return int(m.group(1))
    return None

def _build_trends_sql(year: int, country: str) -> str:
    dataset = "bigquery-public-data.thelook_ecommerce"
    return f"""
SELECT
  EXTRACT(YEAR FROM o.created_at) AS year,
  EXTRACT(MONTH FROM o.created_at) AS month,
  COUNT(DISTINCT o.order_id) AS orders,
  SUM(oi.sale_price) AS revenue
FROM `{dataset}.orders` o
JOIN `{dataset}.order_items` oi
  ON oi.order_id = o.order_id
JOIN `{dataset}.users` u
  ON u.id = o.user_id
WHERE EXTRACT(YEAR FROM o.created_at) = {year}
  AND u.country = '{country}'
GROUP BY year, month
ORDER BY month
""".strip()

def _run_validated_sql_and_append(state: AgentState, sql: str, tool_id_prefix: str):
    state["messages"].append(ToolMessage(content=sql, tool_call_id=f"{tool_id_prefix}_sql"))
    val = validator.invoke({"sql": sql})
    state["messages"].append(ToolMessage(content=val, tool_call_id=f"{tool_id_prefix}_validator"))
    if isinstance(val, str) and val.strip().lower() == "valid":
        out = query_database.invoke({"sql": sql})
        state["messages"].append(ToolMessage(content=out, tool_call_id=f"{tool_id_prefix}_query"))
        return out
    return None

def _compute_metrics_from_rows(rows: List[dict]) -> dict:
    total_orders = sum(r.get("orders", 0) or 0 for r in rows)
    total_revenue = float(sum((r.get("revenue", 0) or 0) for r in rows))
    aov = (total_revenue / total_orders) if total_orders else 0.0
    return {
        "total_orders": int(total_orders),
        "total_revenue": round(total_revenue, 2),
        "aov": round(aov, 2),
    }

def _compute_trend_summary(rows: List[dict]) -> str:
    if not rows:
        return ""
    rows_sorted = sorted(rows, key=lambda r: int(r.get("month", 0)))
    rev = [(int(r["month"]), float(r.get("revenue", 0) or 0)) for r in rows_sorted]
    if not rev:
        return ""
    peak_month, peak_val = max(rev, key=lambda x: x[1])
    first_val = rev[0][1]
    last_val = rev[-1][1]
    direction = "rose" if last_val > first_val else "softened" if last_val < first_val else "held steady"

    def qsum(q):
        months = {1,2,3} if q==1 else {4,5,6} if q==2 else {7,8,9} if q==3 else {10,11,12}
        return sum(v for m,v in rev if m in months)

    q1, q2, q3, q4 = qsum(1), qsum(2), qsum(3), qsum(4)
    seasonality = []
    if q3 >= max(q1,q2,q4): seasonality.append("Q3 peak")
    if q4 >= max(q1,q2,q3): seasonality.append("Q4 peak")
    if q2 > q1: seasonality.append("spring > winter")
    if q3 > q2: seasonality.append("summer > spring")
    if q4 < q3: seasonality.append("Q4 easing vs Q3")

    trend = f"Monthly revenue {direction} from January to December; peak in month {peak_month} (~${peak_val:,.0f})."
    if seasonality:
        trend += " Seasonality: " + ", ".join(seasonality) + "."
    return trend

def _has_tool_message(messages: List[BaseMessage], prefix: str | None = None) -> bool:
    for m in messages:
        if isinstance(m, ToolMessage):
            if prefix is None:
                return True
            if m.tool_call_id and m.tool_call_id.startswith(prefix):
                return True
    return False

def trends_node(state: AgentState):
    sub_state = {
        "messages": state["messages"],
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    }
    result = trends_graph.invoke(sub_state)
    state["messages"] = result["messages"]

    question = state["messages"][0].content if state["messages"] else ""
    ddl_like = re.search(r"\b(select|insert|update|delete|drop|alter|create)\b", question.lower())
    info = _extract_year_and_country(question, state.get("context", {}))
    year = info.get("year") or 2022
    country = info.get("country") or "United States"

    if not _has_tool_message(state["messages"]) and not ddl_like:
        sql = _build_trends_sql(year, country)
        out = _run_validated_sql_and_append(state, sql, "manual_trends")
        if out:
            try:
                rows = json.loads(out) if isinstance(out, str) else []
                metrics = _compute_metrics_from_rows(rows)
                metrics.update({"period": str(year), "country": country})
                state["messages"].append(ToolMessage(content=json.dumps(metrics), tool_call_id="manual_metrics"))
                summary = _compute_trend_summary(rows)
                if summary:
                    state["messages"].append(ToolMessage(content=summary, tool_call_id="manual_summary"))
            except Exception:
                pass

    if question.lower().startswith("analyze") and not ddl_like:
        dataset = "bigquery-public-data.thelook_ecommerce"

        sql_gender = f"""
SELECT u.gender, COUNT(DISTINCT o.order_id) AS orders, SUM(oi.sale_price) AS revenue
FROM `{dataset}.orders` o
JOIN `{dataset}.order_items` oi ON oi.order_id = o.order_id
JOIN `{dataset}.users` u ON u.id = o.user_id
WHERE EXTRACT(YEAR FROM o.created_at) = {year} AND u.country = '{country}'
GROUP BY u.gender
ORDER BY revenue DESC
""".strip()
        _run_validated_sql_and_append(state, sql_gender, "manual_gender")

        sql_age = f"""
SELECT CASE
  WHEN u.age < 18 THEN '<18'
  WHEN u.age BETWEEN 18 AND 24 THEN '18-24'
  WHEN u.age BETWEEN 25 AND 34 THEN '25-34'
  WHEN u.age BETWEEN 35 AND 54 THEN '35-54'
  ELSE '55+'
END AS age_group,
COUNT(DISTINCT o.order_id) AS orders,
SUM(oi.sale_price) AS revenue
FROM `{dataset}.orders` o
JOIN `{dataset}.order_items` oi ON oi.order_id = o.order_id
JOIN `{dataset}.users` u ON u.id = o.user_id
WHERE EXTRACT(YEAR FROM o.created_at) = {year} AND u.country = '{country}'
GROUP BY age_group
ORDER BY revenue DESC
""".strip()
        _run_validated_sql_and_append(state, sql_age, "manual_age")

        sql_categories = f"""
SELECT p.category, COUNT(DISTINCT o.order_id) AS orders, SUM(oi.sale_price) AS revenue
FROM `{dataset}.orders` o
JOIN `{dataset}.order_items` oi ON oi.order_id = o.order_id
JOIN `{dataset}.products` p ON p.id = oi.product_id
JOIN `{dataset}.users` u ON u.id = o.user_id
WHERE EXTRACT(YEAR FROM o.created_at) = {year} AND u.country = '{country}'
GROUP BY p.category
ORDER BY revenue DESC
LIMIT 5
""".strip()
        _run_validated_sql_and_append(state, sql_categories, "manual_categories")

        threshold = _parse_under_threshold(question)
        if threshold:
            sql_under = f"""
SELECT p.name, p.category, COUNT(*) AS items, SUM(oi.sale_price) AS revenue
FROM `{dataset}.orders` o
JOIN `{dataset}.order_items` oi ON oi.order_id = o.order_id
JOIN `{dataset}.products` p ON p.id = oi.product_id
JOIN `{dataset}.users` u ON u.id = o.user_id
WHERE EXTRACT(YEAR FROM o.created_at) = {year} AND u.country = '{country}' AND oi.sale_price < {threshold}
GROUP BY p.name, p.category
ORDER BY revenue DESC
LIMIT 5
""".strip()
            _run_validated_sql_and_append(state, sql_under, "manual_products_under_threshold")

        sql_cities = f"""
SELECT u.city, COUNT(DISTINCT o.order_id) AS orders, SUM(oi.sale_price) AS revenue
FROM `{dataset}.orders` o
JOIN `{dataset}.order_items` oi ON oi.order_id = o.order_id
JOIN `{dataset}.users` u ON u.id = o.user_id
WHERE EXTRACT(YEAR FROM o.created_at) = {year} AND u.country = '{country}'
GROUP BY u.city
ORDER BY revenue DESC
LIMIT 5
""".strip()
        _run_validated_sql_and_append(state, sql_cities, "manual_cities")

        sql_pricebands = f"""
SELECT CASE
  WHEN oi.sale_price < 25 THEN '<$25'
  WHEN oi.sale_price < 50 THEN '$25–$50'
  WHEN oi.sale_price < 100 THEN '$50–$100'
  ELSE '$100+'
END AS price_band,
SUM(oi.sale_price) AS revenue,
COUNT(*) AS items
FROM `{dataset}.orders` o
JOIN `{dataset}.order_items` oi ON oi.order_id = o.order_id
JOIN `{dataset}.users` u ON u.id = o.user_id
WHERE EXTRACT(YEAR FROM o.created_at) = {year} AND u.country = '{country}'
GROUP BY price_band
ORDER BY revenue DESC
""".strip()
        _run_validated_sql_and_append(state, sql_pricebands, "manual_price_bands")

    state["needs_retry"] = False
    return state

geo_graph = get_sub_agent_graph("Geo", "geographic patterns/regions/countries")

def geo_node(state: AgentState):
    sub_state = {
        "messages": state["messages"],
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    }
    result = geo_graph.invoke(sub_state)
    state["messages"] = result["messages"]
    state["needs_retry"] = False
    return state

product_graph = get_sub_agent_graph("Product", "product performance/recommendations/inventory")

def product_node(state: AgentState):
    sub_state = {
        "messages": state["messages"],
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    }
    result = product_graph.invoke(sub_state)
    state["messages"] = result["messages"]
    state["needs_retry"] = False
    return state
