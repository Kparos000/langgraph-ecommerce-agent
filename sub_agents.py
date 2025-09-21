from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from config import get_llm
from tools import query_database, validator, generate_final_answer
from typing import Dict, List, Annotated
from operator import add
from state import AgentState
from langgraph.checkpoint.memory import MemorySaver
import re

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
- ALWAYS use `validator` then `query_database` before `generate_final_answer`.
- If the query includes a country or region, you MUST join `users` and filter using users.country.
  Example: JOIN `bigquery-public-data.thelook_ecommerce.users` u ON o.user_id = u.id AND u.country='United States'.
- For time-based queries, always use created_at for filtering (YEAR, MONTH, QUARTER).
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

def _append_manual_tool_messages(state: AgentState, sql: str):
    val_out = validator.invoke({"sql": sql})
    state["messages"].append(ToolMessage(content=val_out, tool_call_id="manual_validator"))
    if isinstance(val_out, str) and val_out.strip().lower() == "valid":
        q_out = query_database.invoke({"sql": sql})
        state["messages"].append(ToolMessage(content=q_out, tool_call_id="manual_query_database"))

def _has_tool_message(messages: List[BaseMessage]) -> bool:
    return any(isinstance(m, ToolMessage) for m in messages)

def trends_node(state: AgentState):
    sub_state = {
        "messages": state["messages"],
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    }
    result = trends_graph.invoke(sub_state)
    state["messages"] = result["messages"]

    if not _has_tool_message(state["messages"]):
        question = state["messages"][0].content if state["messages"] else ""
        info = _extract_year_and_country(question, state.get("context", {}))
        year = info.get("year")
        country = info.get("country") or "United States"
        if year is None:
            year = 2022
        sql = _build_trends_sql(year, country)
        _append_manual_tool_messages(state, sql)

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
