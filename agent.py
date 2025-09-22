import json
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_bq_client, get_schema, get_context
from state import AgentState
from sub_agents import segmentation_node, trends_node, geo_node, product_node
import structlog

log = structlog.get_logger()

def manager_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Manager Agent. Analyze the input query with reasoning on intent, keywords, memory, context, and schema to delegate to a specialized sub-agent. Available: segmentation (demographics/RFM/behavior), trends (sales/seasonality/growth/trends), geo (geographic/regions/countries/cities), product (performance/recommendations/inventory/products/categories). If query is irrelevant to thelook_ecommerce dataset (e.g., no data-related keywords/intent), delegate to "synthesis" and explain why in reasoning.

Format: Thought: [reason step-by-step on intent/keywords/matching specialty] Final Answer: {{"sub_agent": "value"}}."""),
        ("human", "{input}\n\nMemory: {memory}\nContext: {context}\nSchema: {schema}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "input": state["messages"][-1].content,
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    })
    try:
        final_answer_str = response.content.split("Final Answer:")[-1].strip()
        parsed = json.loads(final_answer_str)
        sub_agent = parsed.get("sub_agent", "geo")
        state["memory"] += f"\nManager Reasoning: {response.content}"
    except json.JSONDecodeError:
        sub_agent = "geo"
        state["memory"] += "\nManager Reasoning: Parse error, default to geo."
    state["remaining_steps"] = sub_agent
    return state

def reflective_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Reflector Agent. Review the sub-agent's output and data from messages for accuracy. Use CoT: Think step-by-step on SQL validity (schema match), data consistency (e.g., dates in context.date_span, countries in context.countries), flag issues (e.g., hallucinations, inconsistencies, no data fetched/SQL executed—must have JSON results from query_database). Update memory with your reasoning. If issue (e.g., no data or text asking for info), append flagged message.

Format: CoT: [detailed reasoning, flag if needed]."""),
        ("human", "{data}\n\nMemory: {memory}\nSchema: {schema}\nContext: {context}")
    ])
    chain = prompt | llm
    data = state["messages"][-1].content
    response = chain.invoke({
        "data": data,
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    })
    cot = response.content
    log.info(event="reflective", reasoning=cot)
    state["memory"] += f"\nCoT: {cot}"

    if ("flag" in cot.lower() or "retry" in cot.lower()) and not state.get("retry_done", False):
        state["retry_done"] = True
        state["messages"].append(HumanMessage(content="Retry: Ensure monthly trends SQL is executed and join users.country when filtering by country."))
        state["remaining_steps"] = state["remaining_steps"]
        return state
    if "issue" in cot.lower() or "flag" in cot.lower():
        state["messages"].append(AIMessage(content="Flagged issue: " + cot))
    return state

def _collect_tool_observations(messages: list[BaseMessage]) -> dict:
    bundle = {
        "monthly_rows": None,
        "metrics": None,
        "trend_summary": None,
        "gender_rows": None,
        "age_rows": None,
        "category_rows": None,
        "city_rows": None,
        "price_band_rows": None,
        "weekday_rows": None
    }
    for m in messages:
        if not isinstance(m, AIMessage) and not isinstance(m, HumanMessage):
            # LangChain ToolMessage implements BaseMessage; we check tool_call_id hints we created
            tool_call_id = getattr(m, "tool_call_id", "") or ""
            content = getattr(m, "content", "")
            if not isinstance(content, str):
                continue
            # Query outputs end with *_query ids in our sub_agents
            if tool_call_id.endswith("manual_trends_query"):
                try:
                    bundle["monthly_rows"] = json.loads(content)
                except Exception:
                    pass
            elif tool_call_id == "manual_metrics":
                try:
                    bundle["metrics"] = json.loads(content)
                except Exception:
                    pass
            elif tool_call_id == "manual_summary":
                bundle["trend_summary"] = content
            elif tool_call_id.endswith("manual_gender_query"):
                try:
                    bundle["gender_rows"] = json.loads(content)
                except Exception:
                    pass
            elif tool_call_id.endswith("manual_age_query"):
                try:
                    bundle["age_rows"] = json.loads(content)
                except Exception:
                    pass
            elif tool_call_id.endswith("manual_categories_query"):
                try:
                    bundle["category_rows"] = json.loads(content)
                except Exception:
                    pass
            elif tool_call_id.endswith("manual_cities_query"):
                try:
                    bundle["city_rows"] = json.loads(content)
                except Exception:
                    pass
            elif tool_call_id.endswith("manual_price_bands_query"):
                try:
                    bundle["price_band_rows"] = json.loads(content)
                except Exception:
                    pass
            elif tool_call_id.endswith("manual_weekday_query"):
                try:
                    bundle["weekday_rows"] = json.loads(content)
                except Exception:
                    pass
    return bundle

def synthesis_node(state: AgentState):
    llm = get_llm()
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Synthesizer Agent. Write a polished, business-ready report using the structured data bundle below.
Follow these rules:
- If the user query starts with "Analyze":
  - Output a concise executive header with: total revenue, total orders, AOV.
  - Then an expressive analysis that ties together: time trend (use monthly), seasonality (quarters), demographics (gender, age), geography (cities), and categories/price bands if available.
  - Use sentences and comparisons (e.g., “Q3 outpaced Q2 by ~12%”; “Females led with $X vs $Y for males”).
  - 2–3 concrete recommendations. No “Risks”.
- If the query starts with "Identify" or "Give me":
  - Output only the exact numbers or a short ranked list/table. No recommendations.
- Never show internal labels or tool IDs. Do not mention weekdays unless the user asked for weekdays.
- If the monthly_rows is missing and there’s no metrics, say: "No data fetched due to missing executed SQL; rephrase query for better results."
"""),
        ("human", "User Query:\n{query}\n\nStructured Data Bundle (JSON):\n{bundle_json}\n\nMemory Hints:\n{memory}")
    ])

    # Build a data bundle from all tool observations in this turn
    bundle = _collect_tool_observations(state["messages"])
    try:
        bundle_json = json.dumps(bundle)
    except Exception:
        bundle_json = "{}"

    query_txt = state["messages"][0].content if state["messages"] else ""
    chain = base_prompt | llm
    response = chain.invoke({
        "query": query_txt,
        "bundle_json": bundle_json,
        "memory": state["memory"]
    })
    state["messages"].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("manager", manager_node)
graph.add_node("segmentation_agent", segmentation_node)
graph.add_node("trends_agent", trends_node)
graph.add_node("geo_agent", geo_node)
graph.add_node("product_agent", product_node)
graph.add_node("reflective", reflective_node)
graph.add_node("synthesis", synthesis_node)

graph.add_edge("__start__", "manager")

def route_to_subagent(state: AgentState):
    sub = state["remaining_steps"]
    if sub == "segmentation":
        return "segmentation_agent"
    elif sub == "trends":
        return "trends_agent"
    elif sub == "geo":
        return "geo_agent"
    elif sub == "product":
        return "product_agent"
    elif sub == "synthesis":
        return "synthesis"
    else:
        return END

graph.add_conditional_edges("manager", route_to_subagent)
graph.add_edge("segmentation_agent", "reflective")
graph.add_edge("trends_agent", "reflective")
graph.add_edge("geo_agent", "reflective")
graph.add_edge("product_agent", "reflective")
graph.add_edge("reflective", "synthesis")
graph.add_edge("synthesis", END)

checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    client = get_bq_client()
    schema = json.dumps(get_schema(client))
    context = get_context(client)
    initial_state = {
        "messages": [HumanMessage(content="Analyze sales in China in 2022")],
        "remaining_steps": "",
        "memory": "",
        "schema": schema,
        "context": context
    }
    config = {"configurable": {"thread_id": "test1"}}
    print(f"Input Query: {initial_state['messages'][0].content}")
    result = compiled_graph.invoke(initial_state, config=config)
    print(f"Delegated to: {result['remaining_steps']}")
    print(f"Final report: {result['messages'][-1].content}")
