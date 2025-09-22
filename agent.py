import json
from typing import List  # <-- fix: import List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
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
        sub_agent = parsed.get("sub_agent", "trends")
        state["memory"] += f"\nManager Reasoning: {response.content}"
    except json.JSONDecodeError:
        sub_agent = "trends"
        state["memory"] += "\nManager Reasoning: Parse error, default to trends."
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
        state["messages"].append(HumanMessage(content="Retry: Ensure query uses schema correctly (join users.country if filtering by country, join products for categories/revenue)."))
        state["remaining_steps"] = state["remaining_steps"]
        return state
    if "issue" in cot.lower() or "flag" in cot.lower():
        state["messages"].append(AIMessage(content="Flagged issue: " + cot))
    return state

def _collect_curated_tooldata(messages: List[BaseMessage]) -> str:
    chunks = []
    last_query_snippet = None
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id:
            tid = m.tool_call_id
            if tid.endswith("_sql"):
                last_query_snippet = m.content
            elif tid.endswith("_query"):
                if any(tid.startswith(prefix) for prefix in [
                    "manual_trends", "manual_gender", "manual_age",
                    "manual_categories", "manual_cities", "manual_price_bands",
                    "manual_products_under_threshold"
                ]):
                    label = tid.replace("_query", "")
                    chunks.append(f"{label} => {m.content}")
            elif tid == "manual_metrics":
                chunks.append(f"manual_metrics => {m.content}")
            elif tid == "manual_summary":
                chunks.append(f"manual_summary => {m.content}")
    if last_query_snippet:
        chunks.insert(0, f"executed_sql => {last_query_snippet}")
    return "\n".join(chunks) if chunks else ""

def synthesis_node(state: AgentState):
    llm = get_llm()
    curated = _collect_curated_tooldata(state["messages"])

    base_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Synthesizer Agent. Use the SQL results (month-by-month time series, plus gender/age/categories/cities/price bands) and reasoning steps (memory) to produce a focused, business-ready report.

Rules:
- If the query starts with "Analyze":
  - Include **Metrics** (total revenue, total orders, AOV).
  - Provide a narrative **Analysis**: describe monthly/quarterly trend (rise/peak/slowdown), then key demographics (gender, age), leading categories/products, and top cities (if present).
  - Close with 2–3 **Recommendations**. Keep them tied to the observed data. No "Risks" section.
  - Do NOT mention weekday performance unless explicitly asked. If any weekday slice appears in the tool data, ignore it.
- If the query starts with "Identify" or "Give me":
  - Return a concise ranked list or table. No recommendations.
- If irrelevant: respond with "I am not trained to answer this type of question. Ask me about bigquery-public-data.thelook_ecommerce."
- If flagged or no usable data: "No data fetched due to [error from memory]; rephrase query for better results."

Strictly avoid including raw JSON blobs in the final answer. Convert numbers into readable sentences and short bullet points when needed."""),
        ("human", "Curated tool data:\n{curated}\n\nMemory:\n{memory}\n\nUser query:\n{query}")
    ])

    chain = base_prompt | llm
    data_for_synth = curated or (state["messages"][-1].content if state["messages"] else "")
    response = chain.invoke({
        "curated": data_for_synth,
        "memory": state["memory"],
        "query": state["messages"][0].content if state["messages"] else ""
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
        "messages": [HumanMessage(content="Analyze sales trends in US in 2022")],
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
