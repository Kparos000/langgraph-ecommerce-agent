import json
import re
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_bq_client, get_schema, get_context
from state import AgentState
from sub_agents import segmentation_node, trends_node, geo_node, product_node
import structlog

log = structlog.get_logger()

def _latest_tool_content(messages):
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return m.content
    return None

def _first_user_query(messages):
    for m in messages:
        if isinstance(m, HumanMessage) and not m.content.strip().lower().startswith("retry:"):
            return m.content.strip()
    return ""

def _looks_like_json(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    if t[0] not in "[{":
        return False
    try:
        json.loads(t)
        return True
    except Exception:
        return False

def manager_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Manager Agent. Analyze the input query with reasoning on intent, keywords, memory, context, and schema to delegate to a specialized sub-agent.

Available sub-agents:
- segmentation: demographics, RFM, behavior
- trends: sales trends, seasonality, growth
- geo: geographic patterns (regions, countries, cities)
- product: product performance, recommendations, inventory

If the query is irrelevant to the thelook_ecommerce dataset, delegate to "synthesis" and explain why in your thought.

Output format (no JSON, no braces):
Thought: <your step-by-step reasoning>
Final Answer: sub_agent=<segmentation|trends|geo|product|synthesis>"""),
        ("human", "{input}\n\nMemory: {memory}\nContext: {context}\nSchema: {schema}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "input": state["messages"][-1].content,
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    })

    text = response.content
    state["memory"] += f"\nManager Reasoning: {text}"

    allowed = ["segmentation", "trends", "geo", "product", "synthesis"]
    sub_agent = None
    m = re.search(r"Final Answer:\s*sub_agent\s*=\s*(\w+)", text, flags=re.IGNORECASE)
    if m:
        cand = m.group(1).lower()
        if cand in allowed:
            sub_agent = cand
    if not sub_agent:
        for a in allowed:
            if re.search(rf"\b{a}\b", text, flags=re.IGNORECASE):
                sub_agent = a
                break
    if not sub_agent:
        sub_agent = "geo"

    state["remaining_steps"] = sub_agent
    return state

def reflective_node(state: AgentState):
    any_tool = any(isinstance(m, ToolMessage) for m in state["messages"])
    if not any_tool and not state.get("retry_done", False):
        state["retry_done"] = True
        state["needs_retry"] = True
        state["messages"].append(HumanMessage(content="Retry: Use tools now. First call validator with a valid BigQuery SQL using the schema; then call query_database; then synthesize. Do not ask me for SQL."))
        state["remaining_steps"] = state["remaining_steps"]
        return state

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Reflector Agent. Review the sub-agent's output and data from messages for accuracy. Use CoT: Think step-by-step on SQL validity (schema match), data consistency (e.g., dates in context.date_span, countries in context.countries), flag issues (e.g., hallucinations, inconsistencies, no data fetched/SQL executed—must have JSON results from query_database). Update memory with your reasoning. If issue (e.g., no data or text asking for info), append flagged message.

Format: CoT: [detailed reasoning, flag if needed]."""),
        ("human", "{data}\n\nMemory: {memory}\nSchema: {schema}\nContext: {context}")
    ])
    chain = prompt | llm
    tool_data = _latest_tool_content(state["messages"])
    data = tool_data if tool_data is not None else state["messages"][-1].content
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
        state["needs_retry"] = True
        state["messages"].append(HumanMessage(content="Retry: Ensure query uses schema correctly (join users.country if filtering by country, join products for categories/revenue). Then call validator and query_database."))
        state["remaining_steps"] = state["remaining_steps"]
        return state

    if "issue" in cot.lower() or "flag" in cot.lower():
        state["messages"].append(AIMessage(content="Flagged issue: " + cot))

    state["needs_retry"] = False
    return state

def synthesis_node(state: AgentState):
    tool_data = _latest_tool_content(state["messages"])

    if tool_data is None:
        state["messages"].append(AIMessage(content="No data fetched due to missing executed SQL; rephrase query for better results."))
        return state

    if not _looks_like_json(tool_data):
        state["messages"].append(AIMessage(content="No data fetched due to missing executed SQL; rephrase query for better results."))
        return state

    llm = get_llm()
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Synthesizer Agent. Use the SQL results (data) and reasoning steps (memory) 
to produce clear, business-ready reports.

Rules:
- If the query starts with "Analyze":
  - Always include **Metrics** (total revenue, total orders, average order value).
  - Always include **Analysis** connecting trends across time, demographics (age/gender), geography (countries/cities), and product categories.
  - Always include **Recommendations** (2–3 actionable, data-backed).
  - Be expressive and connect facts into business insights.
  - IMPORTANT: If SQL provides monthly or quarterly breakdowns, ALWAYS report them. 
    Do NOT claim data is missing if breakdowns exist.
- If the query starts with "Identify" or "Give me":
  - Output a ranked list or table of exact outputs (e.g., top products, top regions).
  - No recommendations, no narrative.
- Do NOT include "Risks".
- If irrelevant: 
  "I am not trained to answer this type of question. Ask me about bigquery-public-data.thelook_ecommerce."
- If flagged or no data:
  "No data fetched due to [error from memory]; rephrase query for better results."

Few-shots:

Query: Analyze sales in China in 2024.
Final Answer:
**Metrics:** $1.3M revenue (+15% YoY), 22,000 orders, $59 AOV
**Analysis:** 
Quarterly growth: Q1 $280K, Q2 $320K, Q3 $380K, Q4 $450K projected.
Females (58%) dominated, led by 18–24 ($416K). 
Shanghai ($585K) and Beijing ($364K) led cities. 
Top <$50 products: Women's Basic Tee ($180K), Phone Case ($150K), Yoga Mat ($120K).
**Recommendations:** Promote affordable fashion bundles for young females; expand Tier-2 city promotions.

Query: Analyze sales trends in US in 2022.
Final Answer:
**Metrics:** $2.5M revenue, 30K orders, $83 AOV
**Analysis:** 
Monthly breakdown: Jan $200K (steady), Q2 rise to $600K, peak Q3 $700K, Q4 $650K. 
Females 25–34 ($1M) led fashion; males 35–54 ($600K) drove tech. 
Top products: Women's Basic Tee ($300K), Phone Case ($220K). 
New York ($800K) and LA ($600K) led cities.
**Recommendations:** Scale Q3 promotions; diversify supply chains to support Q4.

Query: Identify top products by revenue in 2024.
Final Answer:
1. Women's Basic Tee — $450K
2. Phone Case — $400K
3. Yoga Mat — $350K
(No recommendations)

Query: Identify top regions in 2023.
Final Answer:
North America — $3.1M
China — $1.2M
EMEA — $1.1M
(No recommendations)"""),
        ("human", "{data}\n\nMemory: {memory}")
    ])

    first = (base_prompt | llm).invoke({
        "data": tool_data,
        "memory": state["memory"]
    })
    final_text = first.content or ""

    user_query = _first_user_query(state["messages"])
    is_analyze = user_query.lower().startswith("analyze")
    has_metrics = "Metrics" in final_text
    has_analysis = "Analysis" in final_text
    has_recs = "Recommendations" in final_text

    if is_analyze and not (has_metrics and has_analysis and has_recs):
        final_text = (
            "**Metrics:**\n"
            "- See analysis below.\n\n"
            "**Analysis:**\n"
            f"{final_text}\n\n"
            "**Recommendations:**\n"
            "- Align inventory and promotions with observed seasonal trends.\n"
            "- Target high-performing demographics and geographies.\n"
        )

    state["messages"].append(AIMessage(content=final_text))
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

def route_after_reflect(state: AgentState):
    if state.get("needs_retry", False):
        return route_to_subagent(state)
    return "synthesis"

# Restore the crucial edges from sub-agents to reflective
graph.add_edge("segmentation_agent", "reflective")
graph.add_edge("trends_agent", "reflective")
graph.add_edge("geo_agent", "reflective")
graph.add_edge("product_agent", "reflective")

graph.add_conditional_edges("reflective", route_after_reflect)
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
