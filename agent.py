import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_bq_client, get_schema, get_context
from state import AgentState
from sub_agents import segmentation_node, trends_node, geo_node, product_node
import structlog

log = structlog.get_logger()

# Manager node
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

# Reflective node with retry
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

# Synthesis node (enhanced with richer few-shots)
# Synthesis node (enhanced with richer few-shots, corrected monthly handling)
def synthesis_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
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

    chain = prompt | llm
    data = state["messages"][-1].content if "Flagged issue" not in state["messages"][-1].content else state["messages"][-2].content
    response = chain.invoke({
        "data": data,
        "memory": state["memory"]
    })
    state["messages"].append(AIMessage(content=response.content))
    return state

# Graph definition
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
