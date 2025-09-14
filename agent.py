from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from state import AgentState
from sub_agents import segmentation_agent, trends_agent, geo_agent, product_agent
from config import get_llm

llm = get_llm()

# Manager Prompt (simple classification)
manager_prompt = ChatPromptTemplate.from_template("""
You are ManagerAgent. Classify the latest human message in {messages}.
Output **only** "next: [agent_name]" where agent_name is one of: segmentation, trends, geo, product, or synthesis if done.
Example: For "sales trends", output "next: trends".
Example: For "customer segmentation", output "next: segmentation".
Example: For vague, output "next: synthesis".
""")

def manager_node(state: AgentState):
    trimmed_messages = state["messages"][-3:]
    formatted_prompt = manager_prompt.format(messages=trimmed_messages)
    response = llm.invoke(formatted_prompt)
    final_content = response.content.strip()
    if "next: " in final_content:
        next_agent = final_content.split("next: ")[1].strip()
    else:
        next_agent = "synthesis"
    print(f"Manager delegated to: {next_agent}")  # Debug
    state["next"] = next_agent
    state["messages"] = trimmed_messages + [response]
    return state

def invoke_sub_agent(agent, state):
    """Direct invoke sub-agent with higher recursion limit, extract Final Answer JSON."""
    trimmed_state = state.copy()
    trimmed_state["messages"] = state["messages"][-5:]
    try:
        result = agent.invoke(trimmed_state, config={"recursion_limit": 50})
        last_content = result["messages"][-1].content
        # Extract Final Answer JSON
        final_match = re.search(r'Final Answer[:\s]*(\[.*?\])', last_content, re.DOTALL)
        if final_match:
            json_str = final_match.group(1)
        else:
            # Fallback to any JSON array
            array_match = re.search(r'\[.*?\]', last_content, re.DOTALL)
            json_str = array_match.group() if array_match else '[]'
        parsed = json.loads(json_str)
        insights_list = parsed if isinstance(parsed, list) else [str(parsed)]
        print(f"Sub-agent succeeded: {insights_list}")
        # Update result
        result["messages"][-1].content = json.dumps({"insights": insights_list})
        return result, True
    except (ValueError, Exception) as e:
        print(f"Sub-agent failed: {str(e)}")
        fallback_insights = ["Fallback: Unable to generate insights."]
        fallback_result = {"messages": trimmed_state["messages"] + [{"role": "system", "content": json.dumps({"insights": fallback_insights})}]}
        return fallback_result, False

def segmentation_node(state: AgentState):
    result, success = invoke_sub_agent(segmentation_agent, state)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Segmentation", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def trends_node(state: AgentState):
    result, success = invoke_sub_agent(trends_agent, state)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Trends", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def geo_node(state: AgentState):
    result, success = invoke_sub_agent(geo_agent, state)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Geo", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def product_node(state: AgentState):
    result, success = invoke_sub_agent(product_agent, state)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Product", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def synthesis_node(state: AgentState):
    if not state["insights"]:
        state["report"] = "No insights generated—check manager delegation or sub-agent output."
        return state
    trimmed_messages = state["messages"][-5:]
    synth_prompt = ChatPromptTemplate.from_template("""
    Synthesize insights from {insights} for the prompt in {messages}.
    Steps:
    1. Extract all exact figures from insights (e.g., revenue for Q1-Q4 across all years, % differences).
    2. Provide a comprehensive summary covering every insight provided by the sub-agent, ensuring no data is omitted.
    3. Actionable key findings in bullets with exact figures for all quarters and years (e.g., "Q3 2019: $33.31k (highest in 2019, 15% > Q2)").
    4. Markdown format with a detailed table for all quarters and years (Q1-Q4 revenue across available years).
    5. If insights cover top N results (e.g., top 3 countries), note 'Based on top N results; full total queried separately if needed'.
    6. No limitations section—data is complete for all available periods; focus on key findings, table with exact figures, actionable insights.
    Once done, return the report.
    """)
    formatted_insights = json.dumps(state["insights"])
    formatted_prompt = synth_prompt.format(insights=formatted_insights, messages=trimmed_messages)
    state["report"] = llm.invoke(formatted_prompt).content
    return state

# Graph (subs to synthesis directly—no back-edges)
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager_node)
workflow.add_node("segmentation", segmentation_node)
workflow.add_node("trends", trends_node)
workflow.add_node("geo", geo_node)
workflow.add_node("product", product_node)
workflow.add_node("synthesis", synthesis_node)

workflow.add_edge(START, "manager")
workflow.add_conditional_edges(
    "manager",
    lambda s: s["next"],
    {"segmentation": "segmentation", "trends": "trends", "geo": "geo", "product": "product", "synthesis": "synthesis"}
)
workflow.add_edge("segmentation", "synthesis")
workflow.add_edge("trends", "synthesis")
workflow.add_edge("geo", "synthesis")
workflow.add_edge("product", "synthesis")
workflow.add_edge("synthesis", END)

app = workflow.compile(checkpointer=MemorySaver())