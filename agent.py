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
    trimmed_state["messages"] = state["messages"][-10:]  # Changed: Increased from -5 to -10 for more context retention
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

def summarizer_node(state: AgentState):
    """New: Optional summarizer for long histories (conditional before synthesis)."""
    if len(state["messages"]) > 10:
        summary_prompt = ChatPromptTemplate.from_template("Summarize the last 10 messages and insights for context: {messages} {insights}.")
        formatted = summary_prompt.format(messages=state["messages"][-10:], insights=json.dumps(state["insights"]))
        summary = llm.invoke(formatted).content
        state["messages"].append({"role": "system", "content": f"Context Summary: {summary}"})
    return state

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
    Output in a narrative report style for an executive audience: Use paragraphs with bold figures (e.g., **China: $611,205**), no tables, lists, or bullets. Follow the prompt strictly; do not give recommendations unless explicitly asked. Example for "Which two countries had the lowest sales in 2020?":
    Lowest Sales Countries in 2020
    In 2020, our lowest sales regions highlighted areas for potential growth, with distinct patterns in purchasing behavior.
    Austria: The lowest performer was Austria, generating **$154** in sales, primarily from low-volume urban purchases.
    Colombia: Colombia followed with **$409.96** in sales, skewed toward seasonal items in coastal areas.
    Across these regions, sales were concentrated in entry-level products, with limited repeat purchases from urban demographics.
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
workflow.add_node("summarizer", summarizer_node)  # New: Optional summarizer for long histories
workflow.add_node("synthesis", synthesis_node)

workflow.add_edge(START, "manager")
workflow.add_conditional_edges(
    "manager",
    lambda s: s["next"],
    {"segmentation": "segmentation", "trends": "trends", "geo": "geo", "product": "product", "synthesis": "synthesis"}
)
workflow.add_conditional_edges(
    "segmentation",
    lambda s: "summarizer" if len(s["messages"]) > 10 else "synthesis",
    {"summarizer": "summarizer", "synthesis": "synthesis"}
)
workflow.add_conditional_edges(
    "trends",
    lambda s: "summarizer" if len(s["messages"]) > 10 else "synthesis",
    {"summarizer": "summarizer", "synthesis": "synthesis"}
)
workflow.add_conditional_edges(
    "geo",
    lambda s: "summarizer" if len(s["messages"]) > 10 else "synthesis",
    {"summarizer": "summarizer", "synthesis": "synthesis"}
)
workflow.add_conditional_edges(
    "product",
    lambda s: "summarizer" if len(s["messages"]) > 10 else "synthesis",
    {"summarizer": "summarizer", "synthesis": "synthesis"}
)
workflow.add_edge("summarizer", "synthesis")
workflow.add_edge("synthesis", END)

app = workflow.compile(checkpointer=MemorySaver())