from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from sub_agents import segmentation_agent, trends_agent, geo_agent, product_agent
from tools import create_handoff_tool
from config import get_llm

llm = get_llm()

# Manager Prompt (PRD 3.3)
manager_prompt = ChatPromptTemplate.from_template("""
You are ManagerAgent. Classify the latest human message in {messages}.
Steps:
1. Parse for analysis type: segmentation (behavior/demographics), trends (sales/seasonality), geo (location), product (performance/recs).
2. Select 1-4 sub-agents (e.g., 'trends' for seasonality).
3. If multi, sequence (trends first for context).
4. Call handoff tool to delegate.
5. After subs, set next='synthesis'.
Edge: Vague? Reply 'Clarify prompt'.
Once done, return 'complete'.
""")

def manager_node(state: AgentState):
    manager = create_react_agent(
        llm,
        tools=[create_handoff_tool("segmentation"), create_handoff_tool("trends"), 
               create_handoff_tool("geo"), create_handoff_tool("product")],
        prompt=manager_prompt
    )
    result = manager.invoke(state)
    state["next"] = "synthesis" if "complete" in str(result) else result.get("next", "trends")  # Parse from result
    state["messages"] = result["messages"]
    return state

def segmentation_node(state: AgentState):
    result = segmentation_agent.invoke(state)
    state["insights"].append({"agent": "Segmentation", "text": result["messages"][-1].content})
    return state

def trends_node(state: AgentState):
    result = trends_agent.invoke(state)
    state["insights"].append({"agent": "Trends", "text": result["messages"][-1].content})
    return state

def geo_node(state: AgentState):
    result = geo_agent.invoke(state)
    state["insights"].append({"agent": "Geo", "text": result["messages"][-1].content})
    return state

def product_node(state: AgentState):
    result = product_agent.invoke(state)
    state["insights"].append({"agent": "Product", "text": result["messages"][-1].content})
    return state

def synthesis_node(state: AgentState):
    synth_prompt = ChatPromptTemplate.from_template("""
    Synthesize insights from {insights} for the prompt in {messages}.
    Steps:
    1. Combine (e.g., trends + product → recs).
    2. Actionable: Bullets/tables.
    3. Markdown format.
    Edge: Missing data? Note limitations.
    Once done, return the report.
    """)
    formatted_prompt = synth_prompt.format(insights=state["insights"], messages=state["messages"])
    state["report"] = llm.invoke(formatted_prompt).content
    return state

# Graph
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
for sub in ["segmentation", "trends", "geo", "product"]:
    workflow.add_edge(sub, "manager")
workflow.add_edge("synthesis", END)

app = workflow.compile(checkpointer=MemorySaver())