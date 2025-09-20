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

# Manager node (from previous)
def manager_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Manager Agent. Analyze the input query and delegate to a specialized sub-agent based on keywords. Use memory, context, and schema. Output JSON: {{\"sub_agent\": \"geo\"}}. Available: segmentation, trends, geo, product."),
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
        parsed = json.loads(response.content)
        sub_agent = parsed.get("sub_agent", "geo")
    except json.JSONDecodeError:
        sub_agent = "geo"
        if "segment" in state["messages"][-1].content.lower():
            sub_agent = "segmentation"
        elif "trend" in state["messages"][-1].content.lower():
            sub_agent = "trends"
        elif "geo" in state["messages"][-1].content.lower():
            sub_agent = "geo"
        elif "product" in state["messages"][-1].content.lower():
            sub_agent = "product"
    state["remaining_steps"] = sub_agent
    return state

# Reflective node
def reflective_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Reflector Agent. Review the sub-agent's output and data from messages for accuracy. Use CoT: Think step-by-step on SQL validity, data consistency, and flag issues. Update memory with your reasoning."),
        ("human", "{data}\n\nMemory: {memory}\nSchema: {schema}\nContext: {context}")
    ])
    chain = prompt | llm
    data = state["messages"][-1].content  # Assume last message is sub-agent output
    response = chain.invoke({
        "data": data,
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    })
    cot = response.content
    log.info(event="reflective", reasoning=cot)
    state["memory"] += f"\nCoT: {cot}"
    if "issue" in cot.lower():
        state["messages"].append(AIMessage(content="Flagged issue: " + cot))
    return state

# Synthesis node
def synthesis_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Synthesizer Agent. Use data and memory to generate a concise report: Metrics, trends, risks, 2-3 recommendations. Follow few-shot example from Vision: [insert full Vision example here for grounding]."),
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
        "messages": [HumanMessage(content="Test geo query: Top regions in 2023?")],
        "remaining_steps": "",
        "memory": "",
        "schema": schema,
        "context": context
    }
    config = {"configurable": {"thread_id": "test1"}}
    result = compiled_graph.invoke(initial_state, config=config)
    print(f"Delegated to: {result['remaining_steps']}")
    print(f"Final report: {result['messages'][-1].content}")