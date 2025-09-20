from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage
from config import get_llm
from tools import query_database, validator, generate_final_answer
from typing import Dict, List, Annotated
from operator import add
from state import AgentState

class SubAgentState(Dict):
    messages: Annotated[List, add]

tools = [query_database, validator, generate_final_answer]

def get_sub_agent_graph(role: str, specialty: str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a {role} Agent specializing in {specialty}. Use memory, schema, context to reason. Tools: query_database, validator, generate_final_answer. Think step-by-step: Parse → Gen SQL → Validate → Exec → Synthesize. Use only these tables from `bigquery-public-data.thelook_ecommerce`: orders, order_items, products, users. Output tool calls or final answer."),
        ("human", "{input}\nMemory: {memory}\nSchema: {schema}\nContext: {context}")
    ])
    bound_llm = llm.bind_tools(tools)

    def agent_node(state: SubAgentState):
        response = bound_llm.invoke(state["messages"])
        return {"messages": [response]}

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
    return graph.compile()

# Sub-agents
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
    return state

trends_graph = get_sub_agent_graph("Trends", "sales trends/seasonality/growth")

def trends_node(state: AgentState):
    sub_state = {
        "messages": state["messages"],
        "memory": state["memory"],
        "schema": state["schema"],
        "context": state["context"]
    }
    result = trends_graph.invoke(sub_state)
    state["messages"] = result["messages"]
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
    return state