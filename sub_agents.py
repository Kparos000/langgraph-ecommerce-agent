from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from config import get_llm
from tools import query_database, validator, generate_final_answer
from typing import Dict, List, Annotated
from operator import add
from state import AgentState
from langgraph.checkpoint.memory import MemorySaver

# --- SubAgent State ---
class SubAgentState(Dict):
    messages: Annotated[List[BaseMessage], add]
    memory: str
    schema: str
    context: Dict

tools = [query_database, validator, generate_final_answer]

def get_sub_agent_graph(role: str, specialty: str):
    llm = get_llm()
    # Strengthened prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a {role} Agent specializing in {specialty}.
You MUST use the available tools to answer queries. Do not ask the user for more information.
The only dataset is `bigquery-public-data.thelook_ecommerce`.

Tables available with schema:
{{schema}}

Context (date ranges, countries, seasons, age groups, regions):
{{context}}

Rules:
- ALWAYS use `query_database` and `validator` before `generate_final_answer`.
- If the query includes a country or region, you MUST join `users` and filter using users.country. 
  Example: JOIN users u ON o.user_id = u.id AND u.country='United States'.
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

    chain = prompt | llm.bind_tools(tools)

    def agent_node(state: SubAgentState):
        response = chain.invoke({
            "input": state["messages"][-1].content,
            "memory": state["memory"],
            "schema": state["schema"],
            "context": state["context"]
        })
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
    return graph.compile(checkpointer=MemorySaver())

# --- Sub-agents ---
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
