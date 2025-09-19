import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from config import get_llm, get_bq_client, get_schema, get_context
from state import AgentState

# Manager node definition
def manager_node(state: AgentState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Manager Agent. Your task is to analyze the input query and determine which specialized sub-agent to delegate it to, based on keywords. Use the memory, context, and schema to inform your decision. Output a JSON object with a single key 'sub_agent' containing the name of the sub-agent (e.g., {{\"sub_agent\": \"geo\"}}). Available sub-agents: segmentation, trends, geo, product."),
        ("human", "{input}\n\nMemory: {memory}\nContext: {context}\nSchema: {schema}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "input": state["messages"][-1].content,
        "memory": state["memory"],
        "context": state["context"],
        "schema": state["schema"]
    })
    # Parse LLM response to extract sub-agent (attempt JSON parse first)
    try:
        parsed = json.loads(response.content)
        sub_agent = parsed.get("sub_agent", "geo")
    except json.JSONDecodeError:
        sub_agent = "geo"  # Fallback to keyword heuristic if parse fails
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

# Graph definition
graph = StateGraph(AgentState)
graph.add_node("manager", manager_node)
graph.add_edge("__start__", "manager")

def route_to_subagent(state: AgentState):
    return state["remaining_steps"]

graph.add_conditional_edges("manager", route_to_subagent)

if __name__ == "__main__":
    client = get_bq_client()
    schema = json.dumps(get_schema(client))
    context = get_context(client)
    initial_state = {
        "messages": [HumanMessage(content="Test geo query")],
        "remaining_steps": "",
        "memory": "",
        "schema": schema,
        "context": context
    }
    compiled_graph = graph.compile()
    result = compiled_graph.invoke(initial_state)
    print(f"Delegated to: {result['remaining_steps']}")