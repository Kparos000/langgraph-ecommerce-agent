from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # For consistent handling
import json
import re
from state import AgentState
from sub_agents import segmentation_agent, trends_agent, geo_agent, product_agent
from config import get_llm

llm = get_llm()

# Fixed: Proper messages template for manager (Handbook Ch. 4: Use from_messages for chat history; avoids str() mismatch in JSON mode)
manager_system = """You are ManagerAgent. First, check if the latest human message is relevant to e-commerce analysis (customer segmentation, sales trends, geo patterns, product performance). If off-topic, output "next: off_topic" and explain briefly. Otherwise, classify.

Classify the latest human message.
Output **only** "next: [agent_name]" where agent_name is one of: segmentation, trends, geo, product, or synthesis if done.
Example: For "sales trends", output "next: trends".
Example: For "customer segmentation", output "next: segmentation".
Example: For vague, output "next: synthesis"."""
manager_prompt = ChatPromptTemplate.from_messages([
    ("system", manager_system),
    ("human", "{latest_message}"),  # Extract last for simplicity; full history via state if needed
])
manager_chain = manager_prompt | llm  # Chain serializes correctly (Guide p. 9: Robust for decisions)

def manager_node(state: AgentState):
    if not state["messages"]:
        state["next"] = "synthesis"
        return state
    latest_message = state["messages"][-1]  # Last is HumanMessage
    try:
        response = manager_chain.invoke({"latest_message": latest_message.content})
    except Exception as e:
        print(f"Manager chain error: {e}")  # Debug isolation
        response = AIMessage(content="next: synthesis")  # Fallback (PRD: Graceful degradation)
    final_content = response.content.strip()
    # Handle off-topic (Guide p. 24: Halt/transfer)
    if "next: off_topic" in final_content:
        state["next"] = "synthesis"
        state["insights"].append({"agent": "Manager", "text": "Off-topic prompt: Focus on e-commerce analysis."})
        state["messages"] += [response]
        return state
    
    if "next: " in final_content:
        next_agent = final_content.split("next: ")[1].strip()
    else:
        next_agent = "synthesis"
    print(f"Manager delegated to: {next_agent}")  # Debug
    state["next"] = next_agent
    state["messages"] += [response]
    return state

# Fixed: Reflection chain as messages template (prevents same str() issue in retries)
reflect_system = "You are a reflective agent. Error in previous step: {error}. Suggest fix for the task."
reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", reflect_system),
    ("human", "{task_content}"),  # Task from last message
])
reflect_chain = reflect_prompt | llm

def invoke_sub_agent(agent, state, max_retries=3):
    """Direct invoke sub-agent with higher recursion limit, extract Final Answer JSON."""
    trimmed_state = state.copy()
    trimmed_state["messages"] = state["messages"][-10:]
    # Ensure BaseMessage (non-breaking)
    trimmed_state["messages"] = [msg if isinstance(msg, BaseMessage) else HumanMessage(content=str(msg)) if msg[0] == "human" else AIMessage(content=str(msg[1])) for msg in trimmed_state["messages"]]
    for retry in range(max_retries):
        try:
            result = agent.invoke(trimmed_state, config={"recursion_limit": 50})
            last_content = result["messages"][-1].content
            # Extract Final Answer JSON
            final_match = re.search(r'Final Answer[:\s]*(\[.*?\])', last_content, re.DOTALL)
            if final_match:
                json_str = final_match.group(1)
            else:
                array_match = re.search(r'\[.*?\]', last_content, re.DOTALL)
                json_str = array_match.group() if array_match else '[]'
            parsed = json.loads(json_str)
            insights_list = parsed if isinstance(parsed, list) else [str(parsed)]
            print(f"Sub-agent succeeded: {insights_list}")
            result["messages"][-1] = AIMessage(content=json.dumps({"insights": insights_list}))
            if "errors" in state:
                state["errors"] += [{"type": "success", "agent": str(agent), "retry": retry}]
            return result, True
        except (ValueError, Exception) as e:
            print(f"Sub-agent failed (retry {retry+1}): {str(e)}")
            if retry < max_retries - 1:
                # Fixed: Use chain for reflection (Handbook Ch. 4: Adaptation via structured prompts)
                task_content = trimmed_state["messages"][-1].content
                try:
                    reflection = reflect_chain.invoke({"error": str(e), "task_content": task_content})
                except:
                    reflection = AIMessage(content="Retry with corrected SQL.")
                trimmed_state["messages"] += [reflection]
                continue
            print(f"Sub-agent failed after {max_retries} retries: {str(e)}")
            fallback_insights = ["Fallback: Unable to generate insights after retries."]
            fallback_result = {"messages": trimmed_state["messages"] + [AIMessage(content=json.dumps({"insights": fallback_insights}))]}
            if "errors" in state:
                state["errors"] += [{"type": "fallback", "agent": str(agent), "error": str(e), "retries": max_retries}]
            return fallback_result, False

def segmentation_node(state: AgentState):
    result, success = invoke_sub_agent(segmentation_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Segmentation", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def trends_node(state: AgentState):
    result, success = invoke_sub_agent(trends_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Trends", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def geo_node(state: AgentState):
    result, success = invoke_sub_agent(geo_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Geo", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def product_node(state: AgentState):
    result, success = invoke_sub_agent(product_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Product", "text": json.dumps(parsed["insights"])})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

# Fixed: Summary chain as messages (same serialization fix)
summary_system = "Summarize the last 10 messages and insights for context."
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system),
    ("human", "{context}"),  # Combined messages/insights
])
summary_chain = summary_prompt | llm

def summarizer_node(state: AgentState):
    if len(state["messages"]) > 10:
        context = f"Messages: {state['messages'][-10:]} Insights: {json.dumps(state['insights'])}"
        try:
            summary = summary_chain.invoke({"context": context})
        except:
            summary = AIMessage(content="Summary unavailable.")
        state["messages"] += [summary]
    return state

def synthesis_node(state: AgentState):
    if not state["insights"]:
        error_summary = json.dumps(state.get("errors", [])) if "errors" in state else "No errors logged."
        state["report"] = f"No insights generated—check manager delegation or sub-agent output. Errors: {error_summary}"
        return state
    trimmed_messages = state["messages"][-5:]
    synth_prompt = ChatPromptTemplate.from_template("""
    Synthesize insights from {insights} for the prompt in {messages}. If errors occurred ({errors}), note limitations (e.g., 'Query failed due to syntax—retry suggested').
    Output in a narrative report style for an executive audience: Use paragraphs with bold figures (e.g., **China: $611,205**), no tables, lists, or bullets. Follow the prompt strictly; do not give recommendations unless explicitly asked. Example for "Which two countries had the lowest sales in 2020?":
    Lowest Sales Countries in 2020
    In 2020, our lowest sales regions highlighted areas for potential growth, with distinct patterns in purchasing behavior.
    Austria: The lowest performer was Austria, generating **$154** in sales, primarily from low-volume urban purchases.
    Colombia: Colombia followed with **$409.96** in sales, skewed toward seasonal items in coastal areas.
    Across these regions, sales were concentrated in entry-level products, with limited repeat purchases from urban demographics.
    Once done, return the report.
    """)
    formatted_insights = json.dumps(state["insights"])
    formatted_errors = json.dumps(state.get("errors", []))
    formatted_prompt = synth_prompt.format(insights=formatted_insights, messages=trimmed_messages, errors=formatted_errors)
    state["report"] = llm.invoke(formatted_prompt).content
    return state

# Graph unchanged
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager_node)
workflow.add_node("segmentation", segmentation_node)
workflow.add_node("trends", trends_node)
workflow.add_node("geo", geo_node)
workflow.add_node("product", product_node)
workflow.add_node("summarizer", summarizer_node)
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