from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
import json
import re
from state import AgentState
from sub_agents import segmentation_agent, trends_agent, geo_agent, product_agent
from config import get_llm

llm = get_llm()

# Improved Manager: Add few-shot for multi-handoff, ambiguity resolution (e.g., "sales" → revenue), clarification for vague
manager_system = """You are ManagerAgent. Classify latest human message for e-commerce analysis (customer segmentation/behavior, sales trends/seasonality, geo patterns, product performance/recs). 'Sales' = revenue ($). If off-topic, output 'next: off_topic' and explain. If vague/ambiguous, ask clarification via message (e.g., 'Revenue or items?'). Otherwise, classify and handoff to 1-4 subs (sequence if multi, e.g., trends first for context).
Output **only** 'next: [agent_name]' or handoff tools; e.g., 'next: trends' or 'Call TrendsAgent then ProductAgent'.
Examples:
- Normal: "Sales trends summer 2024" → next: trends.
- Sparse: "Low sales 2025" → next: trends (future – proxy latest).
- Extreme (Vague): "Sales?" → Clarify: "Revenue or items? Then next: geo".
- Multi: "Winter product recs by country" → Call TrendsAgent (seasonal) then ProductAgent then GeoAgent.
Context: Dataset 2019-2025; LIMIT 1000; no hallucination."""
manager_prompt = ChatPromptTemplate.from_messages([
    ("system", manager_system),
    ("human", "{latest_message}"),
])
manager_chain = manager_prompt | llm

def manager_node(state: AgentState):
    if not state["messages"]:
        state["next"] = "synthesis"
        return state
    latest_message = state["messages"][-1]
    try:
        response = manager_chain.invoke({"latest_message": latest_message.content})
    except Exception as e:
        print(f"Manager chain error: {e}")
        response = AIMessage(content="next: synthesis")
    final_content = response.content.strip()
    if "next: off_topic" in final_content:
        state["next"] = "synthesis"
        state["insights"].append({"agent": "Manager", "text": "Off-topic prompt: Focus on e-commerce analysis."})
        state["messages"] += [response]
        return state
    
    if "next: " in final_content:
        next_agent = final_content.split("next: ")[1].strip()
    else:
        next_agent = "synthesis"
    print(f"Manager delegated to: {next_agent}")
    state["next"] = next_agent
    state["messages"] += [response]
    return state

# Improved Reflection: Add context for ambiguity (e.g., "sales" = revenue)
reflect_system = "You are reflective. Error: {error}. Suggest fix for task in {task_content}. If ambiguous (e.g., 'sales' = revenue $), clarify metric. Output corrected action or 'retry'."
reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", reflect_system),
    ("human", "{task_content}"),
])
reflect_chain = reflect_prompt | llm

def invoke_sub_agent(agent, state, max_retries=3):
    trimmed_state = state.copy()
    trimmed_state["messages"] = state["messages"][-10:]
    trimmed_state["messages"] = [msg if isinstance(msg, BaseMessage) else HumanMessage(content=str(msg)) if msg[0] == "human" else AIMessage(content=str(msg)) for msg in trimmed_state["messages"]]
    for retry in range(max_retries):
        try:
            result = agent.invoke(trimmed_state, config={"recursion_limit": 50})
            last_content = result["messages"][-1].content
            # Robust extraction
            final_match = re.search(r'Final Answer[:\s]*(\[.*?\])', last_content, re.DOTALL)
            if final_match:
                json_str = final_match.group(1)
            else:
                array_match = re.search(r'\[.*?\]', last_content, re.DOTALL)
                json_str = array_match.group() if array_match else None
                if not json_str:
                    try:
                        parsed_full = json.loads(last_content)
                        json_str = json.dumps(parsed_full.get("insights", []))
                    except:
                        json_str = '[]'
            parsed = json.loads(json_str) if json_str else []
            insights_list = parsed if isinstance(parsed, list) else [str(parsed)]
            print(f"Sub-agent succeeded: {insights_list}")
            result["messages"][-1] = AIMessage(content=json.dumps({"insights": insights_list}))
            if "errors" in state:
                state["errors"] += [{"type": "success", "agent": str(agent), "retry": retry}]
            return result, True
        except (ValueError, Exception) as e:
            print(f"Sub-agent failed (retry {retry+1}): {str(e)}")
            if retry < max_retries - 1:
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

# Improved Summary: Add context for long histories, few-shot for summary
summary_system = "Summarize last 10 messages and insights for context. If >5 insights, 2-3 paras with key patterns; 1 para if sparse. No hallucination."
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system),
    ("human", "{context}"),
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

# Improved Synthesis: Add few-shot for ambiguity ("sales" = revenue), recs inference, 2-3 paras if >5 insights
synth_prompt = ChatPromptTemplate.from_template("""
You are SynthesisAgent. Synthesize insights from {insights} for {messages}. 'Sales' = revenue ($). If errors ({errors}), note limitations (e.g., 'Volume vs revenue ambiguity – retry with $').
Output narrative report for executive: Bold figures (**China: $611,205**), no tables/lists unless asked. If data >5 rows, 2-3 paras with patterns/recs (e.g., Q4 peak → stock boost; infer from patterns even if not asked); 1 para if sparse; no hallucination/filler – stick to parsed data.
Examples:
- Normal: Insights ['China $598k top'] → "China led 2023 sales at **$598,779** (25% share). US followed **$402,856** (growth opportunity in Asia)."
- Sparse: Insights ['No 2025 data'] → "No 2025 sales – proxy 2024: Q4 **$508k** low (adjust inventory)."
- Extreme (Ambiguous): Insights ['China 10k items'] → "Note: Items volume; revenue proxy **$598k** for China (top – expand)."
- Multi: Insights ['Q4 peak $508k', 'Coats $300k'] → 3 paras: "Q4 revenue peaked **$508,801** (39% > Q1 – seasonal boost). Coats drove **$300,000** (high velocity – stock Q4 +50%). Rec: Bundle coats for winter campaigns."
Once done, return the report.
""")

def synthesis_node(state: AgentState):
    if not state["insights"]:
        error_summary = json.dumps(state.get("errors", [])) if "errors" in state else "No errors logged."
        state["report"] = f"No insights generated—check manager delegation or sub-agent output. Errors: {error_summary}"
        return state
    trimmed_messages = state["messages"][-5:]
    formatted_insights = json.dumps(state["insights"])
    formatted_errors = json.dumps(state.get("errors", []))
    formatted_prompt = synth_prompt.format(insights=formatted_insights, messages=trimmed_messages, errors=formatted_errors)
    state["report"] = llm.invoke(formatted_prompt).content
    return state

# Graph
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