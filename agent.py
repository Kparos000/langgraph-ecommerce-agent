from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
import json
import re
from state import AgentState
from sub_agents import segmentation_agent, trends_agent, geo_agent, product_agent
from config import get_llm
import logging
from langchain_core.callbacks import BaseCallbackHandler

llm = get_llm()
logger = logging.getLogger(__name__)

# Custom callback for node/tool traces
class TraceCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized, *, run_id, **kwargs):
        node = serialized.get("name", "unknown")
        input_keys = list(serialized.get('input', {}).keys()) if isinstance(serialized.get('input'), dict) else []
        logger.info(f"Starting {node}: Input keys={input_keys}")

    def on_chain_end(self, output, **kwargs):
        node = kwargs.get("name", "unknown")
        insights_len = len(output.get("insights", [])) if isinstance(output, dict) else 0
        logger.info(f"Ended {node}: Output insights len={insights_len}")

# Improved Manager: Enhanced for multi-period (quarters + seasons), country handling, and clarification with key definitions
manager_system = """You are ManagerAgent. Classify latest human message for e-commerce analysis (customer segmentation/behavior, sales trends/seasonality, geo patterns, product performance/recs). 'Sales' = revenue ($). Use the following context for all decisions:
1. Sex: The dataset includes two categories — Male and Female.
2. Age (Customer Segments): Customer ages are grouped as follows: <18 (Children and teens, 0–17 years), 18–24 (Young adults, including late teens and college-aged), 25–34 (Millennials and young professionals), 35–54 (Gen X and older Millennials; mature adults), >54 (Seniors and retirees).
3. Geographic Location: Map countries to regions: North America (United States, Canada, Mexico), South America (Brazil, Argentina, Chile, Colombia, etc.), EMEA (Europe: UK, Germany, France, etc., Middle East: UAE, Saudi Arabia, Israel, Africa: South Africa, Egypt, Nigeria, etc.), Asia Pacific (APAC: China, India, Japan, Australia, Singapore), China (tracked separately). Dataset countries: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States. If a country is not in the dataset, note it and analyze only valid countries, listing all dataset countries.
4. Season (Four Seasons and Financial Quarters): Use meteorological definitions: Spring (Q1 partial, Mar-May), Summer (Q2, Jun-Aug), Autumn (Fall) (Q3, Sep-Nov), Winter (Q4 partial, Dec-Feb).
5. Exact Countries in the Dataset: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States.
If off-topic, output 'next: off_topic' and explain. If vague/ambiguous (e.g., no period or country), ask clarification via message (e.g., 'Specify period: Q1 (Mar-May) or summer (Jun-Aug)? Specify country from: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States.'). Otherwise, classify and handoff to 1-4 subs (sequence if multi, e.g., trends first for context).
Output **only** 'next: [agent_name]' or handoff tools; e.g., 'next: trends' or 'Call TrendsAgent then ProductAgent'.
Examples:
- Normal: "Sales trends summer 2024" → next: trends.
- Sparse: "Low sales 2025" → next: trends (future – proxy latest).
- Extreme (Vague): "Sales?" → Clarify: "Revenue or items? Specify period: Q1 (Mar-May) or summer (Jun-Aug)? Specify country from: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States."
- Multi (Quarters): "Sales Q1 2022 and Q3 2023" → next: trends.
- Multi (Season+Quarter): "Sales summer 2023 and Q1 2022" → next: trends.
- Geo with Invalid Country: "Sales in Nigeria and China" → next: geo with message: "Nigeria not in dataset; analyzing China. Available countries: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States."
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
        logger.error(f"Manager chain error: {e}")
        response = AIMessage(content="next: synthesis")
    final_content = response.content.strip()
    if "next: off_topic" in final_content:
        state["next"] = "synthesis"
        state["insights"].append({"agent": "Manager", "text": "Off-topic prompt: Focus on e-commerce analysis."})
        state["messages"] += [response]
        logger.warning("Off-topic delegation to synthesis")
        return state
    
    if "Clarify:" in final_content:
        state["next"] = "synthesis"
        state["messages"] += [response]
        logger.info("Manager requested clarification")
        return state
    
    if "next: " in final_content:
        next_agent = final_content.split("next: ")[1].strip()
    else:
        next_agent = "synthesis"
    logger.info(f"Manager delegated to: {next_agent}")
    state["next"] = next_agent
    state["messages"] += [response]
    return state

# Reflection: Enhanced for ambiguity with seasons, ages, and countries
reflect_system = """You are reflective. Error: {error}. Suggest fix for task in {task_content}. Use context:
1. Sex: The dataset includes two categories — Male and Female.
2. Age (Customer Segments): Customer ages are grouped as follows: <18 (Children and teens, 0–17 years), 18–24 (Young adults, including late teens and college-aged), 25–34 (Millennials and young professionals), 35–54 (Gen X and older Millennials; mature adults), >54 (Seniors and retirees).
3. Geographic Location: Map countries to regions: North America (United States, Canada, Mexico), South America (Brazil, Argentina, Chile, Colombia, etc.), EMEA (Europe: UK, Germany, France, etc., Middle East: UAE, Saudi Arabia, Israel, Africa: South Africa, Egypt, Nigeria, etc.), Asia Pacific (APAC: China, India, Japan, Australia, Singapore), China (tracked separately). Dataset countries: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States.
4. Season (Four Seasons and Financial Quarters): Use meteorological definitions: Spring (Q1 partial, Mar-May), Summer (Q2, Jun-Aug), Autumn (Fall) (Q3, Sep-Nov), Winter (Q4 partial, Dec-Feb).
5. Exact Countries in the Dataset: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States.
If ambiguous (e.g., 'sales' = revenue $, 'summer' = Jun-Aug, or invalid country), clarify metric/period/country. Output corrected action or 'retry'."""
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
            
            cot_match = re.search(r'Reasoning[:\s]*([^.\n]*?)(?=\n|Final|$)', last_content, re.DOTALL | re.IGNORECASE)
            cot = cot_match.group(1).strip() if cot_match else ""
            parsed_insights_with_cot = [{"text": i, "cot": cot} for i in insights_list] if isinstance(insights_list, list) else [{"text": str(insights_list), "cot": cot}]
            parsed = {"insights": parsed_insights_with_cot}
            
            logger.info(f"Sub-agent succeeded: {len(parsed_insights_with_cot)} insights; CoT excerpt: {cot[:50]}...")
            result["messages"][-1] = AIMessage(content=json.dumps(parsed))
            if "errors" in state:
                state["errors"] += [{"type": "success", "agent": str(agent), "retry": retry}]
            return result, True
        except (ValueError, Exception) as e:
            logger.warning(f"Sub-agent failed (retry {retry+1}): {str(e)}")
            if retry < max_retries - 1:
                task_content = trimmed_state["messages"][-1].content
                try:
                    reflection = reflect_chain.invoke({"error": str(e), "task_content": task_content})
                    logger.info(f"Reflection generated: {reflection.content[:50]}...")
                except:
                    reflection = AIMessage(content="Retry with corrected SQL.")
                trimmed_state["messages"] += [reflection]
                continue
            logger.error(f"Sub-agent failed after {max_retries} retries: {str(e)}")
            fallback_insights = ["Fallback: Unable to generate insights after retries."]
            fallback_result = {"messages": trimmed_state["messages"] + [AIMessage(content=json.dumps({"insights": [{"text": fallback_insights[0], "cot": ""}]}))]}
            if "errors" in state:
                state["errors"] += [{"type": "fallback", "agent": str(agent), "error": str(e), "retries": max_retries}]
            return fallback_result, False

def segmentation_node(state: AgentState):
    result, success = invoke_sub_agent(segmentation_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Segmentation", "text": [i["text"] for i in parsed["insights"]], "cot": parsed["insights"][0]["cot"] if parsed["insights"] else ""})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def trends_node(state: AgentState):
    result, success = invoke_sub_agent(trends_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Trends", "text": [i["text"] for i in parsed["insights"]], "cot": parsed["insights"][0]["cot"] if parsed["insights"] else ""})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def geo_node(state: AgentState):
    result, success = invoke_sub_agent(geo_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Geo", "text": [i["text"] for i in parsed["insights"]], "cot": parsed["insights"][0]["cot"] if parsed["insights"] else ""})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

def product_node(state: AgentState):
    result, success = invoke_sub_agent(product_agent, state, max_retries=3)
    if success:
        parsed = json.loads(result["messages"][-1].content)
        state["insights"].append({"agent": "Product", "text": [i["text"] for i in parsed["insights"]], "cot": parsed["insights"][0]["cot"] if parsed["insights"] else ""})
    state["next"] = "synthesis"
    state["messages"] = result["messages"][-5:]
    return state

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

synth_prompt = ChatPromptTemplate.from_template("""
You are SynthesisAgent. Synthesize insights from {insights} for {messages}. 'Sales' = revenue ($). Use context:
1. Sex: The dataset includes two categories — Male and Female.
2. Age (Customer Segments): Customer ages are grouped as follows: <18 (Children and teens, 0–17 years), 18–24 (Young adults, including late teens and college-aged), 25–34 (Millennials and young professionals), 35–54 (Gen X and older Millennials; mature adults), >54 (Seniors and retirees).
3. Geographic Location: Map countries to regions: North America (United States, Canada, Mexico), South America (Brazil, Argentina, Chile, Colombia, etc.), EMEA (Europe: UK, Germany, France, etc., Middle East: UAE, Saudi Arabia, Israel, Africa: South Africa, Egypt, Nigeria, etc.), Asia Pacific (APAC: China, India, Japan, Australia, Singapore), China (tracked separately). Dataset countries: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States.
4. Season (Four Seasons and Financial Quarters): Use meteorological definitions: Spring (Q1 partial, Mar-May), Summer (Q2, Jun-Aug), Autumn (Fall) (Q3, Sep-Nov), Winter (Q4 partial, Dec-Feb).
5. Exact Countries in the Dataset: Australia, Austria, Belgium, Brasil, China, Colombia, España, France, Germany, Japan, Poland, South Korea, Spain, United Kingdom, United States.
If errors ({errors}), note limitations (e.g., 'Volume vs revenue ambiguity – retry with $').
Output narrative report for executive: Bold figures (**China: $611,205**), no tables/lists unless asked. If data >5 rows, 2-3 paras with patterns/recs (e.g., Q4 peak → stock boost; infer from patterns even if not asked); 1 para if sparse; no hallucination/filler – stick to parsed data.
Include reasoning if present: {cot}. For Q1, note it covers Mar-May; for summer, note it covers Jun-Aug; for Q3, note it covers Sep-Nov; for Q4, note it covers Dec-Feb.
Examples:
- Normal: Insights ['China $598k top'] → "China led 2023 sales at **$598,779** (25% share, Jun-Aug). US followed **$402,856** (growth opportunity in Asia)."
- Sparse: Insights ['No 2025 data'] → "No 2025 sales – proxy 2024: Q4 **$508k** low (Dec-Feb, adjust inventory)."
- Extreme (Ambiguous): Insights ['China 10k items'] → "Note: Items volume; revenue proxy **$598k** for China (top – expand, Jun-Aug)."
- Multi: Insights ['Q4 peak $508k', 'Coats $300k'] → 3 paras: "Q4 revenue peaked **$508,801** (39% > Q1 Mar-May – seasonal boost). Coats drove **$300,000** (high velocity – stock Dec-Feb +50%). Rec: Bundle coats for winter campaigns."
Once done, return the report.
""")

def synthesis_node(state: AgentState):
    if not state["insights"]:
        error_summary = json.dumps(state.get("errors", [])) if "errors" in state else "No errors logged."
        state["report"] = f"No insights generated—check manager delegation or sub-agent output. Errors: {error_summary}"
        return state
    trimmed_messages = state["messages"][-5:]
    formatted_insights = json.dumps([{k: v for k, v in i.items() if k != 'cot'} for i in state["insights"]])
    formatted_errors = json.dumps(state.get("errors", []))
    formatted_cot = '\n'.join([i.get('cot', '') for i in state["insights"] if i.get('cot')])
    formatted_prompt = synth_prompt.format(insights=formatted_insights, messages=trimmed_messages, errors=formatted_errors, cot=formatted_cot)
    try:
        state["report"] = llm.invoke(formatted_prompt).content + f"\n\n## Reasoning Trace\n{formatted_cot}"
        logger.info("Synthesis complete with CoT weave")
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        state["report"] = "Synthesis failed—raw insights: " + formatted_insights
    return state

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