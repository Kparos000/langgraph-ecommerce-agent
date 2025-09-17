import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage  # New: For proper message format (Handbook Ch. 4: Structured inputs)
from config import get_llm, get_bq_client
from agent import app  # Compiled LangGraph StateGraph
from state import AgentState

# Load .env for API keys
load_dotenv()

def main():
    """CLI interface for e-commerce analysis agent."""
    # Initialize clients with auth validation
    try:
        llm = get_llm()  # Gemini 1.5 Flash
        bq_client = get_bq_client()  # BigQuery client, runs test query
        print("E-Commerce Analysis Agent initialized. Enter a prompt or 'quit' to exit.")
    except Exception as e:
        print(f"Initialization error: {e}. Check .env, API keys, and BigQuery access.")
        return
    # Stateful thread for multi-turn CLI
    thread_id = "1"
    while True:
        # Get user prompt
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == 'quit':
            print("Exiting. Full reports saved to report.md.")
            break
        if not prompt:
            print("Please enter a valid prompt.")
            continue
        # Initialize state for graph invocation
        initial_state: AgentState = {
            "messages": [HumanMessage(content=prompt)],  # Updated: Proper BaseMessage (prevents format/KeyError; Guide p. 9: Consistent messages)
            "df": None,
            "insights": [],
            "next": None,
            "phase": "delegate",
            "report": "",
            "errors": []  # New: Initialize for error tracking (non-breaking, uses get() elsewhere)
        }
        # Invoke graph with stateful config
        config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        try:
            result = app.invoke(initial_state, config)
            report = result.get("report", "No report generated—check prompt relevance.")
            # Print full report (no truncation or labels)
            print(f"\n{report}\nFull report appended to report.md.")
            # Append full report to report.md
            with open("report.md", "a", encoding="utf-8") as f:
                f.write(f"# Analysis for: {prompt}\n{report}\n---\n")
        except Exception as e:
            error_msg = f"Error processing prompt: {e}. Try rephrasing."
            print(error_msg)
            # Append error to report.md
            with open("report.md", "a", encoding="utf-8") as f:
                f.write(f"# Error for: {prompt}\n{error_msg}\n---\n")

if __name__ == "__main__":
    main()