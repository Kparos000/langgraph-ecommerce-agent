import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import logging  # New: Stdlib for traces (Huyen Ch. 6: Observability)
from config import get_llm, get_bq_client
from agent import app
from state import AgentState

# Load .env for API keys
load_dotenv()

def main():
    """CLI interface for e-commerce analysis agent."""
    # New: Setup logging (console + file; aligns with Huyen Ch. 6 observability)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s %(name)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/agent.log"),  # File for persistence
            logging.StreamHandler()  # Console for CLI
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize clients with auth validation
    try:
        llm = get_llm()  # Gemini 1.5 Flash
        bq_client = get_bq_client()  # BigQuery client, runs test query
        logger.info("E-Commerce Analysis Agent initialized. Enter a prompt or 'quit' to exit.")  # Updated: Logger replaces print
    except Exception as e:
        logger.error(f"Initialization error: {e}. Check .env, API keys, and BigQuery access.")  # Updated: Logger
        return
    # Stateful thread for multi-turn CLI
    thread_id = "1"
    while True:
        # Get user prompt
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == 'quit':
            logger.info("Exiting. Full reports saved to report.md.")  # Updated: Logger
            break
        if not prompt:
            logger.warning("Please enter a valid prompt.")  # Updated: Logger
            continue
        # Initialize state for graph invocation
        initial_state: AgentState = {
            "messages": [HumanMessage(content=prompt)],
            "df": None,
            "insights": [],
            "next": None,
            "phase": "delegate",
            "report": "",
            "errors": []
        }
        # Invoke graph with stateful config
        config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        try:
            result = app.invoke(initial_state, config)
            report = result.get("report", "No report generated—check prompt relevance.")
            # Print full report (no truncation or labels) - Keep print for CLI visibility, but log
            print(f"\n{report}\nFull report appended to report.md.")
            logger.info(f"Report generated for prompt: {prompt[:50]}...")  # New: Log success
            # Append full report to report.md
            with open("report.md", "a", encoding="utf-8") as f:
                f.write(f"# Analysis for: {prompt}\n{report}\n---\n")
        except Exception as e:
            error_msg = f"Error processing prompt: {e}. Try rephrasing."
            print(error_msg)
            logger.error(f"Invocation error: {e}")  # New: Log exception
            # Append error to report.md
            with open("report.md", "a", encoding="utf-8") as f:
                f.write(f"# Error for: {prompt}\n{error_msg}\n---\n")

if __name__ == "__main__":
    main()