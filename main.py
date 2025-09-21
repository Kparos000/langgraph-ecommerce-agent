import json
from langchain_core.messages import HumanMessage
from agent import compiled_graph
from config import get_bq_client, get_schema, get_context

def run_cli():
    client = get_bq_client()
    schema = json.dumps(get_schema(client))
    context = get_context(client)

    print("=== LangGraph E-commerce Agent CLI ===")
    print("Ask me questions about the dataset (bigquery-public-data.thelook_ecommerce).")
    print("Type 'exit' to quit.\n")

    # Persistent thread_id keeps memory across turns
    thread_id = "cli_session_1"

    while True:
        query = input("Query: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("Exiting CLI. Goodbye!")
            break

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "remaining_steps": "",
            "memory": "",
            "schema": schema,
            "context": context
        }

        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = compiled_graph.invoke(initial_state, config=config)
            print("\n--- Report ---")
            print(result["messages"][-1].content)
            print("--------------\n")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_cli()
