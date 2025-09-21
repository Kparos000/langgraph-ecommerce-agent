import json
from langchain_core.messages import HumanMessage
from agent import compiled_graph
from config import get_bq_client, get_schema, get_context

def run_query(query: str):
    client = get_bq_client()
    schema = json.dumps(get_schema(client))
    context = get_context(client)
    state = {
        "messages": [HumanMessage(content=query)],
        "remaining_steps": "",
        "memory": "",
        "schema": schema,
        "context": context
    }
    config = {"configurable": {"thread_id": "pytest"}}
    return compiled_graph.invoke(state, config=config)

def test_valid_query():
    result = run_query("Analyze sales trends in US in 2022")
    final_report = result["messages"][-1].content
    assert "Metrics" in final_report
    assert "Analysis" in final_report
    assert "Recommendations" in final_report
    assert "please provide the sql query" not in final_report.lower()

def test_invalid_table():
    result = run_query("Analyze sales in Canada in 2020")
    final_report = result["messages"][-1].content
    assert ("No data fetched" in final_report) or ("not trained" in final_report.lower())

def test_invalid_sql():
    result = run_query("DROP TABLE orders")
    final_report = result["messages"][-1].content
    assert ("No data fetched" in final_report) or ("not trained" in final_report.lower())
