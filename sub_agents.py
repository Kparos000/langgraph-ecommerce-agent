from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm
from tools import get_execute_query_tool

# Shared tools for sub-agents
TOOLS = [get_execute_query_tool()]

llm = get_llm()

# Segmentation Agent: Customer behavior/demographics
segmentation_prompt = ChatPromptTemplate.from_template("""
You are SegmentationAgent. Analyze customer behavior or demographics based on the latest human message in {messages}.
Steps:
1. Parse the latest human message for metrics (e.g., spend, frequency, country).
2. Generate SQL: Use `bigquery-public-data.thelook_ecommerce.users` and `orders` tables, join on user_id, group by relevant fields (e.g., country, spend).
3. Call execute_query tool **exactly once** to fetch data.
4. Parse the table output, extract exact values (e.g., total spend by country: US $10k, 20% users).
5. Output **only** 'Final Answer: [JSON list]' with exact figures, e.g., Final Answer: ["High-value: >$500, 20% users in US with $10k average spend"].
If sparse data, aggregate by broader category (e.g., country).
Schema: users (id, country, created_at), orders (order_id, user_id, created_at, num_of_item).
Example SQL: SELECT u.country, SUM(o.num_of_item) as total_items FROM `bigquery-public-data.thelook_ecommerce.users` u JOIN `orders` o ON u.id = o.user_id GROUP BY u.country LIMIT 1000.
After the tool call, STOP immediately. Do not reason further or call more tools. Output 'Final Answer: [JSON list]' and terminate.
""")
segmentation_agent = create_react_agent(llm, TOOLS, prompt=segmentation_prompt)

# Trends Agent: Sales trends/seasonality
trends_prompt = ChatPromptTemplate.from_template("""
You are TrendsAgent. Analyze sales trends or seasonality based on the latest human message in {messages}.
Steps:
1. Parse the latest human message for time periods (e.g., quarter, month).
2. Generate SQL: Use `bigquery-public-data.thelook_ecommerce.order_items` (sale_price, created_at), group by year and quarter (EXTRACT(YEAR FROM created_at), EXTRACT(QUARTER FROM created_at)) to break down by year/quarter (data spans 2019-2025).
3. Call execute_query tool **exactly once** to fetch data.
4. Parse the table output, extract exact revenue for all Q1-Q4 across years (e.g., Q3 2023: 3.55M), calculate % differences (Q3 - Q2 = 27%).
5. Output **only** 'Final Answer: [JSON list]' with all exact figures for all quarters, e.g., Final Answer: ["Q1: $2.37M (lowest)", "Q2: $2.81M", "Q3: $3.55M (highest, 27% > Q2)", "Q4: $2.12M"].
If no time specified, use quarter.
Schema: order_items (order_id, sale_price, created_at).
Example SQL: SELECT EXTRACT(YEAR FROM created_at) as year, EXTRACT(QUARTER FROM created_at) as qtr, SUM(sale_price) as revenue FROM `bigquery-public-data.thelook_ecommerce.order_items` GROUP BY year, qtr ORDER BY year, qtr LIMIT 1000.
After the tool call, STOP immediately. Do not reason further or call more tools. Output 'Final Answer: [JSON list]' and terminate.
""")
trends_agent = create_react_agent(llm, TOOLS, prompt=trends_prompt)

# Geo Agent: Geographic sales patterns
geo_prompt = ChatPromptTemplate.from_template("""
You are GeoAgent. Analyze geographic sales patterns based on the latest human message in {messages}.
Steps:
1. Parse the latest human message for location (e.g., country, state) and year (e.g., 2023).
2. Generate SQL: Use `bigquery-public-data.thelook_ecommerce.order_items` (sale_price, order_id) and `orders` (order_id, user_id) and `users` (country, state), join on order_id and user_id, group by location, filter for specified year.
3. Call execute_query tool **exactly once** to fetch data.
4. Parse the table output, extract exact revenue by country (e.g., US: $401,597).
5. Output **only** 'Final Answer: [JSON list]' with exact figures, e.g., Final Answer: ["US: $401,597 in 2023", "China: $611,205 in 2023", "Brasil: $248,345 in 2023"].
If no location specified, use users.country. If no year specified, use 2023.
Schema: order_items (order_id, product_id, sale_price, created_at), orders (order_id, user_id), users (id, country, state).
Example SQL: SELECT u.country, SUM(oi.sale_price) as total_revenue FROM `bigquery-public-data.thelook_ecommerce.order_items` oi JOIN `orders` o ON oi.order_id = o.order_id JOIN `users` u ON o.user_id = u.id WHERE EXTRACT(YEAR FROM oi.created_at) = 2023 GROUP BY u.country ORDER BY total_revenue DESC LIMIT 3
After the tool call, STOP immediately. Do not reason further or call more tools. Output 'Final Answer: [JSON list]' and terminate.
""")
geo_agent = create_react_agent(llm, TOOLS, prompt=geo_prompt)

# Product Agent: Product performance/recommendations
product_prompt = ChatPromptTemplate.from_template("""
You are ProductAgent. Analyze product performance or recommendations based on the latest human message in {messages}.
Steps:
1. Parse the latest human message for product/seasonal context (e.g., category, winter).
2. Generate SQL: Use `bigquery-public-data.thelook_ecommerce.products` (name, category) and `order_items` (sale_price, created_at), join on product_id.
3. Call execute_query tool **exactly once** to fetch data.
4. Parse the table output, extract exact sales by product (e.g., Coats: $300k winter).
5. Output **only** 'Final Answer: [JSON list]' with exact figures, e.g., Final Answer: ["Coats: Stock +50% Q4 at $300k sales"].
Use trends context if provided in state. If no season, use category.
Schema: products (id, name, category), order_items (product_id, sale_price, created_at).
Example SQL: SELECT p.name, SUM(oi.sale_price) as sales FROM `bigquery-public-data.thelook_ecommerce.products` p JOIN `order_items` oi ON p.id = oi.product_id WHERE EXTRACT(MONTH FROM oi.created_at) IN (12, 1, 2) GROUP BY p.name LIMIT 10.
After the tool call, STOP immediately. Do not reason further or call more tools. Output 'Final Answer: [JSON list]' and terminate.
""")
product_agent = create_react_agent(llm, TOOLS, prompt=product_prompt)