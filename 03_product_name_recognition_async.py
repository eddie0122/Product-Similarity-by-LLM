import pandas as pd
import asyncio
import asyncpg

from app.preprocess import recognize_text_async, insert_product_trait_text_async


# Connect to your PostgreSQL database
DB_CONFIG = {
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "pgsql",
    "port": "5432"
}

# LLM service configuration
LLM_CONFIG = {
    "role":
"""You are a helpful fashion assistant.
You can analyze and describe fashion items in users' query. Be concise and accurate.
Your responses should be just one json object with the following keys: category1, category2, color, style, material, and occasion. Each key should have only one value and written in Korean.
If you are unsure about any attribute, respond with "unknown" for that attribute.
# Example response 1:
[
    {
        "category1": "셔츠",
        "category2": "티셔츠",
        "color": "흰색",
        "style": "오버사이즈",
        "material": "면",
        "occasion": "캐쥬얼"
    }
]
# Example response 2:
[
    {
        "category1": "바지",
        "category2": "치노",
        "color": "파랑",
        "style": "슬림핏",
        "material": "실크",
        "occasion": "포멀"
    }
]
""",
    "url": "http://ollama:11434/v1",
    "key": "ollama",
    "model": "ebdm/gemma3-enhanced:12b",
    "temperature": 0.0
}

async def main():
    # Connect to PostgreSQL asynchronously
    db_conn = await asyncpg.connect(**DB_CONFIG)

    # Fetch product data
    query = """
        SELECT prd_id, prd_name
        FROM product_similarity.product_raw
        WHERE prd_img IS NOT NULL;
    """
    rows = await db_conn.fetch(query)
    df_prd = pd.DataFrame(rows, columns=["prd_id", "prd_name"])

    # Process each product
    for prd_id, prd_name in df_prd.itertuples(index=False):
        result_recognize = await recognize_text_async(
            service_url=LLM_CONFIG['url'],
            service_key=LLM_CONFIG['key'],
            service_llm=LLM_CONFIG['model'],
            service_temperature=LLM_CONFIG['temperature'],
            service_role=LLM_CONFIG['role'],
            text_query=prd_name
        )
        if result_recognize.get("status"):
            prd_descs = result_recognize.get("return")
            for prd_desc in prd_descs:
                result_insert = await insert_product_trait_text_async(
                    db_conf=DB_CONFIG,
                    prd_id=prd_id,
                    prd_desc=prd_desc
                )
                if not result_insert.get("status"):
                    print(f"Failed to insert trait for {prd_id}: {result_insert.get('return')}")
        else:
            print(f"Failed to recognize traits for {prd_id}: {result_recognize.get('return')}")

    await db_conn.close()

if __name__ == "__main__":
    asyncio.run(main())
