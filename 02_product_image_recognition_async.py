import asyncio
import asyncpg

from app.preprocess import recognize_image_async, insert_product_trait_image_async

# Database configuration
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
You can analyze and describe fashion items in images. Be concise and accurate.
Your responses should be json objects with the following keys: category, color, style, material, and occasion.
If you are unsure about any attribute, respond with "unknown" for that attribute.
# Example response:
[
    {
        "category1": "셔츠",
        "category2": "티셔츠",
        "color": "흰색",
        "style": "오버사이즈",
        "material": "면",
        "occasion": "캐쥬얼"
    },
    {
        "category1": "바지",
        "category2": "치노",
        "color": "파랑",
        "style": "슬림핏",
        "material": "실크",
        "occasion": "포멀"
    },
]
""",
    "url": "http://ollama:11434/v1",
    "key": "ollama",
    "model": "ebdm/gemma3-enhanced:12b",
    "temperature": 0.0
}


async def main():
    conn = await asyncpg.connect(**DB_CONFIG)
    query = """
        SELECT prd_id, prd_img
        FROM product_similarity.product_raw
        WHERE prd_img IS NOT NULL;
    """
    rows = await conn.fetch(query)
    await conn.close()

    for row in rows:
        prd_id = row['prd_id']
        prd_img = row['prd_img']

        result_recognize = await recognize_image_async(
            service_url=LLM_CONFIG['url'],
            service_key=LLM_CONFIG['key'],
            service_llm=LLM_CONFIG['model'],
            service_temperature=LLM_CONFIG['temperature'],
            service_role=LLM_CONFIG['role'],
            img_path=prd_img
        )

        if result_recognize.get('status'):
            prd_descs = result_recognize.get('return')
            for prd_desc in prd_descs:
                result_insert = await insert_product_trait_image_async(
                    user=DB_CONFIG["user"],
                    password=DB_CONFIG["password"],
                    database=DB_CONFIG["database"],
                    host=DB_CONFIG["host"],
                    port=DB_CONFIG["port"],
                    prd_id=prd_id,
                    prd_desc=prd_desc
                )
                if not result_insert.get('status'):
                    print(f"Failed to insert product trait for product ID {prd_id}: {result_insert.get('return')}")
        else:
            print(f"Failed to recognize image for product ID {prd_id}: {result_recognize.get('return')}")

if __name__ == "__main__":
    asyncio.run(main())
