import psycopg2
import pandas as pd

from app.preprocess import recognize_image, insert_product_trait_image


# Connect to your PostgreSQL database
db_conn = psycopg2.connect(
    dbname="mydb",
    user="myuser",
    password="mypassword",
    host="ollama-postgresql-1",
    port="5432"
)
db_cur = db_conn.cursor()

# Fetch product id and path of images from the database
query =\
    """
    SELECT prd_id,
        prd_img
    FROM product_similarity.product_raw
    WHERE prd_img IS NOT NULL;
    """
db_cur.execute(query=query)
rows = db_cur.fetchall()
df_prd = pd.DataFrame(rows, columns=[_[0] for _ in db_cur.description])

# LLM service configuration
llm_role =\
    """
    You are a helpful fashion assistant.
    You can analyze and describe fashion items in images. Be concise and accurate.
    Your responses should be json objects with the following keys: category, color, style, material, and occasion.
    If you are unsure about any attribute, respond with "unknown" for that attribute.
    # Example response:
    [
        {
            "category1": "shirt",
            "category2": "t-shirt",
            "color": "white",
            "style": "oversized",
            "material": "cotton",
            "occasion": "casual"
        },
        {
            "category1": "pants",
            "category2": "chinos",
            "color": "blue",
            "style": "slim fit",
            "material": "silk",
            "occasion": "formal"
        },
    ]
    """
llm_url = "http://ollama:11434/v1"
llm_key = "ollama"
llm_model = "ebdm/gemma3-enhanced:12b"
llm_temperature = 0.0

# Process each product image and insert the recognized traits into the database
for prd_id, prd_img in df_prd.itertuples(index=False):
    result_recognize = recognize_image(
        service_url=llm_url,
        service_key=llm_key,
        service_llm=llm_model,
        service_temperature=llm_temperature,
        service_role=llm_role,
        img_path=prd_img
    )
    if result_recognize.get('status'):
        prd_descs = result_recognize.get('return')
        for prd_desc in prd_descs:
            result_insert = insert_product_trait_image(
                db_cur=db_cur,
                prd_id=prd_id,
                prd_desc=prd_desc
            )
            if not result_insert.get('status'):
                print(
                    f"Failed to insert product trait for product ID {prd_id}: {result_insert.get('return')}")
    else:
        print(
            f"Failed to recognize image for product ID {prd_id}: {result_recognize.get('return')}")

db_conn.close()