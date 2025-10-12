from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import pandas as pd
import psycopg2
import asyncio
import os

# DB connection information
DB_CONFIG = {
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "pgsql",
    "port": "5432"
}

db_conn = psycopg2.connect(**DB_CONFIG)
db_cur = db_conn.cursor()

# Fetch product id, product name and product traits from image
query ="""
SELECT category,
    prd_id,
    prd_name,
    CONCAT_WS(
        ' ',
        CASE WHEN text_cat1 = 'unknown' THEN '' ELSE text_cat1 END,
        CASE WHEN text_cat2 = 'unknown' THEN '' ELSE text_cat2 END,
        CASE WHEN text_style = 'unknown' THEN '' ELSE text_style END,
        CASE WHEN text_occasion = 'unknown' THEN '' ELSE text_occasion END
    ) AS prd_trait_text,
    CONCAT_WS(
        ' ',
        CASE WHEN image_cat1 = 'unknown' THEN '' ELSE image_cat1 END,
        CASE WHEN image_cat2 = 'unknown' THEN '' ELSE image_cat2 END,
        CASE WHEN image_style = 'unknown' THEN '' ELSE image_style END,
        CASE WHEN image_occasion = 'unknown' THEN '' ELSE image_occasion END
    ) AS prd_trait_image
FROM product_similarity.products_trait_information;
"""
db_cur.execute(query=query)
rows = db_cur.fetchall()
df_prd = pd.DataFrame(rows, columns=[_[0] for _ in db_cur.description])
db_conn.close()


# Milvus lite inintial setting
os.makedirs('./milvus_db', exist_ok=True)
connections.connect(
    alias="default",
    uri="./milvus_db/product_similarity.db"
)

fields = [
    FieldSchema(
        name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="prd_id", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="prd_text", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="prd_tag", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
]

schema = CollectionSchema(
    fields,
    description="Embedding vector of product names & traits from images and names"
)

collection = Collection("product_embedding", schema)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)


# Load embedding model & prompt
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
instruct = "패션 의류 및 아이템 상품 유사도 분류"
prompt = "Instruct: {}\nQuery: {}"

# Prepare data for Milvus (product name based data)
df_prd_name = df_prd[['prd_id', 'prd_name']].drop_duplicates('prd_id').copy()
prd_name_ids = df_prd_name['prd_id'].tolist()
prd_name_texts = df_prd_name['prd_name'].tolist()
prd_name_prompts =\
    [prompt.format(instruct, _) for _ in df_prd_name['prd_name'].tolist()]

# Prepare data for Milvus (product image based data)
df_prd_img = df_prd[['prd_id', 'prd_trait_image']].drop_duplicates().copy()
prd_img_ids = df_prd_img['prd_id'].tolist()
prd_img_texts = df_prd_img['prd_trait_image'].tolist()
prd_img_prompts =\
    [prompt.format(instruct, _) for _ in df_prd_img['prd_trait_image'].tolist()]

# Prepare data for Milvus (product text(of name) based data)
df_prd_txt = df_prd[['prd_id', 'prd_trait_text']].drop_duplicates().copy()
prd_txt_ids = df_prd_txt['prd_id'].tolist()
prd_txt_texts = df_prd_txt['prd_trait_text'].tolist()
prd_txt_prompts =\
    [prompt.format(instruct, _) for _ in df_prd_txt['prd_trait_text'].tolist()]


# Batch insert into milvus
async def batch_insert(collection, embedding_model, prd_ids, prd_texts, prd_tag, prd_prompts, batch_size=1000):
    loop = asyncio.get_event_loop()

    for i in range(0, len(prd_ids), batch_size):
        batch_ids = prd_ids[i:i+batch_size]
        batch_texts = prd_texts[i:i+batch_size]
        batch_prompts = prd_prompts[i:i+batch_size]
        batch_tags = [prd_tag] * len(batch_ids)

        embeddings = await loop.run_in_executor(None, embedding_model.encode, batch_prompts)
        await loop.run_in_executor(None, collection.insert, [batch_ids, batch_texts, batch_tags, embeddings])
    await loop.run_in_executor(None, collection.flush)


# Run batch insert
async def main():
    await asyncio.gather(
        batch_insert(
            collection=collection,
            embedding_model=embedding_model,
            prd_ids=prd_name_ids,
            prd_texts=prd_name_texts,
            prd_tag='product_name',
            prd_prompts=prd_name_prompts,
        ),
        batch_insert(
            collection=collection,
            embedding_model=embedding_model,
            prd_ids=prd_img_ids,
            prd_texts=prd_img_texts,
            prd_tag='product_image',
            prd_prompts=prd_img_prompts,
        ),
        batch_insert(
            collection=collection,
            embedding_model=embedding_model,
            prd_ids=prd_txt_ids,
            prd_texts=prd_txt_texts,
            prd_tag='product_text',
            prd_prompts=prd_txt_prompts,
        ),
    )
asyncio.run(main())
