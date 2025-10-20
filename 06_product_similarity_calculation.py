import asyncio
from pymilvus import connections, Collection
import psycopg2
import psycopg2.extras

# Variable to store the connection to Milvus DB
DB_CONFIG = {
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "pgsql",
    "port": "5432"
}

milvus_uri = "./milvus_db/product_similarity.db"
collection_name = "product_embedding"

batch_size = 500
max_concurrency = 20

def _get_embedding(collection, prd_id, prd_tag='product_name'):
    return collection.query(
        expr=f'prd_tag == "{prd_tag}" and prd_id == "{prd_id}"',
        output_fields=["embedding"]
    )

def _calculate_similarity(collection, prd_id, embedding, prd_tag='product_image', top_k=10):
    return collection.search(
        expr=f'prd_tag == "{prd_tag}" and prd_id == "{prd_id}"',
        anns_field="embedding",
        data=[embedding],
        param={"metric_type": "COSINE", "params": {"nprobe": 8}},
        output_fields=["prd_id", "prd_text", "prd_tag"],
        limit=top_k
    )

def _get_product_similarity_inner(collection, prd_id):
    result = _get_embedding(collection, prd_id, 'product_name')
    embedding_prd_name = result[0].get('embedding')

    result = _get_embedding(collection, prd_id, 'product_text')
    embedding_prd_text = result[0].get('embedding')

    result = _calculate_similarity(collection, prd_id, embedding_prd_name, 'product_text')
    similarity_name_text = max([_.get('distance') for _ in result[0]])

    result = _calculate_similarity(collection, prd_id, embedding_prd_name, 'product_image')
    similarity_name_image = max([_.get('distance') for _ in result[0]])

    result = _calculate_similarity(collection, prd_id, embedding_prd_text, 'product_image')
    similarity_text_image = max([_.get('distance') for _ in result[0]])

    return (prd_id, similarity_name_text, similarity_name_image, similarity_text_image)

def insert_batch_similarities(db_cursor, similarities):
    query = """
       INSERT INTO product_similarity.products_similarity_score_inner
       (prd_id, similarity_name_text, similarity_name_image, similarity_text_image)
       VALUES %s"""
    psycopg2.extras.execute_values(
        db_cursor, query, similarities
    )

# ──────────────────────────────────────────────
# ASYNC WRAPPERS
# ──────────────────────────────────────────────

async def get_product_similarity_async(collection, prd_id):
    return await asyncio.to_thread(_get_product_similarity_inner, collection, prd_id)

async def insert_batch_async(db_cursor, similarities):
    await asyncio.to_thread(insert_batch_similarities, db_cursor, similarities)

# ──────────────────────────────────────────────
# ASYNC BATCH PROCESSOR
# ──────────────────────────────────────────────

async def process_batch(collection, db_cursor, prd_ids, semaphore):
    async def process_one(prd_id):
        async with semaphore:
            return await get_product_similarity_async(collection, prd_id)

    tasks = [process_one(pid) for pid in prd_ids]
    results = await asyncio.gather(*tasks)
    await insert_batch_async(db_cursor, results)

# ──────────────────────────────────────────────
# MAIN LOGIC
# ──────────────────────────────────────────────

async def main():
    # Connect to DB
    db_conn = psycopg2.connect(**DB_CONFIG)
    db_cur = db_conn.cursor()

    # Load product IDs
    db_cur.execute("""
        SELECT DISTINCT prd_id
        FROM product_similarity.products_trait_information
    """)
    prd_ids = [row[0] for row in db_cur.fetchall()]

    # Connect to Milvus
    connections.connect("default", uri=milvus_uri)
    collection = Collection(collection_name)
    collection.load()

    # Create batches
    semaphore = asyncio.Semaphore(max_concurrency)
    batches = [prd_ids[i:i + batch_size] for i in range(0, len(prd_ids), batch_size)]

    for batch_num, batch in enumerate(batches, start=1):
        print(f"Processing batch {batch_num}/{len(batches)} ...")
        await process_batch(collection, db_cur, batch, semaphore)
        db_conn.commit()  # Commit after each batch
        print(f"Batch {batch_num} committed.")

    # Cleanup
    db_cur.close()
    db_conn.close()
    print("All done.")

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
