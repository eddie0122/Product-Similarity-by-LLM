import asyncio
import psycopg2
import psycopg2.extras
from pymilvus import connections, Collection
from app.embedding import get_product_similarity_inner, insert_batch_similarities


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


# Asynchronous wrapper
async def get_product_similarity_async(collection, prd_id):
    return await asyncio.to_thread(get_product_similarity_inner, collection, prd_id)


async def insert_batch_async(db_cursor, similarities):
    await asyncio.to_thread(insert_batch_similarities, db_cursor, similarities)

# Asynchronous process (similarity calculation and insertion)
async def process_batch(collection, db_cursor, prd_ids, semaphore):
    async def process_one(prd_id):
        async with semaphore:
            return await get_product_similarity_async(collection, prd_id)

    tasks = [process_one(pid) for pid in prd_ids]
    results = await asyncio.gather(*tasks)
    await insert_batch_async(db_cursor, results)

# Asynchronous process (entire process)
async def main():
    db_conn = psycopg2.connect(**DB_CONFIG)
    db_cur = db_conn.cursor()

    db_cur.execute("""
        SELECT DISTINCT prd_id
        FROM product_similarity.products_trait_information
    """)
    prd_ids = [row[0] for row in db_cur.fetchall()]

    connections.connect("default", uri=milvus_uri)
    collection = Collection(collection_name)
    collection.load()

    semaphore = asyncio.Semaphore(max_concurrency)
    batches = [prd_ids[i:i + batch_size] for i in range(0, len(prd_ids), batch_size)]

    for batch_num, batch in enumerate(batches, start=1):
        print(f"Processing batch {batch_num}/{len(batches)} ...")
        await process_batch(collection, db_cur, batch, semaphore)
        db_conn.commit()
        print(f"Batch {batch_num} committed.")

    db_cur.close()
    db_conn.close()

# Run
if __name__ == "__main__":
    asyncio.run(main())
