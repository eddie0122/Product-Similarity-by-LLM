import asyncio
import asyncpg
import pandas as pd
from pymilvus import connections, Collection

# Variable of postgresql & milvus connection config
DB_CONFIG = {
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "pgsql",
    "port": "5432"
}

milvus_uri = "./milvus_db/product_similarity.db"
collection_name = "product_embedding"

# Get product ids and names from PostgreSQL
async def fetch_products():
    conn = await asyncpg.connect(**DB_CONFIG)
    query = """
        SELECT DISTINCT prd_id, prd_name
        FROM product_similarity.products_trait_information
    """
    rows = await conn.fetch(query)
    await conn.close()
    return pd.DataFrame(rows, columns=["prd_id", "prd_name"])

# Connect to milvus lite
async def connect_milvus():
    await asyncio.to_thread(connections.connect, "default", uri=milvus_uri)
    return Collection(collection_name)


async def get_high_ranked_image_only(collection, prd_id, prd_tag="product_image"):
    """Delete all but the top-ranked image embeddings for a product."""
    def _sync_op():
        # Query the main embedding for the product
        embedding_name = collection.query(
            expr=f'prd_tag == "product_name" and prd_id == "{prd_id}"',
            output_fields=["embedding"]
        )

        if not embedding_name:
            return None

        # Search for product image embeddings by similarity
        results = collection.search(
            expr=f'prd_tag == "{prd_tag}" and prd_id == "{prd_id}"',
            anns_field="embedding",
            data=[embedding_name[0].get("embedding")],
            param={"metric_type": "COSINE", "params": {"nprobe": 8}},
            output_fields=["prd_id", "prd_id_seq", "prd_text", "prd_tag"],
            limit=100
        )

        seqs_delete = [_.get("prd_id_seq") for _ in results[0]][1:]
        if seqs_delete:
            return collection.delete(
                f'prd_id == "{prd_id}" and prd_tag == "{prd_tag}" and prd_id_seq in {seqs_delete}'
            )
        return None

    return await asyncio.to_thread(_sync_op)


async def process_all_products(df_prd):
    """Run deletion tasks concurrently for all products."""
    collection = await connect_milvus()
    await asyncio.to_thread(collection.load)

    tasks = [
        get_high_ranked_image_only(collection, prd_id)
        for prd_id in df_prd["prd_id"].tolist()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    await asyncio.to_thread(collection.release)
    await asyncio.to_thread(connections.disconnect)

    return results


async def main():
    print("Fetching product data from PostgreSQL...")
    df_prd = await fetch_products()
    print(f"Fetched {len(df_prd)} products.")

    print("Connecting to Milvus and cleaning up embeddings...")
    results = await process_all_products(df_prd)
    print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(main())
