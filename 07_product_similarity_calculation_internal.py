import asyncio
import asyncpg
from pymilvus import connections, Collection
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

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

# Load milvus db (sync operation - done once at startup)
connections.connect("default", uri=milvus_uri)
collection = Collection(collection_name)
collection.load()

# Thread pool executor for Milvus operations (thread-safe)
executor = ThreadPoolExecutor(max_workers=10)


async def get_embedding(prd_id: str, prd_tag: str) -> Optional[List[float]]:
    """Fetch embedding from Milvus in a thread-safe way."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: collection.query(
                expr=f'prd_id == "{prd_id}" and prd_tag == "{prd_tag}"',
                output_fields=['embedding']
            )
        )
        return result[0].get('embedding') if result else None
    except Exception as e:
        print(f"Error fetching embedding for {prd_id} ({prd_tag}): {e}")
        return None


async def calculate_similarity(prd_id: str, embedding: List[float], target_tag: str) -> Optional[float]:
    """Calculate similarity score between embedding and target tag in Milvus."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: collection.search(
                expr=f'prd_id == "{prd_id}" and prd_tag == "{target_tag}"',
                anns_field="embedding",
                data=[embedding],
                param={"metric_type": "COSINE", "params": {"nprobe": 8}},
                output_fields=["prd_id", "prd_text", "prd_tag"],
                limit=10
            )
        )
        return result[0][0].get('distance') if result and result[0] else None
    except Exception as e:
        print(f"Error calculating similarity for {prd_id} ({target_tag}): {e}")
        return None


async def process_product(prd_id: str, db_pool: asyncpg.Pool, max_retries: int = 3) -> bool:
    """Process a single product: calculate similarities and insert into database.

    Returns True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            # Fetch embeddings concurrently
            prd_name_embedding, prd_text_embedding = await asyncio.gather(
                get_embedding(prd_id, "product_name"),
                get_embedding(prd_id, "product_text")
            )

            if not prd_name_embedding or not prd_text_embedding:
                print(f"Warning: Missing embeddings for product {prd_id}")
                return False

            # Calculate all similarities concurrently
            similarity_name_text, similarity_name_image, similarity_text_image = await asyncio.gather(
                calculate_similarity(prd_id, prd_name_embedding, "product_text"),
                calculate_similarity(prd_id, prd_name_embedding, "product_image"),
                calculate_similarity(prd_id, prd_text_embedding, "product_image")
            )

            # Check if all similarities were calculated
            if None in (similarity_name_text, similarity_name_image, similarity_text_image):
                print(f"Warning: Missing similarity scores for product {prd_id}")
                return False

            # Insert into database
            async with db_pool.acquire() as conn:
                query = """
                    INSERT INTO product_similarity.products_similarity_score_inner
                    VALUES ($1, $2, $3, $4)
                """
                await conn.execute(query, prd_id, similarity_name_text, similarity_name_image, similarity_text_image)

            print(f"Processed product {prd_id}")
            return True

        except asyncpg.PostgresError as e:
            if attempt < max_retries - 1:
                print(f"Database error for product {prd_id} (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"Failed to process product {prd_id} after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            print(f"Unexpected error processing product {prd_id}: {e}")
            return False

    return False


async def process_products_batch(prd_ids: List[str], db_pool: asyncpg.Pool, batch_size: int = 10) -> tuple[int, int]:
    """Process products in batches to control concurrency.

    Returns tuple of (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    for i in range(0, len(prd_ids), batch_size):
        batch = prd_ids[i:i + batch_size]
        results = await asyncio.gather(*[process_product(prd_id, db_pool) for prd_id in batch])

        successful += sum(results)
        failed += len(results) - sum(results)

        print(f"Completed batch {i//batch_size + 1}/{(len(prd_ids) + batch_size - 1)//batch_size} "
              f"(Success: {successful}, Failed: {failed})")

    return successful, failed


async def main():
    """Main async function to orchestrate the similarity calculation process."""
    # Create PostgreSQL connection pool
    db_pool = await asyncpg.create_pool(
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        min_size=5,
        max_size=20,
        command_timeout=60
    )

    try:
        # Fetch distinct product ids from the database
        query = """
        SELECT DISTINCT prd_id
        FROM product_similarity.products_trait_information
        ORDER BY prd_id
        """
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query)

        prd_ids = [row['prd_id'] for row in rows]
        print(f"Found {len(prd_ids)} products to process")

        # Process products in batches (adjust batch_size based on your system resources)
        successful, failed = await process_products_batch(prd_ids, db_pool, batch_size=500)

        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(prd_ids)}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"Fatal error in main: {e}")
        raise
    finally:
        # Close the connection pool and executor
        await db_pool.close()
        executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
