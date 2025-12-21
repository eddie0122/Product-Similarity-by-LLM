import asyncpg
import base64
import json
import re
import yaml
import os
import asyncio
from openai import AsyncOpenAI

# Set root path
root_path = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/"

# Configuration for batch processing
MAX_CONCURRENT_REQUESTS = 5  # Maximum number of concurrent API requests
BATCH_SIZE = 20  # Number of products to process in each batch
LIMIT_ROWS = None  # Set to None to process all rows, or a number to limit for testing

# Helper function : encoding image
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        image_encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return image_encoded

# Helper function : extract JSON from text
def extract_json(text):
    matches = re.findall(r'\{.*?\}', text, re.DOTALL)
    json_objects = []
    for match in matches:
        try:
            obj = json.loads(match)
            json_objects.append(obj)
        except json.JSONDecodeError:
            continue
    return json_objects

# Main function : recognize product images
async def recognize_image(client, image_path, llm_prompt, model_name, temperature, semaphore):
    async with semaphore:  # Limit concurrent requests
        try:
            encoded_image = encode_image(image_path)
            messages = [
                {
                    "role": "system",
                    "content":
                        [
                            {
                                "type": "text",
                                "text": llm_prompt
                            }
                        ],
                },
                {
                    "role": "user",
                    "content":
                        [
                            {
                                "type": "text",
                                "text": "What can you tell me about this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                            },
                        ],
                }
            ]
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            return {"status": True, "content": extract_json(response.choices[0].message.content)}
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {"status": False}

# Process a single product image
async def process_product(client, prd_id, prd_img, llm_prompt, model_name, temperature, db_pool, semaphore):
    result_recognize = await recognize_image(
        client=client,
        image_path=prd_img,
        llm_prompt=llm_prompt,
        model_name=model_name,
        temperature=temperature,
        semaphore=semaphore,
    )

    if result_recognize.get('status'):
        prd_descs = result_recognize.get('content')
        async with db_pool.acquire() as conn:
            for prd_desc in prd_descs:
                query_image = '''
                    INSERT INTO product_similarity.product_image
                    (category1, category2, color_tone, style, occasion, prd_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                '''
                await conn.execute(
                    query_image,
                    prd_desc.get('category1'),
                    prd_desc.get('category2'),
                    prd_desc.get('color_tone'),
                    prd_desc.get('style'),
                    prd_desc.get('occasion'),
                    prd_id
                )

    return {"prd_id": prd_id, "status": result_recognize.get('status')}

# Process products in batches
async def process_batch(client, products, llm_prompt, model_name, temperature, db_pool, semaphore, batch_num, total_batches):
    print(f"Processing batch {batch_num}/{total_batches} ({len(products)} products)...")

    tasks = [
        process_product(
            client=client,
            prd_id=product['prd_id'],
            prd_img=product['prd_img'],
            llm_prompt=llm_prompt,
            model_name=model_name,
            temperature=temperature,
            db_pool=db_pool,
            semaphore=semaphore
        )
        for product in products
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if not isinstance(r, Exception) and r.get('status', False))
    print(f"Batch {batch_num}/{total_batches} completed: {successful}/{len(products)} successful")

    return results

# Main async function
async def main():
    # Load configuration
    with open(f"{root_path}/configuration/config.yml", "r") as f:
        conf = yaml.safe_load(f)

    # Load prompts
    with open(f"{root_path}/configuration/prompt.yml", "r") as f:
        prompts = yaml.safe_load(f)

    # Create async OpenAI client
    client = AsyncOpenAI(**conf['ollama'])

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Connect to PostgreSQL database using asyncpg
    db_pool = await asyncpg.create_pool(
        host=conf['postgresql']['host'],
        port=conf['postgresql'].get('port', 5432),
        user=conf['postgresql']['user'],
        password=conf['postgresql']['password'],
        database=conf['postgresql']['database'],
        min_size=2,
        max_size=10,
    )

    try:
        # Fetch product id and path of images from the database
        query = """
            SELECT prd_id,
                prd_img
            FROM product_similarity.product_information
            WHERE prd_img IS NOT NULL
        """

        # Add LIMIT clause if LIMIT_ROWS is set
        if LIMIT_ROWS is not None:
            query += f"\n            LIMIT {LIMIT_ROWS}"

        query += ";"

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query)

        # Convert to list of dictionaries
        products = [{"prd_id": row['prd_id'], "prd_img": row['prd_img']} for row in rows]
        total_products = len(products)

        if LIMIT_ROWS is not None:
            print(f"[TEST MODE] Limiting to first {LIMIT_ROWS} rows")
        print(f"Found {total_products} products to process")

        # Split products into batches
        batches = [products[i:i + BATCH_SIZE] for i in range(0, len(products), BATCH_SIZE)]
        total_batches = len(batches)

        print(f"Processing in {total_batches} batches with max {MAX_CONCURRENT_REQUESTS} concurrent requests")

        # Process batches sequentially
        all_results = []
        for batch_num, batch in enumerate(batches, 1):
            batch_results = await process_batch(
                client=client,
                products=batch,
                llm_prompt=prompts['recognize_image']['content'],
                model_name=conf['model']['gemma3_12b'],
                temperature=conf['model']['temperature'],
                db_pool=db_pool,
                semaphore=semaphore,
                batch_num=batch_num,
                total_batches=total_batches
            )
            all_results.extend(batch_results)

        # Print final summary
        total_successful = sum(1 for r in all_results if not isinstance(r, Exception) and r.get('status', False))
        print(f"\n=== Final Summary ===")
        print(f"Total products processed: {total_products}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_products - total_successful}")

    finally:
        # Close database pool and client
        await db_pool.close()
        await client.close()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
