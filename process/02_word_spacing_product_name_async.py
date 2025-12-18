import pandas as pd
import asyncio
import asyncpg
import aiohttp
import base64
import yaml
import json
import re
import os

# Set root path
root_path = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/"

# Load configuration
with open(f"{root_path}/configuration/config.yml", "r") as f:
    conf = yaml.safe_load(f)

# Load prompts
with open(f"{root_path}/configuration/prompt.yml", "r") as f:
    prompts = yaml.safe_load(f)

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

# Helper function : drop duplicate words in product name
def drop_duplicate_words(product_name):
    words = [None if _ == '' else _.strip() for _ in product_name.split(' ')]
    return ' '.join(list(filter(None, list(set(words)))))

# Main function : cleansing product name (separate words and drop duplicate words)
async def split_words_product_name(user_query, user_config, llm_prompt, llm_name, session, timeout):
    url = f"{user_config['ollama']['base_url']}/chat/completions"

    payload = {
        "model": user_config['model'][llm_name],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": llm_prompt['split_words']['content']}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f'user_input:"{user_query}"'}]
            }
        ],
        "temperature": user_config['model']['temperature']
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {user_config['ollama']['api_key']}"
    }

    async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
        result = await response.json()
        content = result['choices'][0]['message']['content']

        # Try to parse as JSON first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON using regex
            json_results = extract_json(content)
            if not json_results:
                raise ValueError(f"No JSON found in response: {content}")
            return json_results[0]

# Async function to process a single product
async def process_product(row, db_pool, user_config, llm_prompt, llm_name, session, timeout, semaphore):
    async with semaphore:  # Limit concurrent requests
        result = await split_words_product_name(
            user_query=row['prd_name'],
            user_config=user_config,
            llm_prompt=llm_prompt,
            llm_name=llm_name,
            session=session,
            timeout=timeout
        )
        result_text = result.get('result')
        if not result_text:
            raise ValueError(f"No 'result' field in LLM response for product {row['prd_id']}: {result}")
        prd_rename = drop_duplicate_words(result_text)

        query_rename = '''
            INSERT INTO product_similarity.product_rename (prd_id, prd_rename)
            VALUES ($1, $2)
        '''
        async with db_pool.acquire() as conn:
            await conn.execute(query_rename, row['prd_id'], prd_rename)

        return prd_rename

# Main async function
async def main():
    print("Starting async product information recognition...")

    try:
        # Create asyncpg connection pool
        print("Connecting to database...")
        db_pool = await asyncpg.create_pool(
            host=conf["postgresql"]["host"],
            port=conf["postgresql"]["port"],
            user=conf["postgresql"]["user"],
            password=conf["postgresql"]["password"],
            database=conf["postgresql"]["database"]
        )
        print("Database connection established")

        try:
            # Fetch product information
            print("Fetching product information...")
            query = '''
                SELECT *
                FROM product_similarity.product_information
            '''
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(query)

            print(f"Found {len(rows)} products to process")

            # Convert to pandas DataFrame
            df_prd_info = pd.DataFrame(
                [dict(row) for row in rows]
            )

            # Set timeout and concurrency limits
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes per request
            max_concurrent_requests = 10  # Limit concurrent requests to avoid overwhelming the server
            semaphore = asyncio.Semaphore(max_concurrent_requests)

            # Create aiohttp session for concurrent LLM requests
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Process all products concurrently (but limited by semaphore)
                print(f"Processing products with LLM (max {max_concurrent_requests} concurrent requests)...")
                tasks = [
                    process_product(
                        row=row,
                        db_pool=db_pool,
                        user_config=conf,
                        llm_prompt=prompts,
                        llm_name='qwen3_4b_instruct',
                        session=session,
                        timeout=timeout,
                        semaphore=semaphore
                    )
                    for idx, row in df_prd_info.iterrows()
                ]

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check for errors
                errors = [r for r in results if isinstance(r, Exception)]
                successes = [r for r in results if not isinstance(r, Exception)]

                print(f"\nProcessing complete:")
                print(f"  - Successfully processed: {len(successes)} products")
                if errors:
                    print(f"  - Errors encountered: {len(errors)}")
                    for i, error in enumerate(errors, 1):
                        print(f"    Error {i}: {type(error).__name__}: {error}")

        finally:
            # Close the connection pool
            print("Closing database connection...")
            await db_pool.close()
            print("Done!")

    except Exception as e:
        print(f"Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
