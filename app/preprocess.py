import base64
import json
import re
from openai import OpenAI
from openai import AsyncOpenAI
import asyncpg

def encode_image(image_path):
    """
    Load an image file and encode it to base64.
    Args:
        image_path (str): Path to the image file.
    Returns:
        byte: Base64 encoded image data.
    """
    with open(image_path, "rb") as img_file:
        image_encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return image_encoded


def extract_json(text):
    """
    Extract JSON objects from a text string.
    Args:
        text (str): Input text containing JSON objects.
    Returns:
        list: A list of extracted JSON objects.
    """
    matches = re.findall(r'\{.*?\}', text, re.DOTALL)
    json_objects = []
    for match in matches:
        try:
            obj = json.loads(match)
            json_objects.append(obj)
        except json.JSONDecodeError:
            continue
    return json_objects


def recognize_image(service_url, service_key, service_llm, service_temperature, service_role, img_path):
    """
    Recognizes the content of an image using a language model.
    Args:
        service_url (str): URL for LLM service
        service_key (str): API key for LLM service
        service_llm (str): Name of the language model
        service_role (str): System role description for the LLM
        service_temperature (float): Temperature setting for the LLM
        img_path (str): Path to the image file
    Returns:
        status (boolean): Status of the operation (True/False)
        return (list): Extracted JSON content or error message
        
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
    try:
        client = OpenAI(
            base_url=service_url,
            api_key=service_key,
        )
        encoded_image = encode_image(img_path)
        messages = [
            {
                "role": "system",
                "content":
                    [
                        {
                            "type": "text",
                            "text": service_role
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
        response = client.chat.completions.create(
            model=service_llm,
            messages=messages,
            temperature=service_temperature,
        )
        return {"status": True, "return": extract_json(response.choices[0].message.content)}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": False, "return": str(e)}


def recognize_text(service_url, service_key, service_llm, service_temperature, service_role, text_query):
    """
    Recognizes the content of an text using a language model.
    Args:
        service_url (str): URL for LLM service
        service_key (str): API key for LLM service
        service_llm (str): Name of the language model
        service_role (str): System role description for the LLM
        service_temperature (float): Temperature setting for the LLM
    Returns:
        status (boolean): Status of the operation (True/False)
        return (list): Extracted JSON content or error message
        
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
    try:
        client = OpenAI(
            base_url=service_url,
            api_key=service_key,
        )
        messages = [
            {
                "role": "system",
                "content":
                    [
                        {
                            "type": "text",
                            "text": service_role
                        }
                    ],
            },
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": f'What can you tell me about "{text_query}" ?'
                        }
                    ],
            }
        ]
        response = client.chat.completions.create(
            model=service_llm,
            messages=messages,
            temperature=service_temperature,
        )
        return {"status": True, "return": extract_json(response.choices[0].message.content)}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": False, "return": str(e)}


async def recognize_text_async(service_url, service_key, service_llm, service_temperature, service_role, text_query):
    """
    Asynchronously recognizes the content of a text using a language model.
    Args:
        service_url (str): URL for LLM service
        service_key (str): API key for LLM service
        service_llm (str): Name of the language model
        service_role (str): System role description for the LLM
        service_temperature (float): Temperature setting for the LLM
    Returns:
        dict: {
            "status": True/False,
            "return": Extracted JSON content or error message
        }
    """
    try:
        client = AsyncOpenAI(
            base_url=service_url,
            api_key=service_key,
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": service_role
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'What can you tell me about "{text_query}" ?'
                    }
                ],
            }
        ]
        response = await client.chat.completions.create(
            model=service_llm,
            messages=messages,
            temperature=service_temperature,
        )
        return {"status": True, "return": extract_json(response.choices[0].message.content)}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": False, "return": str(e)}


async def recognize_image_async(service_url, service_key, service_llm, service_temperature, service_role, img_path):
    """
    Asynchronously recognizes the content of an image using a language model.
    Args:
        service_url (str): URL for LLM service
        service_key (str): API key for LLM service
        service_llm (str): Name of the language model
        service_role (str): System role description for the LLM
        service_temperature (float): Temperature setting for the LLM
        img_path (str): Path to the image file
    Returns:
        dict: {
            "status": True/False,
            "return": list of extracted JSON content or error message
        }
    """
    try:
        client = AsyncOpenAI(
            base_url=service_url,
            api_key=service_key,
        )
        encoded_image = encode_image(img_path)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": service_role}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What can you tell me about this image?"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"}}
                ],
            }
        ]
        response = await client.chat.completions.create(
            model=service_llm,
            messages=messages,
            temperature=service_temperature,
        )
        return {"status": True, "return": extract_json(response.choices[0].message.content)}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": False, "return": str(e)}


def insert_product_trait_image(db_cur, prd_id, prd_desc):
    """
    Inserts product image trait data into the database.
    Args:
        db_cur: Database cursor
        prd_id (str): Product ID
        prd_desc (dict): Product description containing traits
    """
    try:
        query =\
            """
            INSERT INTO product_similarity.products_trait_image
            (category1, category2, color, style, material, occasion, prd_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
        db_cur.execute(
            query,
            (
                prd_desc.get("category1"),
                prd_desc.get("category2"),
                prd_desc.get("color"),
                prd_desc.get("style"),
                prd_desc.get("material"),
                prd_desc.get("occasion"),
                prd_id,
            )
        )
        db_cur.connection.commit()
        return {"status": True, "return": f"Insert successful : {prd_id}"}
    except Exception as e:
        return {"status": False, "return": f"Insert failed : {prd_id}\n{str(e)}"}


def insert_product_trait_text(db_cur, prd_id, prd_desc):
    """
    Inserts product text trait data into the database.
    Args:
        db_cur: Database cursor
        prd_id (str): Product ID
        prd_desc (dict): Product description containing traits
    """
    try:
        query =\
            """
            INSERT INTO product_similarity.products_trait_text
            (category1, category2, color, style, material, occasion, prd_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
        db_cur.execute(
            query,
            (
                prd_desc.get("category1"),
                prd_desc.get("category2"),
                prd_desc.get("color"),
                prd_desc.get("style"),
                prd_desc.get("material"),
                prd_desc.get("occasion"),
                prd_id,
            )
        )
        db_cur.connection.commit()
        return {"status": True, "return": f"Insert successful : {prd_id}"}
    except Exception as e:
        return {"status": False, "return": f"Insert failed : {prd_id}\n{str(e)}"}


async def insert_product_trait_text_async(db_conf, prd_id, prd_desc):
    """
    Asynchronously inserts product text trait data into the database.
    Args:
        db_conn: Async database connection (e.g., asyncpg connection)
        prd_id (str): Product ID
        prd_desc (dict): Product description containing traits
    Returns:
        dict: {
            "status": True/False,
            "return": Success message or error details
        }
    """
    try:
        conn = await asyncpg.connect(**db_conf)
        query = """
            INSERT INTO product_similarity.products_trait_text
            (category1, category2, color, style, material, occasion, prd_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7);
        """
        await conn.execute(
            query,
            prd_desc.get("category1"),
            prd_desc.get("category2"),
            prd_desc.get("color"),
            prd_desc.get("style"),
            prd_desc.get("material"),
            prd_desc.get("occasion"),
            prd_id,
        )
        await conn.close()
        return {"status": True, "return": f"Insert successful : {prd_id}"}
    except Exception as e:
        return {"status": False, "return": f"Insert failed : {prd_id}\n{str(e)}"}


async def insert_product_trait_image_async(user, password, database, host, port, prd_id, prd_desc):
    """
    Asynchronously inserts product trait data into the database.
    Args:
        user (str): Database username
        password (str): Database password
        database (str): Database name
        host (str): Database host
        port (int): Database port
        prd_id (str): Product ID
        prd_desc (dict): Product description contains traits
    Returns:
        dict: status and message
    """
    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=database,
            host=host,
            port=port
        )
        query = """
            INSERT INTO product_similarity.products_trait_image
            (category1, category2, color, style, material, occasion, prd_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7);
        """
        await conn.execute(
            query,
            prd_desc.get("category1"),
            prd_desc.get("category2"),
            prd_desc.get("color"),
            prd_desc.get("style"),
            prd_desc.get("material"),
            prd_desc.get("occasion"),
            prd_id,
        )
        await conn.close()
        return {"status": True, "return": f"Insert successful : {prd_id}"}
    except Exception as e:
        return {"status": False, "return": f"Insert failed : {prd_id}\n{str(e)}"}
