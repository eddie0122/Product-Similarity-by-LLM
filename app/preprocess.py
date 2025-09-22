import base64
import json
import re
from openai import OpenAI
from openai import AsyncOpenAI


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


def recognize_image(service_url, service_key, llm, llm_temperature, system_role, img_path):
    """
    Recognizes the content of an image using a language model.
    Args:
        service_url (str): URL for LLM service
        service_key (str): API key for LLM service
        llm (str): Name of the language model
        system_role (str): System role description for the LLM
        llm_temperature (float): Temperature setting for the LLM
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
                            "text": system_role
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
            model=llm,
            messages=messages,
            temperature=llm_temperature,
        )
        return {"status": True, "return": extract_json(response.choices[0].message.content)}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": False, "return": str(e)}


async def recognize_image_async(service_url, service_key, llm, llm_temperature, system_role, img_path):
    """
    Asynchronously recognizes the content of an image using a language model.
    Args:
        service_url (str): URL for LLM service
        service_key (str): API key for LLM service
        llm (str): Name of the language model
        system_role (str): System role description for the LLM
        llm_temperature (float): Temperature setting for the LLM
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
                    {"type": "text", "text": system_role}
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
            model=llm,
            messages=messages,
            temperature=llm_temperature,
        )
        return {"status": True, "return": extract_json(response.choices[0].message.content)}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": False, "return": str(e)}
