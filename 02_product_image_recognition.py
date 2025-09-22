from openai import OpenAI
import base64
import json
import re
import os


def _encode_image(image_path):
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


def _extract_json(text):
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
        encoded_image = _encode_image(img_path)
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
        return {"status": True, "return": _extract_json(response.choices[0].message.content)}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": False, "return": str(e)}


# Directory containing product images
dir_path = os.path\
    .dirname(os.path.realpath(__file__)) + "/product_information/prd_img"
file_names = os.listdir(dir_path)


ollama_role =\
    '''
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
    '''
ollama_url = "http://ollama:11434/v1"
ollama_key = "ollama"
ollama_model = "ebdm/gemma3-enhanced:12b"
ollama_temperature = 0.0
img_path = f"{dir_path}/{file_names[6978]}"

result = recognize_image(
    service_url=ollama_url,
    service_key=ollama_key,
    llm=ollama_model,
    llm_temperature=ollama_temperature,
    system_role=ollama_role,
    img_path=img_path
)
print(result)
