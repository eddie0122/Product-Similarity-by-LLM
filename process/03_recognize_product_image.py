import psycopg2
import pandas as pd
import base64
import json
import re
import yaml
import os
from openai import OpenAI

# Set root path
root_path = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/"

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
def recognize_image(user_config, image_path, llm_prompt, llm_name):
    try:
        client = OpenAI(**user_config['ollama'])
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
        response = client.chat.completions.create(
            model=user_config['model'][llm_name],
            messages=messages,
            temperature=user_config['model']['temperature'],
        )
        return {"status": True, "content": extract_json(response.choices[0].message.content)}
    except:
        return {"status": False}

# Load configuration
with open(f"{root_path}/configuration/config.yml", "r") as f:
    conf = yaml.safe_load(f)

# Load prompts
with open(f"{root_path}/configuration/prompt.yml", "r") as f:
    prompts = yaml.safe_load(f)

# Connect to your PostgreSQL database
db_conn = psycopg2.connect(**conf["postgresql"])
db_cur = db_conn.cursor()

# Fetch product id and path of images from the database
query =\
    """
    SELECT prd_id,
        prd_img
    FROM product_similarity.product_information
    WHERE prd_img IS NOT NULL;
    """
db_cur.execute(query=query)
rows = db_cur.fetchall()
df_prd = pd.DataFrame(rows, columns=[_[0] for _ in db_cur.description])

# Recognize product images and insert the recognized traits into the database
query_image =\
    '''
    INSERT INTO product_similarity.product_image (category1, category2, color_tone, style, occasion, prd_id) VALUES ('{}', '{}', '{}', '{}', '{}', '{}')
    '''
for prd_id, prd_img in df_prd.itertuples(index=False):
    result_recognize = recognize_image(
        user_config=conf,
        image_path=prd_img,
        llm_prompt=prompts['recognize_image']['content'],
        llm_name='gemma3_12b',
    )
    if result_recognize.get('status'):
        prd_descs = result_recognize.get('content')
        for prd_desc in prd_descs:
            db_cur.execute(
                query_image.format(
                    prd_desc.get('category1'),
                    prd_desc.get('category2'),
                    prd_desc.get('color_tone'),
                    prd_desc.get('style'),
                    prd_desc.get('occasion'),
                    prd_id
                )
            )
            db_conn.commit()
db_conn.close()
