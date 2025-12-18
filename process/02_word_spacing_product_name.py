import pandas as pd
import os
import psycopg2
import yaml
from openai import OpenAI

import base64
import json
import re

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
def split_words_product_name(user_query, user_config, llm_prompt, llm_name):
    client = OpenAI(**user_config["ollama"])
    messages = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": llm_prompt['split_words']['content']}], },
        {"role": "user",
         "content": [{"type": "text",
                      "text": f'user_input:"{user_query}"'}], }
    ]
    response = client.chat.completions.create(
        model=user_config['model'][llm_name],
        messages=messages,
        temperature=user_config['model']['temperature'],
    )
    return extract_json(response.choices[0].message.content)[0]


# Connect to PostgreSQL database
db_conn = psycopg2.connect(**conf["postgresql"])
db_cur = db_conn.cursor()

# Fetch product information
query =\
    '''
    SELECT *
    FROM product_similarity.product_information
    '''
db_cur.execute(query)
df_prd_info = pd.DataFrame(
    db_cur.fetchall(), columns=[_[0] for _ in db_cur.description])

# Process each product and store renamed(cleansing) products
query_rename =\
    '''
    INSERT INTO product_similarity.product_rename (prd_id, prd_rename) VALUES ('{}', '{}')
    '''
for idx, row in df_prd_info.iterrows():
    prd_rename = drop_duplicate_words(
        split_words_product_name(
            user_query=row['prd_name'],
            user_config=conf,
            llm_prompt=prompts,
            llm_name='qwen3_4b_instruct'
        ).get('result')
    )
    db_cur.execute(query_rename.format(row['prd_id'], prd_rename))
    db_conn.commit()

db_conn.close()
