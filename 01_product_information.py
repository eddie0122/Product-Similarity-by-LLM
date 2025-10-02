import requests
import pandas as pd
import numpy as np
import os
import time
import psycopg2
import psycopg2.extras

# Load product information data
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/product_information/'
file_name = 'product_information.csv'
df_prd = pd.read_csv(f"{dir_path}{file_name}", dtype=str)

# Map category links to category names
prd_dict = {
    'https://www.coupang.com/np/categories/498918?channel=plp_C2':'티셔츠',
    'https://www.coupang.com/np/categories/498919?channel=plp_C2':'맨투맨/후디',
    'https://www.coupang.com/np/categories/498922?channel=plp_C2':'셔츠',
    'https://www.coupang.com/np/categories/498930?channel=plp_C2':'니트/조끼',
    'https://www.coupang.com/np/categories/498934?channel=plp_C2':'아우터',
    'https://www.coupang.com/np/categories/498929?channel=plp_C2':'후드집업/집업',
    'https://www.coupang.com/np/categories/498924?channel=plp_C2':'바지',
    'https://www.coupang.com/np/categories/498923?channel=plp_C2':'정장/세트',
    'https://www.coupang.com/np/categories/498928?channel=plp_C2':'트레이닝복',
    'https://www.coupang.com/np/categories/565081?channel=plp_C2':'속옷/잠옷',
    'https://www.coupang.com/np/categories/498938?channel=plp_C2':'비치웨어',
    'https://www.coupang.com/np/categories/498940?channel=plp_C2':'테마의류',
    'https://www.coupang.com/np/categories/498943?channel=plp_C2':'커플/패밀리룩',
    'https://www.coupang.com/np/categories/498946?channel=plp_C2':'스포츠의류',
    'https://www.coupang.com/np/categories/498939?channel=plp_C2':'한복/수의',
    'https://www.coupang.com/np/categories/498974?channel=plp_C2':'신발',
    'https://www.coupang.com/np/categories/499007?channel=plp_C2':'가방/잡화',
}
df_prd['category'] = df_prd['category-link-0-href'].map(prd_dict)

# Clean 'rating' column by removing parentheses
df_prd['rating'] = df_prd['rating']\
    .map(lambda x: x.replace('(', '').replace(')', '') if pd.notna(x) else x)

# Clean 'price' column by removing the currency symbol
df_prd['price3'] = df_prd['price3']\
    .map(lambda x: x.replace('원', '').replace(',', '') if pd.notna(x) else x)

# Rename columns for clarity
df_prd = df_prd.rename(
    columns={
        'web-scraper-order': 'prd_id',
        'image-src': 'prd_img',
        'image2-src': 'delivery_type',
        'rating': 'review',
        'ratingValue': 'review_rating',
        'price3': 'price',
        'name': 'prd_name',
    }
)

# Download product images (It takes long long time)
os.makedirs(f"{dir_path}/prd_img/", exist_ok=True)
for idx, row in df_prd.iterrows():
    img_url = row['prd_img']
    img_id = row['prd_id']
    response = requests.get(img_url)
    if response.status_code == 200:
        with open(f"{dir_path}/prd_img/{img_id}.jpg", "wb") as f:
            f.write(response.content)
    time.sleep(np.random.uniform(0.1, 0.35))

# Add product image paths
df_prd['prd_path'] = df_prd['prd_id']\
    .apply(lambda x: f"{dir_path}prd_img/{x}.jpg")

# Handle missing values as white spaces
df_prd.loc[df_prd['review'].isnull(), 'review'] = ''
df_prd.loc[df_prd['review_rating'].isnull(), 'review_rating'] = ''

# Prepare data for database insertion
cols = ['prd_id', 'category', 'prd_name', 
        'price', 'review', 'review_rating', 'prd_path']
db_insert = [tuple(_) for _ in df_prd[cols].to_numpy()]
db_insert = [tuple(None if __ == '' else __ for __ in _) for _ in db_insert]

# Connect to your PostgreSQL database
DB_CONFIG = {
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "pgsql",
    "port": "5432"
}
db_conn = psycopg2.connect(**DB_CONFIG)
db_cur = db_conn.cursor()

# Insert data into the table
query =\
    """
    INSERT INTO product_similarity.product_raw
    (prd_id, category, prd_name, price, review, review_rating, prd_img)
    VALUES %s
    ON CONFLICT (prd_id) DO NOTHING;
    """
psycopg2.extras.execute_values(
    cur=db_cur,
    sql=query,
    argslist=db_insert
)
db_conn.commit()
db_conn.close()
