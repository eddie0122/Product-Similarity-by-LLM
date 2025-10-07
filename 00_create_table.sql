CREATE SCHEMA IF NOT EXISTS product_similarity;

CREATE TABLE IF NOT EXISTS product_similarity.product_raw (
    prd_id VARCHAR(30) PRIMARY KEY,
    category VARCHAR(20),
    prd_name TEXT,
    price NUMERIC,
    review NUMERIC,
    review_rating NUMERIC,
    prd_img TEXT
);

CREATE TABLE IF NOT EXISTS product_similarity.products_trait_image (
    category1 VARCHAR(30),
    category2 VARCHAR(30),
    color VARCHAR(50),
    style VARCHAR(50),
    material VARCHAR(50),
    occasion VARCHAR(50),
    prd_id VARCHAR(30) NOT NULL
);

CREATE TABLE IF NOT EXISTS product_similarity.products_trait_text (
    category1 VARCHAR(30),
    category2 VARCHAR(30),
    color VARCHAR(50),
    style VARCHAR(50),
    material VARCHAR(50),
    occasion VARCHAR(50),
    prd_id VARCHAR(30) NOT NULL
);

CREATE TABLE IF NOT EXISTS product_similarity.products_trait_information (
    prd_id VARCHAR(30) NOT NULL,
    category VARCHAR(20) NOT NULL,
    prd_name TEXT NOT NULL,
    text_cat1 VARCHAR(30),
    text_cat2 VARCHAR(30),
    text_color VARCHAR(50),
    text_style VARCHAR(50),
    text_material VARCHAR(50),
    text_occasion VARCHAR(50),
    image_cat1 VARCHAR(30),
    image_cat2 VARCHAR(30),
    image_color VARCHAR(50),
    image_style VARCHAR(50),
    image_material VARCHAR(50),
    image_occasion VARCHAR(50)
);
