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
    price NUMERIC NOT NULL,
    review NUMERIC,
    review_rating NUMERIC,
    prd_img_path TEXT,
    category_high_img VARCHAR(30),
    category_low_img VARCHAR(30),
    color_img VARCHAR(50),
    style_img VARCHAR(50),
    material_img VARCHAR(50),
    occasion_img VARCHAR(50)
);
