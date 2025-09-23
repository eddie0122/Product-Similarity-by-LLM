CREATE SCHEMA product_similarity;

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
    category1 VARCHAR(30) NOT NULL,
    category2 VARCHAR(30) NOT NULL,
    color VARCHAR(50) NOT NULL,
    style VARCHAR(50) NOT NULL,
    material VARCHAR(50) NOT NULL,
    occasion VARCHAR(50) NOT NULL,
    prd_id VARCHAR(30) NOT NULL
);