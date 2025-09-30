INSERT INTO product_similarity.products_trait_information (
        SELECT prd.prd_id,
            prd.category,
            prd.prd_name,
            prd.price,
            prd.review,
            prd.review_rating,
            prd.prd_img AS prd_img_path,
            pri.category1 AS category_high_img,
            pri.category2 AS category_low_img,
            pri.color AS color_img,
            pri.style AS style_img,
            pri.material AS material_img,
            pri.occasion AS occasion_img
        FROM product_similarity.product_raw AS prd
            LEFT JOIN product_similarity.products_trait_image AS pri ON prd.prd_id = pri.prd_id
        WHERE pri.prd_id IS NOT NULL
    );