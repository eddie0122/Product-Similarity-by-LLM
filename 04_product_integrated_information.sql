INSERT INTO product_similarity.products_trait_information (
        SELECT prw.prd_id,
            prw.category,
            prw.prd_name,
            ptt.category1 AS text_cat1,
            ptt.category2 AS text_cat2,
            ptt.color AS text_color,
            ptt.style AS text_style,
            ptt.material AS text_material,
            ptt.occasion AS text_occasion,
            pti.category1 AS image_cat1,
            pti.category2 AS image_cat2,
            pti.color AS image_color,
            pti.style AS image_style,
            pti.material AS image_material,
            pti.occasion AS image_occasion
        FROM product_similarity.product_raw AS prw
            LEFT JOIN product_similarity.products_trait_text AS ptt ON prw.prd_id = ptt.prd_id
            LEFT JOIN product_similarity.products_trait_image AS pti ON prw.prd_id = pti.prd_id
        WHERE ptt.prd_id IS NOT NULL
            AND pti.prd_id IS NOT NULL
    );
