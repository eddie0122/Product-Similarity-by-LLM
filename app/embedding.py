import psycopg2
import psycopg2.extras


def _get_embedding(collection, prd_id, prd_tag='product_name'):
    return collection.query(
        expr=f'prd_tag == "{prd_tag}" and prd_id == "{prd_id}"',
        output_fields=["embedding"]
    )


def _calculate_similarity(collection, prd_id, embedding, prd_tag='product_image', top_k=10):
    return collection.search(
        expr=f'prd_tag == "{prd_tag}" and prd_id == "{prd_id}"',
        anns_field="embedding",
        data=[embedding],
        param={"metric_type": "COSINE", "params": {"nprobe": 8}},
        output_fields=["prd_id", "prd_text", "prd_tag"],
        limit=top_k
    )


def get_product_similarity_inner(collection, prd_id):
    result = _get_embedding(collection, prd_id, 'product_name')
    embedding_prd_name = result[0].get('embedding')

    result = _get_embedding(collection, prd_id, 'product_text')
    embedding_prd_text = result[0].get('embedding')

    result = _calculate_similarity(
        collection, prd_id, embedding_prd_name, 'product_text')
    similarity_name_text = max([_.get('distance') for _ in result[0]])

    result = _calculate_similarity(
        collection, prd_id, embedding_prd_name, 'product_image')
    similarity_name_image = max([_.get('distance') for _ in result[0]])

    result = _calculate_similarity(
        collection, prd_id, embedding_prd_text, 'product_image')
    similarity_text_image = max([_.get('distance') for _ in result[0]])

    return (prd_id, similarity_name_text, similarity_name_image, similarity_text_image)


def insert_batch_similarities(db_cursor, similarities):
    query = """
       INSERT INTO product_similarity.products_similarity_score_inner
       (prd_id, similarity_name_text, similarity_name_image, similarity_text_image)
       VALUES %s"""
    psycopg2.extras.execute_values(
        db_cursor, query, similarities
    )
