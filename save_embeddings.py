import numpy as np

def serialize_faiss_index(vectorstore):
    pkl = vectorstore.serialize_to_bytes()  # serializes the faiss
    return pkl


def save_index_to_database(serialized_index, conn, identifier="unique_identifier_for_your_vector_store"):
    
    cursor = conn.cursor()
    blob_data = serialized_index if isinstance(serialized_index, np.ndarray) else serialized_index
    print('Inserting..')
    cursor.execute("""
        INSERT INTO vector_stores (id, vector_store) VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE vector_store = %s
    """, (identifier, blob_data, blob_data))

    conn.commit()