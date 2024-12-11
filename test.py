from sentence_transformers import SentenceTransformer

# Load the model
model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
model = SentenceTransformer(model_name)

print("ready")
# Define the queries and documents
queries = ['what is snowflake?', 'Where can I get the best tacos?']
documents = ['The Data Cloud!', 'Mexico City of Course!']

# Compute embeddings: use `prompt_name="query"` to encode queries!
query_embeddings = model.encode(queries, prompt_name="query") 
document_embeddings = model.encode(documents)

# Compute cosine similarity scores
scores = model.similarity(query_embeddings, document_embeddings)

# Output the results
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    print("Query:", query)
    for document, score in doc_score_pairs:
        print(score, document)
