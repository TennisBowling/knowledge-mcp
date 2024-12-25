from sanic import Sanic, json
from sanic.response import json as json_response
from sentence_transformers import SentenceTransformer
import asyncpg
import json
import time
from contextlib import contextmanager

app = Sanic("knowledge_graph")

model = None
db = None

@contextmanager
def timer(description):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.3f} seconds")

@app.listener('before_server_start')
async def setup_db(app, loop):
    global db, model
    print("Loading model...")
    with timer("Model loading time"):
        model = SentenceTransformer("/home/tennisbowling/snowflake-arctic-embed-l-v2.0")
    print("Model loaded")
    
    # Setup database connection pool
    db = await asyncpg.create_pool(
        database="tennisbowling",
        user="tennisbowling",
        host="mediacenter2",
        password="tennispass"
    )

    print("Connected to postgres")

@app.listener('after_server_stop')
async def close_db(app, loop):
    if db:
        await db.close()
        

@app.route("/set_entity", methods=["POST"])
async def set_entity(request):
    data = request.json
    name = data["name"]
    entity_type = data["entity_type"]
    new_content = data["content"]

    # First try to get existing content
    with timer("Fetch existing content query"):
        existing_content = await db.fetchval("SELECT content FROM entities WHERE name = $1;", name)

    if existing_content:
        updated_content = existing_content + new_content
        
        # Create embedding based on old + new content
        combined = json.dumps({"name": name, "entity_type": entity_type, "content": updated_content})
        with timer("Model encoding for update"):
            embedding = str(model.encode(combined).tolist())
        
        with timer("Update entity query"):
            id = await db.fetchval(
                "UPDATE entities SET content = $1, embedding = $2 WHERE name = $3 RETURNING id;",
                updated_content, embedding, name
            )
        return json_response({"message": f"Updated existing entity '{id}' with new content."})
    else:
        # Create new entry with just the new content
        combined = json.dumps({"name": name, "entity_type": entity_type, "content": new_content})
        with timer("Model encoding for new entity"):
            embedding = str(model.encode(combined).tolist())
        
        with timer("Insert new entity query"):
            id = await db.fetchval(
                "INSERT INTO entities (name, type, content, embedding) VALUES ($1, $2, $3, $4) RETURNING id;",
                name, entity_type, new_content, embedding
            )
        return json_response({"message": f"Inserted new entity as id {id} into memory."})

@app.route("/set_relationship", methods=["POST"])
async def set_relationship(request):
    data = request.json
    entity1 = data["source_entity_id"]
    entity2 = data["target_entity_id"]
    relation_type = data["relation_type"]

    with timer("Insert relationship query"):
        await db.execute(
            "INSERT INTO relations (source_entity_id, target_entity_id, relation_type) VALUES ($1, $2, $3);",
            entity1, entity2, relation_type
        )

    return json_response({"message": "Created relationship."})

@app.route("/search_entity", methods=["POST"])
async def search_entity(request):
    data = request.json
    query = data["query"]
    
    with timer("Model encoding for search"):
        embedding = str(model.encode(query).tolist())
    
    # First get the initial matches
    with timer("Search query execution"):
        rows = await db.fetch(f"""
            WITH initial_matches AS (
                SELECT id, name, type, content 
                FROM entities
                WHERE embedding <=> '{embedding}' < 1.2
                ORDER BY embedding <=> '{embedding}' 
                LIMIT 3
            ),
            related_entities AS (
                -- Get entities connected by outgoing relations
                SELECT DISTINCT e.id, e.name, e.type, e.content
                FROM initial_matches im
                JOIN relations r ON im.id = r.source_entity_id
                JOIN entities e ON r.target_entity_id = e.id
                UNION
                -- Get entities connected by incoming relations
                SELECT DISTINCT e.id, e.name, e.type, e.content
                FROM initial_matches im
                JOIN relations r ON im.id = r.target_entity_id
                JOIN entities e ON r.source_entity_id = e.id
            )
            SELECT id, name, type, content
            FROM (
                SELECT * FROM initial_matches
                UNION
                SELECT * FROM related_entities
            ) combined;
        """)

    if not rows:
        return json_response({"message": "No matches found."})

    ret = []
    for row in rows:
        ret.append({
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "content": row["content"]
        })

    return json_response(ret)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)