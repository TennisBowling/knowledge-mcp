from sentence_transformers import SentenceTransformer
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import asyncpg
import json

server = Server("example-server")

print("loading model")
model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
model = SentenceTransformer(model_name)
print("loaded model")
db: asyncpg.Pool = None # type: ignore

# Add prompt capabilities
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="set-entity",
            description="Saves an entity in the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the entity, like the person, project, or anything else."
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "What the entity is, like a person, project, business, class, etc."
                    },
                    "content": {
                        "type": "list of strings",
                        "description": "List of strings in python form that describes properties of the entity (like what they do, description, etc)."
                    }
                },
                "required": ["name", "entity_type", "content"]
            }
        ),
        types.Tool(
            name="set-relationship",
            description="Adds a relationship between an entity and another entity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_entity_id": {
                        "type": "id",
                        "description": "Id of the source/from entity. Id must come from set-entity."
                    },
                    "target_entity_id": {
                        "type": "id",
                        "description": "Id of the target/to entity. Id must come from set-entity."
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Description of the relationship, such as uses, created, leads, etc."
                    }
                },
                "required": ["source_entity_id", "target_entity_id", "relation_type"]
            }
        ),
        types.Tool(
            name="search-entity",
            description="Searches for an entity in the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Name of the entity to search for, like the person, project, or anything else. Will try to find the most relevant info."
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_get_prompt(
    name: str,
    arguments: dict[str, str]
) -> list[types.TextContent]:
    global db
    if name == "set-entity":            
        name = arguments["name"]
        entity_type = arguments["entity_type"]
        new_content = arguments["content"]

        # First try to get existing content
        existing_content = await db.fetchval("SELECT content FROM entities WHERE name = $1;", name)

        if existing_content:
            updated_content = existing_content + new_content
            
            # Create embedding based on old + new content
            combined = json.dumps({"name": name, "entity_type": entity_type, "content": updated_content})
            embedding = str(model.encode(combined).tolist())
            
            id = await db.fetchval("UPDATE entities SET content = $1, embedding = $2 WHERE name = $3 RETURNING id;", updated_content, embedding, name)
            return [types.TextContent(type="text", text=f"Updated existing entity '{id}' with new content.")]
        else:
            # Create new entry with just the new content
            combined = json.dumps({"name": name, "entity_type": entity_type, "content": new_content})
            embedding = str(model.encode(combined).tolist())
            
            id = await db.fetchval("INSERT INTO entities (name, type, content, embedding) VALUES ($1, $2, $3, $4) RETURNING id;", name, entity_type, new_content, embedding)
            return [types.TextContent(type="text", text=f"Inserted new entity as id {id} into memory.")]
        

    elif name == "set-relationship":
        entity1 = arguments["source_entity_id"]
        entity2 = arguments["target_entity_id"]
        relation_type = arguments["relation_type"]

        await db.execute("INSERT INTO relations (source_entity_id, target_entity_id, relation_type) VALUES ($1, $2, $3);", entity1, entity2, relation_type)

        return [types.TextContent(type="text", text=f"Created relationship.")]

    elif name == "search-entity":
        query = arguments["query"]
        embedding = str(model.encode(query).tolist())
        
        # First get the initial matches
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
            return [types.TextContent(type="text", text="No matches found.")]

        ret = []
        for row in rows:
            ret.append({"id": row["id"], "name": row["name"], "type": row["type"], "content": row["content"]})

        return [types.TextContent(type="text", text=str(ret))]

    else:
        return [types.TextContent(type="text", text="Something went wrong")]
    


async def run():
    global db
    db = await asyncpg.create_pool(database="tennisbowling", user="tennisbowling", host="mediacenter2", password="tennispass")

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="Knowledge Graph",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())