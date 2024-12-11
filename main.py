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
            description="Saves an entity in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "name of the entity, like the person or project or whatever"
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "What the entity is, like a person, project, business, class, etc"
                    },
                    "content": {
                        "type": "list of strings",
                        "description": "list of strings in python form that describes properties of the entity (like what they do, description, etc)"
                    }
                },
                "required": ["name", "entity_type", "content"]
            }
        ),
        types.Tool(
            name="search-entity",
            description="Searches for an entity in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "name of the entity to search for, like the person or project or whatever"
                    }
                },
                "required": ["name"]
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
        content = arguments["content"]

        combined = json.dumps({"name": name, "entity_type": entity_type, "content": content})
        embedding = str(model.encode(combined).tolist())
        
        await db.execute("INSERT INTO entities (name, type, content, embedding) VALUES ($1, $2, $3, $4);", name, entity_type, content, embedding)

        return [types.TextContent(type="text", text="Inserted entity into memory.")]
    elif name == "search-entity":
        query = arguments["query"]
        embedding = str(model.encode(query).tolist())

        rows = await db.fetchmany("SELECT name, type, content FROM entities ORDER BY embedding <=> $1 LIMIT 3;", embedding)
        ret = []
        for row in rows:
            ret.append({"name": row["name"], "type": row["type"], "content": row["content"]})

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