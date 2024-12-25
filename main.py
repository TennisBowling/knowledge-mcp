from sentence_transformers import SentenceTransformer
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import asyncpg
import json
import aiohttp

server = Server("knowledge-graph")

client = None


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
    global client
    try:
        if name == "set-entity":            

            # arguments is the same as what the server will try to decode
            async with client.post("http://enzopc:8000/set_entity", json=arguments) as resp:
                return [types.TextContent(type="text", text=(await resp.json())["message"])]
            

        elif name == "set-relationship":
            async with client.post("http://enzopc:8000/set_relationship", json=arguments) as resp:
                return [types.TextContent(type="text", text=(await resp.json())["message"])]

        elif name == "search-entity":
            async with client.post("http://enzopc:8000/search_entity", json=arguments) as resp:
                return [types.TextContent(type="text", text=await resp.text())]
        
        else:
            return [types.TextContent(type="text", text="Could not find tool requested.")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"An error occurred: {e}")]
    


async def run():
    global client
    client = aiohttp.ClientSession()
    
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
