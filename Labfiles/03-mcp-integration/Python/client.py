import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam


load_dotenv(Path(__file__).with_name(".env"))
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
tenant_id = os.getenv("AZURE_TENANT_ID")

missing_settings = [
    name
    for name, value in (
        ("PROJECT_ENDPOINT", project_endpoint),
        ("MODEL_DEPLOYMENT_NAME", model_deployment),
        ("AZURE_TENANT_ID", tenant_id),
    )
    if not value
]
if missing_settings:
    raise ValueError(
        f"Set these missing .env settings before running this script: {', '.join(missing_settings)}"
    )


async def connect_to_server(exit_stack: AsyncExitStack) -> ClientSession:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("server.py"))],
    )
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport

    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    response = await session.list_tools()
    print("\nConnected to server with tools:", [tool.name for tool in response.tools])
    return session


def _tool_output_text(result) -> str:
    if result.is_error:
        return "Tool call failed: " + " ".join(
            item.text for item in result.content if hasattr(item, "text")
        )

    text_parts = [item.text for item in result.content if hasattr(item, "text")]
    if text_parts:
        return "\n".join(text_parts)
    if result.structured_content is not None:
        return json.dumps(result.structured_content)
    return ""


async def chat_loop(session: ClientSession) -> None:
    with (
        InteractiveBrowserCredential(tenant_id=tenant_id) as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        response = await session.list_tools()
        tools = response.tools

        async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
            result = await session.call_tool(tool_name, arguments)
            return _tool_output_text(result)

        functions_dict = {tool.name: tool for tool in tools}
        mcp_function_tools = [
            FunctionTool(
                name=tool.name,
                description=tool.description or tool.name,
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=True,
            )
            for tool in tools
        ]

        agent = project_client.agents.create_version(
            agent_name="inventory-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions=(
                    "You are an inventory assistant. Use the inventory and weekly sales tools "
                    "to answer questions. Recommend restock if item inventory is below 10 and "
                    "weekly sales are above 15. Recommend clearance if item inventory is "
                    "above 20 and weekly sales are below 5."
                ),
                tools=mcp_function_tools,
            ),
        )
        print(f"Inventory agent created (id: {agent.id}, version: {agent.version})")

        try:
            conversation = openai_client.conversations.create()
            while True:
                user_input = input(
                    "Enter a prompt for the inventory agent. Use 'quit' to exit.\nUSER: "
                ).strip()
                if user_input.lower() == "quit":
                    print("Exiting chat.")
                    break
                if not user_input:
                    continue

                response = openai_client.responses.create(
                    conversation=conversation.id,
                    input=user_input,
                    extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                )

                while True:
                    if response.status == "failed":
                        print(f"Response failed: {response.error}")
                        break

                    function_outputs: ResponseInputParam = []
                    for item in response.output:
                        if item.type != "function_call":
                            continue

                        if item.name not in functions_dict:
                            output_text = f"Unknown MCP tool: {item.name}"
                        else:
                            result = await call_mcp_tool(item.name, json.loads(item.arguments))
                            output_text = result

                        function_outputs.append(
                            FunctionCallOutput(
                                type="function_call_output",
                                call_id=item.call_id,
                                output=output_text,
                            )
                        )

                    if not function_outputs:
                        print(f"Agent response: {response.output_text}")
                        break

                    response = openai_client.responses.create(
                        input=function_outputs,
                        conversation=conversation.id,
                        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                    )
        finally:
            project_client.agents.delete_version(
                agent_name=agent.name,
                agent_version=agent.version,
            )
            print("Deleted inventory agent.")


async def main() -> None:
    async with AsyncExitStack() as exit_stack:
        session = await connect_to_server(exit_stack)
        await chat_loop(session)


if __name__ == "__main__":
    asyncio.run(main())
