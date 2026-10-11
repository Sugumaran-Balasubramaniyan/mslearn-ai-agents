import os
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MCPTool, PromptAgentDefinition
from openai.types.responses.response_input_param import McpApprovalResponse, ResponseInputParam


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


with (
    InteractiveBrowserCredential(tenant_id=tenant_id) as credential,
    AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
    project_client.get_openai_client() as openai_client,
):
    mcp_tool = MCPTool(
        server_label="api-specs",
        server_url="https://learn.microsoft.com/api/mcp",
        require_approval="always",
    )

    agent = project_client.agents.create_version(
        agent_name="MyAgent",
        definition=PromptAgentDefinition(
            model=model_deployment,
            instructions=(
                "You are a helpful agent that can use MCP tools to assist users. "
                "Use the available MCP tools to answer questions and perform tasks."
            ),
            tools=[mcp_tool],
        ),
    )
    print(f"Agent created (id: {agent.id}, name: {agent.name}, version: {agent.version})")

    try:
        conversation = openai_client.conversations.create()
        print(f"Created conversation (id: {conversation.id})")

        response = openai_client.responses.create(
            conversation=conversation.id,
            input="Give me the Azure CLI commands to create an Azure Container App with a managed identity.",
            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
        )

        while True:
            input_list: ResponseInputParam = []
            for item in response.output:
                if item.type == "mcp_approval_request" and item.server_label == "api-specs" and item.id:
                    input_list.append(
                        McpApprovalResponse(
                            type="mcp_approval_response",
                            approve=True,
                            approval_request_id=item.id,
                        )
                    )

            if not input_list:
                break

            response = openai_client.responses.create(
                input=input_list,
                previous_response_id=response.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            )

        print(f"\nAgent response: {response.output_text}")
    finally:
        project_client.agents.delete_version(
            agent_name=agent.name,
            agent_version=agent.version,
        )
        print("Agent deleted")
