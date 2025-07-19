from typing import Dict, Any, List
import httpx
from models import MCPTool
from config import DJANGO_BASE_URL, HTTP_CLIENT_TIMEOUT, logger


class MCPServer:
    """MCP Server implementation that calls Django backend for tools"""

    def __init__(self):
        self.django_base_url = DJANGO_BASE_URL
        self.tools: Dict[str, MCPTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default MCP tools - schemas only, execution handled by Django"""
        
        # Calculator tool
        self.register_tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        )

        # GEOSPATIAL ANALYSIS TOOLS
        self._register_geospatial_tools()
        
        # HEAP VOLUME ANALYSIS TOOLS
        self._register_heap_volume_tools()

    def _register_geospatial_tools(self):
        """Register geospatial analysis tools"""
        
        # List available layers
        self.register_tool(
            name="list_layers",
            description="List available geospatial layers with their names and IDs",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of layers to return (default: 10, max: 50)",
                        "default": 10,
                    },
                },
            },
        )

        # Find layer by name
        self.register_tool(
            name="find_layer_by_name",
            description="Find a layer by its name and get basic information",
            parameters={
                "type": "object",
                "properties": {
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to find (case-insensitive partial match)",
                    }
                },
                "required": ["layer_name"],
            },
        )

        # Get layer information
        self.register_tool(
            name="get_layer_info",
            description="Get detailed information about a specific layer",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "string",
                        "description": "UUID of the layer to analyze",
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to analyze (alternative to layer_id)",
                    },
                },
            },
        )

    def _register_heap_volume_tools(self):
        """Register heap volume analysis tools"""
        
        # List heap clusters
        self.register_tool(
            name="list_heap_clusters",
            description="List heap clusters with volume summaries, optionally filtered by site or iteration",
            parameters={
                "type": "object",
                "properties": {
                    "site_name": {
                        "type": "string",
                        "description": "Filter by site name (case-insensitive partial match)",
                    },
                    "iteration_name": {
                        "type": "string", 
                        "description": "Filter by iteration name (case-insensitive partial match)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of heap clusters to return (default: 10, max: 50)",
                        "default": 10,
                    },
                },
            },
        )

        # Get heap cluster details
        self.register_tool(
            name="get_heap_cluster_details",
            description="Get detailed information about a specific heap cluster including all volume data",
            parameters={
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "string",
                        "description": "UUID of the heap cluster to analyze",
                    },
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the heap cluster (alternative to cluster_id)",
                    },
                },
            },
        )

        # List heaps in cluster
        self.register_tool(
            name="list_heaps_in_cluster", 
            description="List individual heaps within a specific heap cluster with their volume data",
            parameters={
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "string",
                        "description": "UUID of the heap cluster",
                    },
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the heap cluster (alternative to cluster_id)",
                    },
                },
            },
        )

        # Get volume summary by site
        self.register_tool(
            name="get_volume_summary_by_site",
            description="Get aggregated volume summary for all heap clusters in a site",
            parameters={
                "type": "object",
                "properties": {
                    "site_name": {
                        "type": "string",
                        "description": "Name of the site to analyze",
                    },
                    "site_id": {
                        "type": "string", 
                        "description": "UUID of the site (alternative to site_name)",
                    },
                },
            },
        )

        # Get heap clusters from multiple iterations
        self.register_tool(
            name="get_heap_clusters_from_multiple_iterations",
            description="Get heap clusters data from multiple iterations at once",
            parameters={
                "type": "object",
                "properties": {
                    "iteration_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of iteration names to fetch heap clusters from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of heap clusters to return per iteration (default: 10, max: 20)",
                        "default": 10,
                    },
                },
                "required": ["iteration_names"],
            },
        )

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any]):
        """Register a new MCP tool"""
        self.tools[name] = MCPTool(
            name=name, description=description, parameters=parameters
        )
        logger.info(f"Registered MCP tool: {name}")

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], user_query: str = ""
    ) -> Dict[str, Any]:
        """Execute an MCP tool by calling Django backend"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                logger.info(f"Calling Django MCP tool: {tool_name} with arguments: {arguments}")
                
                # Call Django MCP endpoint
                response = await client.post(
                    f"{self.django_base_url}/ai-chat/api/mcp/execute_tool/",
                    json={"tool_name": tool_name, "arguments": arguments, "user_query": user_query},
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Django MCP tool {tool_name} executed successfully")
                    
                    # Check if the result contains HTML content or chart data from Django
                    if result.get("content_type") == "text/html":
                        logger.info(f"Tool {tool_name} returned HTML content for charts")
                        # Return the HTML content directly as the tool result
                        return {
                            "success": True,
                            "result": result.get("result", ""),
                            "content_type": "text/html"
                        }
                    elif result.get("content_type") == "application/json" and result.get("chart_type"):
                        logger.info(f"Tool {tool_name} returned chart data: {result.get('chart_type')}")
                        # Return the chart data directly as the tool result
                        return {
                            "success": True,
                            "result": result.get("result", ""),
                            "content_type": "application/json",
                            "chart_type": result.get("chart_type")
                        }
                    else:
                        # Return the regular tool result
                        return result
                else:
                    error_msg = f"Django MCP tool {tool_name} failed with status {response.status_code}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tools schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ] 