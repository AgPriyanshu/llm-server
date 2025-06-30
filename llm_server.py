import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import uuid
import re
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import ollama
from contextlib import asynccontextmanager
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    stream: bool = Field(default=False, description="Whether to stream the response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="MCP tools available")

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

class MCPTool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class MCPToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

class MCPServer:
    """MCP Server implementation for tool integration"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.tool_handlers: Dict[str, callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default MCP tools"""
        # File operations tool
        self.register_tool(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["file_path"]
            },
            handler=self._read_file_handler
        )
        
        # Enhanced web search tool
        self.register_tool(
            name="web_search",
            description="Search the web for information using DuckDuckGo",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            handler=self._web_search_handler
        )
        
        # Browser page content tool
        self.register_tool(
            name="get_page_content",
            description="Fetch and extract text content from a web page",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the web page to fetch"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum length of content to return (default: 2000)",
                        "default": 2000
                    }
                },
                "required": ["url"]
            },
            handler=self._get_page_content_handler
        )
        
        # Calculator tool
        self.register_tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            handler=self._calculate_handler
        )
        
        # URL validation tool
        self.register_tool(
            name="validate_url",
            description="Check if a URL is accessible and get basic information",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to validate"
                    }
                },
                "required": ["url"]
            },
            handler=self._validate_url_handler
        )
        
        # GEOSPATIAL ANALYSIS TOOLS
        
        # List available layers
        self.register_tool(
            name="list_layers",
            description="List all available geospatial layers with their names and IDs",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._list_layers_handler
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
                        "description": "Name of the layer to find (case-insensitive partial match)"
                    }
                },
                "required": ["layer_name"]
            },
            handler=self._find_layer_by_name_handler
        )
        
        # Get layer information
        self.register_tool(
            name="get_layer_info",
            description="Get detailed information about a specific layer",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer to analyze"
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to analyze (alternative to layer_id)"
                    }
                }
            },
            handler=self._get_layer_info_handler
        )
        
        # Analyze population data
        self.register_tool(
            name="analyze_population",
            description="Analyze population data in a layer and highlight features based on population thresholds",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer containing population data"
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer containing population data (alternative to layer_id)"
                    },
                    "population_field": {
                        "type": "string",
                        "description": "Name of the field containing population data (e.g., 'population', 'pop_est')"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Population threshold for highlighting (optional)"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["greater_than", "less_than", "between", "top_n"],
                        "description": "Type of analysis operation",
                        "default": "greater_than"
                    },
                    "threshold_max": {
                        "type": "number",
                        "description": "Maximum threshold for 'between' operation"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of top features for 'top_n' operation"
                    }
                },
                "required": ["population_field"]
            },
            handler=self._analyze_population_handler
        )
        
        # Filter features by attributes
        self.register_tool(
            name="filter_features",
            description="Filter features in a layer based on attribute conditions",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer to filter"
                    },
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "operator": {"type": "string", "enum": ["eq", "gt", "lt", "gte", "lte", "contains", "in"]},
                                "value": {"type": ["string", "number", "array"]}
                            }
                        },
                        "description": "Array of filter conditions"
                    }
                },
                "required": ["layer_id", "filters"]
            },
            handler=self._filter_features_handler
        )
        
        # Get attribute statistics
        self.register_tool(
            name="get_attribute_stats",
            description="Get statistical analysis of a specific attribute in a layer",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer"
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer (alternative to layer_id)"
                    },
                    "attribute_key": {
                        "type": "string",
                        "description": "Name of the attribute to analyze"
                    }
                },
                "required": ["attribute_key"]
            },
            handler=self._get_attribute_stats_handler
        )
        
        # Spatial analysis
        self.register_tool(
            name="spatial_analysis",
            description="Perform spatial analysis operations like buffer, intersection, proximity",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the primary layer"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["buffer", "intersection", "within_distance", "centroid", "area_calculation"],
                        "description": "Type of spatial operation"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Operation-specific parameters (distance for buffer, target_layer_id for intersection, etc.)"
                    }
                },
                "required": ["layer_id", "operation"]
            },
            handler=self._spatial_analysis_handler
        )
        
        # Create visualization
        self.register_tool(
            name="create_map_visualization",
            description="Create a map visualization with styled features based on analysis results",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer to visualize"
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to visualize (alternative to layer_id)"
                    },
                    "style_field": {
                        "type": "string",
                        "description": "Field to use for styling (optional)"
                    },
                    "style_type": {
                        "type": "string",
                        "enum": ["choropleth", "categorical", "graduated", "simple"],
                        "description": "Type of visualization style",
                        "default": "simple"
                    },
                    "color_scheme": {
                        "type": "string",
                        "enum": ["blues", "reds", "greens", "viridis", "plasma"],
                        "description": "Color scheme for visualization",
                        "default": "blues"
                    },
                    "feature_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific feature IDs to highlight (optional)"
                    }
                }
            },
            handler=self._create_visualization_handler
        )
        
        # Comprehensive layer analysis
        self.register_tool(
            name="analyze_layer_attributes",
            description="Perform comprehensive attribute analysis on a layer including statistics, data types, and insights",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer to analyze"
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to analyze (alternative to layer_id)"
                    },
                    "include_statistics": {
                        "type": "boolean",
                        "description": "Include detailed statistics for numeric attributes",
                        "default": True
                    },
                    "include_samples": {
                        "type": "boolean", 
                        "description": "Include sample values for each attribute",
                        "default": True
                    }
                }
            },
            handler=self._analyze_layer_attributes_handler
        )
    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], handler: callable):
        """Register a new MCP tool"""
        self.tools[name] = MCPTool(
            name=name,
            description=description,
            parameters=parameters
        )
        self.tool_handlers[name] = handler
        logger.info(f"Registered MCP tool: {name}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool"""
        if tool_name not in self.tool_handlers:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        try:
            handler = self.tool_handlers[tool_name]
            result = await handler(arguments)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # GEOSPATIAL ANALYSIS HANDLERS
    
    async def _resolve_layer_id(self, layer_id: Optional[int] = None, layer_name: Optional[str] = None) -> Optional[int]:
        """Resolve layer name to ID, or return provided ID"""
        if layer_id:
            return layer_id
        
        if not layer_name:
            return None
            
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get("http://localhost:8000/map/layers/")
                
                if response.status_code == 200:
                    response_data = response.json()
                    layers = response_data.get('data', []) if isinstance(response_data, dict) else response_data
                    for layer in layers:
                        if layer_name.lower() in layer['name'].lower():
                            return layer['id']
                    return None
                else:
                    return None
        except Exception as e:
            logger.error(f"Error resolving layer name: {str(e)}")
            return None
    
    async def _list_layers_handler(self, args: Dict[str, Any]) -> str:
        """Handler for listing all available layers"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get("http://localhost:8000/map/layers/")
                
                if response.status_code == 200:
                    response_data = response.json()
                    layers = response_data.get('data', []) if isinstance(response_data, dict) else response_data
                    if layers:
                        result = "Available Geospatial Layers:\n"
                        for layer in layers:
                            result += f"- ID: {layer['id']}, Name: {layer['name']}\n"
                            result += f"  Description: {layer.get('description', 'No description')}\n"
                            result += f"  Features: {layer.get('feature_count', 'Unknown')} features\n"
                            result += f"  Created: {layer.get('created_at', 'Unknown')}\n\n"
                        return result
                    else:
                        return "No geospatial layers found. Upload a shapefile to get started."
                else:
                    return f"Error fetching layers: HTTP {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error listing layers: {str(e)}")
            return f"Error listing layers: {str(e)}"
    
    async def _find_layer_by_name_handler(self, args: Dict[str, Any]) -> str:
        """Handler for finding a layer by name"""
        layer_name = args.get("layer_name")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get("http://localhost:8000/map/layers/")
                
                if response.status_code == 200:
                    response_data = response.json()
                    layers = response_data.get('data', []) if isinstance(response_data, dict) else response_data
                    matches = []
                    
                    for layer in layers:
                        if layer_name.lower() in layer['name'].lower():
                            matches.append(layer)
                    
                    if not matches:
                        return f"No layers found matching '{layer_name}'"
                    
                    result = f"Found {len(matches)} layer(s) matching '{layer_name}':\n\n"
                    for layer in matches:
                        result += f"- ID: {layer['id']}, Name: {layer['name']}\n"
                        result += f"  Description: {layer.get('description', 'No description')}\n"
                        result += f"  Features: {layer.get('feature_count', 'Unknown')} features\n"
                        result += f"  Created: {layer.get('created_at', 'Unknown')}\n\n"
                    
                    return result
                else:
                    return f"Error fetching layers: HTTP {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error finding layer by name: {str(e)}")
            return f"Error finding layer by name: {str(e)}"
    
    async def _get_layer_info_handler(self, args: Dict[str, Any]) -> str:
        """Handler for getting detailed layer information"""
        layer_id = await self._resolve_layer_id(
            args.get("layer_id"), 
            args.get("layer_name")
        )
        
        if not layer_id:
            return "Please provide either a layer_id or layer_name"
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Get layer details
                layer_response = await client.get(f"http://localhost:8000/map/layers/{layer_id}/")
                
                if layer_response.status_code != 200:
                    return f"Layer {layer_id} not found"
                
                layer_response_data = layer_response.json()
                layer_data = layer_response_data.get('data', {}) if isinstance(layer_response_data, dict) else layer_response_data
                
                # Get attribute summary
                attr_response = await client.get(f"http://localhost:8000/map/layers/{layer_id}/attributes/summary/")
                
                result = f"Layer Information:\n"
                result += f"ID: {layer_data['id']}\n"
                result += f"Name: {layer_data['name']}\n"
                result += f"Description: {layer_data.get('description', 'No description')}\n"
                result += f"Created: {layer_data.get('created_at', 'Unknown')}\n\n"
                
                if attr_response.status_code == 200:
                    attr_response_data = attr_response.json()
                    attr_data = attr_response_data.get('data', {}) if isinstance(attr_response_data, dict) else attr_response_data
                    if attr_data.get('attribute_summary'):
                        result += "Available Attributes:\n"
                        for attr_name, attr_info in attr_data['attribute_summary'].items():
                            result += f"- {attr_name}: {attr_info.get('data_type', 'unknown')} "
                            result += f"({attr_info.get('count', 0)} values)\n"
                            if attr_info.get('sample_values'):
                                result += f"  Sample values: {', '.join(map(str, attr_info['sample_values'][:3]))}\n"
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting layer info: {str(e)}")
            return f"Error getting layer info: {str(e)}"
    
    async def _analyze_population_handler(self, args: Dict[str, Any]) -> str:
        """Handler for population analysis"""
        layer_id = await self._resolve_layer_id(
            args.get("layer_id"), 
            args.get("layer_name")
        )
        
        if not layer_id:
            return "Please provide either a layer_id or layer_name"
            
        population_field = args.get("population_field")
        threshold = args.get("threshold")
        operation = args.get("operation", "greater_than")
        threshold_max = args.get("threshold_max")
        n = args.get("n")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # First, get attribute statistics to understand the data
                stats_response = await client.get(
                    f"http://localhost:8000/map/layers/{layer_id}/attributes/{population_field}/stats/"
                )
                
                if stats_response.status_code != 200:
                    return f"Population field '{population_field}' not found in layer {layer_id}"
                
                stats_response_data = stats_response.json()
                stats_data = stats_response_data.get('data', {}) if isinstance(stats_response_data, dict) else stats_response_data
                statistics = stats_data.get('statistics', {})
                
                # Build filter based on operation
                filters = []
                
                if operation == "greater_than" and threshold is not None:
                    filters = [{"field": population_field, "operator": "gt", "value": threshold}]
                elif operation == "less_than" and threshold is not None:
                    filters = [{"field": population_field, "operator": "lt", "value": threshold}]
                elif operation == "between" and threshold is not None and threshold_max is not None:
                    filters = [
                        {"field": population_field, "operator": "gte", "value": threshold},
                        {"field": population_field, "operator": "lte", "value": threshold_max}
                    ]
                elif operation == "top_n" and n is not None:
                    # For top_n, we'll need to use the statistics to determine threshold
                    if statistics.get('max'):
                        # Use a high threshold to get top features (this is a simplified approach)
                        percentile_90 = statistics.get('max', 0) * 0.9
                        filters = [{"field": population_field, "operator": "gte", "value": percentile_90}]
                
                if not filters:
                    return "Invalid parameters for population analysis. Please provide appropriate threshold values."
                
                # Filter features
                filter_response = await client.post(
                    f"http://localhost:8000/map/layers/{layer_id}/features/filter/",
                    json={"filters": filters}
                )
                
                if filter_response.status_code != 200:
                    return f"Error filtering features: {filter_response.text}"
                
                filter_response_data = filter_response.json()
                filter_data = filter_response_data.get('data', {}) if isinstance(filter_response_data, dict) else filter_response_data
                
                # Build result
                result = f"Population Analysis Results:\n"
                result += f"Field analyzed: {population_field}\n"
                result += f"Operation: {operation}\n"
                
                if statistics:
                    result += f"Population Statistics:\n"
                    result += f"- Min: {statistics.get('min', 'N/A'):,.0f}\n"
                    result += f"- Max: {statistics.get('max', 'N/A'):,.0f}\n"
                    result += f"- Average: {statistics.get('avg', 'N/A'):,.0f}\n"
                    result += f"- Total features: {statistics.get('count', 'N/A')}\n\n"
                
                filtered_features = filter_data.get('features', [])
                result += f"Features matching criteria: {len(filtered_features)}\n\n"
                
                if filtered_features:
                    result += "Matching Features:\n"
                    for feature in filtered_features[:10]:  # Show first 10
                        name = "Unknown"
                        population = "Unknown"
                        
                        for attr in feature.get('attributes', []):
                            if attr['key'].lower() in ['name', 'country', 'admin', 'name_en']:
                                name = attr['value']
                            elif attr['key'] == population_field:
                                population = f"{float(attr['value']):,.0f}"
                        
                        result += f"- {name}: {population}\n"
                    
                    if len(filtered_features) > 10:
                        result += f"... and {len(filtered_features) - 10} more features\n"
                
                return result
                
        except Exception as e:
            logger.error(f"Error in population analysis: {str(e)}")
            return f"Error in population analysis: {str(e)}"
    
    async def _filter_features_handler(self, args: Dict[str, Any]) -> str:
        """Handler for filtering features by attributes"""
        layer_id = args.get("layer_id")
        filters = args.get("filters", [])
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"http://localhost:8000/map/layers/{layer_id}/features/filter/",
                    json={"filters": filters}
                )
                
                if response.status_code != 200:
                    return f"Error filtering features: {response.text}"
                
                response_data = response.json()
                data = response_data.get('data', {}) if isinstance(response_data, dict) else response_data
                features = data.get('features', [])
                
                result = f"Filter Results:\n"
                result += f"Applied filters: {len(filters)} conditions\n"
                result += f"Matching features: {len(features)}\n\n"
                
                if features:
                    result += "Sample of matching features:\n"
                    for feature in features[:5]:  # Show first 5
                        result += f"Feature ID {feature['id']}:\n"
                        for attr in feature.get('attributes', []):
                            result += f"  {attr['key']}: {attr['value']}\n"
                        result += "\n"
                
                return result
                
        except Exception as e:
            logger.error(f"Error filtering features: {str(e)}")
            return f"Error filtering features: {str(e)}"
    
    async def _get_attribute_stats_handler(self, args: Dict[str, Any]) -> str:
        """Handler for getting attribute statistics"""
        layer_id = await self._resolve_layer_id(
            args.get("layer_id"), 
            args.get("layer_name")
        )
        
        if not layer_id:
            return "Please provide either a layer_id or layer_name"
            
        attribute_key = args.get("attribute_key")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(
                    f"http://localhost:8000/map/layers/{layer_id}/attributes/{attribute_key}/stats/"
                )
                
                if response.status_code != 200:
                    return f"Attribute '{attribute_key}' not found in layer {layer_id}"
                
                response_data = response.json()
                data = response_data.get('data', {}) if isinstance(response_data, dict) else response_data
                statistics = data.get('statistics', {})
                
                result = f"Attribute Statistics for '{attribute_key}':\n"
                result += f"Data type: {statistics.get('data_type', 'unknown')}\n"
                result += f"Total values: {statistics.get('count', 0)}\n"
                
                if statistics.get('data_type') in ['integer', 'float']:
                    result += f"Minimum: {statistics.get('min', 'N/A')}\n"
                    result += f"Maximum: {statistics.get('max', 'N/A')}\n"
                    result += f"Average: {statistics.get('avg', 'N/A'):.2f}\n"
                    result += f"Sum: {statistics.get('sum', 'N/A')}\n"
                
                if statistics.get('unique_values'):
                    result += f"Unique values: {len(statistics['unique_values'])}\n"
                    result += f"Sample values: {', '.join(map(str, statistics['unique_values'][:10]))}\n"
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting attribute stats: {str(e)}")
            return f"Error getting attribute stats: {str(e)}"
    
    async def _spatial_analysis_handler(self, args: Dict[str, Any]) -> str:
        """Handler for spatial analysis operations"""
        layer_id = args.get("layer_id")
        operation = args.get("operation")
        parameters = args.get("parameters", {})
        
        try:
            # This would require additional Django endpoints for spatial operations
            # For now, we'll provide a basic response
            result = f"Spatial Analysis Request:\n"
            result += f"Layer ID: {layer_id}\n"
            result += f"Operation: {operation}\n"
            result += f"Parameters: {parameters}\n\n"
            
            if operation == "buffer":
                distance = parameters.get("distance", 1000)
                result += f"Buffer analysis would create {distance}m buffers around all features.\n"
            elif operation == "area_calculation":
                result += "Area calculation would compute the area of all polygon features.\n"
            elif operation == "centroid":
                result += "Centroid calculation would find the center point of all features.\n"
            else:
                result += f"Spatial operation '{operation}' is planned but not yet implemented.\n"
            
            result += "Note: Advanced spatial operations require additional backend implementation."
            
            return result
            
        except Exception as e:
            logger.error(f"Error in spatial analysis: {str(e)}")
            return f"Error in spatial analysis: {str(e)}"
    
    async def _create_visualization_handler(self, args: Dict[str, Any]) -> str:
        """Handler for creating map visualizations"""
        layer_id = await self._resolve_layer_id(
            args.get("layer_id"), 
            args.get("layer_name")
        )
        
        if not layer_id:
            return "Please provide either a layer_id or layer_name"
            
        style_field = args.get("style_field")
        style_type = args.get("style_type", "simple")
        color_scheme = args.get("color_scheme", "blues")
        feature_ids = args.get("feature_ids")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Get layer GeoJSON
                response = await client.get(f"http://localhost:8000/map/layers/{layer_id}/geojson/")
                
                if response.status_code != 200:
                    return f"Error getting layer data for visualization: {response.text}"
                
                geojson_data = response.json()
                
                visualization_config = {
                    "type": "map_visualization",
                    "layer_id": layer_id,
                    "style_type": style_type,
                    "color_scheme": color_scheme,
                    "geojson": geojson_data,
                    "feature_count": len(geojson_data.get('features', []))
                }
                
                if style_field:
                    visualization_config["style_field"] = style_field
                
                if feature_ids:
                    visualization_config["highlighted_features"] = feature_ids
                
                result = f"Map Visualization Created:\n"
                result += f"Layer ID: {layer_id}\n"
                result += f"Style Type: {style_type}\n"
                result += f"Color Scheme: {color_scheme}\n"
                result += f"Features: {visualization_config['feature_count']}\n"
                
                if style_field:
                    result += f"Styled by: {style_field}\n"
                
                if feature_ids:
                    result += f"Highlighted features: {len(feature_ids)}\n"
                
                # The frontend will receive this visualization data through the chat context
                result += f"\nVisualization data ready for map display."
                
                return result
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return f"Error creating visualization: {str(e)}"
    
    async def _analyze_layer_attributes_handler(self, args: Dict[str, Any]) -> str:
        """Handler for comprehensive layer attribute analysis"""
        layer_id = await self._resolve_layer_id(
            args.get("layer_id"), 
            args.get("layer_name")
        )
        
        if not layer_id:
            # If no layer specified, list available layers
            return await self._list_layers_handler({})
        
        include_statistics = args.get("include_statistics", True)
        include_samples = args.get("include_samples", True)
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Get layer basic info
                layer_response = await client.get(f"http://localhost:8000/map/layers/{layer_id}/")
                
                if layer_response.status_code != 200:
                    return f"Layer {layer_id} not found"
                
                layer_response_data = layer_response.json()
                layer_data = layer_response_data.get('data', {}) if isinstance(layer_response_data, dict) else layer_response_data
                
                # Get attribute summary
                attr_response = await client.get(f"http://localhost:8000/map/layers/{layer_id}/attributes/summary/")
                
                result = f"📊 COMPREHENSIVE LAYER ANALYSIS\n"
                result += f"{'='*50}\n\n"
                
                result += f"🗺️  LAYER INFORMATION:\n"
                result += f"   Name: {layer_data['name']}\n"
                result += f"   ID: {layer_data['id']}\n"
                result += f"   Description: {layer_data.get('description', 'No description')}\n"
                result += f"   Geometry Type: {layer_data.get('geometry_type', 'Unknown')}\n"
                result += f"   Created: {layer_data.get('created_at', 'Unknown')}\n\n"
                
                if attr_response.status_code == 200:
                    attr_response_data = attr_response.json()
                    attr_data = attr_response_data.get('data', {}) if isinstance(attr_response_data, dict) else attr_response_data
                    attributes = attr_data.get('attribute_summary', {})
                    
                    if attributes:
                        result += f"📋 ATTRIBUTE ANALYSIS:\n"
                        result += f"   Total Attributes: {len(attributes)}\n\n"
                        
                        # Categorize attributes by type
                        numeric_attrs = []
                        text_attrs = []
                        date_attrs = []
                        
                        for attr_name, attr_info in attributes.items():
                            data_type = attr_info.get('data_type', 'unknown')
                            if data_type in ['integer', 'float']:
                                numeric_attrs.append((attr_name, attr_info))
                            elif data_type in ['date', 'datetime']:
                                date_attrs.append((attr_name, attr_info))
                            else:
                                text_attrs.append((attr_name, attr_info))
                        
                        # Numeric attributes analysis
                        if numeric_attrs:
                            result += f"🔢 NUMERIC ATTRIBUTES ({len(numeric_attrs)}):\n"
                            for attr_name, attr_info in numeric_attrs:
                                result += f"   • {attr_name}: {attr_info.get('data_type', 'unknown')} "
                                result += f"({attr_info.get('count', 0)} values)\n"
                                
                                if include_samples and attr_info.get('sample_values'):
                                    samples = ', '.join(map(str, attr_info['sample_values'][:3]))
                                    result += f"     Sample values: {samples}\n"
                                
                                # Get detailed statistics for numeric fields
                                if include_statistics:
                                    try:
                                        stats_response = await client.get(
                                            f"http://localhost:8000/map/layers/{layer_id}/attributes/{attr_name}/stats/"
                                        )
                                        if stats_response.status_code == 200:
                                            stats_response_data = stats_response.json()
                                            stats_data = stats_response_data.get('data', {}) if isinstance(stats_response_data, dict) else stats_response_data
                                            statistics = stats_data.get('statistics', {})
                                            if statistics:
                                                result += f"     Min: {statistics.get('min', 'N/A'):,.2f}, "
                                                result += f"Max: {statistics.get('max', 'N/A'):,.2f}, "
                                                result += f"Avg: {statistics.get('avg', 'N/A'):,.2f}\n"
                                    except Exception:
                                        pass
                                result += "\n"
                        
                        # Text attributes analysis  
                        if text_attrs:
                            result += f"📝 TEXT ATTRIBUTES ({len(text_attrs)}):\n"
                            for attr_name, attr_info in text_attrs:
                                result += f"   • {attr_name}: {attr_info.get('data_type', 'unknown')} "
                                result += f"({attr_info.get('count', 0)} values)\n"
                                
                                if include_samples and attr_info.get('sample_values'):
                                    samples = ', '.join(map(str, attr_info['sample_values'][:3]))
                                    result += f"     Sample values: {samples}\n"
                                result += "\n"
                        
                        # Date attributes analysis
                        if date_attrs:
                            result += f"📅 DATE ATTRIBUTES ({len(date_attrs)}):\n"
                            for attr_name, attr_info in date_attrs:
                                result += f"   • {attr_name}: {attr_info.get('data_type', 'unknown')} "
                                result += f"({attr_info.get('count', 0)} values)\n"
                                
                                if include_samples and attr_info.get('sample_values'):
                                    samples = ', '.join(map(str, attr_info['sample_values'][:3]))
                                    result += f"     Sample values: {samples}\n"
                                result += "\n"
                        
                        # Analysis insights
                        result += f"💡 ANALYSIS INSIGHTS:\n"
                        if numeric_attrs:
                            result += f"   • Found {len(numeric_attrs)} numeric attribute(s) suitable for statistical analysis\n"
                            result += f"   • Can perform population analysis, filtering, and visualization\n"
                        if text_attrs:
                            result += f"   • Found {len(text_attrs)} text attribute(s) for categorical analysis\n" 
                        if date_attrs:
                            result += f"   • Found {len(date_attrs)} date attribute(s) for temporal analysis\n"
                        
                        result += f"\n🔧 SUGGESTED ACTIONS:\n"
                        if numeric_attrs:
                            top_numeric = numeric_attrs[0][0]  # First numeric attribute
                            result += f"   • Try: analyze_population(layer_name='{layer_data['name']}', population_field='{top_numeric}')\n"
                            result += f"   • Try: get_attribute_stats(layer_name='{layer_data['name']}', attribute_key='{top_numeric}')\n"
                        result += f"   • Try: create_map_visualization(layer_name='{layer_data['name']}')\n"
                    else:
                        result += "No attributes found for this layer.\n"
                else:
                    result += f"Could not retrieve attribute information (HTTP {attr_response.status_code})\n"
                
                return result
                
        except Exception as e:
            logger.error(f"Error in layer analysis: {str(e)}")
            return f"Error in layer analysis: {str(e)}"

    async def _read_file_handler(self, args: Dict[str, Any]) -> str:
        """Handler for file reading tool"""
        file_path = args.get("file_path")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File content:\n{content}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def _web_search_handler(self, args: Dict[str, Any]) -> str:
        """Handler for web search tool with real search functionality"""
        query = args.get("query")
        max_results = args.get("max_results", 5)
        
        if not query:
            return "Error: No search query provided"
        
        try:
            # Use DuckDuckGo Instant Answer API for search
            async with httpx.AsyncClient(timeout=120.0) as client:
                # DuckDuckGo Instant Answer API
                ddg_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
                response = await client.get(ddg_url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    results = []
                    
                    # Add abstract if available
                    if data.get("Abstract"):
                        results.append(f"Summary: {data['Abstract']}")
                        if data.get("AbstractURL"):
                            results.append(f"Source: {data['AbstractURL']}")
                    
                    # Add instant answer if available
                    if data.get("Answer"):
                        results.append(f"Answer: {data['Answer']}")
                    
                    # Add definition if available
                    if data.get("Definition"):
                        results.append(f"Definition: {data['Definition']}")
                        if data.get("DefinitionURL"):
                            results.append(f"Source: {data['DefinitionURL']}")
                    
                    # Add related topics
                    if data.get("RelatedTopics"):
                        topics = data["RelatedTopics"][:max_results]
                        for i, topic in enumerate(topics, 1):
                            if isinstance(topic, dict) and topic.get("Text"):
                                results.append(f"{i}. {topic['Text']}")
                                if topic.get("FirstURL"):
                                    results.append(f"   URL: {topic['FirstURL']}")
                    
                    if results:
                        return f"Search results for '{query}':\n" + "\n".join(results)
                    else:
                        # Fallback to web scraping search
                        return await self._fallback_web_search(query, max_results)
                else:
                    return await self._fallback_web_search(query, max_results)
                    
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return f"Error performing web search: {str(e)}"
    
    async def _fallback_web_search(self, query: str, max_results: int) -> str:
        """Fallback web search using HTML scraping"""
        try:
            async with httpx.AsyncClient(
                timeout=120.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            ) as client:
                # Use DuckDuckGo HTML search as fallback
                search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                response = await client.get(search_url)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = []
                    
                    # Find search result elements
                    result_elements = soup.find_all('div', class_='result')[:max_results]
                    
                    for i, element in enumerate(result_elements, 1):
                        title_elem = element.find('a', class_='result__a')
                        snippet_elem = element.find('a', class_='result__snippet')
                        
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            url = title_elem.get('href', '')
                            
                            result_text = f"{i}. {title}"
                            if snippet_elem:
                                snippet = snippet_elem.get_text(strip=True)
                                result_text += f"\n   {snippet}"
                            if url:
                                result_text += f"\n   URL: {url}"
                            
                            results.append(result_text)
                    
                    if results:
                        return f"Search results for '{query}':\n" + "\n\n".join(results)
                    else:
                        return f"No search results found for '{query}'"
                else:
                    return f"Search request failed with status code: {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error in fallback web search: {str(e)}")
            return f"Error performing fallback web search: {str(e)}"
    
    async def _get_page_content_handler(self, args: Dict[str, Any]) -> str:
        """Handler for fetching web page content"""
        url = args.get("url")
        max_length = args.get("max_length", 2000)
        
        if not url:
            return "Error: No URL provided"
        
        try:
            async with httpx.AsyncClient(
                timeout=180.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            ) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Truncate if too long
                    if len(text) > max_length:
                        text = text[:max_length] + "..."
                    
                    return f"Content from {url}:\n\n{text}"
                else:
                    return f"Error: Unable to fetch page. Status code: {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error fetching page content: {str(e)}")
            return f"Error fetching page content: {str(e)}"
    
    async def _validate_url_handler(self, args: Dict[str, Any]) -> str:
        """Handler for URL validation"""
        url = args.get("url")
        
        if not url:
            return "Error: No URL provided"
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.head(url)
                
                info = {
                    "url": url,
                    "status_code": response.status_code,
                    "accessible": response.status_code < 400,
                    "content_type": response.headers.get("content-type", "Unknown"),
                    "content_length": response.headers.get("content-length", "Unknown")
                }
                
                return f"URL validation for {url}:\n" + "\n".join([f"{k}: {v}" for k, v in info.items()])
                
        except Exception as e:
            logger.error(f"Error validating URL: {str(e)}")
            return f"Error validating URL: {str(e)}"
    
    async def _calculate_handler(self, args: Dict[str, Any]) -> str:
        """Handler for calculator tool"""
        expression = args.get("expression")
        try:
            # Safe evaluation of mathematical expressions
            allowed_names = {
                k: v for k, v in __builtins__.items() if k in ['abs', 'round', 'min', 'max']
            }
            allowed_names.update({
                'pow': pow,
                'sqrt': lambda x: x ** 0.5,
                'pi': 3.14159265359,
                'e': 2.71828182846
            })
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tools schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]

class OllamaLLMServer:
    """Main LLM Server class with Ollama integration"""
    
    def __init__(self):
        self.ollama_client = ollama.AsyncClient()
        self.mcp_server = MCPServer()
        self.available_models = []
        
    async def initialize(self):
        """Initialize the server and load available models"""
        try:
            models = await self.ollama_client.list()
            self.available_models = [model['name'] for model in models['models']]
            logger.info(f"Available Ollama models: {self.available_models}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            self.available_models = []
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request"""
        if request.model not in self.available_models:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
        
        # Convert messages to Ollama format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Add tools to the conversation if provided
        tools_context = ""
        if request.tools:
            tools_schema = self.mcp_server.get_tools_schema()
            tools_context = f"""

You have access to the following tools. When the user asks for information that requires these tools, you MUST use them by calling the function in this format: function_name(parameter1="value1", parameter2="value2")

Available tools:
"""
            for tool in tools_schema:
                func = tool["function"]
                tools_context += f"- {func['name']}: {func['description']}\n"
                if func.get('parameters', {}).get('properties'):
                    tools_context += f"  Parameters: {', '.join(func['parameters']['properties'].keys())}\n"
            
            tools_context += """
When a user asks about layers, geospatial data, or analysis, you should use the appropriate tools. For example:
- "What layers are available?" -> use list_layers()
- "Analyze layer X" -> use find_layer_by_name(layer_name="X") then analyze_layer_attributes(layer_name="X")
- "Population statistics" -> use get_attribute_stats() or analyze_population()

Always use tools when the user's question requires data that you don't have directly.
"""
            messages[-1]["content"] += tools_context
        
        try:
            response = await self.ollama_client.chat(
                model=request.model,
                messages=messages,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            )
            
            # Process tool calls if present
            assistant_message = response['message']['content']
            
            # Check if we need to force tool usage based on user query
            user_query = messages[-1]["content"].lower()
            forced_tools = self._get_forced_tools(user_query)
            
            tool_calls = self._extract_tool_calls(assistant_message)
            
            # Add forced tools if no tools were detected but they should be used
            if not tool_calls and forced_tools:
                tool_calls = forced_tools
            
            if tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    result = await self.mcp_server.execute_tool(
                        tool_call['function']['name'],
                        tool_call['function']['arguments']
                    )
                    tool_results.append(result)
                
                # Add tool results to conversation and get final response
                messages.append({"role": "assistant", "content": assistant_message})
                messages.append({
                    "role": "user", 
                    "content": f"Tool execution results: {json.dumps(tool_results, indent=2)}"
                })
                
                final_response = await self.ollama_client.chat(
                    model=request.model,
                    messages=messages,
                    options={
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    }
                )
                assistant_message = final_response['message']['content']
            
            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_message
                    },
                    "finish_reason": "stop"
                }]
            )
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion request"""
        if request.model not in self.available_models:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Add tools context if provided
        if request.tools:
            tools_schema = self.mcp_server.get_tools_schema()
            tools_context = f"""

You have access to the following tools. When the user asks for information that requires these tools, you MUST use them by calling the function in this format: function_name(parameter1="value1", parameter2="value2")

Available tools:
"""
            for tool in tools_schema:
                func = tool["function"]
                tools_context += f"- {func['name']}: {func['description']}\n"
                if func.get('parameters', {}).get('properties'):
                    tools_context += f"  Parameters: {', '.join(func['parameters']['properties'].keys())}\n"
            
            tools_context += """
When a user asks about layers, geospatial data, or analysis, you should use the appropriate tools. For example:
- "What layers are available?" -> use list_layers()
- "Analyze layer X" -> use find_layer_by_name(layer_name="X") then analyze_layer_attributes(layer_name="X")
- "Population statistics" -> use get_attribute_stats() or analyze_population()

Always use tools when the user's question requires data that you don't have directly.
"""
            messages[-1]["content"] += tools_context
        
        try:
            stream = await self.ollama_client.chat(
                model=request.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            )
            
            chunk_id = str(uuid.uuid4())
            
            async for chunk in stream:
                content = chunk['message']['content']
                
                response_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None
                    }]
                }
                
                yield f"data: {json.dumps(response_chunk)}\n\n"
            
            # Send final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming completion: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from assistant message"""
        tool_calls = []
        
        # Look for function calls in various formats
        import re
        
        # Pattern 1: function_name(arguments)
        pattern1 = r'(\w+)\(([^)]*)\)'
        matches1 = re.findall(pattern1, content)
        
        # Pattern 2: explicit tool usage mentions
        pattern2 = r'(?:use|call|execute)\s+(\w+)(?:\(([^)]*)\))?'
        matches2 = re.findall(pattern2, content, re.IGNORECASE)
        
        # Pattern 3: "I'll use X tool" or "Let me check with X"
        pattern3 = r'(?:I\'ll use|let me (?:use|call|check with))\s+(\w+)'
        matches3 = re.findall(pattern3, content, re.IGNORECASE)
        
        all_matches = []
        
        # Process pattern 1 matches
        for function_name, args_str in matches1:
            if function_name in self.mcp_server.tools:
                all_matches.append((function_name, args_str))
        
        # Process pattern 2 matches  
        for function_name, args_str in matches2:
            if function_name in self.mcp_server.tools:
                all_matches.append((function_name, args_str or ""))
        
        # Process pattern 3 matches (no arguments)
        for function_name in matches3:
            if function_name in self.mcp_server.tools:
                all_matches.append((function_name, ""))
        
        # Also check if the user's question implies tool usage
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in ['what layers', 'available layers', 'list layers']):
            if not any(match[0] == 'list_layers' for match in all_matches):
                all_matches.append(('list_layers', ''))
        
        if any(phrase in content_lower for phrase in ['analyze', 'analysis', 'attributes']):
            # Look for layer names in quotes or after "layer"
            layer_match = re.search(r'(?:layer\s+["\']?(\w+)["\']?|["\']([^"\']+)["\']?\s+layer)', content_lower)
            if layer_match and not any(match[0].startswith('analyze') for match in all_matches):
                layer_name = layer_match.group(1) or layer_match.group(2)
                all_matches.append(('analyze_layer_attributes', f'layer_name="{layer_name}"'))
        
        for function_name, args_str in all_matches:
            try:
                # Parse arguments (key=value format)
                args = {}
                if args_str.strip():
                    # Handle both key=value and positional arguments
                    if '=' in args_str:
                        for arg in args_str.split(','):
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key.strip()] = value.strip().strip('"\'')
                    else:
                        # For simple cases like layer names
                        if function_name in ['find_layer_by_name', 'analyze_layer_attributes']:
                            args['layer_name'] = args_str.strip().strip('"\'')
                
                tool_calls.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": args
                    }
                })
            except Exception as e:
                logger.error(f"Error parsing tool call: {str(e)}")
        
        return tool_calls
    
    def _get_forced_tools(self, user_query: str) -> List[Dict[str, Any]]:
        """Force tool usage based on user query patterns"""
        forced_tools = []
        
        # Force list_layers for layer discovery queries
        if any(phrase in user_query for phrase in [
            'what layers', 'available layers', 'list layers', 'show layers',
            'layers are available', 'which layers', 'what maps', 'available maps'
        ]):
            forced_tools.append({
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": "list_layers",
                    "arguments": {}
                }
            })
        
        # Force layer analysis for analysis queries
        elif any(phrase in user_query for phrase in ['analyze', 'analysis', 'attributes']):
            # Try to extract layer name
            import re
            layer_patterns = [
                r'(?:layer\s+["\']?(\w+)["\']?)',
                r'(?:["\']([^"\']+)["\']?\s+layer)',
                r'(?:of\s+(?:the\s+)?["\']?([^"\']+?)["\']?(?:\s+layer)?)',
                r'(?:the\s+([a-zA-Z_]+)\s+(?:layer|data))'
            ]
            
            layer_name = None
            for pattern in layer_patterns:
                match = re.search(pattern, user_query, re.IGNORECASE)
                if match:
                    layer_name = match.group(1)
                    break
            
            if layer_name:
                forced_tools.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "analyze_layer_attributes",
                        "arguments": {"layer_name": layer_name}
                    }
                })
            else:
                # If no specific layer, list available layers first
                forced_tools.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "list_layers",
                        "arguments": {}
                    }
                })
        
        # Force population analysis for population queries
        elif any(phrase in user_query for phrase in [
            'population', 'demographic', 'people', 'inhabitants'
        ]):
            if any(phrase in user_query for phrase in ['statistics', 'stats', 'data']):
                # Try to extract layer name
                import re
                layer_match = re.search(r'(?:of|in|for)\s+(?:the\s+)?([a-zA-Z_]+)', user_query, re.IGNORECASE)
                if layer_match:
                    layer_name = layer_match.group(1)
                    forced_tools.append({
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": "get_attribute_stats",
                            "arguments": {"layer_name": layer_name, "attribute_key": "population"}
                        }
                    })
        
        return forced_tools

# Initialize the LLM server
llm_server = OllamaLLMServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await llm_server.initialize()
    yield
    # Shutdown
    pass

# Create FastAPI app
app = FastAPI(
    title="LLM Server with Ollama and MCP",
    description="A comprehensive LLM server with Ollama integration and Model Context Protocol support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "LLM Server with Ollama and MCP",
        "version": "1.0.0",
        "available_models": llm_server.available_models,
        "available_tools": list(llm_server.mcp_server.tools.keys())
    }

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "ollama"
            }
            for model in llm_server.available_models
        ]
    }

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {
        "tools": llm_server.mcp_server.get_tools_schema()
    }

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if request.stream:
        return StreamingResponse(
            llm_server.stream_chat_completion(request),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        return await llm_server.chat_completion(request)

@app.post("/mcp/execute_tool")
async def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]):
    """Execute an MCP tool directly"""
    try:
        result = await llm_server.mcp_server.execute_tool(tool_name, arguments)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ollama_connected": len(llm_server.available_models) > 0
    }

# Add Ollama-compatible endpoints for GUI integration
@app.get("/api/tags")
async def ollama_tags():
    """Ollama-compatible tags endpoint for GUI"""
    try:
        models = await llm_server.ollama_client.list()
        return {
            "models": [
                {
                    "name": model["name"],
                    "model": model["name"],
                    "modified_at": model.get("modified_at", datetime.now().isoformat()),
                    "size": model.get("size", 0),
                    "digest": model.get("digest", ""),
                    "details": model.get("details", {})
                }
                for model in models.get("models", [])
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching tags: {str(e)}")
        return {"models": []}

@app.get("/tags")
async def tags_redirect():
    """Redirect /tags to /api/tags for compatibility"""
    return await ollama_tags()

@app.get("/api/version")
async def ollama_version():
    """Ollama-compatible version endpoint"""
    return {
        "version": "0.1.0",
        "build": "enhanced-llm-server"
    }

@app.post("/api/generate")
async def ollama_generate(request: Request):
    """Ollama-compatible generate endpoint"""
    try:
        body = await request.json()
        
        # Convert Ollama generate format to chat completion format
        chat_request = ChatCompletionRequest(
            model=body.get("model", "qwen3:8b"),
            messages=[{"role": "user", "content": body.get("prompt", "")}],
            stream=body.get("stream", False),
            temperature=body.get("options", {}).get("temperature", 0.7),
            max_tokens=body.get("options", {}).get("num_predict", None)
        )
        
        if chat_request.stream:
            async def generate_stream():
                async for chunk in llm_server.stream_chat_completion(chat_request):
                    # Convert to Ollama format
                    if chunk.startswith("data: "):
                        chunk_data = chunk[6:]
                        if chunk_data.strip() == "[DONE]":
                            yield json.dumps({"done": True}) + "\n"
                        else:
                            try:
                                data = json.loads(chunk_data)
                                if "choices" in data and data["choices"]:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        yield json.dumps({
                                            "model": chat_request.model,
                                            "created_at": datetime.now().isoformat(),
                                            "response": content,
                                            "done": False
                                        }) + "\n"
                            except json.JSONDecodeError:
                                continue
            
            return StreamingResponse(
                generate_stream(),
                media_type="application/x-ndjson"
            )
        else:
            response = await llm_server.chat_completion(chat_request)
            content = response.choices[0]["message"]["content"] if response.choices else ""
            
            return {
                "model": chat_request.model,
                "created_at": datetime.now().isoformat(),
                "response": content,
                "done": True,
                "context": [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0
            }
            
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Ollama-compatible chat endpoint"""
    try:
        body = await request.json()
        
        # Convert Ollama chat format to our format
        messages = body.get("messages", [])
        chat_request = ChatCompletionRequest(
            model=body.get("model", "qwen3:8b"),
            messages=[ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages],
            stream=body.get("stream", False),
            temperature=body.get("options", {}).get("temperature", 0.7),
            max_tokens=body.get("options", {}).get("num_predict", None),
            tools=llm_server.mcp_server.get_tools_schema() if body.get("tools", True) else None
        )
        
        if chat_request.stream:
            async def chat_stream():
                async for chunk in llm_server.stream_chat_completion(chat_request):
                    # Convert to Ollama format
                    if chunk.startswith("data: "):
                        chunk_data = chunk[6:]
                        if chunk_data.strip() == "[DONE]":
                            yield json.dumps({
                                "model": chat_request.model,
                                "created_at": datetime.now().isoformat(),
                                "message": {"role": "assistant", "content": ""},
                                "done": True
                            }) + "\n"
                        else:
                            try:
                                data = json.loads(chunk_data)
                                if "choices" in data and data["choices"]:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        yield json.dumps({
                                            "model": chat_request.model,
                                            "created_at": datetime.now().isoformat(),
                                            "message": {"role": "assistant", "content": content},
                                            "done": False
                                        }) + "\n"
                            except json.JSONDecodeError:
                                continue
            
            return StreamingResponse(
                chat_stream(),
                media_type="application/x-ndjson"
            )
        else:
            response = await llm_server.chat_completion(chat_request)
            content = response.choices[0]["message"]["content"] if response.choices else ""
            
            return {
                "model": chat_request.model,
                "created_at": datetime.now().isoformat(),
                "message": {"role": "assistant", "content": content},
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0
            }
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pull")
async def ollama_pull(request: Request):
    """Ollama-compatible pull endpoint"""
    try:
        body = await request.json()
        model_name = body.get("name", "")
        
        # Forward to actual Ollama instance
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/pull",
                json=body,
                timeout=300.0
            )
            
            if body.get("stream", False):
                async def pull_stream():
                    async for chunk in response.aiter_lines():
                        yield chunk + "\n"
                
                return StreamingResponse(
                    pull_stream(),
                    media_type="application/x-ndjson"
                )
            else:
                return response.json()
                
    except Exception as e:
        logger.error(f"Error in pull endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete")
async def ollama_delete(request: Request):
    """Ollama-compatible delete endpoint"""
    try:
        body = await request.json()
        
        # Forward to actual Ollama instance
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                "http://127.0.0.1:11434/api/delete",
                json=body
            )
            return response.json()
            
    except Exception as e:
        logger.error(f"Error in delete endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/show")
async def ollama_show(request: Request):
    """Ollama-compatible show endpoint"""
    try:
        model_name = request.query_params.get("name", "")
        
        # Forward to actual Ollama instance
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/show",
                json={"name": model_name}
            )
            return response.json()
            
    except Exception as e:
        logger.error(f"Error in show endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add direct endpoint redirects for GUI compatibility
@app.post("/chat")
async def chat_redirect(request: Request):
    """Redirect /chat to /api/chat for GUI compatibility"""
    return await ollama_chat(request)

@app.post("/generate")
async def generate_redirect(request: Request):
    """Redirect /generate to /api/generate for GUI compatibility"""
    return await ollama_generate(request)

@app.get("/version")
async def version_redirect():
    """Redirect /version to /api/version for GUI compatibility"""
    return await ollama_version()

@app.post("/pull")
async def pull_redirect(request: Request):
    """Redirect /pull to /api/pull for GUI compatibility"""
    return await ollama_pull(request)

@app.delete("/delete")
async def delete_redirect(request: Request):
    """Redirect /delete to /api/delete for GUI compatibility"""
    return await ollama_delete(request)

@app.post("/show")
async def show_redirect(request: Request):
    """Redirect /show to /api/show for GUI compatibility"""
    return await ollama_show(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "llm_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
