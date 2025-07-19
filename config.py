import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DJANGO_BASE_URL = "http://localhost:80"
PORT = int(os.getenv("PORT", 8001))

# Timeout settings
HTTP_CLIENT_TIMEOUT = 300.0  # 5 minutes for database operations

# Tool Context Template
TOOLS_CONTEXT_TEMPLATE = """
You have access to the following tools. When the user asks for information that requires these tools, you MUST use them by calling the function in this format: function_name(parameter1="value1", parameter2="value2")

Available tools:
{tools_list}

When a user asks about layers, geospatial data, heap volumes, or analysis, you MUST use the appropriate tools. For example:
- "What layers are available?" -> use list_layers()
- "Find layer X" -> use find_layer_by_name(layer_name="X")
- "Information about layer X" -> use get_layer_info(layer_name="X")
- "Calculate 2+2" -> use calculate(expression="2+2")
- "Show heap clusters" -> use list_heap_clusters()
- "Heap clusters in site X" -> use list_heap_clusters(site_name="X")
- "Heap clusters in iteration Y" -> use list_heap_clusters(iteration_name="Y")
- "Show heap data for iteration Z" -> use list_heap_clusters(iteration_name="Z")
- "Get heap clusters from iterations A, B, C" -> use list_heap_clusters(iteration_name="A") then list_heap_clusters(iteration_name="B") then list_heap_clusters(iteration_name="C")
- "Details of heap cluster X" -> use get_heap_cluster_details(cluster_name="X")
- "Heaps in cluster X" -> use list_heaps_in_cluster(cluster_name="X")
- "Volume summary for site X" -> use get_volume_summary_by_site(site_name="X")
- "Analyze volume data for all heap clusters" -> use list_heap_clusters()
- "Volume breakdown by site" -> use list_heap_clusters()
- "Volume analysis with charts" -> use list_heap_clusters()

IMPORTANT: Always use tools when the user's question requires data that you don't have directly. Never provide placeholder data or instructions on how to implement charts manually. The tools will return actual data with properly formatted charts.

When users mention specific iteration names, always use the iteration_name parameter in list_heap_clusters to filter the results.
When users mention multiple iteration names, use get_heap_clusters_from_multiple_iterations with the iteration_names parameter.
""" 