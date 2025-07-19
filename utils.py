import re
import uuid
from typing import Dict, List, Any
from config import logger


def clean_response(content: str) -> str:
    """Clean the response by removing thinking tags and other unwanted content"""
    # Remove thinking tags and their contents
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    
    # Remove any remaining XML-like tags that might be thinking related
    content = re.sub(r'</?(?:think|thinking|reason|reasoning|internal)>', '', content, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()
    
    return content


def extract_tool_calls(content: str, available_tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from assistant message"""
    tool_calls = []

    # Look for function calls in various formats
    # Pattern 1: function_name(arguments)
    pattern1 = r"(\w+)\(([^)]*)\)"
    matches1 = re.findall(pattern1, content)

    # Pattern 2: explicit tool usage mentions
    pattern2 = r"(?:use|call|execute)\s+(\w+)(?:\(([^)]*)\))?"
    matches2 = re.findall(pattern2, content, re.IGNORECASE)

    # Pattern 3: "I'll use X tool" or "Let me check with X"
    pattern3 = r"(?:I\'ll use|let me (?:use|call|check with))\s+(\w+)"
    matches3 = re.findall(pattern3, content, re.IGNORECASE)

    all_matches = []

    # Process pattern 1 matches
    for function_name, args_str in matches1:
        if function_name in available_tools:
            all_matches.append((function_name, args_str))

    # Process pattern 2 matches
    for function_name, args_str in matches2:
        if function_name in available_tools:
            all_matches.append((function_name, args_str or ""))

    # Process pattern 3 matches (no arguments)
    for function_name in matches3:
        if function_name in available_tools:
            all_matches.append((function_name, ""))

    for function_name, args_str in all_matches:
        try:
            # Parse arguments (key=value format)
            args = {}
            if args_str.strip():
                # Handle both key=value and positional arguments
                if "=" in args_str:
                    for arg in args_str.split(","):
                        if "=" in arg:
                            key, value = arg.split("=", 1)
                            args[key.strip()] = value.strip().strip("\"'")
                else:
                    # For simple cases like layer names
                    if function_name in [
                        "find_layer_by_name",
                    ]:
                        args["layer_name"] = args_str.strip().strip("\"'")
                    elif function_name in [
                        "get_heap_cluster_details",
                        "list_heaps_in_cluster",
                    ]:
                        args["cluster_name"] = args_str.strip().strip("\"'")
                    elif function_name in [
                        "get_volume_summary_by_site",
                    ]:
                        args["site_name"] = args_str.strip().strip("\"'")
                    elif function_name in [
                        "list_heap_clusters",
                    ]:
                        # Could be site_name or iteration_name, try to guess from context
                        arg_value = args_str.strip().strip("\"'")
                        if "site" in function_name.lower() or "site" in arg_value.lower():
                            args["site_name"] = arg_value
                        elif "iteration" in function_name.lower() or "iteration" in arg_value.lower():
                            args["iteration_name"] = arg_value
                        else:
                            args["site_name"] = arg_value  # Default to site_name

            tool_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": function_name, "arguments": args},
                }
            )
        except Exception as e:
            logger.error(f"Error parsing tool call: {str(e)}")

    return tool_calls


def should_use_list_layers(user_query: str) -> bool:
    """Check if the user query should trigger list_layers tool"""
    user_query_lower = user_query.lower()
    
    # Look for exact phrases that indicate listing layers
    layer_list_phrases = [
        "list layers",
        "show layers", 
        "what layers",
        "available layers",
        "layers available",
        "which layers",
        "see layers",
        "display layers"
    ]
    
    return any(phrase in user_query_lower for phrase in layer_list_phrases)


def should_use_heap_tools(user_query: str) -> str:
    """Check if the user query should trigger heap analysis tools and return the appropriate tool"""
    user_query_lower = user_query.lower()
    
    # Look for phrases that indicate heap cluster listing
    heap_list_phrases = [
        "list heap clusters",
        "show heap clusters", 
        "what heap clusters",
        "available heap clusters",
        "heap clusters available",
        "which heap clusters",
        "see heap clusters",
        "display heap clusters",
        "show heaps",
        "list heaps",
        "all heap clusters",
        "heap cluster data",
        "analyze heap clusters",
        "heap cluster analysis"
    ]
    
    # Look for phrases that indicate iteration-specific queries
    iteration_phrases = [
        "iteration",
        "iterations",
        "from iteration",
        "in iteration",
        "for iteration",
        "iteration data",
        "heap data for iteration",
        "clusters in iteration",
        "clusters from iteration",
        "show iteration",
        "get iteration",
        "analyze iteration"
    ]
    
    # Look for phrases that indicate volume summary by site
    volume_summary_phrases = [
        "volume summary",
        "total volume",
        "site volume",
        "volume by site",
        "aggregated volume",
        "volume analysis",
        "volume data",
        "analyze volume",
        "breakdown by site",
        "volume breakdown",
        "site breakdown",
        "volume per site",
        "volume for each site",
        "volume distribution"
    ]
    
    # Look for phrases that indicate heap cluster details
    cluster_detail_phrases = [
        "heap cluster details",
        "cluster information",
        "details of heap cluster",
        "information about cluster",
        "specific heap cluster",
        "individual heap cluster"
    ]
    
    # Priority order: More specific queries first
    
    # Check for iteration-specific queries (highest priority)
    if any(iter_phrase in user_query_lower for iter_phrase in iteration_phrases) and \
       any(heap_phrase in user_query_lower for heap_phrase in ["heap", "cluster", "volume"]):
        return "list_heap_clusters"
    
    # Check for volume analysis with site breakdown (most specific)
    elif any(vol_phrase in user_query_lower for vol_phrase in volume_summary_phrases) and \
         any(heap_phrase in user_query_lower for heap_phrase in ["heap", "cluster"]):
        return "list_heap_clusters"  # Start with listing all heap clusters to get comprehensive data
    
    # Check for heap cluster listing
    elif any(phrase in user_query_lower for phrase in heap_list_phrases):
        return "list_heap_clusters"
    
    # Check for volume summary by site
    elif any(phrase in user_query_lower for phrase in volume_summary_phrases):
        return "get_volume_summary_by_site"
    
    # Check for heap cluster details
    elif any(phrase in user_query_lower for phrase in cluster_detail_phrases):
        return "get_heap_cluster_details"
    
    return None


def extract_heap_tool_arguments(user_query: str) -> Dict[str, Any]:
    """Extract iteration names, site names, and other arguments from user query"""
    arguments = {}
    user_query_lower = user_query.lower()
    
    # Look for multiple iteration names in various formats
    multi_iteration_patterns = [
        r"iterations?\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])*",  # iterations "name1", "name2"
        r"iterations?\s+([A-Za-z0-9_-]+)(?:\s*,\s*([A-Za-z0-9_-]+))*",      # iterations name1, name2
        r"from\s+iterations?\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])*",  # from iterations "name1", "name2"
        r"from\s+iterations?\s+([A-Za-z0-9_-]+)(?:\s*,\s*([A-Za-z0-9_-]+))*",      # from iterations name1, name2
    ]
    
    # First try to find multiple iterations
    multiple_iterations = []
    for pattern in multi_iteration_patterns:
        matches = re.findall(pattern, user_query, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    multiple_iterations.extend([m for m in match if m.strip()])
                else:
                    multiple_iterations.append(match)
            break
    
    # Also look for comma-separated lists
    comma_separated_patterns = [
        r"iterations?\s+([A-Za-z0-9_-]+(?:\s*,\s*[A-Za-z0-9_-]+)+)",  # iterations name1, name2, name3
        r"from\s+iterations?\s+([A-Za-z0-9_-]+(?:\s*,\s*[A-Za-z0-9_-]+)+)",  # from iterations name1, name2, name3
    ]
    
    for pattern in comma_separated_patterns:
        matches = re.findall(pattern, user_query, re.IGNORECASE)
        if matches:
            for match in matches:
                names = [name.strip() for name in match.split(',')]
                multiple_iterations.extend(names)
            break
    
    if multiple_iterations:
        # Clean up the list and remove duplicates
        unique_iterations = list(set([name.strip() for name in multiple_iterations if name.strip()]))
        arguments["iteration_names"] = unique_iterations
    else:
        # Look for single iteration names in various formats
        iteration_patterns = [
            r"iteration\s+['\"]([^'\"]+)['\"]",  # iteration "name"
            r"iteration\s+([A-Za-z0-9_-]+)",     # iteration name
            r"from\s+iteration\s+['\"]([^'\"]+)['\"]",  # from iteration "name"
            r"from\s+iteration\s+([A-Za-z0-9_-]+)",     # from iteration name
            r"in\s+iteration\s+['\"]([^'\"]+)['\"]",    # in iteration "name"
            r"in\s+iteration\s+([A-Za-z0-9_-]+)",       # in iteration name
            r"for\s+iteration\s+['\"]([^'\"]+)['\"]",   # for iteration "name"
            r"for\s+iteration\s+([A-Za-z0-9_-]+)",      # for iteration name
        ]
        
        for pattern in iteration_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            if matches:
                # Take the first match
                arguments["iteration_name"] = matches[0]
                break
    
    # Look for site names in various formats (only if no iteration names found)
    if "iteration_name" not in arguments and "iteration_names" not in arguments:
        site_patterns = [
            r"site\s+['\"]([^'\"]+)['\"]",  # site "name"
            r"site\s+([A-Za-z0-9_-]+)",     # site name
            r"from\s+site\s+['\"]([^'\"]+)['\"]",  # from site "name"
            r"from\s+site\s+([A-Za-z0-9_-]+)",     # from site name
            r"in\s+site\s+['\"]([^'\"]+)['\"]",    # in site "name"
            r"in\s+site\s+([A-Za-z0-9_-]+)",       # in site name
            r"for\s+site\s+['\"]([^'\"]+)['\"]",   # for site "name"
            r"for\s+site\s+([A-Za-z0-9_-]+)",      # for site name
        ]
        
        for pattern in site_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            if matches:
                # Take the first match
                arguments["site_name"] = matches[0]
                break
    
    return arguments 