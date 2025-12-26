"""Weather tool for the LLM agent."""

from langchain.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

