from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_openai import ChatOpenAI

from app.core import settings
from app.services.vector_db import vector_db
from app.tools import get_weather


def _format_doc_for_prompt(doc) -> str:
    """Format a retrieved doc for the LLM; image chunks become [Figure on page N: ...]."""
    meta = getattr(doc, "metadata", None) or {}
    content_type = meta.get("content_type", "text")
    page_content = doc.page_content or ""
    if content_type == "image":
        page_num = meta.get("page_number")
        caption = page_content.strip() if page_content else "see document"
        if page_num is not None:
            return f"[Figure on page {page_num}: {caption}]"
        return f"[Figure: {caption}]"
    return page_content


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject RAG context into the system prompt (text, tables, and image refs)."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_db.similarity_search(last_query)

    docs_content = "\n\n".join(_format_doc_for_prompt(doc) for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


def create_llm_agent():
    """Create and configure the LLM agent."""
    # vLLM provides OpenAI-compatible API
    model = ChatOpenAI(
        model=settings.model_name,
        base_url=f"{settings.inference_server_url}/v1",
        api_key="not-needed",  # vLLM doesn't require API key
    )

    return create_agent(
        model=model,
        tools=[get_weather],
        middleware=[prompt_with_context],
    )


# Module-level agent instance
agent = create_llm_agent()

