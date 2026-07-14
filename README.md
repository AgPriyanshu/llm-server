**TODO
- Create docker compose for this project - Done
- Add vector db for embeddings - Done
- Create websocket for chat - Done
- Enable model streaming - Done
- Visualize embeddings
- Create dataset for training
- Create training pipline

### Caching model weights

Models are cached under `llm_server/.cache/models` (or `MODEL_CACHE_DIR`) so they are not re-downloaded. To pre-download all pipeline models once (e.g. in Docker build or before first run):

```bash
python scripts/download_models.py
```
