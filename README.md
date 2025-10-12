# LangGraph Chatbot

This repository contains a small prototype chatbot built with a local backend and a Streamlit frontend. It was created as an exploratory project to combine a lightweight conversational backend (`langgraph_backend.py`) with an interactive UI (`streamlit_frontend.py`) and a design notebook (`chatbot_initial_design.ipynb`).

Contents
- `langgraph_backend.py` — backend logic for the chatbot (conversation flow, model integration hooks, data processing).
- `streamlit_frontend.py` — Streamlit-based frontend to interact with the chatbot locally.
- `chatbot_initial_design.ipynb` — design notes, experiments, and early iterations captured in a Jupyter notebook.

Quick start

1. Create and activate a Python virtual environment (recommended):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

This repository includes a `requirements.txt` with pinned versions captured from the project's `venv`. Install all dependencies with:

```cmd
pip install -r requirements.txt
```

3. Configure environment variables

Create a `.env` file from `.env.sample` in the project root and update relevant variables with your keys.

Optionally if your backend uses any external API keys or configuration, update the `.env` file and add variables such as:

```
# .env (example)
OPENAI_API_KEY=sk-...
OTHER_API_KEY=...
```

4. Run the backend server

```cmd
python langgraph_backend.py
```

5. Run the Streamlit frontend

```cmd
streamlit run streamlit_frontend.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

Ollama (local model host)
-------------------------

This project uses an Ollama-hosted model running locally (see `langgraph_backend.py`: it constructs a `ChatOllama` with `model="llama3.1:8b"`). Below are basic steps to get Ollama running and make the model available to the backend.

1. Install Ollama

Visit https://ollama.ai and follow the platform-specific installation instructions for your OS. On Windows, Ollama provides an installer and a CLI executable.

2. Start the Ollama daemon

After installation, start the Ollama service (daemon). On most systems it's started automatically; if not, use the Ollama CLI to start it. For example:

```cmd
ollama run
```

3. Pull or run the model used by this project

The backend expects `llama3.1:8b` (as configured in `langgraph_backend.py`). Pull or run that model with Ollama, for example:

```cmd
ollama pull llama3.1:8b
# or run the model directly
ollama run llama3.1:8b
```

4. Configure `OLLAMA_HOST`

By default the backend uses `http://localhost:11434` (see `langgraph_backend.py`). If your Ollama daemon is reachable at a different host/port, set the `OLLAMA_HOST` environment variable in your `.env` file:

```
OLLAMA_HOST=http://localhost:11434
```

5. Test connectivity

You can confirm the model is reachable by using the Ollama CLI or sending a small request from Python. If there are network or firewall restrictions, ensure the port is open and reachable by the backend process.

Quick test examples

curl (test a simple prompt):

```cmd
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Explain the concept of quantum entanglement in simple terms.",
  "stream": false
}'
```

Python (requests) example:

```python
import os
import requests

host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
url = f"{host}/api/chat?model=llama3.1:8b"
payload = {"messages": [{"role": "user", "content": "Hello from test"}]}
resp = requests.post(url, json=payload)
print(resp.status_code, resp.text)
```

Notes
- Ollama may require specific versions or a model license for larger models; check Ollama's docs for model availability and hardware requirements.
- If you want to use a different model, update the `model` argument in `langgraph_backend.py`.

Project contract and assumptions
- Inputs: user text from the Streamlit UI.
- Outputs: model reply shown in the UI and optionally logged by the backend.
- Error modes: missing API keys or network problems will cause the backend to raise an error; the frontend should display the error message.

Assumptions made while writing this README
- `langgraph_backend.py` implements the core conversation logic, but may require wiring to a specific LLM provider or local model.

Development notes and next steps
- Add simple unit tests for `langgraph_backend.py` functions.
- Add long term memory using a vector store (e.g., FAISS, Pinecone).
- Add instructions and helper script to obtain and store API keys securely (or integrate with a secrets manager).
- Explore deployment options (e.g., Streamlit Cloud, Heroku) for hosting the chatbot online.
