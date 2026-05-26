from api.ollama import chat_completion, list_local_models, resolve_chat_model
from api.orchestrator import build_tool_schemas
import json, traceback

print('models:', list_local_models())
print('resolved:', resolve_chat_model())

messages = [
    {"role": "system", "content": "Testing tool schemas for Ollama"},
    {"role": "user", "content": "what kind of career do physical activities"},
]

try:
    resp = chat_completion(messages, tools=build_tool_schemas())
    print('RESP:')
    print(json.dumps(resp, indent=2))
except Exception:
    traceback.print_exc()
