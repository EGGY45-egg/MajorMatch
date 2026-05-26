from api.ollama import chat_completion, list_local_models, resolve_chat_model
import traceback

print('models:', list_local_models())
print('resolved:', resolve_chat_model())
try:
    resp = chat_completion([{"role":"user","content":"Hello"}])
    print('OK', resp)
except Exception:
    traceback.print_exc()
