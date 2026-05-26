import api.ollama as ollama


class FakeStreamResponse:
    def __init__(self, lines):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)


def test_chat_completion_stream_yields_text_fragments(monkeypatch):
    lines = [
        b'{"model":"llama2:latest","message":{"role":"assistant","content":"B"},"done":false}\n',
        b'{"model":"llama2:latest","message":{"role":"assistant","content":"ased"},"done":false}\n',
        b'{"model":"llama2:latest","message":{"role":"assistant","content":" on"},"done":false}\n',
        b'{"model":"llama2:latest","message":{"role":"assistant","content":""},"done":true}\n',
    ]

    class FakeUrlopen:
        def __call__(self, request, timeout=None):
            return FakeStreamResponse(lines)

    monkeypatch.setattr(ollama.urllib.request, "urlopen", FakeUrlopen())

    chunks = list(ollama.chat_completion_stream([{ "role": "user", "content": "hello" }], model="test-model"))

    assert chunks == ["B", "ased", " on"]
    assert "{" not in "".join(chunks)