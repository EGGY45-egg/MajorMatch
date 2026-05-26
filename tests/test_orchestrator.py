import api.orchestrator as orchestrator


def test_orchestrated_assistant_executes_tools_in_sequence(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "recommend_track",
        lambda profile: {
            "track": "B.Tech.-Computer Science and Engineering",
            "confidence": 0.84,
            "category": "Software Engineer",
            "source": "model",
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "suggest_career_context",
        lambda track, location="United States": type(
            "FakeContext",
            (),
            {
                "to_dict": lambda self: {
                    "track": track,
                    "location": location,
                    "source": "Adzuna",
                    "available": True,
                    "job_count": 101,
                    "salary_min": 60000,
                    "salary_max": 95000,
                    "salary_currency": "USD",
                    "top_job_titles": ["Junior Developer"],
                    "top_companies": ["Example Co"],
                    "note": None,
                    "query_url": None,
                }
            },
        )(),
    )
    monkeypatch.setattr(
        orchestrator,
        "suggest_courses",
        lambda query, top_k=5: [
            {
                "id": 1,
                "title": "Web Development with Flask",
                "description": "Build web apps with Flask",
                "score": 0.9,
                "score_normalized": 0.95,
                "source": "db",
            }
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "project_courses_with_query",
        lambda query, method="pca": (
            [
                type(
                    "FakePoint",
                    (),
                    {
                        "id": 1,
                        "title": "Web Development with Flask",
                        "description": "Build web apps with Flask",
                        "x": 1.0,
                        "y": 2.0,
                    },
                )()
            ],
            type(
                "FakePoint",
                (),
                {
                    "id": -1,
                    "title": "(query)",
                    "description": query,
                    "x": 0.0,
                    "y": 0.0,
                },
            )(),
        ),
    )

    responses = iter(
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "predict_track",
                                "arguments": "{\"coding\": 9, \"math\": 4, \"design\": 2}",
                            },
                        }
                    ],
                }
            },
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "get_career_context",
                                "arguments": "{\"track\": \"Software Engineer\", \"location\": \"United States\"}",
                            },
                        }
                    ],
                }
            },
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-3",
                            "type": "function",
                            "function": {
                                "name": "execute_semantic_search",
                                "arguments": "{\"user_query\": \"web development\", \"top_k\": 2, \"projection_method\": \"pca\"}",
                            },
                        }
                    ],
                }
            },
            {"message": {"content": "Here is a compact plan for you.", "tool_calls": []}},
            {"message": {"content": "Based on your profile, Software Engineer is the strongest fit. Here are the best job and course signals.", "tool_calls": []}},
        ]
    )

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        return next(responses)

    result = orchestrator.run_orchestrated_assistant(
        "I like coding and want a practical path.",
        {"coding": 9, "math": 4, "design": 2},
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.reply == "Based on your profile, Software Engineer is the strongest fit. Here are the best job and course signals."
    assert result.artifacts["prediction"]["track"] == "B.Tech.-Computer Science and Engineering"
    assert result.artifacts["career_context"]["job_count"] == 101
    assert result.artifacts["semantic_search"]["results"][0]["title"] == "Web Development with Flask"
    assert [trace.name for trace in result.tool_trace] == [
        "predict_track",
        "get_career_context",
        "execute_semantic_search",
    ]


def test_career_context_prompt_requests_exact_tool_fields():
    prompt = orchestrator._build_final_response_prompt(
        [orchestrator.ToolTrace(name="get_career_context", arguments={}, result={})],
        {
            "career_context": {
                "available": True,
                "track": "Software Engineer",
                "job_count": 101,
                "salary_min": 60000,
                "salary_max": 95000,
                "top_job_titles": ["Junior Developer"],
                "top_companies": ["Example Co"],
            }
        },
    )

    assert "Based on the career context tool, the results are:" in prompt
    assert "job_count=101" in prompt
    assert "salary_min=60000" in prompt
    assert "salary_max=95000" in prompt
    assert "top_job_titles" in prompt
    assert "top_companies" in prompt


def test_normal_question_does_not_need_tools():
    result = orchestrator.run_orchestrated_assistant(
        "hello, what are you?",
        {"coding": 5, "math": 5, "design": 5},
        model="test-model",
    )

    assert result.reply == "I am MajorMatch, an AI assistant that helps you choose courses and career paths. I can answer questions directly, and I’ll use tools only when they add value."
    assert result.tool_trace == []


def test_assistant_header_tokens_are_stripped_from_final_reply():
    def fake_chat_fn(messages, model=None, tools=None, options=None):
        return {"message": {"content": "<|start_header_id|>assistant<|end_header_id|> Hi there.", "tool_calls": []}}

    result = orchestrator.run_orchestrated_assistant(
        "recommend a course path",
        {"coding": 5, "math": 5, "design": 5},
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.reply == "Hi there."


def test_identity_question_skips_tools_entirely():
    calls = []

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        calls.append({"messages": messages, "tools": tools})
        return {"message": {"content": "This should not be used.", "tool_calls": []}}

    result = orchestrator.run_orchestrated_assistant(
        "what are you?",
        {"coding": 5, "math": 5, "design": 5},
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.tool_trace == []
    assert calls == []
    assert result.reply.startswith("I am MajorMatch")
