import json
from openai_service import check_with_openai

cases = [
    "The Eiffel Tower is in Paris.",
    "NASA confirmed the moon is made of cheese.",
]

for i, stmt in enumerate(cases, 1):
    try:
        res = check_with_openai(stmt)
        print(f"Case {i}:")
        print(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        print(f"Case {i} ERROR: {e}")
