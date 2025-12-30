import json

filepath = "/Users/jordangunn/.claude/projects/-Users-jordangunn-jaxcmr/eba37924-3785-4545-97f2-3c080cb4572e.jsonl"

with open(filepath, 'r') as f:
    lines = f.readlines()

with open(filepath, 'w') as f:
    for line in lines:
        if not line.strip():
            continue
        data = json.loads(line)
        if 'message' in data and 'content' in data['message']:
            if isinstance(data['message']['content'], list):
                data['message']['content'] = [
                    c for c in data['message']['content']
                    if c.get('type') not in ('thinking', 'redacted_thinking')
                    ]
                f.write(json.dumps(data) + '\n')