import json

with open('test.json', 'r', encoding='utf8') as f:
    data = json.load(f)

entities = data['entities']
mentions = data['mentions']

for r in data['relations']:
    if r['type'] != 'same gateway':
        continue
    print(r)
