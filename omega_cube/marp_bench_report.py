import json
import os

log_file = os.path.expanduser('~/.hermes/logs/marp_router/marp_20260713.jsonl')
entries = []
with open(log_file) as f:
    for line in f:
        l = line.strip()
        if l:
            entries.append(json.loads(l))

total = len(entries)
print('Total queries:', total)

latencies = []
for e in entries:
    latencies.append(e['latency_us'])
avg_lat = round(sum(latencies) / len(latencies), 1)
min_lat = round(min(latencies), 1)
max_lat = round(max(latencies), 1)
print('Avg latency (us):', avg_lat)
print('Min latency (us):', min_lat)
print('Max latency (us):', max_lat)

kw_count = 0
sm_count = 0
for e in entries:
    if e['model_used'] == 'keyword':
        kw_count += 1
    elif e['model_used'] == 'smollm2':
        sm_count += 1
print('Keyword hits:', kw_count)
print('SmolLM2 hits:', sm_count)

total_savings = 0
for e in entries:
    total_savings += e['token_savings']
print('Total token savings:', round(total_savings, 2))

doms = {}
for e in entries:
    for d in e['domains']:
        if d in doms:
            doms[d] = doms[d] + 1
        else:
            doms[d] = 1
sorted_doms = sorted(doms.items(), key=lambda x: x[1], reverse=True)
print('Top 10 domains:')
for d, c in sorted_doms[:10]:
    pct = round(100 * c / total, 1)
    print('  ' + d + ': ' + str(c) + ' (' + str(pct) + '%)')
