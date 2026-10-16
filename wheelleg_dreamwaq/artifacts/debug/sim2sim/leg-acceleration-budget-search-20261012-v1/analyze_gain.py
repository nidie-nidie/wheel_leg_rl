from pathlib import Path
import re,json
out=Path('artifacts/debug/sim2sim/leg-acceleration-budget-search-20261012-v1')
rows={}
for arm in ('80','75','70','85','90'):
 p=out/f'budget{arm}-200-train.log'
 if not (out/(arm+'-result.json')).is_file(): continue
 s=p.read_text(errors='replace')
 next_values=[float(v) for v in re.findall(r'Acceleration/gain_next:\s+([0-9.]+)',s)]
 used_values=[float(v) for v in re.findall(r'Acceleration/gain_used:\s+([0-9.]+)',s)]
 rows[arm]={'iterations_with_gain_logged':len(next_values),
            'gain_next_min':min(next_values),'gain_next_max':max(next_values),
            'gain_next_at_1_fraction':sum(v==1 for v in next_values)/len(next_values),
            'gain_used_first':used_values[0], 'gain_used_after_first_at_1_fraction':sum(v==1 for v in used_values[1:])/len(used_values[1:])}
(out/'gain-log-analysis.json').write_text(json.dumps(rows,indent=2))
print(rows)

