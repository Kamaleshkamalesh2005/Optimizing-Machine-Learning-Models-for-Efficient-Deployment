import requests
import time

r = requests.get('http://127.0.0.1:8000/api/compress', stream=True, timeout=1200)
print('status', r.status_code)
start = time.time()
for line in r.iter_lines(decode_unicode=True):
    if not line:
        continue
    print(line)
    if '"status": "done"' in line or '"status": "error"' in line:
        break
    if time.time() - start > 600:
        print('timeout break')
        break
