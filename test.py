import requests
resp = requests.post("http://localhost:11434/api/generate", json={"model": "llama2:7b", "prompt": "Hello", "stream": False})
print(resp.status_code, resp.text)