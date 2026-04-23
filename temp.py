import requests

url = "http://localhost:12434/engines/v1/chat/completions"

data = {
    "model": "sha256:1163f19dcd973b865c35d8e1a2c03736f4eb0a98c71e2b4425b7f84d183a423f",
    "messages": [
        {"role": "system", "content": "you are an assistant"},
        {"role": "user", "content": "Hello"},
    ],
}

response = requests.post(url, json=data)
response.raise_for_status()

print(response.json()["choices"][0]["message"]["content"])
