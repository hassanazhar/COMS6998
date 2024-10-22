import requests

BASE_URL = "http://localhost:8000"

def test_single_chat():
    url = f"{BASE_URL}/chat"
    payload = {"text": "Hello"}
    response = requests.post(url, json=payload)
    print_response("Single Chat", response)


def test_batch_chat():
    url = f"{BASE_URL}/batch_chat"
    payload = {"texts": ["Hi", "How are you?", "What is your name?"]}
    response = requests.post(url, json=payload)
    print_response("Batch Chat", response)

def test_ray_batch_chat():
    url = f"{BASE_URL}/batch_chat_ray"
    payload = {"texts": ["Good morning", "Tell me a joke", "Where are you from?"]}
    response = requests.post(url, json=payload)
    print_response("Ray Batch Chat", response)

def test_torchx_batch_chat():
    url = f"{BASE_URL}/torchx_batch_chat"
    payload = {"texts": ["Hello there", "Tell me something interesting", "Goodbye"]}
    response = requests.post(url, json=payload)
    print_response("TorchX Batch Chat", response)

def print_response(test_name, response):
    if response.status_code == 200:
        result = response.json()
        print(f"{test_name} Passed: {result['response']}")
        print(f"Time taken: {result['time_taken']:.4f} seconds")
    else:
        print(f"{test_name} Failed:", response.status_code, response.text)

if __name__ == "__main__":
    print("Running Chatbot Tests...")
    test_single_chat()
    test_batch_chat()
    test_ray_batch_chat()
    test_torchx_batch_chat()
