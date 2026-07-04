import requests

API_URL = "http://127.0.0.1:8000/query"


def ask_question(question):

    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=120,
    )

    return response.json()["answer"]