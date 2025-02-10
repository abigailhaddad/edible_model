import requests
response = requests.post('http://localhost:8000/predict', 
                        json={'text': 'carrot stick'})
print(response.json())