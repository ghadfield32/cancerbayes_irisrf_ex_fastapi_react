import requests
import json

def get_token():
    auth_data = {
        'username': 'alice',
        'password': 'supersecretvalue'
    }
    response = requests.post(
        'http://localhost:8000/api/token',
        data=auth_data
    )
    print("\nToken Response:")
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"Body: {response.text}")

    try:
        data = response.json()
        return data.get('access_token')
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return None

def test_health():
    response = requests.get('http://localhost:8000/api/health')
    print("\nHealth Check:")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response.text}")

def test_iris_predict(token):
    if not token:
        print("No token available - skipping iris prediction")
        return

    headers = {'Authorization': f'Bearer {token}'}
    data = {
        "model_type": "rf",  # Added model_type
        "samples": [{
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }]
    }
    response = requests.post(
        'http://localhost:8000/api/iris/predict',
        json=data,
        headers=headers
    )
    print("\nIris Prediction:")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response.text}")

def test_cancer_predict(token):
    if not token:
        print("No token available - skipping cancer prediction")
        return

    headers = {'Authorization': f'Bearer {token}'}
    data = {
        "model_type": "bayes",  # Added model_type
        "samples": [{
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001.0,
            "mean_smoothness": 0.1184,
            "mean_compactness": 0.2776,
            "mean_concavity": 0.3001,
            "mean_concave_points": 0.1471,
            "mean_symmetry": 0.2419,
            "mean_fractal_dimension": 0.07871,
            # SE features
            "se_radius": 1.095,
            "se_texture": 0.9053,
            "se_perimeter": 8.589,
            "se_area": 153.4,
            "se_smoothness": 0.006399,
            "se_compactness": 0.04904,
            "se_concavity": 0.05373,
            "se_concave_points": 0.01587,
            "se_symmetry": 0.03003,
            "se_fractal_dimension": 0.006193,
            # Worst features
            "worst_radius": 25.38,
            "worst_texture": 17.33,
            "worst_perimeter": 184.6,
            "worst_area": 2019.0,
            "worst_smoothness": 0.1622,
            "worst_compactness": 0.6656,
            "worst_concavity": 0.7119,
            "worst_concave_points": 0.2654,
            "worst_symmetry": 0.4601,
            "worst_fractal_dimension": 0.1189
        }]
    }
    response = requests.post(
        'http://localhost:8000/api/cancer/predict',
        json=data,
        headers=headers
    )
    print("\nCancer Prediction:")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response.text}")

if __name__ == '__main__':
    token = get_token()
    test_health()
    test_iris_predict(token)
    test_cancer_predict(token) 
