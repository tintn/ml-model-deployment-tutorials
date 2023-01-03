import requests
import numpy as np
from locust import HttpUser, task, between

class HelloWorldUser(HttpUser):
    wait_time = between(0.01, 0.1)

    arr = np.random.randint(0, 256, (1, 3, 32, 32)).tolist()

    @task
    def infer(self):
        payload = {"inputs":
            [
                {
                    "name":"image__0",
                    "datatype":"UINT8",
                    "shape": [1, 3, 32, 32],
                    "data": self.arr
                }
            ]
        }
        url = "/seldon/default/cifar10/v2/models/cifar10-pytorch/infer"
        r = self.client.post(url, json=payload)
        # print(r.content)
        assert r.status_code == 200


def send_request():
    arr = np.random.randint(0, 256, (1, 3, 32, 32)).tolist()
    payload = {"inputs":
            [
                {
                    "name":"image__0",
                    "datatype":"UINT8",
                    "shape": [1, 3, 32, 32],
                    "data": arr
                }
            ]
        }
    url = 'http://localhost:8080/seldon/default/cifar10/v2/models/cifar10-pytorch/infer'
    r = requests.post(url, json=payload)
    print(r.status_code)
    print(r.content)


if __name__ == '__main__':
    send_request()
