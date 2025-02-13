# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
from time import sleep
import requests


if __name__ == "__main__":
    url = sys.argv[1]
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}

    # while True:
    #     sentence = input("Enter prompt: ")
    #     tokens_to_generate = int(eval(input("Enter number of tokens to generate: ")))

    sentences = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\nCreate a detailed description for the following product: ABC, belonging to category: Gas Station"
        # "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\nCreate a detailed description for the following product: CG8565, belonging to category: Desktop Computer"
    ]
    for sentence in sentences:
        data = {"prompts": [sentence], "tokens_to_generate": 200}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
        print("--------------------")
        sleep(5)
