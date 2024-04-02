import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key= "sk-YXYnEtqklrD16nNlIj50z6T3SNuSgvL5P1Vy2q00as2mMNrq",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.cn/v1"
)

def obj(x,y):
    return (x-2)**2+(y-3)**2

with open("prompt.txt","r",encoding="utf-8") as file:
    says = file.read()

def chat(says):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": says,
            }
        ],
        model="gpt-3.5-turbo-0125",
    )
    print("bots: "+chat_completion.choices[0].message.content)
    return  chat_completion.choices[0].message.content

while True:
    str = chat(says)
    s = str.find("{")
    e = str.find("}")
    str = str[s:e+1]
    data = json.loads(str)
    numbers = list(data.values())
    loss = obj(numbers[0][0],numbers[0][1])
    loss = np.round(loss,2)
    text  = str + " - [\"loss\": {}]".format(loss)
    says = (says[:-198] + "\n"+ text  +says[-198:])
    print("user: "+says)
    if loss == 0:
        break
