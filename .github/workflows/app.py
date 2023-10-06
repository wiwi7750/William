# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:49:16 2023

@author: william.chen
"""

from flask import Flask, request, jsonify
from botbuilder.schema import Activity, ActivityTypes
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
import openai
import pandas as pd
from scipy import spatial
import tiktoken

# 集中管理配置
OPENAI_API_KEY = "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"
BOT_APP_ID = "1ee97037-cd6f-4c54-931d-4030151897d8"
BOT_APP_PASSWORD = "Kxm8Q~Sg_~DEflbxYzoV5DxQFMSjSWMtv9X9uac."
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

app = Flask(__name__)

# 初始化OpenAI
openai.api_key = OPENAI_API_KEY

# 載入資料
df = pd.read_pickle('ecv_billing_qa.pk')

# 設定Bot Framework Adapter
bot_settings = BotFrameworkAdapterSettings(BOT_APP_ID, BOT_APP_PASSWORD)
adapter = BotFrameworkAdapter(bot_settings)

@app.route("/api/messages", methods=["POST"])
def messages():
    if "application/json" in request.headers["Content-Type"]:
        body = request.json
    else:
        return jsonify(status=415)

    activity = Activity().deserialize(body)
    auth_header = (
        request.headers["Authorization"] if "Authorization" in request.headers else ""
    )

    async def on_turn(context):
        if context.activity.type == ActivityTypes.message:
            # 處理使用者輸入的訊息
            user_message = context.activity.text
            response = ask(user_message)
            await context.send_activity(response)

    adapter.process_activity(activity, auth_header, on_turn)
    return ""

def get_embedding(text):
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
    return response["data"][0]["embedding"]

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below eCloudvalley billing team FAQ doc to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nQ&A:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    messages = [
        {"role": "system", "content": "You answer questions about the billing team questions, try to rephrase into a better answer and format as bullet form."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

if __name__ == "__main__":
    app.run()
