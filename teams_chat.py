# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:23:32 2023

@author: william.chen
"""

import os
import openai
import pandas as pd
from scipy import spatial
from aiohttp import web
from botbuilder.schema import Activity, ActivityTypes
from botbuilder.core import BotFrameworkAdapter, TurnContext
from botbuilder.core import MessageFactory

# 设置应用 ID 和密码，替换为您的 Bot 凭据
APP_ID = "YourAppId"
APP_PASSWORD = "YourAppPassword"

# 创建 Bot 实例
adapter = BotFrameworkAdapter(APP_ID, APP_PASSWORD)

# OpenAI 设置
openai.api_key = "YOUR_OPENAI_API_KEY"
df = pd.read_pickle('ecv_billing_qa.pkl')

# OpenAI Chatbot 相关函数
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 10
) -> tuple[list[str], list[float]]:
    query_embedding_response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, ast.literal_eval(row["embedding"])))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    return len(openai.Completion.create(model=model, prompt=text).choices[0].text)

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
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question

def ask_gpt(
    query: str,
    df: pd.DataFrame = df,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    response = openai.Completion.create(
        model=model,
        prompt=message,
        temperature=0
    )
    response_message = response.choices[0].text
    return response_message

# 处理消息的函数
async def on_message_activity(context: TurnContext):
    message_text = context.activity.text
    gpt_answer = ask_gpt(message_text)
    
    reply_activity = MessageFactory.text(gpt_answer)
    
    await context.send_activity(reply_activity)

async def on_turn(context: TurnContext):
    if context.activity.type == ActivityTypes.message:
        await on_message_activity(context)

# 创建 Web 服务以监听消息
app = web.Application()
app.router.add_post('/api/messages', messages)

async def messages(request):
    if "application/json" in request.headers["Content-Type"]:
        data = await request.json()
        activity = Activity().deserialize(data)
        context = TurnContext(adapter, activity)
        await adapter.process_activity(context, on_turn)
        return web.Response(status=201)
    else:
        return web.Response(status=415)

if __name__ == '__main__':
    web.run_app(app)
