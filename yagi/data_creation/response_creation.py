import concurrent.futures
from tqdm.auto import tqdm, trange
import time
from openai import AzureOpenAI
import random
import os
import numpy as np
import pandas as pd
from datasets import load_dataset

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version = "2024-05-01-preview",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
)

def gen_responses(prompt_text):

    response = client.chat.completions.create(
      model="gpt4o-2024-05-13",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt_text
            }
          ]
        }
      ],
      temperature=1,
      max_tokens=2048,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )

    return response


def run_gen_responses(prompt_text):
    try:
        time.sleep(6 * random.random())
        r = gen_responses(prompt_text)
        return {
            "gen_prompts": r.choices[0].message.content,
            "finish_reason": r.choices[0].finish_reason,
        }
    except Exception as e:
        print(f"Failed - {prompt_text}")
        print(e)
        print()
        return None

def save_dataset(yagi_df, dataset_name):

    chunk_size = 500

    for i in trange(0, yagi_df.shape[0], chunk_size):

        batch_df = yagi_df.iloc[i:i+chunk_size]
        prompt_texts = batch_df["prompt_list"].tolist()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            response_results = list(tqdm(executor.map(run_gen_responses, prompt_texts), total=len(prompt_texts)))

        batch_df.loc[:, "responses"] = response_results
        batch_df.to_parquet(f"~/{dataset_name}/{str(i).zfill(6)}.parquet")


    yagi_dataset = load_dataset("parquet", data_files={"train": f"~/{dataset_name}/*.parquet"}, split="train")
    yagi_dataset.push_to_hub(f"lightblue/{dataset_name}", private=True)