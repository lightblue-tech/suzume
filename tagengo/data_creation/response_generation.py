import pandas as pd
from openai import AzureOpenAI
from datasets import load_dataset, Dataset
from glob import glob
from tqdm.auto import tqdm
import os

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
)

def get_openai_response(input_text, model_name):
    try:
        response = client.chat.completions.create(
          model=model_name,
          messages=[
            {
              "role": "user",
              "content": input_text
            }
          ],
          temperature=0,
          max_tokens=2048,
        )
        print(
            str(
                round(
                    float(response.usage.completion_tokens * (30 / 1_000_000)) + float(response.usage.prompt_tokens * (10 / 1_000_000)),
                    3
                )
            ) + "$"
        )

        output_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        return output_text, finish_reason
    except Exception as e:
        print("ERROR!")
        print(e)
        return None, None

def main():

    prompt_dataset = load_dataset("lightblue/multilingual_prompts_25k_max", split="train")

    step_size = 1000

    for i in range(0, len(prompt_dataset), step_size):
        batch_dataset = prompt_dataset.select(
            range(i, min(i+step_size, len(prompt_dataset)))
        ).map(
            lambda x: {
                "response": get_openai_response(x["conversation"][0]["content"], "gpt-4-1106-preview")
            }, num_proc=4
        )
        
        batch_dataset.to_json(f"/home/jupyter/gpt_multiling_saved_4/{str(i).zfill(6)}.json")

    # ## Load ###

    paths = glob("gpt_multiling_saved_4/*.json")

    df = pd.concat([pd.read_json(p, lines=True) for p in tqdm(paths)])

    df["conversations"] = df.apply(lambda x: [
        {"from": "human", "value": x["conversation"][0]["content"]},
        {"from": "gpt", "value": x["response"][0]},
    ], axis=1)

    keep_col = ["conversation_id", "conversations", "language", "lang_detect_result", "response"]

    df = df[keep_col]

    Dataset.from_pandas(df).select_columns(keep_col).push_to_hub("lightblue/tagengo-gpt4", private=True)

if __name__ == "__main__": 
    main()