from openai import AzureOpenAI
from datasets import load_dataset
from tqdm.auto import trange, tqdm
import os
import concurrent.futures
import time
import random
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4')

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version = "2024-05-01-preview",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
)

def get_chatty_revision(input_text, output_text, language):
    
    output_response_multiplier = 2.0
    
    original_token_count = len(tokenizer.encode(input_text) + tokenizer.encode(output_text))
    output_tokens = min(4095, int(output_response_multiplier * original_token_count))
    
    response = client.chat.completions.create(
      model="gpt4o-2024-05-13",
      messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": f"Given an input and output, correct and rewrite the text so that it is a more natural interaction between an AI and a user. Make the response helpful and detailed but also concise. Write them in natural, fluent {language}.\n\nFormat your output like so:\n\n<REVISED INPUT>\nYour revised input here\n\n<REVISED OUTPUT>\nYour revised output here"
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"<INPUT>\n{input_text}\n\n<OUTPUT>\n{output_text}"
            }
          ]
        }
      ],
      temperature=0,
      max_tokens=output_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    print(language)
    return {
        "gen_input_outputs": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
    }

def chatty_handler(input_data_items):
    input_text, output_text, language = input_data_items
    time.sleep(8 * random.random())
    
    try:
        return get_chatty_revision(input_text, output_text, language)
    except Exception as e:
        print(e)
        return {
            "gen_input_outputs": None,
            "finish_reason": None,
        }
    
aya_dataset = load_dataset("CohereForAI/aya_dataset", split="train")
chunk_size = 1_000

for i in trange(38_000, len(aya_dataset), chunk_size):
    batch_dataset = aya_dataset.select(range(i, min(i+chunk_size, len(aya_dataset))))

    input_data = list(zip(batch_dataset["inputs"], batch_dataset["targets"], batch_dataset["language"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        response_results = list(tqdm(executor.map(chatty_handler, input_data), total=len(input_data)))

    batch_dataset = batch_dataset.add_column(f"gen_input_outputs", [x["gen_input_outputs"] for x in response_results])
    batch_dataset = batch_dataset.add_column(f"finish_reason", [x["finish_reason"] for x in response_results])
    batch_dataset.to_parquet(f"chatty_aya_{str(i).zfill(6)}.parquet")