from openai import AzureOpenAI
from datasets import load_dataset
from tqdm.auto import trange, tqdm
import os
import concurrent.futures

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version = "2024-05-01-preview",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
)

def get_chatty_revision(input_text, output_text, language):

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": f"Given an input and output, correct and rewrite the text so that it is a more natural interaction between an AI and a user. Make the response helpful and detailed but also concise. Write them in the natural, fluent {language}.\n\nFormat your output like so:\n\n<REVISED INPUT>\nYour revised input here\n\n<REVISED OUTPUT>\nYour revised output here"
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"<INPUT>\n{input_text}？\t\n\n<OUTPUT>\n{output_text}"
            }
          ]
        }
      ],
      temperature=0,
      max_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return {
        "gen_input_outputs": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
    }

def chatty_handler(input_data_items):
    input_text, output_text, language = input_data_items

    try:
        return get_chatty_revision(input_text, output_text, language)
    except:
        return {
            "gen_input_outputs": None,
            "finish_reason": None,
        }
    
aya_dataset = load_dataset("CohereForAI/aya_dataset", split=True)
chunk_size = 1_000

for i in trange(0, len(aya_dataset), chunk_size):
    batch_dataset = aya_dataset.select(range(i, min(i+chunk_size, len(aya_dataset))))

    input_data = list(zip(aya_dataset["inputs"], aya_dataset["targets"], aya_dataset["language"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        response_results = list(tqdm(executor.map(chatty_handler, input_data), total=len(input_data)))

    batch_dataset = batch_dataset.add_column(f"gen_input_outputs", [x["gen_input_outputs"] for x in response_results])
    batch_dataset = batch_dataset.add_column(f"finish_reason", [x["finish_reason"] for x in response_results])
    batch_dataset.to_parquet(f"chatty_aya_{str(i).zfill(6)}.parquet")