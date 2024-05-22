from datasets import load_dataset
from openai import AzureOpenAI
import os
import random

def get_system_message(num_models = 7):
    # Make a list of capital alphabet characters A-G then shuffle them randomly
    alphabet_labels = [chr(x) for x in range(65, 65 + num_models)]
    random_order = ", ".join(random.sample(alphabet_labels, num_models))

    example_responses = "\n\n".join([f"<<<RESPONSE {x}>>>\nEXAMPLE RESPONSE {x}" for x in alphabet_labels])

    return f"""You are an evaluator AI. Your task is to rank multiple responses to a given prompt from best to worst.
You will first be given the original prompt, and then seven possible responses to that prompt, labelled alphabetically.
You should first write a very brief (<40 words per model) explanation of the merits and drawbacks of the responses, before giving the ranking itself.
This explanation of each response should be in a randomised order (go in the order of '{random_order}').
Make sure you explain and rank all responses, do not leave any out in your explanation or ranking.
The ranking should be a list of alphabet characters that describe the ranking, with '>' denoting the left item is ranked higher than the right item and '=' denoting that the items are of equal ranking (e.g. 'Z>Y>X=W>V>U=T').

The user input will look like this:

```
<<<PROMPT>>>
AN EXAMPLE USER PROMPT

{example_responses}
```

and your output should look like this:

```
<<<EXPLANATION>>>
[SHORT EXPLANATION OF THE RANKING]

<<<RANKING>>>
[SEPARATED LIST OF ALPHABET CHARACTERS THAT DESCRIBE THE RANKING]
```

The evaluation rubric is as follows:

* Is the response relevant? The response should be the best possible answer.
* Is the response truthful?
* Is the response accurate? The response should accurately fulfill the prompt’s request.
* If a creative answer is expected, is the response creative? If an analytical answer is expected, is the response factual/objectively correct?
* Is the response written naturally and fluently in the language that the prompter would expect?
* Is the response detailed? The response should at minimum satisfy the full level of detail required by the prompt."""

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
)

def get_openai_response(input_text, model_name, num_models = 7):
    system_message = get_system_message(num_models = num_models)
    
    try:
        response = client.chat.completions.create(
          model=model_name,
          messages=[
            {
              "role": "system",
              "content": system_message
            },
            {
              "role": "user",
              "content": input_text
            }
          ],
          temperature=0,
          max_tokens=1024,
        )
    except Exception as e:
        print("OpenAI Exception:")
        print(e)
        return None, None
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

def make_eval_input_prompt(row):
    initial_prompt = row["conversation"][0]["content"]

    llm_responses = row["llm_responses"]
    model_list = list(llm_responses.keys())
    model_list = random.sample(model_list, len(model_list))

    model_id_list = [(str(chr(i + 65)), model_list[i]) for i in range(len(model_list))]

    eval_prompt = f"""<<<PROMPT>>>
{initial_prompt}"""

    for model_id, model_name in model_id_list:
        eval_prompt += f"""

<<<RESPONSE {model_id}>>>
{llm_responses[model_name]['content']}"""

    return model_id_list, eval_prompt

def eval_row(row, model_name):
    model_id_list, eval_input = make_eval_input_prompt(row)

    eval_response, eval_finish_reason = get_openai_response(
        eval_input, model_name, num_models = len(row["llm_responses"].keys())
        )

    if "model_evals" not in row.keys():
        row["model_evals"] = []

    return {
        f"model_evals": row["model_evals"] + [{
            "model_id_list": model_id_list,
            "eval_response": eval_response,
            "eval_finish_reason": eval_finish_reason,
            "eval_model_name": model_name,
        }]
    }

if __name__ == "__main__": 

    eval_model_name = "gpt-4-0125-preview"

    dataset = load_dataset("lightblue/mitsu", split="train")
    dataset = dataset.filter(lambda x: not any([x["content"] is None for x in x["llm_responses"].values()]))

    dataset.push_to_hub("lightblue/mitsu")
    dataset = dataset.map(lambda x: eval_row(x, eval_model_name), num_proc=4)
    dataset.push_to_hub("lightblue/mitsu")
    dataset = dataset.map(lambda x: eval_row(x, eval_model_name), num_proc=4)
    dataset.push_to_hub("lightblue/mitsu")
    dataset = dataset.map(lambda x: eval_row(x, eval_model_name), num_proc=4)
    dataset.push_to_hub("lightblue/mitsu")
    dataset = dataset.map(lambda x: eval_row(x, eval_model_name), num_proc=4)
    dataset.push_to_hub("lightblue/mitsu")
    dataset = dataset.map(lambda x: eval_row(x, eval_model_name), num_proc=4)
    dataset.push_to_hub("lightblue/mitsu")