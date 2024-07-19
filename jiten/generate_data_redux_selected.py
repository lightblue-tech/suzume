import random
from transformers import GPT2TokenizerFast
from tqdm.auto import tqdm
from openai import AzureOpenAI, BadRequestError
import concurrent.futures
from tqdm.auto import tqdm
import os
import json
import argparse
from datasets import load_dataset
import pandas as pd

def make_sys_message(l, t, c): 
    return f"""Given a Wikipedia article in {l}, write a prompt in {l} that a user may ask a language model about that topic. 
Then write a truthful response in {l} from a helpful AI assistant. 
The prompt and response can be tangentially related or directly related to the Wikipedia article. 

# Prompt requirements
Make the prompt unique to the general theme of the article.
Make the prompt a practical prompt that a user may actually ask an LLM.
The requested task could ask the assistant to {c} something or another task that could be asked by a real user.
Write the prompt so it does not refer to any knowledge that is assumed from the article. 
Write the prompt so that it could be given without ever having read the passage.
The prompt should be written as a {t} in fluent {l}.

Do not make the prompt trivial. 
For example, do not simply make the prompt "Tell me about XYZ" when given an article about XYZ. 
Do not simply ask for clarification or explanation about a certain aspect in the article. 
Do not make a prompt that simply asks about the significance of certain concepts.

# Response requirements
The response must be written in fluent, natural {l}.
The Wikipedia article contains correct factual information, so if your response requires factual content, use the given article as the basis for those facts where possible.
If a short response will suffice, then write a short response.
If a long answer is required, then write a long answer.

# Output format

You should output your prompt and response like so:

<<PROMPT>>
Your prompt

<<RESPONSE>>
Your response"""

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4')
num_max_tokens = 4096
truncate_text = lambda x: tokenizer.decode(tokenizer.encode(x)[:num_max_tokens])

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version = "2024-04-01-preview",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
)

def get_openai_response(title_text, input_text, input_language, prompt_type, prompt_category, max_tokens = 2048):

    response = client.chat.completions.create(
        model="gpt4o-2024-05-13",
        messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": make_sys_message(input_language, prompt_type, prompt_category)
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": title_text + "\n\n" + input_text
            }
          ]
        }
        ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=0
    )
    choice = response.choices[0]
    
    return {"finish_reason": choice.finish_reason, "content": choice.message.content}

def handle_openai_call(input_data_item):
    title_text, input_text, input_language, prompt_type, prompt_category = input_data_item
    # return get_openai_response(title_text, input_text, input_language, prompt_type)
    try:
        return get_openai_response(title_text, input_text, input_language, prompt_type, prompt_category)
    except BadRequestError as bre:
        print(json.loads(bre.response.content))
        return {"bad_request": str(json.loads(bre.response.content)["error"]["innererror"])}
    except Exception as e:
        print(e)
        return {"error": str(e)}

supported_language_map = {'ab': 'Abkhaz', 'ace': 'Acehnese', 'ady': 'Adyghe', 'af': 'Afrikaans', 'als': 'Alemannic German', 'alt': 'Southern Altai', 'am': 'Amharic', 'ami': 'Amis', 'an': 'Aragonese', 'ang': 'Old English', 'anp': 'Angika', 'ar': 'Arabic', 'arc': 'Aramaic', 'ary': 'Moroccan Arabic', 'arz': 'Egyptian Arabic', 'as': 'Assamese', 'ast': 'Asturian', 'atj': 'Atikamekw', 'av': 'Avar', 'avk': 'Kotava', 'awa': 'Awadhi', 'ay': 'Aymara', 'az': 'Azerbaijani', 'azb': 'South Azerbaijani', 'ba': 'Bashkir', 'ban': 'Balinese', 'bar': 'Bavarian', 'bat-smg': 'Samogitian', 'bcl': 'Central Bikol', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bh': 'Bihari', 'bi': 'Bislama', 'bjn': 'Banjarese', 'blk': "Pa'O", 'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Lhasa Tibetan', 'bpy': 'Bishnupriya Manipuri', 'br': 'Breton', 'bs': 'Bosnian', 'bug': 'Buginese', 'bxr': 'Buryat', 'ca': 'Catalan', 'cbk-zam': 'Zamboanga Chavacano', 'cdo': 'Eastern Min', 'ce': 'Chechen', 'ceb': 'Cebuano', 'ch': 'Chamorro', 'chr': 'Cherokee', 'chy': 'Cheyenne', 'ckb': 'Kurdish (Sorani)', 'co': 'Corsican', 'cr': 'Cree', 'crh': 'Crimean Tatar', 'cs': 'Czech', 'csb': 'Kashubian', 'cu': 'Old Church Slavonic', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'dag': 'Dagbani', 'de': 'German', 'din': 'Dinka', 'diq': 'Zaza', 'dsb': 'Lower Sorbian', 'dty': 'Doteli', 'dv': 'Maldivian', 'dz': 'Dzongkha', 'ee': 'Ewe', 'el': 'Greek', 'eml': 'Emilian–Romagnol', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'ext': 'Extremaduran', 'fa': 'Persian', 'fat': 'Fante', 'ff': 'Fula', 'fi': 'Finnish', 'fiu-vro': 'Võro', 'fj': 'Fijian', 'fo': 'Faroese', 'fon': 'Fon (Dahomean)', 'fr': 'French', 'frp': 'Franco-Provençal', 'frr': 'North Frisian', 'fur': 'Friulian', 'fy': 'West Frisian', 'ga': 'Irish', 'gag': 'Gagauz', 'gan': 'Gan Chinese', 'gcr': 'French Guianese Creole', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'glk': 'Gilaki', 'gn': 'Guarani', 'gom': 'Konkani', 'gor': 'Gorontalo', 'got': 'Gothic', 'gpe': 'Ghanaian Pidgin English', 'gu': 'Gujarati', 'guc': 'Wayuu', 'gur': 'Farefare', 'guw': 'Gun', 'gv': 'Manx', 'ha': 'Hausa', 'hak': 'Hakka Chinese', 'haw': 'Hawaiian', 'he': 'Hebrew', 'hi': 'Hindi', 'hif': 'Fiji Hindi', 'hr': 'Croatian', 'hsb': 'Upper Sorbian', 'ht': 'Haitian Creole', 'hu': 'Hungarian', 'hy': 'Armenian', 'hyw': 'Western Armenian', 'ia': 'Interlingua', 'id': 'Indonesian', 'ie': 'Interlingue', 'ig': 'Igbo', 'ik': 'Iñupiaq', 'ilo': 'Ilocano', 'inh': 'Ingush', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese', 'jam': 'Jamaican Patois', 'jbo': 'Lojban', 'jv': 'Javanese', 'ka': 'Georgian', 'kaa': 'Karakalpak', 'kab': 'Kabyle', 'kbd': 'Kabardian', 'kbp': 'Kabiye', 'kcg': 'Tyap', 'kg': 'Kongo', 'ki': 'Kikuyu', 'kk': 'Kazakh', 'kl': 'Greenlandic', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'koi': 'Komi-Permyak', 'krc': 'Karachay-Balkar', 'ks': 'Kashmiri', 'ksh': 'Ripuarian', 'ku': 'Kurdish (Kurmanji)', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lad': 'Judaeo-Spanish', 'lb': 'Luxembourgish', 'lbe': 'Lak', 'lez': 'Lezgian', 'lfn': 'Lingua Franca Nova', 'lg': 'Luganda', 'li': 'Limburgish', 'lij': 'Ligurian', 'lld': 'Ladin', 'lmo': 'Lombard', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'ltg': 'Latgalian', 'lv': 'Latvian', 'mad': 'Madurese', 'mai': 'Maithili', 'map-bms': 'Banyumasan', 'mdf': 'Moksha', 'mg': 'Malagasy', 'mhr': 'Meadow Mari', 'mi': 'Māori', 'min': 'Minangkabau', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mni': 'Meitei', 'mnw': 'Mon', 'mr': 'Marathi', 'mrj': 'Hill Mari', 'ms': 'Malay', 'mt': 'Maltese', 'mwl': 'Mirandese', 'my': 'Burmese', 'myv': 'Erzya', 'mzn': 'Mazanderani', 'nah': 'Nahuatl', 'nap': 'Neapolitan', 'nds-nl': 'Dutch Low Saxon', 'nds': 'Low German', 'ne': 'Nepali', 'new': 'Newar', 'nia': 'Nias', 'nl': 'Dutch', 'nn': 'Norwegian (Nynorsk)', 'no': 'Norwegian (Bokmål)', 'nov': 'Novial', 'nqo': "N'Ko", 'nrm': 'Norman', 'nso': 'Northern Sotho', 'nv': 'Navajo', 'ny': 'Chewa', 'oc': 'Occitan', 'olo': 'Livvi-Karelian', 'om': 'Oromo', 'or': 'Odia', 'os': 'Ossetian', 'pa': 'Punjabi', 'pag': 'Pangasinan', 'pam': 'Kapampangan', 'pap': 'Papiamento', 'pcd': 'Picard', 'pcm': 'Nigerian Pidgin', 'pdc': 'Pennsylvania Dutch', 'pfl': 'Palatine German', 'pi': 'Pali', 'pih': 'Norfuk', 'pl': 'Polish', 'pms': 'Piedmontese', 'pnb': 'Western Punjabi', 'pnt': 'Pontic Greek', 'ps': 'Pashto', 'pt': 'Portuguese', 'pwn': 'Paiwan', 'qu': 'Quechua', 'rm': 'Romansh', 'rmy': 'Romani', 'rn': 'Kirundi', 'ro': 'Romanian', 'roa-rup': 'Aromanian', 'roa-tara': 'Tarantino', 'ru': 'Russian', 'rue': 'Rusyn', 'rw': 'Kinyarwanda', 'sa': 'Sanskrit', 'sah': 'Yakut', 'sat': 'Santali', 'sc': 'Sardinian', 'scn': 'Sicilian', 'sco': 'Scots', 'sd': 'Sindhi', 'se': 'Northern Sámi', 'sg': 'Sango', 'sh': 'Serbo-Croatian', 'shi': 'Shilha', 'shn': 'Shan', 'si': 'Sinhala', 'simple': 'Simple English', 'sk': 'Slovak', 'skr': 'Saraiki', 'sl': 'Slovene', 'sm': 'Samoan', 'smn': 'Inari Sámi', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'srn': 'Sranan Tongo', 'ss': 'Swazi', 'st': 'Sotho', 'stq': 'Saterland Frisian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'szl': 'Silesian', 'szy': 'Sakizaya', 'ta': 'Tamil', 'tay': 'Atayal', 'tcy': 'Tulu', 'te': 'Telugu', 'tet': 'Tetum', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tly': 'Talysh', 'tn': 'Tswana', 'to': 'Tongan', 'tpi': 'Tok Pisin', 'tr': 'Turkish', 'trv': 'Seediq', 'ts': 'Tsonga', 'tt': 'Tatar', 'tum': 'Tumbuka', 'tw': 'Twi', 'ty': 'Tahitian', 'tyv': 'Tuvan', 'udm': 'Udmurt', 'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vec': 'Venetian', 'vep': 'Veps', 'vi': 'Vietnamese', 'vls': 'West Flemish', 'vo': 'Volapük', 'wa': 'Walloon', 'war': 'Waray', 'wo': 'Wolof', 'wuu': 'Wu Chinese', 'xal': 'Kalmyk Oirat', 'xh': 'Xhosa', 'xmf': 'Mingrelian', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang', 'zea': 'Zeelandic', 'zh-classical': 'Classical Chinese', 'zh-min-nan': 'Southern Min', 'zh-yue': 'Cantonese', 'zh': 'Chinese', 'zu': 'Zulu'}

def run_generation(cluster_num):

    # ds = load_dataset("parquet", data_files={"train": f"./cluster_500_article_texts_{cluster_num}.parquet"}, split="train", num_proc=16)
    # ds = ds.shuffle(seed=42).to_parquet(f"./cluster_500_article_texts_{cluster_num}_shuffled.parquet")
    ds = load_dataset("parquet", data_files={"train": f"./cluster_500_article_texts_{cluster_num}_shuffled.parquet"}, split="train", num_proc=16)
    
    ds = ds.filter(
        lambda x: x["language_code"] in ["ar", "zh", "ja", "ko", "ru", "es", "pt", "hi", "id"], num_proc=16
    )
    ds = ds.map(lambda x: {"language_name": supported_language_map[x["language_code"]]}, num_proc=16)
    ds = ds.map(lambda x: {"prompt_type": "command" if random.random() >= 0.5 else "question"}, num_proc=16)
    ds = ds.map(lambda x: {"prompt_category": random.choice(
        ["explain", "describe", "analyze", "calculate", "synthesize", "summarize", "hypothesize"]
            ) if x["prompt_type"] == "question" else random.choice(
        ["reformat", "create", "combine", "give details about", "rewrite", "improve", "pretend"]
    )}, num_proc=16)
    
    chunk_size = 1_000

    for i in range(183_000, len(ds), chunk_size):
        chunk_ds = ds[i:i+chunk_size]

        input_data = list(zip(
            chunk_ds["title"],
            [truncate_text(x) for x in tqdm(chunk_ds["text"])],
            chunk_ds["language_name"],
            chunk_ds["prompt_type"],
            chunk_ds["prompt_category"]
        ))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            response_results = list(tqdm(executor.map(handle_openai_call, input_data), total=len(input_data)))

        chunk_ds["generated_prompt_response"] = response_results

        pd.DataFrame(chunk_ds).to_parquet(f"./gen_data/redux_selected_generated_jiten_{str(i).zfill(7)}_{cluster_num}.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples_per_cluster', type=int, required=False, default=1)

    # Parse the arguments
    args = parser.parse_args()

    run_generation(args.num_samples_per_cluster)
