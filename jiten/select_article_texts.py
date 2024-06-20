from datasets import load_dataset, concatenate_datasets
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

def gen_article_texts(num_per_cluster, lang_codes=None):

    dataset_list = []

    paths = glob(f"./title_cluster_ids_*_500.parquet")

    for p in tqdm(sorted(paths)):

        lang_code = p.split("_50.parquet")[-2].split("_")[-1]
        
        if lang_codes is not None and lang_code not in lang_codes:
            continue

        print(lang_code + str(num_per_cluster))

        df = pd.read_parquet(p)
        
        sampled_titles = set(df.groupby("label").apply(lambda x: x.sample(
                                n=min(x.shape[0], num_per_cluster),
                                random_state=0
                            )).title.values)

        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang_code}", split="train", num_proc=32)

        sampled_dataset = dataset.filter(lambda x: x["title"] in sampled_titles, num_proc=32)

        sampled_dataset = sampled_dataset.add_column("language_code", [lang_code] * len(sampled_dataset))

        dataset_list.append(sampled_dataset)

    concatenate_datasets(dataset_list).to_parquet(f"cluster_500_article_texts_{num_per_cluster}.parquet")

if __name__ == '__main__':
    gen_article_texts(num_per_cluster = 1)
    
    popular_language_codes = set(['en', 'de', 'fr', 'es', 'ru', 'pt', 'it', 'pl', 'nl', 'uk', 'fi', 'hu', 'et', 'tr', 'uz', 'az', 'kk', 'ar', 'he', 'am', 'my', 'sw', 'zu', 'xh', 'yo', 'ig', 'ta', 'ml', 'te', 'kn', 'id', 'vi', 'th', 'ms', 'jv', 'tl', 'km', 'su', 'hi', 'mr', 'gu', 'pa', 'bn', 'ja', 'ko', 'mi', 'sm', 'ur', 'zh', 'zh_min_nan', 'pcm', 'ha', 'om', 'ro', 'el', 'sv', 'cs', 'fa'])

    gen_article_texts(num_per_cluster = 10, lang_codes=popular_language_codes)
