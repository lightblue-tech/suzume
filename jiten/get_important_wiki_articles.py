import re
import pandas as pd
from tqdm.auto import tqdm
import subprocess
import os
from FlagEmbedding import BGEM3FlagModel
import torch
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def count_file_lines(file_path):
    print(f"Calculating the number of lines of {file_path}")
    with open(file_path, 'r') as file:
        file_len = 0
        for line in tqdm(file):
            file_len += 1
    return file_len

def add_to_value_counts(all_vc, match_list):
    match_df = pd.DataFrame(match_list,
                            columns=["pl_from", "pl_namespace", "pl_title", "last_char", "pl_from_namespace", "pl_target_id"]
                            )
    article_df = match_df[(match_df["pl_namespace"] == "0") & (match_df["pl_from_namespace"] == "0")]
    vc = article_df["pl_title"].value_counts()
    all_vc = vc if all_vc is None else (all_vc + vc).fillna(all_vc).fillna(vc)
    return all_vc

make_url = lambda l, ddt: f"https://dumps.wikimedia.org/{l}wiki/{ddt}/{l}wiki-{ddt}-pagelinks.sql.gz"

def write_link_counts(lang, data_dump_time = "20240501"):
    url = make_url(lang, data_dump_time)
    filename = f"{lang}wiki-{data_dump_time}-pagelinks"

    result = subprocess.run(['wget', url], check=True)
    result = subprocess.run (['gzip', '-d', f'{filename}.sql.gz'])

    # File path to the SQL file
    file_path = f'{filename}.sql'

    regex_pattern = r"\((?P<item1>\d+),(?P<item2>\d+),(?P<item3>(?<!\\')'([^']|\\')+(?<!\\)'),(?P<item4>\d+),(?P<item5>\d+)\)"
    regex_compiled = re.compile(regex_pattern, re.VERBOSE)

    file_len = count_file_lines(file_path)

    all_vc = None

    with open(file_path, 'r') as file:
        match_list = []
        for i, line in enumerate(tqdm(file, total=file_len)):
            matches = regex_compiled.findall(line)
            match_list.extend(matches)

            if len(match_list) > 10_000_000:
                all_vc = add_to_value_counts(all_vc, match_list)
                match_list = []

    all_vc = add_to_value_counts(all_vc, match_list)
    match_list = []
    
    os.remove(file_path)
    pd.DataFrame(all_vc).to_parquet(f"counts_{lang}.parquet")

def get_sim_matrix(embeddings):
    embeddings_cuda = torch.Tensor(embeddings).to("cuda")
    similarity = (embeddings_cuda @ embeddings_cuda.T).cpu().numpy()
    return similarity

def get_first_from_each_label(top_df):
    first_df = top_df.sort_values(
      "count", ascending=False
    ).reset_index(
        drop=False
    ).groupby(
        "label"
    ).first().sort_values(
        "count", ascending=False
    )
    return first_df

def get_top_article_names(language_code, model, initial_cluster_num=25_000, final_selected_num=5_000):

    df = pd.read_parquet(f"counts_{language_code}.parquet")

    top_df = df.sort_values(
        "count", ascending=False
        ).iloc[:initial_cluster_num].sample(frac=1.0)
    popular_articles = top_df.index
    popular_articles = [x.replace("_", " ") for x in popular_articles]

    embeddings = model.encode(popular_articles, batch_size=256)['dense_vecs']

    similarity = get_sim_matrix(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=final_selected_num, 
        metric='precomputed', 
        linkage='average'
    )
    clustering = clustering.fit(1 - similarity)

    top_df["label"] = clustering.labels_
    first_df = get_first_from_each_label(top_df)
    first_df.to_parquet(f"important_{language_code}_{final_selected_num}.parquet")

language_codes = ['aa', 'ab', 'ace', 'ady', 'af', 'ak', 'als', 'alt', 'am', 'ami', 'an', 'ang', 'anp', 'ar', 'arc', 'ary', 'arz', 'as', 'ast', 'atj', 'av', 'avk', 'awa', 'ay', 'az', 'azb', 'ba', 'ban', 'bar', 'bat_smg', 'bbc', 'bcl', 'be', 'bew', 'bg', 'bh', 'bi', 'bjn', 'blk', 'bm', 'bn', 'bo', 'bpy', 'br', 'bs', 'bug', 'bxr', 'ca', 'cbk_zam', 'cdo', 'ce', 'ceb', 'ch', 'cho', 'chr', 'chy', 'ckb', 'co', 'cr', 'crh', 'cs', 'csb', 'cu', 'cv', 'cy', 'da', 'dag', 'de', 'dga', 'din', 'diq', 'dsb', 'dtp', 'dty', 'dv', 'dz', 'ee', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'fat', 'ff', 'fi', 'fiu_vro', 'fj', 'fo', 'fon', 'fr', 'frp', 'frr', 'fur', 'fy', 'ga', 'gag', 'gan', 'gcr', 'gd', 'gl', 'glk', 'gn', 'gom', 'gor', 'got', 'gpe', 'gu', 'guc', 'gur', 'guw', 'gv', 'ha', 'hak', 'haw', 'he', 'hi', 'hif', 'ho', 'hr', 'hsb', 'ht', 'hu', 'hy', 'hyw', 'hz', 'ia', 'id', 'ie', 'ig', 'igl', 'ii', 'ik', 'ilo', 'inh', 'io', 'is', 'it', 'iu', 'ja', 'jam', 'jbo', 'jv', 'ka', 'kaa', 'kab', 'kbd', 'kbp', 'kcg', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'koi', 'kr', 'krc', 'ks', 'ksh', 'ku', 'kus', 'kv', 'kw', 'ky', 'la', 'lad', 'lb', 'lbe', 'lez', 'lfn', 'lg', 'li', 'lij', 'lld', 'lmo', 'ln', 'lo', 'lrc', 'lt', 'ltg', 'lv', 'mad', 'mai', 'map_bms', 'mdf', 'meta', 'mg', 'mh', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mni', 'mnw', 'mr', 'mrj', 'ms', 'mt', 'mus', 'mwl', 'my', 'myv', 'mzn', 'na', 'nah', 'nap', 'nds', 'nds_nl', 'ne', 'new', 'ng', 'nia', 'nl', 'nn', 'no', 'nov', 'nqo', 'nrm', 'nso', 'nv', 'ny', 'oc', 'olo', 'om', 'or', 'os', 'pa', 'pag', 'pam', 'pap', 'pcd', 'pcm', 'pdc', 'pfl', 'pi', 'pih', 'pl', 'pms', 'pnb', 'pnt', 'ps', 'pt', 'pwn', 'qu', 'rm', 'rmy', 'rn', 'ro', 'roa_rup', 'roa_tar', 'roa_tara', 'ru', 'rue', 'rw', 'sa', 'sah', 'sat', 'sc', 'scn', 'sco', 'sd', 'se', 'sg', 'sh', 'shi', 'shn', 'si', 'simple', 'sk', 'skr', 'sl', 'sm', 'smn', 'sn', 'so', 'sq', 'sr', 'srn', 'ss', 'st', 'stq', 'su', 'sv', 'sw', 'szl', 'szy', 'ta', 'tay', 'tcy', 'te', 'ten', 'test', 'tet', 'tg', 'th', 'ti', 'tk', 'tl', 'tly', 'tn', 'to', 'tpi', 'tr', 'trv', 'ts', 'tt', 'tum', 'tw', 'ty', 'tyv', 'udm', 'ug', 'uk', 'ur', 'uz', 've', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wo', 'wuu', 'xal', 'xh', 'xmf', 'yi', 'yo', 'za', 'ze', 'zea', 'zgh', 'zh', 'zh_classical', 'zh_min_nan', 'zh_yue', 'zu']

popular_language_codes = ['en', 'de', 'fr', 'es', 'ru', 'pt', 'it', 'pl', 'nl', 'uk', 'fi', 'hu', 'et', 'tr', 'uz', 'az', 'kk', 'ar', 'he', 'am', 'my', 'sw', 'zu', 'xh', 'yo', 'ig', 'ta', 'ml', 'te', 'kn', 'id', 'vi', 'th', 'ms', 'jv', 'tl', 'km', 'su', 'hi', 'mr', 'gu', 'pa', 'bn', 'ja', 'ko', 'mi', 'sm', 'ur', 'zh', 'zh_min_nan', 'pcm', 'ha', 'om', 'ro', 'el', 'sv', 'cs', 'fa']

for l in language_codes:
    write_link_counts(l)

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

for l in language_codes:
    get_top_article_names(l, model, final_selected_num=500)

for l in popular_language_codes:
    get_top_article_names(l, model, final_selected_num=5_000)