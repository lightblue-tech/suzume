import re
import pandas as pd
from tqdm.auto import tqdm
import subprocess
import os

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
    pd.DataFrame(all_vc).to_parquet(f"{lang}.parquet")
    
language_codes = ['ady', 'af', 'alt', 'am', 'an', 'ang', 'ar', 'arc', 'ary', 'as', 'ast', 'awa', 'ay', 'az', 'azb', 'bcl', 'be', 'bew', 'bg', 'bi', 'bjn', 'bm', 'bn', 'bpy', 'br', 'bs', 'bxr', 'ca', 'ce', 'ceb', 'chr', 'ckb', 'cs', 'csb', 'cv', 'cy', 'dag', 'de', 'diq', 'dsb', 'dty', 'dv', 'ee', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'gag', 'gcr', 'gl', 'got', 'gu', 'haw', 'he', 'hi', 'hif', 'hr', 'hsb', 'hu', 'hy', 'ia', 'id', 'ig', 'igl', 'ilo', 'io', 'is', 'it', 'ja', 'jam', 'jv', 'ka', 'kab', 'kc', 'kg', 'kk', 'kl', 'km', 'kn', 'ko', 'ku', 'kus', 'ky', 'la', 'lad', 'lb', 'ln', 'lt', 'lv', 'mdf', 'meta', 'mg', 'mi', 'mk', 'ml', 'mni', 'mnw', 'mr', 'ms', 'mus', 'my', 'mzn', 'na', 'nah', 'nds', 'nl', 'nn', 'no', 'nv', 'oc', 'os', 'pa', 'pcd', 'pfl', 'pih', 'pl', 'pt', 'qu', 'rm', 'rmy', 'rn', 'ro', 'roa_tar', 'ru', 'rue', 'rw', 'sah', 'sc', 'scn', 'sd', 'se', 'sg', 'sh', 'shn', 'simple', 'sk', 'sl', 'sm', 'smn', 'sq', 'sr', 'srn', 'st', 'su', 'sv', 'sw', 'szl', 'szy', 'ta', 'te', 'test', 'th', 'ti', 'tl', 'tn', 'tpi', 'tr', 'tt', 'ty', 'uk', 'ur', 'uz', 've', 'vec', 'vep', 'vi', 'vls', 'vo', 'war', 'wo', 'xal', 'yi', 'ze', 'zh', 'zh_min_nan'] #'btm', 

for l in language_codes:
    write_link_counts(l)