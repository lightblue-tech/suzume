import subprocess
from tqdm.auto import trange
from multiprocessing import Pool

def download(inputs):
    lang, idx = inputs
    i_str = str(idx).zfill(4)
    print(lang)
    print(idx)
    print()
    url = f"https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/resolve/main/{lang}/{i_str}.parquet"
    result = subprocess.run(["wget", url, "-P", f"data/{lang}/{i_str}.parquet"], capture_output=True, text=True)

def download_all_datasets(parallel_lang_proc_num = 256):
    dataset_file_counts = {'ab': 0, 'ace': 0, 'ady': 0, 'af': 4, 'als': 2, 'alt': 0, 'am': 0, 'ami': 0, 'an': 1, 'ang': 0, 'anp': 0, 'ar': 36, 'arc': 0, 'ary': 0, 'arz': 10, 'as': 0, 'ast': 10, 'atj': 0, 'av': 0, 'avk': 0, 'awa': 0, 'ay': 0, 'az': 8, 'azb': 2, 'ba': 4, 'ban': 0, 'bar': 0, 'bat-smg': 0, 'bcl': 0, 'be-x-old': 3, 'be': 8, 'bg': 14, 'bh': 0, 'bi': 0, 'bjn': 0, 'blk': 0, 'bm': 0, 'bn': 7, 'bo': 0, 'bpy': 0, 'br': 2, 'bs': 3, 'bug': 0, 'bxr': 0, 'ca': 41, 'cbk-zam': 0, 'cdo': 0, 'ce': 10, 'ceb': 98, 'ch': 0, 'chr': 0, 'chy': 0, 'ckb': 1, 'co': 0, 'cr': 0, 'crh': 0, 'cs': 31, 'csb': 0, 'cu': 0, 'cv': 1, 'cy': 7, 'da': 12, 'dag': 0, 'de': 207, 'din': 0, 'diq': 0, 'dsb': 0, 'dty': 0, 'dv': 0, 'dz': 0, 'ee': 0, 'el': 15, 'eml': 0, 'en': 414, 'eo': 12, 'es': 129, 'et': 10, 'eu': 13, 'ext': 0, 'fa': 20, 'fat': 0, 'ff': 0, 'fi': 24, 'fiu-vro': 0, 'fj': 0, 'fo': 0, 'fon': 0, 'fr': 178, 'frp': 0, 'frr': 0, 'fur': 0, 'fy': 2, 'ga': 1, 'gag': 0, 'gan': 0, 'gcr': 0, 'gd': 0, 'gl': 10, 'glk': 0, 'gn': 0, 'gom': 0, 'gor': 0, 'got': 0, 'gpe': 0, 'gu': 0, 'guc': 0, 'gur': 0, 'guw': 0, 'gv': 0, 'ha': 1, 'hak': 0, 'haw': 0, 'he': 29, 'hi': 5, 'hif': 0, 'hr': 9, 'hsb': 0, 'ht': 0, 'hu': 29, 'hy': 13, 'hyw': 0, 'ia': 0, 'id': 23, 'ie': 0, 'ig': 1, 'ik': 0, 'ilo': 0, 'inh': 0, 'io': 1, 'is': 1, 'it': 104, 'iu': 0, 'ja': 66, 'jam': 0, 'jbo': 0, 'jv': 1, 'ka': 5, 'kaa': 0, 'kab': 0, 'kbd': 0, 'kbp': 0, 'kcg': 0, 'kg': 0, 'ki': 0, 'kk': 6, 'kl': 0, 'km': 0, 'kn': 3, 'ko': 15, 'koi': 0, 'krc': 0, 'ks': 0, 'ksh': 0, 'ku': 0, 'kv': 0, 'kw': 0, 'ky': 2, 'la': 3, 'lad': 0, 'lb': 2, 'lbe': 0, 'lez': 0, 'lfn': 0, 'lg': 0, 'li': 0, 'lij': 0, 'lld': 1, 'lmo': 1, 'ln': 0, 'lo': 0, 'lt': 7, 'ltg': 0, 'lv': 4, 'mad': 0, 'mai': 0, 'map-bms': 0, 'mdf': 0, 'mg': 1, 'mhr': 0, 'mi': 0, 'min': 3, 'mk': 7, 'ml': 3, 'mn': 1, 'mni': 0, 'mnw': 0, 'mr': 2, 'mrj': 0, 'ms': 8, 'mt': 0, 'mwl': 0, 'my': 2, 'myv': 0, 'mzn': 0, 'nah': 0, 'nap': 0, 'nds-nl': 0, 'nds': 1, 'ne': 0, 'new': 1, 'nia': 0, 'nl': 61, 'nn': 5, 'no': 22, 'nov': 0, 'nqo': 0, 'nrm': 0, 'nso': 0, 'nv': 0, 'ny': 0, 'oc': 2, 'olo': 0, 'om': 0, 'or': 0, 'os': 0, 'pa': 1, 'pag': 0, 'pam': 0, 'pap': 0, 'pcd': 0, 'pcm': 0, 'pdc': 0, 'pfl': 0, 'pi': 0, 'pih': 0, 'pl': 59, 'pms': 0, 'pnb': 3, 'pnt': 0, 'ps': 1, 'pt': 56, 'pwn': 0, 'qu': 0, 'rm': 0, 'rmy': 0, 'rn': 0, 'ro': 17, 'roa-rup': 0, 'roa-tara': 0, 'ru': 137, 'rue': 0, 'rw': 0, 'sa': 0, 'sah': 0, 'sat': 0, 'sc': 0, 'scn': 0, 'sco': 0, 'sd': 0, 'se': 0, 'sg': 0, 'sh': 11, 'shi': 0, 'shn': 0, 'si': 1, 'simple': 6, 'sk': 8, 'skr': 0, 'sl': 9, 'sm': 0, 'smn': 0, 'sn': 0, 'so': 0, 'sq': 3, 'sr': 21, 'srn': 0, 'ss': 0, 'st': 0, 'stq': 0, 'su': 1, 'sv': 49, 'sw': 1, 'szl': 0, 'szy': 0, 'ta': 6, 'tay': 0, 'tcy': 0, 'te': 6, 'tet': 0, 'tg': 1, 'th': 8, 'ti': 0, 'tk': 0, 'tl': 1, 'tly': 0, 'tn': 0, 'to': 0, 'tpi': 0, 'tr': 17, 'trv': 0, 'ts': 0, 'tt': 11, 'tum': 0, 'tw': 0, 'ty': 0, 'tyv': 0, 'udm': 0, 'ug': 0, 'uk': 69, 'ur': 4, 'uz': 8, 've': 0, 'vec': 0, 'vep': 0, 'vi': 28, 'vls': 0, 'vo': 0, 'wa': 0, 'war': 11, 'wo': 0, 'wuu': 0, 'xal': 0, 'xh': 0, 'xmf': 0, 'yi': 0, 'yo': 0, 'za': 0, 'zea': 0, 'zh-classical': 0, 'zh-min-nan': 2, 'zh-yue': 0, 'zh': 27, 'zu': 0}

    dataset_ids = []

    for l, c in dataset_file_counts.items():
        dataset_ids.extend([(l, i) for i in range(c+1)])

    with Pool(parallel_lang_proc_num) as p:
        p.map(download, dataset_ids)

if __name__ == '__main__':
    download_all_datasets()