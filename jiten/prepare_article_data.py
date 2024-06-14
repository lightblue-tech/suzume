from glob import glob
from datasets import load_dataset
import pandas as pd

supported_language_map = {'ab': 'Abkhaz', 'ace': 'Acehnese', 'ady': 'Adyghe', 'af': 'Afrikaans', 'als': 'Alemannic German', 'alt': 'Southern Altai', 'am': 'Amharic', 'ami': 'Amis', 'an': 'Aragonese', 'ang': 'Old English', 'anp': 'Angika', 'ar': 'Arabic', 'arc': 'Aramaic', 'ary': 'Moroccan Arabic', 'arz': 'Egyptian Arabic', 'as': 'Assamese', 'ast': 'Asturian', 'atj': 'Atikamekw', 'av': 'Avar', 'avk': 'Kotava', 'awa': 'Awadhi', 'ay': 'Aymara', 'az': 'Azerbaijani', 'azb': 'South Azerbaijani', 'ba': 'Bashkir', 'ban': 'Balinese', 'bar': 'Bavarian', 'bat-smg': 'Samogitian', 'bcl': 'Central Bikol', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bh': 'Bihari', 'bi': 'Bislama', 'bjn': 'Banjarese', 'blk': "Pa'O", 'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Lhasa Tibetan', 'bpy': 'Bishnupriya Manipuri', 'br': 'Breton', 'bs': 'Bosnian', 'bug': 'Buginese', 'bxr': 'Buryat', 'ca': 'Catalan', 'cbk-zam': 'Zamboanga Chavacano', 'cdo': 'Eastern Min', 'ce': 'Chechen', 'ceb': 'Cebuano', 'ch': 'Chamorro', 'chr': 'Cherokee', 'chy': 'Cheyenne', 'ckb': 'Kurdish (Sorani)', 'co': 'Corsican', 'cr': 'Cree', 'crh': 'Crimean Tatar', 'cs': 'Czech', 'csb': 'Kashubian', 'cu': 'Old Church Slavonic', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'dag': 'Dagbani', 'de': 'German', 'din': 'Dinka', 'diq': 'Zaza', 'dsb': 'Lower Sorbian', 'dty': 'Doteli', 'dv': 'Maldivian', 'dz': 'Dzongkha', 'ee': 'Ewe', 'el': 'Greek', 'eml': 'Emilian–Romagnol', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'ext': 'Extremaduran', 'fa': 'Persian', 'fat': 'Fante', 'ff': 'Fula', 'fi': 'Finnish', 'fiu-vro': 'Võro', 'fj': 'Fijian', 'fo': 'Faroese', 'fr': 'French', 'frp': 'Franco-Provençal', 'frr': 'North Frisian', 'fur': 'Friulian', 'fy': 'West Frisian', 'ga': 'Irish', 'gag': 'Gagauz', 'gan': 'Gan Chinese', 'gcr': 'French Guianese Creole', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'glk': 'Gilaki', 'gn': 'Guarani', 'gom': 'Konkani', 'gor': 'Gorontalo', 'got': 'Gothic', 'gu': 'Gujarati', 'guc': 'Wayuu', 'gur': 'Farefare', 'guw': 'Gun', 'gv': 'Manx', 'ha': 'Hausa', 'hak': 'Hakka Chinese', 'haw': 'Hawaiian', 'he': 'Hebrew', 'hi': 'Hindi', 'hif': 'Fiji Hindi', 'hr': 'Croatian', 'hsb': 'Upper Sorbian', 'ht': 'Haitian Creole', 'hu': 'Hungarian', 'hy': 'Armenian', 'hyw': 'Western Armenian', 'ia': 'Interlingua', 'id': 'Indonesian', 'ie': 'Interlingue', 'ig': 'Igbo', 'ik': 'Iñupiaq', 'ilo': 'Ilocano', 'inh': 'Ingush', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese', 'jam': 'Jamaican Patois', 'jbo': 'Lojban', 'jv': 'Javanese', 'ka': 'Georgian', 'kaa': 'Karakalpak', 'kab': 'Kabyle', 'kbd': 'Kabardian', 'kbp': 'Kabiye', 'kcg': 'Tyap', 'kg': 'Kongo', 'ki': 'Kikuyu', 'kk': 'Kazakh', 'kl': 'Greenlandic', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'koi': 'Komi-Permyak', 'krc': 'Karachay-Balkar', 'ks': 'Kashmiri', 'ksh': 'Ripuarian', 'ku': 'Kurdish (Kurmanji)', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lad': 'Judaeo-Spanish', 'lb': 'Luxembourgish', 'lbe': 'Lak', 'lez': 'Lezgian', 'lfn': 'Lingua Franca Nova', 'lg': 'Luganda', 'li': 'Limburgish', 'lij': 'Ligurian', 'lld': 'Ladin', 'lmo': 'Lombard', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'ltg': 'Latgalian', 'lv': 'Latvian', 'mad': 'Madurese', 'mai': 'Maithili', 'map-bms': 'Banyumasan', 'mdf': 'Moksha', 'mg': 'Malagasy', 'mhr': 'Meadow Mari', 'mi': 'Māori', 'min': 'Minangkabau', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mni': 'Meitei', 'mnw': 'Mon', 'mr': 'Marathi', 'mrj': 'Hill Mari', 'ms': 'Malay', 'mt': 'Maltese', 'mwl': 'Mirandese', 'my': 'Burmese', 'myv': 'Erzya', 'mzn': 'Mazanderani', 'nah': 'Nahuatl', 'nap': 'Neapolitan', 'nds-nl': 'Dutch Low Saxon', 'nds': 'Low German', 'ne': 'Nepali', 'new': 'Newar', 'nia': 'Nias', 'nl': 'Dutch', 'nn': 'Norwegian (Nynorsk)', 'no': 'Norwegian (Bokmål)', 'nov': 'Novial', 'nqo': "N'Ko", 'nrm': 'Norman', 'nso': 'Northern Sotho', 'nv': 'Navajo', 'ny': 'Chewa', 'oc': 'Occitan', 'olo': 'Livvi-Karelian', 'om': 'Oromo', 'or': 'Odia', 'os': 'Ossetian', 'pa': 'Punjabi', 'pag': 'Pangasinan', 'pam': 'Kapampangan', 'pap': 'Papiamento', 'pcd': 'Picard', 'pcm': 'Nigerian Pidgin', 'pdc': 'Pennsylvania Dutch', 'pfl': 'Palatine German', 'pi': 'Pali', 'pih': 'Norfuk', 'pl': 'Polish', 'pms': 'Piedmontese', 'pnb': 'Western Punjabi', 'pnt': 'Pontic Greek', 'ps': 'Pashto', 'pt': 'Portuguese', 'pwn': 'Paiwan', 'qu': 'Quechua', 'rm': 'Romansh', 'rmy': 'Romani', 'rn': 'Kirundi', 'ro': 'Romanian', 'roa-rup': 'Aromanian', 'roa-tara': 'Tarantino', 'ru': 'Russian', 'rue': 'Rusyn', 'rw': 'Kinyarwanda', 'sa': 'Sanskrit', 'sah': 'Yakut', 'sat': 'Santali', 'sc': 'Sardinian', 'scn': 'Sicilian', 'sco': 'Scots', 'sd': 'Sindhi', 'se': 'Northern Sámi', 'sg': 'Sango', 'sh': 'Serbo-Croatian', 'shi': 'Shilha', 'shn': 'Shan', 'si': 'Sinhala', 'simple': 'Simple English', 'sk': 'Slovak', 'skr': 'Saraiki', 'sl': 'Slovene', 'sm': 'Samoan', 'smn': 'Inari Sámi', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'srn': 'Sranan Tongo', 'ss': 'Swazi', 'st': 'Sotho', 'stq': 'Saterland Frisian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'szl': 'Silesian', 'szy': 'Sakizaya', 'ta': 'Tamil', 'tay': 'Atayal', 'tcy': 'Tulu', 'te': 'Telugu', 'tet': 'Tetum', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tn': 'Tswana', 'to': 'Tongan', 'tpi': 'Tok Pisin', 'tr': 'Turkish', 'trv': 'Seediq', 'ts': 'Tsonga', 'tt': 'Tatar', 'tum': 'Tumbuka', 'tw': 'Twi', 'ty': 'Tahitian', 'tyv': 'Tuvan', 'udm': 'Udmurt', 'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vec': 'Venetian', 'vep': 'Veps', 'vi': 'Vietnamese', 'vls': 'West Flemish', 'vo': 'Volapük', 'wa': 'Walloon', 'war': 'Waray', 'wo': 'Wolof', 'wuu': 'Wu Chinese', 'xal': 'Kalmyk Oirat', 'xh': 'Xhosa', 'xmf': 'Mingrelian', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang', 'zea': 'Zeelandic', 'zh-classical': 'Classical Chinese', 'zh-min-nan': 'Southern Min', 'zh-yue': 'Cantonese', 'zh': 'Chinese', 'zu': 'Zulu'}

def generate_dataset(paths, dataset_name):

    datasets_list = []

    for path in paths:
        lang_code = path.split("_")[-2]

        if lang_code not in supported_language_map.keys():
            print(f"{lang_code} not supported!")
            continue

        print(lang_code)
        df = pd.read_parquet(path)
        top_titles = set(df["pl_title"].str.strip("'").str.replace("_", " ").tolist())
        print(f"% articles with more than 1 link: {100*(df['count'] > 1).mean():.4}%")
        dataset = load_dataset("graelo/wikipedia", f"20230601.{lang_code}", split="train", trust_remote_code=True)
        top_dataset = dataset.filter(lambda x: x["title"] in top_titles, num_proc=12)
        print(len(top_dataset))
        df = top_dataset.to_pandas()
        df["language_code"] = lang_code
        df["language"] = supported_language_map[lang_code]
        datasets_list.append(df)
        # ! rm -r ~/.cache/huggingface

    concat_df = pd.concat(datasets_list)
    c.to_parquet(f"wiki_texts_{dataset_name}.parquet")

paths = sorted(glob("./*_500.parquet"))
generate_dataset(paths, "500")

paths = sorted(glob("./*_5000.parquet"))
generate_dataset(paths, "5000")

jiten_283 = load_dataset(
    "parquet", 
    data_files={"train": "wiki_texts_500.parquet"}, 
    split="train"
)
jiten_283.push_to_hub("lightblue/jiten_283_preprocessed", private=True)

jiten_57 = load_dataset(
    "parquet", 
    data_files={"train": "wiki_texts_5000.parquet"}, 
    split="train"
)
jiten_57.push_to_hub("lightblue/jiten_57_preprocessed", private=True)