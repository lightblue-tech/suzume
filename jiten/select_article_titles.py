from datasets import load_dataset
import pandas as pd
from tqdm.auto import trange, tqdm
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from multiprocessing import Pool
import os
from glob import glob
import random

num_proc = 32
seed = 0

def get_long_articles(data_file_paths):
    # Get id stubs
    vcs = None
    lang = data_file_paths[0].split("/")[-2]
    for path in tqdm(data_file_paths, desc=f"Long articles {lang}"):
        df = pd.read_parquet(path)
        batch_vc = df._id.str.split("_").str[:-1].str.join("_").value_counts()
        vcs = batch_vc if vcs is None else (batch_vc + vcs).fillna(batch_vc).fillna(vcs)
    # Select all articles that have more than 1 entry (i.e. long articles)
    long_articles = set(vcs[vcs > 1].index)
    return long_articles

def select_first_chunk_long_articles(df, long_article_ids):
    underbar_split = df._id.str.split("_")
    df["id_stem"] = underbar_split.str[:-1].str.join("_")
    first_chunk_mask = underbar_split.str[-1] == "0"
    long_article_mask = df["id_stem"].isin(long_article_ids)
    mask = first_chunk_mask & long_article_mask
    df = df[mask]
    return df

def calculate_kmeans(data_file_paths, long_article_ids, n_clusters=500, batch_size=1_000):
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=batch_size,
        n_init="auto"
    )
    
    all_df = None
    did_clustering = False
    lang = data_file_paths[0].split("/")[-2]
    for path in tqdm(data_file_paths, desc=f"K means {lang}"):
        df = pd.read_parquet(path)
        df = select_first_chunk_long_articles(df, long_article_ids)
        
        all_df = df if all_df is None else pd.concat([all_df, df])
        if all_df.shape[0] >= n_clusters:
            first_embeddings = np.stack(all_df.sample(frac=1.0, random_state=seed).emb.values)
            kmeans = kmeans.partial_fit(first_embeddings)
            all_df = None
            did_clustering = True
    
    if all_df is not None and all_df.shape[0] >= n_clusters:
        first_embeddings = np.stack(all_df.sample(frac=1.0, random_state=seed).emb.values)
        kmeans = kmeans.partial_fit(first_embeddings)
    elif not did_clustering:
        print("Did not cluster for:\n" + "\n".join(data_file_paths))
        kmeans = None
        
    return kmeans

def get_title_df(data_file_paths, long_article_ids, kmeans):
    title_labels = []
    lang = data_file_paths[0].split("/")[-2]
    for path in tqdm(data_file_paths, desc=f"Title select {lang}"):
        df = pd.read_parquet(path)
        df = select_first_chunk_long_articles(df, long_article_ids)
        if df.shape[0] < 1:
            continue
        elif kmeans is None:
            df["label"] = range(df.shape[0])
        else:
            first_embeddings = np.stack(df.emb.values)
            df["label"] = kmeans.predict(first_embeddings)
        title_labels.append(df[["title", "label", "url"]])
    return pd.concat(title_labels)

def save_title_df(data_folder, n_clusters=5000):
    data_file_path = os.path.join(data_folder, "*.parquet")
    data_file_paths = glob(data_file_path)
    random.Random(seed).shuffle(data_file_paths)
    
    lang = data_folder.split("/")[-1]
    
    print(lang)
    # Remove short articles
    long_article_ids = get_long_articles(data_file_paths)
                
    # Calculate kmeans
    kmeans = calculate_kmeans(data_file_paths, long_article_ids, n_clusters=n_clusters)

    # Add the cluster id for each title and create the df
    title_df = get_title_df(data_file_paths, long_article_ids, kmeans)
    
    title_df.to_parquet(f"title_cluster_ids_{lang}_{n_clusters}.parquet")

if __name__ == '__main__':
    parallel_lang_proc_num = 8
    paths = sorted(glob("./data/*"))
    with Pool(parallel_lang_proc_num) as p:
        p.map(save_title_df, paths)
