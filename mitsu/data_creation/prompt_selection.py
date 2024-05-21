from datasets import load_dataset, Dataset

if __name__ == "__main__": 
    # See tagengo/data_creation/tagengo_prompt_preparation.ipynb for how we prepared these prompts
    dataset = load_dataset("lightblue/multilingual_prompts_25k_max", split="train")
    df = dataset.to_pandas()
    # Sample 100 prompts per language
    sampled_df = df.groupby("language").apply(lambda x: x.sample(n=min(x.shape[0], 100)))
    # Push df to huggingface hub
    Dataset.from_pandas(
        sampled_df.drop("__index_level_0__", axis=1)
    ).remove_columns(
        ["__index_level_0__"]
    ).push_to_hub("lightblue/multilingual_prompts_100_sample", private=True)