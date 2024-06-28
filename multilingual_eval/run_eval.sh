models=("Qwen/Qwen2-7B-Instruct" "meta-llama/Meta-Llama-3-8B-Instruct" "google/gemma-2-9b-it" "microsoft/Phi-3-small-8k-instruct" "mistralai/Mistral-7B-Instruct-v0.3")

# Loop over each string in the list
for model in "${models[@]}"; do
    HF_HOME=/workspace/hf_home HF_HUB_ENABLE_HF_TRANSFER=1 accelerate launch -m lm_eval --model hf --model_args pretrained=$model --tasks belebele,xcopa,xwinograd --batch_size auto --output_path ./results
    HF_HOME=/workspace/hf_home HF_HUB_ENABLE_HF_TRANSFER=1 accelerate launch -m lm_eval --model vllm --model_args pretrained=$model --tasks mgsm_direct,ifeval --batch_size auto --output_path ./results
done