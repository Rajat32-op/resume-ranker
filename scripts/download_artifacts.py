from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Rajat3206/redrob-artifacts",
    repo_type="dataset",
    filename="candidate_embeddings.npy",
    local_dir="artifacts"
)

hf_hub_download(
    repo_id="Rajat3206/redrob-artifacts",
    repo_type="dataset",
    filename="candidate_ids.npy",
    local_dir="artifacts"
)