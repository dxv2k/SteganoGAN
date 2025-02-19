import kagglehub

# Download latest version
path = kagglehub.dataset_download("dntai2/cocostuff-10k-v1-1")

print("Path to dataset files:", path)
