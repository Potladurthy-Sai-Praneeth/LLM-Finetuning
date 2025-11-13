import google.cloud.aiplatform as aiplatform

# --- 1. Define your job configuration ---
PROJECT_ID = "finetuning-477501"
REGION = "us-west1"
BUCKET_URI = "gs://gemma3_finetuning_psp"
JOB_NAME = "FSDP-QLoRA-Finetune-Job"

# The new generic image URI
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/finetuning/llm_finetuning-v4:latest"
GPU_COUNT = 4

SCRIPT_ARGS = [
    f"--nproc_per_node={GPU_COUNT}",
    "-m",
    "finetune.deepspeed_finetune",
]


aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "n1-standard-16", 
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": GPU_COUNT,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": IMAGE_URI,
            "args": SCRIPT_ARGS 
        },
    }
]

job = aiplatform.CustomJob(
    display_name=JOB_NAME,
    worker_pool_specs=worker_pool_specs,
)

print(f"Submitting Vertex AI Custom Job: {JOB_NAME}")
print(f"Will run command: torchrun {' '.join(SCRIPT_ARGS)}")
job.run(sync=False) 
print(f"Job submitted. View in console: {job._dashboard_uri}")