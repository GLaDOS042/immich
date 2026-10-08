# Immich Machine Learning

- CLIP embeddings
- Facial recognition

# Setup

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/), so be sure to install it first.
Running `uv sync --extra cpu` will install everything you need in an isolated virtual environment.
CUDA, ROCM and OpenVINO are supported as acceleration APIs. To use them, you can replace `--extra cpu` with either of `--extra cuda`, `--extra rocm` or `--extra openvino`. In the case of CUDA, a [compute capability](https://developer.nvidia.com/cuda-gpus) of 5.2 or higher is required.

To add or remove dependencies, you can use the commands `uv add $PACKAGE_NAME` and `uv remove $PACKAGE_NAME`, respectively.
Be sure to commit the `uv.lock` and `pyproject.toml` files with `uv lock` to reflect any changes in dependencies.

# Load Testing

To measure inference throughput and latency, you can use [Locust](https://locust.io/) using the provided `locustfile.py`.
Locust works by querying the model endpoints and aggregating their statistics, meaning the app must be deployed.
You can change the models or adjust options like score thresholds through the Locust UI.

To get started, you can simply run `locust --web-host 127.0.0.1` and open `localhost:8089` in your browser to access the UI. See the [Locust documentation](https://docs.locust.io/en/stable/index.html) for more info on running Locust.

Note that in Locust's jargon, concurrency is measured in `users`, and each user runs one task at a time. To achieve a particular per-endpoint concurrency, multiply that number by the number of endpoints to be queried. For example, if there are 3 endpoints and you want each of them to receive 8 requests at a time, you should set the number of users to 24.

# Facial Recognition

## Acknowledgements

This project utilizes facial recognition models from the [InsightFace](https://github.com/deepinsight/insightface/tree/master/model_zoo) project. We appreciate the work put into developing these models, which have been beneficial to the machine learning part of this project.

### Used Models

- antelopev2
- buffalo_l
- buffalo_m
- buffalo_s

## License and Use Restrictions

We have received permission to use the InsightFace facial recognition models in our project, as granted via email by Jia Guo (guojia@insightface.ai) on 18th March 2023. However, it's important to note that this permission does not extend to the redistribution or commercial use of their models by third parties. Users and developers interested in using these models should review the licensing terms provided in the InsightFace GitHub repository.

For more information on the capabilities of the InsightFace models and to ensure compliance with their license, please refer to their [official repository](https://github.com/deepinsight/insightface). Adhering to the specified licensing terms is crucial for the respectful and lawful use of their work.

# EmbeddingGemma 2 Smart Search (experimental)

This fork can route one existing 768-dimensional Immich Smart Search model name to EmbeddingGemma 2 while preserving the current Immich machine-learning `/predict` API and response format. This allows the stock Immich server image to remain unchanged.

The backend uses the ONNX Community `embeddinggemma-2-ONNX` text and vision graphs through Immich's existing ONNX Runtime execution-provider stack. On the ROCm image this means `MIGraphXExecutionProvider` is used in the same way as other Immich ML models.

## Configuration

Choose an existing Immich Smart Search model with a 768-dimensional embedding. The recommended alias is:

```text
ViT-B-16-SigLIP-256__webli
```

Configure the custom ML container with:

```yaml
environment:
  MACHINE_LEARNING_EMBEDDING_GEMMA2_ALIAS: ViT-B-16-SigLIP-256__webli
  MACHINE_LEARNING_EMBEDDING_GEMMA2_VISION_TOKENS: 280
```

Then select `ViT-B-16-SigLIP-256__webli` as the Smart Search model in Immich. The Immich server still sees a supported 768-dimensional model, while this ML service routes visual and textual Smart Search requests to EmbeddingGemma 2.

Do not use an alias whose Immich embedding dimension is not 768. Changing the alias or enabling this backend requires re-running the Smart Search jobs so image and query embeddings come from the same model.

Supported vision token budgets are `70`, `140`, `280`, `560`, and `1120`. `280` is the upstream default. Lower values reduce image inference cost; higher values preserve more spatial detail.

Additional optional settings:

```text
MACHINE_LEARNING_EMBEDDING_GEMMA2_REPO=onnx-community/embeddinggemma-2-ONNX
MACHINE_LEARNING_EMBEDDING_GEMMA2_REVISION=main
MACHINE_LEARNING_EMBEDDING_GEMMA2_TEXT_CONTEXT=128
MACHINE_LEARNING_EMBEDDING_GEMMA2_VISUAL_CONTEXT=384
```

## ROCm build

Build the same ROCm target Immich already uses:

```bash
cd machine-learning
docker build --build-arg DEVICE=rocm -t immich-machine-learning:embeddinggemma2-rocm .
```

Example Compose service fragment:

```yaml
immich-machine-learning:
  image: immich-machine-learning:embeddinggemma2-rocm
  devices:
    - /dev/kfd:/dev/kfd
    - /dev/dri:/dev/dri
  group_add:
    - video
    - render
  environment:
    MACHINE_LEARNING_EMBEDDING_GEMMA2_ALIAS: ViT-B-16-SigLIP-256__webli
    MACHINE_LEARNING_EMBEDDING_GEMMA2_VISION_TOKENS: 280
  volumes:
    - model-cache:/cache
```

The first text+vision load downloads the fp32 ONNX text and vision artifacts into the normal ML cache. The normal Immich model TTL/cache behavior remains in effect.
