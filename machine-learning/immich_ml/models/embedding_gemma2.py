from __future__ import annotations

import hashlib
import json
import math
from functools import cache, cached_property
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, cast

import numpy as np
import onnx
from huggingface_hub import snapshot_download
from immich_model.runtime import RewritePlan
from numpy.typing import NDArray
from PIL import Image
from tokenizers import Tokenizer

from immich_ml.config import clean_name, log, settings
from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import clean_text, decode_pil, serialize_np_array
from immich_ml.schemas import (
    ModelFormat,
    ModelGraph,
    ModelSession,
    ModelSource,
    ModelTask,
    ModelType,
    Options,
    Shape,
    TextualOptions,
    VisualOptions,
)
from immich_ml.sessions.ort import GraphSpec, OrtGraph, _disabled_optimizers_default, _providers_default

_EMBEDDING_DIM = 768
_SUPPORTED_VISION_TOKENS = (70, 140, 280, 560, 1120)
_COMMON_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",
)
_VISION_FILES = (
    "onnx/vision_encoder.onnx",
    "onnx/vision_encoder.onnx_data",
)


def is_embedding_gemma2_alias(model_name: str) -> bool:
    alias = settings.embedding_gemma2_alias
    return alias is not None and clean_name(alias) == clean_name(model_name)


def _target_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    target_px = max_patches * patch_size**2
    factor = math.sqrt(target_px / (height * width))
    side_mult = pooling_kernel_size * patch_size
    target_height = math.floor((factor * height) / side_mult) * side_mult
    target_width = math.floor((factor * width) / side_mult) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "EmbeddingGemma 2 image resize produced 0x0; "
            f"image sides must allow a multiple of {side_mult} pixels"
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(math.floor(width / height) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(math.floor(height / width) * side_mult, max_side_length)

    return target_height, target_width


def _prepare_image(
    image: Image.Image,
    max_soft_tokens: int,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
    rescale_factor: float = 1 / 255,
) -> tuple[NDArray[np.float32], NDArray[np.int64], int]:
    if max_soft_tokens not in _SUPPORTED_VISION_TOKENS:
        raise ValueError(f"Unsupported vision token budget: {max_soft_tokens}")

    image = image.convert("RGB")
    max_patches = max_soft_tokens * pooling_kernel_size**2
    target_height, target_width = _target_size(
        image.height,
        image.width,
        patch_size,
        max_patches,
        pooling_kernel_size,
    )
    if image.height != target_height or image.width != target_width:
        image = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)

    pixels = np.asarray(image, dtype=np.float32) * rescale_factor
    patch_height = target_height // patch_size
    patch_width = target_width // patch_size
    patch_dim = patch_size * patch_size * 3
    patches = (
        pixels.reshape(patch_height, patch_size, patch_width, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, patch_dim)
    )

    padded = np.zeros((max_patches, patch_dim), dtype=np.float32)
    padded[: patches.shape[0]] = patches

    positions = np.full((max_patches, 2), -1, dtype=np.int64)
    columns, rows = np.meshgrid(
        np.arange(patch_width, dtype=np.int64),
        np.arange(patch_height, dtype=np.int64),
        indexing="xy",
    )
    real_positions = np.stack((columns, rows), axis=-1).reshape(-1, 2)
    positions[: real_positions.shape[0]] = real_positions

    num_soft_tokens = patches.shape[0] // pooling_kernel_size**2
    return padded[None], positions[None], num_soft_tokens


def _tokenize(
    tokenizer: Tokenizer,
    text: str,
    context_length: int,
    *,
    truncate: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    encoding = tokenizer.encode(text)
    ids = np.asarray(encoding.ids, dtype=np.int64)

    # Gemma tokenizers add a BOS token by default. The low-level tokenizers
    # JSON may or may not encode that policy in its post-processor, so make
    # the behavior explicit while avoiding a duplicate BOS if it is present.
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id is not None and (ids.size == 0 or int(ids[0]) != bos_id):
        ids = np.concatenate((np.asarray([bos_id], dtype=np.int64), ids))

    if ids.size > context_length:
        if not truncate:
            raise ValueError(
                f"EmbeddingGemma 2 input requires {ids.size} tokens, "
                f"but context is configured for {context_length}"
            )
        ids = ids[:context_length]

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    input_ids = np.full((1, context_length), pad_id, dtype=np.int64)
    attention_mask = np.zeros((1, context_length), dtype=np.int64)
    input_ids[0, : ids.size] = ids
    attention_mask[0, : ids.size] = 1
    return input_ids, attention_mask


def _run_output(graph: ModelGraph, output_name: str, feed: dict[str, Any]) -> NDArray[Any]:
    outputs = {node.name for node in graph.get_outputs()}
    if output_name not in outputs:
        raise RuntimeError(f"EmbeddingGemma 2 graph has no output named '{output_name}'")

    inputs = {node.name for node in graph.get_inputs()}
    missing = inputs - feed.keys()
    if missing:
        raise RuntimeError(f"EmbeddingGemma 2 graph is missing input '{next(iter(missing))}'")

    filtered = {name: feed[name] for name in inputs}
    return graph.run([output_name], filtered)[0]


def _normalize_embedding(output: NDArray[Any]) -> NDArray[np.float32]:
    embedding = np.asarray(output, dtype=np.float32)
    if embedding.ndim == 0:
        raise RuntimeError("EmbeddingGemma 2 returned a scalar instead of an embedding")
    embedding = embedding.reshape(-1, embedding.shape[-1])[0]
    if embedding.shape[0] != _EMBEDDING_DIM:
        raise RuntimeError(
            f"EmbeddingGemma 2 returned {embedding.shape[0]} dimensions; Immich alias requires {_EMBEDDING_DIM}"
        )
    if not np.all(np.isfinite(embedding)):
        raise RuntimeError("EmbeddingGemma 2 returned non-finite embedding values")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("EmbeddingGemma 2 returned a zero or invalid embedding norm")
    normalized: NDArray[np.float32] = np.asarray(embedding / norm, dtype=np.float32)
    return normalized


@cache
def _model_input_shapes(model_path: Path) -> dict[str, tuple[int | str | None, ...]]:
    model = onnx.load(model_path.as_posix(), load_external_data=False)
    shapes: dict[str, tuple[int | str | None, ...]] = {}
    for node in model.graph.input:
        dims: list[int | str | None] = []
        for dim in node.type.tensor_type.shape.dim:
            if dim.dim_param:
                dims.append(dim.dim_param)
            elif dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        shapes[node.name] = tuple(dims)
    return shapes


def _shape_overrides(model_path: Path, feed: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    declared_shapes = _model_input_shapes(model_path)
    overrides: dict[str, int] = {}
    for input_name, declared in declared_shapes.items():
        if input_name not in feed:
            continue
        actual = np.asarray(feed[input_name]).shape
        if len(actual) != len(declared):
            raise RuntimeError(
                f"EmbeddingGemma 2 input '{input_name}' has rank {len(actual)}, expected {len(declared)}"
            )
        for axis, (declared_dim, actual_dim) in enumerate(zip(declared, actual, strict=True)):
            size = int(actual_dim)
            if isinstance(declared_dim, int):
                if declared_dim != size:
                    raise RuntimeError(
                        f"EmbeddingGemma 2 input '{input_name}' axis {axis} is {size}, expected {declared_dim}"
                    )
            elif isinstance(declared_dim, str):
                if declared_dim in overrides and overrides[declared_dim] != size:
                    raise RuntimeError(
                        f"EmbeddingGemma 2 symbolic dimension '{declared_dim}' has conflicting sizes "
                        f"{overrides[declared_dim]} and {size}"
                    )
                overrides[declared_dim] = size
            else:
                raise RuntimeError(
                    f"EmbeddingGemma 2 input '{input_name}' axis {axis} is dynamic but has no symbolic name; "
                    "MIGraphX requires it to be pinned"
                )
    return tuple(sorted(overrides.items()))


class EmbeddingGemma2GraphSpec(GraphSpec):
    """Prepare the external graph without Immich-specific narrowing or rewrites."""

    @cached_property
    def half(self) -> bool:
        return False

    @property
    def plan(self) -> RewritePlan:
        return RewritePlan((), "embeddinggemma2-none")


class PinnedOrtSession:
    """Build one concrete-shape ORT graph per EmbeddingGemma request shape."""

    def __init__(self, model_path: Path | str, cache_marker: int, threads: int = 2) -> None:
        self.model_path = Path(model_path)
        self.cache_marker = cache_marker
        self.providers = _providers_default()
        self.disabled_optimizers = _disabled_optimizers_default(self.providers)
        self.threads = threads
        self.graphs: dict[tuple[tuple[str, int], ...], OrtGraph] = {}
        self.lock = Lock()
        self.shapes = (Shape(batch=1),)
        self.batches = (1,)

    def for_feed(self, feed: dict[str, Any]) -> ModelGraph:
        overrides = _shape_overrides(self.model_path, feed)
        if (graph := self.graphs.get(overrides)) is not None:
            return graph

        with self.lock:
            if (graph := self.graphs.get(overrides)) is not None:
                return graph

            digest = hashlib.sha256(repr(overrides).encode()).digest()
            shape_marker = int.from_bytes(digest[:8], "big")
            pins = {
                "embedding_gemma2_component": self.cache_marker,
                "embedding_gemma2_shape": shape_marker,
            }
            log.info(f"Specializing {self.model_path} for {self.providers[0]} {dict(overrides)}")
            spec = EmbeddingGemma2GraphSpec(
                self.model_path,
                pins,
                list(overrides),
                self.providers,
                self.disabled_optimizers,
                self.threads,
            )
            graph = self.graphs[overrides] = OrtGraph(spec)
            return graph

    def for_shape(self, shape: Shape) -> ModelGraph:
        if len(self.graphs) == 1:
            return next(iter(self.graphs.values()))
        raise RuntimeError("EmbeddingGemma 2 needs the concrete input feed before selecting a MIGraphX graph")

    def warm(self) -> None:
        # A valid warm-up needs the real multimodal feed so every symbolic dimension can be specialized.
        pass


class BaseEmbeddingGemma2Encoder[O: Options](InferenceModel[O]):
    sources = (ModelSource.OPENCLIP, ModelSource.MCLIP)
    required_files: ClassVar[tuple[str, ...]] = _COMMON_FILES
    threads = 4

    @property
    def _model_format_default(self) -> ModelFormat:
        return ModelFormat.ONNX

    @property
    def _cache_dir_default(self) -> Path:
        return settings.cache_folder / ModelTask.SEARCH.value / "embeddinggemma-2" / settings.embedding_gemma2_revision

    @property
    def model_dir(self) -> Path:
        return self.cache_dir / "onnx"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.onnx"

    @property
    def cached(self) -> bool:
        return all((self.cache_dir / path).is_file() for path in self.required_files)

    def download(self) -> None:
        if self.cached:
            return
        log.info(f"Downloading EmbeddingGemma 2 from '{settings.embedding_gemma2_repo}' to {self.cache_dir}")
        snapshot_download(
            settings.embedding_gemma2_repo,
            revision=settings.embedding_gemma2_revision,
            cache_dir=self.cache_dir,
            local_dir=self.cache_dir,
            allow_patterns=list(self.required_files),
        )
        if not self.cached:
            raise FileNotFoundError(f"EmbeddingGemma 2 files are incomplete in {self.cache_dir}")

    @cached_property
    def model_cfg(self) -> dict[str, Any]:
        with (self.cache_dir / "config.json").open(encoding="utf-8") as file:
            return cast(dict[str, Any], json.load(file))

    @cached_property
    def processor_cfg(self) -> dict[str, Any]:
        with (self.cache_dir / "processor_config.json").open(encoding="utf-8") as file:
            return cast(dict[str, Any], json.load(file))

    @cached_property
    def tokenizer(self) -> Tokenizer:
        return Tokenizer.from_file((self.cache_dir / "tokenizer.json").as_posix())

    @property
    def hidden_size(self) -> int:
        return int(self.model_cfg["text_config"]["hidden_size"])

    @property
    def context_length(self) -> int:
        raise NotImplementedError

    def _load(self) -> ModelSession:
        return PinnedOrtSession(self.model_path, cache_marker=1, threads=self.threads)

    def build(self) -> None:
        # Graph creation is deferred until the first real feed so MIGraphX sees only concrete dimensions.
        self.load()

    def _main_embedding(
        self,
        input_ids: NDArray[np.int64],
        attention_mask: NDArray[np.int64],
        *,
        image_features: NDArray[np.float32] | None = None,
    ) -> str:
        empty = np.empty((0, self.hidden_size), dtype=np.float32)
        feed: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_features": empty if image_features is None else image_features.astype(np.float32, copy=False),
            "video_features": empty,
            "audio_features": empty,
        }
        session = cast(PinnedOrtSession, self.session)
        graph = session.for_feed(feed)
        embedding = _normalize_embedding(_run_output(graph, "sentence_embedding", feed))
        return serialize_np_array(embedding)


class EmbeddingGemma2TextualEncoder(BaseEmbeddingGemma2Encoder[TextualOptions]):
    depends = []
    identity = (ModelType.TEXTUAL, ModelTask.SEARCH)

    @property
    def context_length(self) -> int:
        return settings.embedding_gemma2_text_context

    def _predict(self, inputs: str, options: TextualOptions) -> str:
        query = f"task: search result | query: {clean_text(inputs)}"
        input_ids, attention_mask = _tokenize(self.tokenizer, query, self.context_length, truncate=True)
        return self._main_embedding(input_ids, attention_mask)


class EmbeddingGemma2VisualEncoder(BaseEmbeddingGemma2Encoder[VisualOptions]):
    depends = []
    identity = (ModelType.VISUAL, ModelTask.SEARCH)
    required_files = (*_COMMON_FILES, *_VISION_FILES)
    threads = 2

    @property
    def vision_tokens(self) -> int:
        return settings.embedding_gemma2_vision_tokens

    @property
    def context_length(self) -> int:
        return max(settings.embedding_gemma2_visual_context, self.vision_tokens + 32)

    @property
    def vision_model_path(self) -> Path:
        return self.model_dir / "vision_encoder.onnx"

    def _load(self) -> ModelSession:
        self.vision_session = PinnedOrtSession(self.vision_model_path, cache_marker=2, threads=self.threads)
        return super()._load()

    def unload(self) -> None:
        if hasattr(self, "vision_session"):
            del self.vision_session
        super().unload()

    def _predict(self, inputs: Image.Image | bytes, options: VisualOptions) -> str:
        image = decode_pil(inputs)
        image_cfg = self.processor_cfg.get("image_processor", {})
        vision_cfg = self.model_cfg["vision_config"]
        patch_size = int(image_cfg.get("patch_size", vision_cfg.get("patch_size", 16)))
        pooling = int(image_cfg.get("pooling_kernel_size", vision_cfg.get("pooling_kernel_size", 3)))
        rescale_factor = float(image_cfg.get("rescale_factor", 1 / 255))

        pixel_values, position_ids, num_soft_tokens = _prepare_image(
            image,
            self.vision_tokens,
            patch_size=patch_size,
            pooling_kernel_size=pooling,
            rescale_factor=rescale_factor,
        )

        vision_feed: dict[str, Any] = {
            "pixel_values": pixel_values,
            "pixel_position_ids": position_ids,
        }
        vision_graph = self.vision_session.for_feed(vision_feed)
        image_features = np.asarray(_run_output(vision_graph, "image_features", vision_feed), dtype=np.float32)
        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features[0]
        image_features = image_features.reshape(-1, image_features.shape[-1])
        if image_features.shape[0] < num_soft_tokens:
            raise RuntimeError(
                f"EmbeddingGemma 2 vision encoder returned {image_features.shape[0]} features "
                f"for {num_soft_tokens} image tokens"
            )
        image_features = image_features[:num_soft_tokens]

        prompt = "<|image>" + "<|image|>" * num_soft_tokens + "<image|>"
        input_ids, attention_mask = _tokenize(self.tokenizer, prompt, self.context_length, truncate=False)
        image_token_id = int(self.model_cfg["image_token_id"])
        if int(np.count_nonzero(input_ids == image_token_id)) != num_soft_tokens:
            raise RuntimeError("EmbeddingGemma 2 tokenizer produced the wrong number of image placeholders")

        return self._main_embedding(input_ids, attention_mask, image_features=image_features)
