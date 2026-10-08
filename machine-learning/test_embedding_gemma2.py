from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image
from pytest_mock import MockerFixture

from immich_ml.config import settings
from immich_ml.models.clip.textual import OpenClipTextualEncoder
from immich_ml.models.clip.visual import OpenClipVisualEncoder
from immich_ml.models.embedding_gemma2 import (
    EmbeddingGemma2TextualEncoder,
    EmbeddingGemma2VisualEncoder,
    PinnedOrtSession,
    _normalize_embedding,
    _prepare_image,
    _shape_overrides,
    _tokenize,
    is_embedding_gemma2_alias,
)
from immich_ml.pipeline import Clip, PipelineRequest, Slot
from immich_ml.schemas import TextualOptions, VisualOptions

ALIAS = "ViT-B-16-SigLIP-256__webli"


def _write_shape_model(path: Path) -> None:
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["image_tokens", 512]),
        helper.make_tensor_value_info("video_features", TensorProto.FLOAT, ["video_tokens", 512]),
        helper.make_tensor_value_info("audio_features", TensorProto.FLOAT, ["audio_tokens", 512]),
    ]
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    value = helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node("Constant", inputs=[], outputs=["out"], value=value)
    graph = helper.make_graph([node], "embeddinggemma-shape-test", inputs, [output])
    onnx.save(helper.make_model(graph), path)


def _text_feed(image_tokens: int = 0) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
        "image_features": np.empty((image_tokens, 512), dtype=np.float32),
        "video_features": np.empty((0, 512), dtype=np.float32),
        "audio_features": np.empty((0, 512), dtype=np.float32),
    }


def test_embedding_gemma_alias_is_opt_in(mocker: MockerFixture) -> None:
    mocker.patch.object(settings, "embedding_gemma2_alias", None)
    assert not is_embedding_gemma2_alias(ALIAS)

    mocker.patch.object(settings, "embedding_gemma2_alias", ALIAS)
    assert is_embedding_gemma2_alias(ALIAS)


def test_embedding_gemma_alias_routes_both_clip_encoders(mocker: MockerFixture) -> None:
    mocker.patch.object(settings, "embedding_gemma2_alias", ALIAS)
    request = PipelineRequest(
        clip=Clip(
            visual=Slot(ALIAS, VisualOptions()),
            textual=Slot(ALIAS, TextualOptions()),
        )
    )

    entries = list(request.entries())

    assert entries[0].model is EmbeddingGemma2VisualEncoder
    assert entries[1].model is EmbeddingGemma2TextualEncoder


def test_other_models_keep_the_upstream_clip_encoders(mocker: MockerFixture) -> None:
    mocker.patch.object(settings, "embedding_gemma2_alias", ALIAS)
    model = "ViT-B-32__openai"
    request = PipelineRequest(
        clip=Clip(
            visual=Slot(model, VisualOptions()),
            textual=Slot(model, TextualOptions()),
        )
    )

    entries = list(request.entries())

    assert entries[0].model is OpenClipVisualEncoder
    assert entries[1].model is OpenClipTextualEncoder


def test_image_preprocessing_matches_gemma4_soft_token_layout() -> None:
    image = Image.new("RGB", (640, 480))

    pixels, positions, soft_tokens = _prepare_image(image, 280)

    assert pixels.shape == (1, 2520, 768)
    assert positions.shape == (1, 2520, 2)
    assert soft_tokens == 266


def test_square_image_uses_256_soft_tokens_at_default_budget() -> None:
    image = Image.new("RGB", (480, 480))

    _, _, soft_tokens = _prepare_image(image, 280)

    assert soft_tokens == 256


def test_tokenize_adds_single_gemma_bos_and_padding(mocker: MockerFixture) -> None:
    tokenizer = mocker.Mock()
    tokenizer.encode.return_value.ids = [101, 102]
    tokenizer.token_to_id.side_effect = lambda token: {"<bos>": 2, "<pad>": 0}.get(token)

    input_ids, attention_mask = _tokenize(tokenizer, "query", 5, truncate=True)

    assert input_ids.tolist() == [[2, 101, 102, 0, 0]]
    assert attention_mask.tolist() == [[1, 1, 1, 0, 0]]


def test_tokenize_does_not_duplicate_existing_bos(mocker: MockerFixture) -> None:
    tokenizer = mocker.Mock()
    tokenizer.encode.return_value.ids = [2, 101]
    tokenizer.token_to_id.side_effect = lambda token: {"<bos>": 2, "<pad>": 0}.get(token)

    input_ids, _ = _tokenize(tokenizer, "query", 4, truncate=True)

    assert input_ids.tolist() == [[2, 101, 0, 0]]


def test_normalize_embedding_enforces_768_dimensions_and_unit_norm() -> None:
    output = np.arange(1, 769, dtype=np.float32)[None, :]

    embedding = _normalize_embedding(output)

    assert embedding.shape == (768,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_normalize_embedding_rejects_wrong_dimension() -> None:
    with pytest.raises(RuntimeError, match="768"):
        _normalize_embedding(np.ones((1, 512), dtype=np.float32))


def test_normalize_embedding_rejects_non_finite_or_zero_vectors() -> None:
    non_finite = np.ones((1, 768), dtype=np.float32)
    non_finite[0, 1] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _normalize_embedding(non_finite)

    with pytest.raises(RuntimeError, match="zero or invalid"):
        _normalize_embedding(np.zeros((1, 768), dtype=np.float32))


def test_shape_overrides_pin_zero_length_media_dimensions(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    _write_shape_model(model_path)

    overrides = dict(_shape_overrides(model_path, _text_feed()))

    assert overrides == {
        "audio_tokens": 0,
        "batch": 1,
        "image_tokens": 0,
        "sequence": 128,
        "video_tokens": 0,
    }


def test_shape_overrides_change_with_image_token_count(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    _write_shape_model(model_path)

    text = dict(_shape_overrides(model_path, _text_feed()))
    image = dict(_shape_overrides(model_path, _text_feed(image_tokens=266)))

    assert text["image_tokens"] == 0
    assert image["image_tokens"] == 266


def test_multimodal_shapes_get_distinct_migraphx_cache_keys(tmp_path: Path, mocker: MockerFixture) -> None:
    model_path = tmp_path / "model.onnx"
    _write_shape_model(model_path)
    graph = mocker.patch("immich_ml.models.embedding_gemma2.OrtGraph")
    mocker.patch(
        "immich_ml.models.embedding_gemma2._providers_default",
        return_value=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
    )
    mocker.patch("immich_ml.models.embedding_gemma2._disabled_optimizers_default", return_value=[])

    session = PinnedOrtSession(model_path, cache_marker=1)
    session.for_feed(_text_feed())
    session.for_feed(_text_feed(image_tokens=266))

    text_spec = graph.call_args_list[0].args[0]
    image_spec = graph.call_args_list[1].args[0]
    assert dict(text_spec.overrides)["image_tokens"] == 0
    assert dict(image_spec.overrides)["image_tokens"] == 266
    assert text_spec.directory != image_spec.directory


def test_multimodal_components_use_separate_ort_cache_markers(tmp_path: Path, mocker: MockerFixture) -> None:
    model_path = tmp_path / "model.onnx"
    _write_shape_model(model_path)
    graph = mocker.patch("immich_ml.models.embedding_gemma2.OrtGraph")
    mocker.patch("immich_ml.models.embedding_gemma2._providers_default", return_value=["CPUExecutionProvider"])
    mocker.patch("immich_ml.models.embedding_gemma2._disabled_optimizers_default", return_value=[])

    text = PinnedOrtSession(model_path, cache_marker=1)
    vision = PinnedOrtSession(model_path, cache_marker=2)
    text.for_feed(_text_feed())
    vision.for_feed(_text_feed())

    text_spec = graph.call_args_list[0].args[0]
    vision_spec = graph.call_args_list[1].args[0]
    assert text_spec.pins["embedding_gemma2_component"] == 1
    assert vision_spec.pins["embedding_gemma2_component"] == 2
    assert text_spec.directory != vision_spec.directory


def test_embedding_gemma_disables_generic_fp16_and_rewrites_on_migraphx(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    model_path = tmp_path / "model.onnx"
    _write_shape_model(model_path)
    graph = mocker.patch("immich_ml.models.embedding_gemma2.OrtGraph")
    mocker.patch(
        "immich_ml.models.embedding_gemma2._providers_default",
        return_value=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
    )
    mocker.patch("immich_ml.models.embedding_gemma2._disabled_optimizers_default", return_value=[])
    mocker.patch.object(settings, "model_revision", "v2")

    session = PinnedOrtSession(model_path, cache_marker=1)
    session.for_feed(_text_feed())

    spec = graph.call_args.args[0]
    assert spec.provider == "MIGraphXExecutionProvider"
    assert spec.half is False
    assert spec.plan.rewrites == ()
