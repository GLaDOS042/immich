import numpy as np
import pytest
from PIL import Image
from pytest_mock import MockerFixture

from immich_ml.config import settings
from immich_ml.models.clip.textual import OpenClipTextualEncoder
from immich_ml.models.clip.visual import OpenClipVisualEncoder
from immich_ml.models.embedding_gemma2 import (
    DynamicOrtSession,
    EmbeddingGemma2TextualEncoder,
    EmbeddingGemma2VisualEncoder,
    _normalize_embedding,
    _prepare_image,
    _tokenize,
    is_embedding_gemma2_alias,
)
from immich_ml.pipeline import Clip, PipelineRequest, Slot
from immich_ml.schemas import TextualOptions, VisualOptions

ALIAS = "ViT-B-16-SigLIP-256__webli"


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


def test_multimodal_components_use_separate_ort_cache_markers(mocker: MockerFixture) -> None:
    graph = mocker.patch("immich_ml.models.embedding_gemma2.OrtGraph")
    mocker.patch("immich_ml.models.embedding_gemma2._providers_default", return_value=["CPUExecutionProvider"])
    mocker.patch("immich_ml.models.embedding_gemma2._disabled_optimizers_default", return_value=[])

    DynamicOrtSession("/cache/onnx/model.onnx", cache_marker=1)
    DynamicOrtSession("/cache/onnx/vision_encoder.onnx", cache_marker=2)

    text_spec = graph.call_args_list[0].args[0]
    vision_spec = graph.call_args_list[1].args[0]
    assert text_spec.pins == {"embedding_gemma2_component": 1}
    assert vision_spec.pins == {"embedding_gemma2_component": 2}
    assert text_spec.directory != vision_spec.directory


def test_embedding_gemma_disables_generic_fp16_and_rewrites_on_migraphx(mocker: MockerFixture) -> None:
    graph = mocker.patch("immich_ml.models.embedding_gemma2.OrtGraph")
    mocker.patch(
        "immich_ml.models.embedding_gemma2._providers_default",
        return_value=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
    )
    mocker.patch("immich_ml.models.embedding_gemma2._disabled_optimizers_default", return_value=[])
    mocker.patch.object(settings, "model_revision", "v2")

    DynamicOrtSession("/cache/onnx/model.onnx", cache_marker=1)

    spec = graph.call_args.args[0]
    assert spec.provider == "MIGraphXExecutionProvider"
    assert spec.half is False
    assert spec.plan.rewrites == ()
