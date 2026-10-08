from PIL import Image
from pytest_mock import MockerFixture

from immich_ml.config import settings
from immich_ml.models.clip.textual import OpenClipTextualEncoder
from immich_ml.models.clip.visual import OpenClipVisualEncoder
from immich_ml.models.embedding_gemma2 import (
    EmbeddingGemma2TextualEncoder,
    EmbeddingGemma2VisualEncoder,
    _prepare_image,
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
