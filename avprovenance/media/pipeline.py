"""Route a normalized plan to the matching media implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor
from .audio_ops import render_audio
from .errors import MediaOperationFailure
from .inspector import inspect_media
from .video_ops import render_video
from .views import DecodedMediaView


@dataclass(frozen=True)
class RenderOutcome:
    data: bytes
    descriptor: MediaDescriptor
    operation_types: list[str]


def apply_media_operations(
    primary_view: DecodedMediaView,
    operations: list[dict[str, Any]],
    secondary_view: DecodedMediaView | None,
) -> RenderOutcome:
    primary = primary_view.descriptor
    if secondary_view is not None and secondary_view.kind != primary.kind:
        raise MediaOperationFailure(
            "MEDIA_KIND_MISMATCH",
            "Primary and secondary media must share a media kind.",
        )

    if primary.kind == "audio":
        output = render_audio(primary_view, operations, secondary_view)
    elif primary.kind == "video":
        output = render_video(primary_view, operations, secondary_view)
    else:
        raise UnsupportedMedia("The primary media kind has no renderer.")

    descriptor = inspect_media(output)
    if descriptor.kind != primary.kind:
        raise UnsupportedMedia("The renderer changed the media kind unexpectedly.")
    return RenderOutcome(
        data=output,
        descriptor=descriptor,
        operation_types=[str(operation["type"]) for operation in operations],
    )
