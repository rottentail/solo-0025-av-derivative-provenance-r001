"""Read-only decoded media views and their shared, reference-counted registry.

Ownership model
---------------

A :class:`DecodedMediaView` is an immutable, read-only interpretation of one
content-addressed blob (raw bytes, probe descriptor, and the decoded clip).
Views carry no per-request state and are safe to share across threads:

* Audio accessors return a fresh mutable copy of every channel, so one request
  trimming a clip can never alter the samples another request is gaining or
  mixing.
* Video clips are composed only of frozen frames and immutable ``bytes``
  planes; every video operation builds new clip objects instead of mutating
  them, so sharing one view cannot couple concurrent requests either.

A :class:`MediaViewLease` is the single authority that can end one usage of a
view. Media operations deliberately receive views, never leases, so they have
no way to release a view that another request is still replaying. Leases are
context managers and release idempotently; a ``__del__`` backstop guarantees
that a lease abandoned by a failed or disconnected request cannot pin memory
forever.

The :class:`MediaViewRegistry` owns the shared, process-local entries. It only
guards its bookkeeping (the entry map and reference counts); loading and
decoding happen outside its lock, so requests sharing one asset never run
media work serially. An entry leaves the registry only when its last lease is
returned, while any lease still in flight keeps its own view object alive.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from ..errors import PersistenceFailure
from ..models import MediaDescriptor
from .errors import MediaOperationFailure
from .inspector import inspect_media
from .wav import AudioClip, decode_wav
from .y4m import Y4mClip, decode_y4m


@dataclass(frozen=True)
class DecodedMediaView:
    """An immutable decoded interpretation of one stored media blob."""

    content_key: str
    raw: bytes
    descriptor: MediaDescriptor
    _audio_clip: AudioClip | None
    _video_clip: Y4mClip | None

    @property
    def kind(self) -> str:
        return self.descriptor.kind

    def audio_clip(self) -> AudioClip:
        """Return an independent, mutable copy of the decoded PCM channels.

        Each call allocates new channel lists so callers may transform the
        clip without affecting any other holder of the shared view.
        """
        if self._audio_clip is None:
            raise MediaOperationFailure(
                "MEDIA_VIEW_KIND_MISMATCH",
                "The shared media view does not decode to an audio clip.",
            )
        return AudioClip(
            sample_rate=self._audio_clip.sample_rate,
            channels=tuple(
                list(channel) for channel in self._audio_clip.channels
            ),
        )

    def video_clip(self) -> Y4mClip:
        """Return the shared decoded Y4M clip.

        The clip and every frame/plane it references are immutable, and the
        video operations replace rather than mutate them, so sharing it across
        requests cannot couple their results.
        """
        if self._video_clip is None:
            raise MediaOperationFailure(
                "MEDIA_VIEW_KIND_MISMATCH",
                "The shared media view does not decode to a video clip.",
            )
        return self._video_clip


def build_decoded_view(content_key: str, data: bytes) -> DecodedMediaView:
    """Probe and decode one blob exactly once into a read-only view."""
    descriptor = inspect_media(data)
    if descriptor.kind == "audio":
        return DecodedMediaView(
            content_key=content_key,
            raw=data,
            descriptor=descriptor,
            _audio_clip=decode_wav(data),
            _video_clip=None,
        )
    return DecodedMediaView(
        content_key=content_key,
        raw=data,
        descriptor=descriptor,
        _audio_clip=None,
        _video_clip=decode_y4m(data),
    )


@dataclass
class _RegistryEntry:
    view: DecodedMediaView
    reference_count: int


class MediaViewLease:
    """A single request's exclusive authority to end one media usage."""

    __slots__ = ("_registry", "_content_key", "_view", "_released")

    def __init__(
        self,
        registry: "MediaViewRegistry",
        content_key: str,
        view: DecodedMediaView,
    ) -> None:
        self._registry = registry
        self._content_key = content_key
        self._view = view
        self._released = False

    @property
    def content_key(self) -> str:
        return self._content_key

    @property
    def view(self) -> DecodedMediaView:
        if self._released:
            raise PersistenceFailure(
                "The media view lease has already been released."
            )
        return self._view

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._registry._return_lease(self._content_key)

    def __enter__(self) -> "MediaViewLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __del__(self) -> None:
        # Backstop: a lease dropped by a failed or disconnected request still
        # returns its reference so the registry cannot leak the view forever.
        try:
            self.release()
        except Exception:
            pass


class MediaViewRegistry:
    """Process-local cache of read-only views with explicit lease lifetimes."""

    def __init__(self, loader: Callable[[str], bytes]) -> None:
        self._loader = loader
        self._entries: dict[str, _RegistryEntry] = {}
        self._lock = threading.Lock()
        self._closed = False

    def _acquire(
        self,
        content_key: str,
        builder: Callable[[], DecodedMediaView],
    ) -> MediaViewLease:
        with self._lock:
            if self._closed:
                raise PersistenceFailure(
                    "The media view registry has been closed."
                )
            entry = self._entries.get(content_key)
            if entry is not None:
                entry.reference_count += 1
                return MediaViewLease(self, content_key, entry.view)

        # Loading from disk and decoding are intentionally outside the
        # registry lock so concurrent requests never serialize media work.
        fresh_view = builder()

        with self._lock:
            if self._closed:
                raise PersistenceFailure(
                    "The media view registry has been closed."
                )
            entry = self._entries.get(content_key)
            if entry is None:
                self._entries[content_key] = _RegistryEntry(
                    view=fresh_view,
                    reference_count=1,
                )
                view = fresh_view
            else:
                entry.reference_count += 1
                view = entry.view
            return MediaViewLease(self, content_key, view)

    def acquire(self, content_key: str) -> MediaViewLease:
        """Lease the cached view for a content key, loading it on a miss."""

        def build_from_storage() -> DecodedMediaView:
            return build_decoded_view(content_key, self._loader(content_key))

        return self._acquire(content_key, build_from_storage)

    def register_bytes(self, content_key: str, data: bytes) -> MediaViewLease:
        """Lease a view for bytes already in hand (the ingestion path)."""
        return self._acquire(
            content_key,
            lambda: build_decoded_view(content_key, data),
        )

    def _return_lease(self, content_key: str) -> None:
        with self._lock:
            entry = self._entries.get(content_key)
            if entry is None:
                return
            entry.reference_count -= 1
            if entry.reference_count <= 0:
                # In-flight leases still reference their view object directly,
                # so removing the registry entry can never free data they are
                # actively reading.
                del self._entries[content_key]

    def active_view_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def reference_count(self, content_key: str) -> int:
        with self._lock:
            entry = self._entries.get(content_key)
            return entry.reference_count if entry is not None else 0

    def close(self) -> None:
        """Drop every shared entry. Outstanding leases finish on their own."""
        with self._lock:
            self._closed = True
            self._entries.clear()
