"""Camera entities for RTSP streams and intercom video."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

import aiohttp
from homeassistant.components.camera import Camera, CameraEntityFeature
from homeassistant.components.camera.webrtc import (
    WebRTCAnswer,
    WebRTCCandidate,
    WebRTCError,
    WebRTCSendMessage,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from webrtc_models import RTCIceCandidateInit

if TYPE_CHECKING:
    from collections.abc import Callable

try:  # HA 2026.11+: lets the built-in camera dialog offer a microphone
    from homeassistant.components.camera import (  # type: ignore[attr-defined,unused-ignore]
        CameraEntityCapabilities,
    )
except ImportError:  # pragma: no cover - older Home Assistant
    CameraEntityCapabilities = None  # type: ignore[assignment,misc,unused-ignore]

from .camera_utils import get_rtsp_url
from .const import DOMAIN, MANUFACTURER
from .coordinator import ComelitLocalConfigEntry, ComelitLocalCoordinator, go2rtc_endpoint
from .entity import ComelitEntity
from .models import Camera as CameraModel
from .models import PushEvent
from .placeholder import PLACEHOLDER_JPEG

_LOGGER = logging.getLogger(__name__)

# Grace period before hanging up a viewer-started call once the last viewer
# leaves — a page refresh or reopening the dialog reuses the running call.
VIEWER_HANGUP_DELAY = 5.0

# A call started for a stream_source() consumer (HomeKit's ffmpeg, HA's HLS
# worker for Google) is hung up once no RTSP client has been connected for
# this long. The first client gets STREAM_CONNECT_GRACE to show up.
STREAM_IDLE_HANGUP_DELAY = 10.0
STREAM_CONNECT_GRACE = 20.0
STREAM_IDLE_POLL = 1.0

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ComelitLocalConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up camera entities from device config."""
    coordinator = entry.runtime_data
    config = coordinator.device_config
    if not config:
        return

    entities: list[Camera] = [ComelitCamera(coordinator, cam, entry.entry_id) for cam in config.cameras if cam.rtsp_url]

    # Add intercom camera if there are doors (i.e. the device has an intercom)
    if config.doors:
        entities.append(ComelitIntercomCamera(coordinator, entry.entry_id))

    async_add_entities(entities)


class ComelitCamera(Camera):
    """Camera entity that provides an RTSP stream."""

    _attr_has_entity_name = True
    _attr_supported_features = CameraEntityFeature.STREAM

    def __init__(
        self,
        coordinator: ComelitLocalCoordinator,
        camera: CameraModel,
        entry_id: str,
    ) -> None:
        """Initialize the camera entity."""
        super().__init__()
        self._coordinator = coordinator
        self._camera = camera
        self._entry_id = entry_id
        self._attr_unique_id = f"{entry_id}_camera_{camera.id}"
        self._attr_name = camera.name

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info linking this camera to its own device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry_id}_camera_{self._camera.id}")},
            manufacturer=MANUFACTURER,
            name=self._camera.name,
            via_device=(DOMAIN, self._entry_id),
        )

    async def stream_source(self) -> str | None:
        """Return the RTSP stream URL for HA's stream integration."""
        url = get_rtsp_url(self._camera, self._coordinator.host)
        return url or None


class ComelitIntercomCamera(ComelitEntity, Camera):
    """Camera entity for live intercom video and audio via go2rtc/WebRTC.

    Serves a persistent local RTSP stream (started at integration load) so
    go2rtc can connect immediately. When a video call session is active,
    H.264 video and G.711 audio flow through to the browser via WebRTC.

    `CameraEntityFeature.STREAM` is required for go2rtc's WebRTC path to
    work — without it, HA falls back to MJPEG polling via
    `async_camera_image()` at ~2 fps, wasting the 25 fps we get from the
    device.  The 24 s first-call HLS warmup that the stream worker would
    otherwise cause is avoided by gating `stream_source()` — it returns
    None until a video session is active, so the stream worker cannot
    connect at HA boot and freeze its codec context with pix_fmt=-1.
    """

    _attr_translation_key = "intercom_camera"
    _attr_supported_features = CameraEntityFeature.STREAM

    def __init__(
        self,
        coordinator: ComelitLocalCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the intercom camera entity."""
        super().__init__(coordinator, entry_id)
        Camera.__init__(self)
        self._attr_unique_id = f"{entry_id}_intercom_camera"
        # Talk-back goes browser mic → WebRTC → go2rtc → our RTSP ONVIF
        # backchannel → device RTPC.  Declaring it lets HA 2026.11's own
        # more-info dialog show a mic (WebRTC only, behind the Labs
        # "two-way audio" preview).  The panel only plays it once the call
        # is answered (Answer doorbell button / an answered ring).
        if CameraEntityCapabilities is not None:
            self._attr_camera_entity_capabilities = CameraEntityCapabilities(two_way_audio=True)
        self._remove_push_cb: Callable[[], None] | None = None
        self._remove_stop_video_cb: Callable[[], None] | None = None
        self._remove_state_cb: Callable[[], None] | None = None
        # Active go2rtc WS signaling sessions: session_id -> (ws, listen task)
        self._webrtc_sessions: dict[str, tuple[aiohttp.ClientWebSocketResponse, asyncio.Task[None]]] = {}
        # Client candidates that trickle in while the intercom call is still
        # starting (before the go2rtc session exists): session_id -> candidates
        self._pending_candidates: dict[str, list[str]] = {}
        # True while the running call was started by a WebRTC viewer (not a
        # ring or the Start button), so closing the last viewer may end it.
        self._viewer_started_video = False
        self._hangup_task: asyncio.Task[None] | None = None
        self._stream_hangup_task: asyncio.Task[None] | None = None

    @property
    def is_streaming(self) -> bool:
        """Reflect whether a video session is currently active.

        HA maps this to the entity state ("streaming" vs "idle"), which the
        Lovelace card uses to decide between the live picture-entity and
        the idle thumbnail.  Without a truthful is_streaming, picture-entity
        locks onto the transport it picked at first stream_source() call —
        MJPEG if the session wasn't ready yet — and never upgrades.
        """
        return self.coordinator.video_session is not None

    async def stream_source(self) -> str | None:
        """Return the RTSP URL, starting the intercom call if none is running.

        HomeKit (ffmpeg) and Google Assistant (HA's HLS worker) pull video
        through this method instead of the WebRTC offer path, and the
        intercom only streams during a call — so, like a WebRTC viewer, a
        pull with no session starts one.  The URL is returned only once the
        session is ready: a consumer connecting during the 2-3 s CTPP
        handshake would probe a video-less stream and back off ~10 s.

        This camera has a native WebRTC implementation, so HA's provider
        discovery never calls stream_source(); only real stream consumers
        do.  A call started here is hung up by _hangup_when_stream_idle.
        """
        if self.coordinator.video_session is not None:
            return self.coordinator.rtsp_url
        try:
            started = await self.coordinator.async_ensure_video()
        except Exception as err:
            _LOGGER.warning("Could not start intercom video for stream consumer: %s", err)
            return None
        if self.coordinator.video_session is None:
            return None
        if started:
            self._schedule_stream_idle_hangup()
        return self.coordinator.rtsp_url

    async def async_camera_image(self, width: int | None = None, height: int | None = None) -> bytes | None:
        """Return the latest JPEG frame, or placeholder when video is off."""
        session = self.coordinator.video_session
        if not session or not session.active:
            return PLACEHOLDER_JPEG
        rtp_receiver = session.rtp_receiver
        if not rtp_receiver:
            return PLACEHOLDER_JPEG
        try:
            async with asyncio.timeout(2.0):
                return await rtp_receiver.get_jpeg_frame()
        except TimeoutError:
            return rtp_receiver.latest_frame

    async def async_added_to_hass(self) -> None:
        """Register for push events when entity is added."""
        self._remove_push_cb = self.coordinator.add_push_callback(self._on_push)
        self._remove_stop_video_cb = self.coordinator.add_stop_video_callback(self._async_stop_ha_stream)
        self._remove_state_cb = self.coordinator.add_video_state_change_callback(self._async_video_state_changed)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister callbacks when entity is removed."""
        if self._remove_push_cb:
            self._remove_push_cb()
            self._remove_push_cb = None
        if self._remove_stop_video_cb:
            self._remove_stop_video_cb()
            self._remove_stop_video_cb = None
        if self._remove_state_cb:
            self._remove_state_cb()
            self._remove_state_cb = None
        if self._stream_hangup_task is not None:
            self._stream_hangup_task.cancel()
            self._stream_hangup_task = None
        await self.coordinator.async_stop_video()

    async def _async_video_state_changed(self) -> None:
        """Push a fresh state to HA when video session starts or stops.

        Also re-runs provider discovery.  HA caches the WebRTC provider lookup
        at entity-add time by calling `stream_source()` once — when the
        session isn't active yet it returns None, so go2rtc is never attached
        and `frontend_stream_types` only advertises HLS.  Refreshing here
        lets go2rtc claim the camera on the first video session, which
        upgrades the picture-entity card from HLS to WebRTC.
        """
        await self.async_refresh_providers()
        self.async_write_ha_state()

    async def _async_stop_ha_stream(self) -> None:
        """Tear down HA's cached Stream so the worker thread exits cleanly.

        Called from the coordinator before it forces any RTSP client
        disconnect.  Clearing `self.stream` ensures HA re-invokes
        `stream_source()` on the next Start, which properly waits on
        `_video_ready_event`.
        """
        stream = self.stream
        if stream is None:
            return
        self.stream = None
        try:
            await stream.stop()
        except Exception:
            _LOGGER.debug("Error stopping HA stream", exc_info=True)

    async def async_handle_async_webrtc_offer(
        self,
        offer_sdp: str,
        session_id: str,
        send_message: WebRTCSendMessage,
    ) -> None:
        """Open a go2rtc WebSocket signaling session for this offer.

        Uses go2rtc's /api/ws trickle-ICE protocol (same as HA core's go2rtc
        provider): the offer goes up immediately, and answer + ICE candidates
        stream back as they arrive. Client candidates are forwarded in
        async_on_webrtc_candidate. The previous one-shot HTTP POST had no
        candidate path — every trickled candidate raised "Cannot handle
        WebRTC candidate" and killed the session (observed live 2026-08-27).
        """
        # Viewing the camera starts the intercom call — there is no video
        # to relay otherwise.  Done before go2rtc connects so its RTSP pull
        # finds a stream that already carries video.  The browser keeps
        # trickling candidates meanwhile; buffer them until go2rtc is up.
        self._pending_candidates[session_id] = []
        self._cancel_hangup()
        try:
            if await self.coordinator.async_ensure_video():
                self._viewer_started_video = True
        except Exception as err:
            self._pending_candidates.pop(session_id, None)
            _LOGGER.warning("Could not start intercom video for WebRTC viewer: %s", err)
            send_message(WebRTCError(code="video_start_failed", message=str(err)))
            return
        if session_id not in self._pending_candidates:
            _LOGGER.debug("WebRTC %s closed while video was starting", session_id)
            self._schedule_hangup_if_unwatched()
            return
        name = f"comelit_man_{self._entry_id}"
        session, base_url = go2rtc_endpoint(self.hass)
        try:
            ws = await session.ws_connect(
                f"{base_url}/api/ws?src={name}",
                timeout=aiohttp.ClientWSTimeout(ws_close=5.0),
            )
        except Exception as err:
            self._pending_candidates.pop(session_id, None)
            send_message(WebRTCError(code="go2rtc_error", message=str(err)))
            return

        async def listen() -> None:
            try:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    data = msg.json()
                    mtype = data.get("type")
                    value = data.get("value")
                    if mtype == "webrtc/answer":
                        _LOGGER.debug("WebRTC %s: answer from go2rtc (%d bytes)", session_id, len(value or ""))
                        send_message(WebRTCAnswer(answer=value))
                    elif mtype == "webrtc/candidate" and value:
                        _LOGGER.debug("WebRTC %s: candidate from go2rtc: %.60s", session_id, value)
                        send_message(WebRTCCandidate(candidate=RTCIceCandidateInit(value)))
                    elif mtype == "error":
                        _LOGGER.debug("WebRTC %s: go2rtc error: %s", session_id, value)
                        send_message(WebRTCError(code="go2rtc_error", message=str(value)))
                    else:
                        _LOGGER.debug("WebRTC %s: unhandled go2rtc message type=%s", session_id, mtype)
            except Exception:
                _LOGGER.debug("go2rtc signaling session %s ended", session_id, exc_info=True)

        task = self.hass.async_create_background_task(listen(), f"comelit-webrtc-{session_id}")
        self._webrtc_sessions[session_id] = (ws, task)
        buffered = self._pending_candidates.pop(session_id, [])
        try:
            await ws.send_json({"type": "webrtc/offer", "value": offer_sdp})
            for candidate in buffered:
                await ws.send_json({"type": "webrtc/candidate", "value": candidate})
        except Exception as err:
            send_message(WebRTCError(code="go2rtc_error", message=str(err)))
            self.close_webrtc_session(session_id)

    async def async_on_webrtc_candidate(self, session_id: str, candidate: RTCIceCandidateInit) -> None:
        """Forward a client ICE candidate to the go2rtc signaling session."""
        entry = self._webrtc_sessions.get(session_id)
        if entry is None:
            pending = self._pending_candidates.get(session_id)
            if pending is not None:
                pending.append(candidate.candidate)
                return
            _LOGGER.debug("Unknown WebRTC session %s — ignoring candidate", session_id)
            return
        ws, _task = entry
        try:
            _LOGGER.debug("WebRTC %s: candidate from client: %.60s", session_id, candidate.candidate)
            await ws.send_json({"type": "webrtc/candidate", "value": candidate.candidate})
        except Exception:
            _LOGGER.debug("Failed to forward candidate for session %s", session_id, exc_info=True)

    def close_webrtc_session(self, session_id: str) -> None:
        """Close a go2rtc signaling session (called on WS unsubscribe)."""
        was_starting = self._pending_candidates.pop(session_id, None) is not None
        entry = self._webrtc_sessions.pop(session_id, None)
        if entry is not None:
            ws, task = entry
            task.cancel()

            async def _close() -> None:
                with contextlib.suppress(Exception):
                    await ws.close()

            self.hass.async_create_background_task(_close(), f"comelit-webrtc-close-{session_id}")
        elif not was_starting:
            return
        self._schedule_hangup_if_unwatched()

    def _schedule_hangup_if_unwatched(self) -> None:
        """Arm the hangup timer when a viewer-started call has no viewers left.

        Reached both from a normal close and from an offer whose viewer
        closed while the call was still starting — that close found the
        session only in the candidate buffer, before the call was marked
        viewer-started, so the offer re-checks once the call is up.
        """
        if self._viewer_started_video and not self._webrtc_sessions and not self._pending_candidates:
            self._cancel_hangup()
            self._hangup_task = self.hass.async_create_background_task(
                self._hangup_after_last_viewer(), "comelit-webrtc-hangup"
            )

    def _cancel_hangup(self) -> None:
        if self._hangup_task is not None and not self._hangup_task.done():
            self._hangup_task.cancel()
        self._hangup_task = None

    async def _hangup_after_last_viewer(self) -> None:
        """End a viewer-started call once nobody has watched it for a moment.

        Without this the intercom stays in a call until the session's own
        120 s timeout.  Calls started by a ring (or still ringing) are left
        alone — hanging those up would cut the visitor off.
        """
        await asyncio.sleep(VIEWER_HANGUP_DELAY)
        if self._webrtc_sessions or self._pending_candidates:
            return
        self._viewer_started_video = False
        if self.coordinator.video_session is None or self.coordinator.inbound_ring_pending:
            return
        _LOGGER.debug("Last WebRTC viewer left — ending intercom call")
        self.coordinator.request_video_stop()
        await self.coordinator.async_stop_video()

    def _schedule_stream_idle_hangup(self) -> None:
        if self._stream_hangup_task is not None and not self._stream_hangup_task.done():
            return
        self._stream_hangup_task = self.hass.async_create_background_task(
            self._hangup_when_stream_idle(), "comelit-stream-hangup"
        )

    async def _hangup_when_stream_idle(self) -> None:
        """End a call started for a stream consumer once no RTSP client is left.

        Unlike WebRTC viewers, HomeKit and HLS consumers give no close
        callback — the RTSP client count is the only signal that they are
        gone.  The count also covers go2rtc (which holds an RTSP client per
        WebRTC viewer), so a dashboard viewer keeps the call up too.
        Without this the call would run to its 120 s timeout.  Calls with
        an unanswered ring are left alone, as in _hangup_after_last_viewer.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STREAM_CONNECT_GRACE
        idle_since: float | None = None
        while self.coordinator.video_session is not None:
            server = self.coordinator.rtsp_server
            now = loop.time()
            if server is not None and server.client_count > 0:
                deadline = 0.0
                idle_since = None
            elif now >= deadline:
                idle_since = idle_since if idle_since is not None else now
                if now - idle_since >= STREAM_IDLE_HANGUP_DELAY:
                    break
            await asyncio.sleep(STREAM_IDLE_POLL)
        else:
            return
        if self._webrtc_sessions or self._pending_candidates or self.coordinator.inbound_ring_pending:
            return
        _LOGGER.debug("No RTSP clients left — ending intercom call started for stream consumer")
        self.coordinator.request_video_stop()
        await self.coordinator.async_stop_video()

    def _on_push(self, event: PushEvent) -> None:
        """Handle push events — no auto-start; user controls video via button or automation."""
