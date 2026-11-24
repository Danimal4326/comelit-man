"""Door open during a ring: answer first, open in-call, hang up; inbound callee."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.comelit_man import coordinator as coord_mod
from custom_components.comelit_man.models import Door
from custom_components.comelit_man.protocol import encode_door_open_during_video


def _door() -> Door:
    return Door(id=0, index=0, name="Gate", output_index=1, apt_address="00000100", module_index=0)


def _coord():
    from tests.test_coordinator import _make_coordinator

    coord = _make_coordinator(with_client=True)
    coord._config.apt_address = "00000001"
    coord._config.apt_subaddress = 3
    coord._config.caller_address = "00000100"
    return coord


class TestEncoderCallee:
    def test_default_callee_is_entrance(self):
        payload = encode_door_open_during_video("000000013", "00000100", 1, 1)
        assert payload[-10:].rstrip(b"\x00") == b"00000100"

    def test_inbound_callee_is_apartment(self):
        payload = encode_door_open_during_video("000000013", "00000100", 1, 1, callee="00000001")
        assert payload[-10:].rstrip(b"\x00") == b"00000001"
        assert len(payload) == len(encode_door_open_during_video("000000013", "00000100", 1, 1))


class TestSessionInboundCallee:
    @pytest.mark.asyncio
    async def test_inbound_session_uses_apartment_callee(self):
        from custom_components.comelit_man.video_call import VideoCallSession

        session = VideoCallSession.__new__(VideoCallSession)
        session._client = MagicMock()
        session._client.get_channel = MagicMock(return_value=MagicMock())
        session._client.send_binary = AsyncMock()
        session._active = True
        session._ctpp_lock = asyncio.Lock()
        session._call_counter = 0x10000000
        session._config = MagicMock(apt_address="00000001")
        session._inbound = True
        await session.async_open_door_on_ctpp("000000013", "00000100", 1)
        payload = session._client.send_binary.call_args[0][1]
        assert payload[-10:].rstrip(b"\x00") == b"00000001"


class TestOpenDoorDuringRing:
    @pytest.mark.asyncio
    async def test_unanswered_ring_is_answered_then_opened_then_hung_up(self):
        coord = _coord()
        session = MagicMock(active=True)
        session.async_open_door_on_ctpp = AsyncMock()
        coord._video_session = session
        coord._pending_inbound_ring = "00000100"

        async def _answer():
            coord._inbound_answered = True
            coord._pending_inbound_ring = None

        with patch.object(coord, "async_answer_inbound", side_effect=_answer) as answer:
            await coord.async_open_door(_door())
        answer.assert_awaited_once()
        session.async_open_door_on_ctpp.assert_awaited_once_with("000000013", "00000100", 1)
        coord.config_entry.async_create_background_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_answered_call_opens_without_answering_or_hangup(self):
        coord = _coord()
        session = MagicMock(active=True)
        session.async_open_door_on_ctpp = AsyncMock()
        coord._video_session = session
        coord._inbound_answered = True
        with patch.object(coord, "async_answer_inbound", new_callable=AsyncMock) as answer:
            await coord.async_open_door(_door())
        answer.assert_not_awaited()
        coord.config_entry.async_create_background_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_waits_for_ring_video_then_answers(self):
        coord = _coord()
        coord._last_ring_mono = coord_mod.time.monotonic()
        session = MagicMock(active=True)
        session.async_open_door_on_ctpp = AsyncMock()

        async def _video_arrives():
            await asyncio.sleep(0)
            coord._video_session = session
            coord._pending_inbound_ring = "00000100"
            coord._video_ready_event.set()

        async def _answer():
            coord._inbound_answered = True

        await coord._video_start_lock.acquire()
        try:
            asyncio.get_running_loop().create_task(_video_arrives())
            with patch.object(coord, "async_answer_inbound", side_effect=_answer):
                await coord.async_open_door(_door())
        finally:
            coord._video_start_lock.release()
        session.async_open_door_on_ctpp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_ring_uses_plain_open(self):
        coord = _coord()
        with patch.object(coord_mod, "open_door", new_callable=AsyncMock) as plain:
            await coord.async_open_door(_door())
        plain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ring_video_wait_times_out_to_plain_open(self):
        coord = _coord()
        coord._last_ring_mono = coord_mod.time.monotonic()
        await coord._video_start_lock.acquire()
        try:
            with (
                patch.object(coord_mod, "DOOR_RING_VIDEO_WAIT", 0.01),
                patch.object(coord_mod, "open_door", new_callable=AsyncMock) as plain,
            ):
                await coord.async_open_door(_door())
        finally:
            coord._video_start_lock.release()
        plain.assert_awaited_once()


class TestHangupAfterDoorOpen:
    @pytest.mark.asyncio
    async def test_hangs_up_when_unwatched(self):
        coord = _coord()
        with (
            patch.object(coord_mod.asyncio, "sleep", new_callable=AsyncMock),
            patch.object(coord, "async_stop_video", new_callable=AsyncMock) as stop,
        ):
            await coord._hangup_after_door_open()
        stop.assert_awaited_once()
        assert coord._video_stopped_by_user is True

    @pytest.mark.asyncio
    async def test_keeps_call_when_watched(self):
        coord = _coord()
        coord._rtsp_server = MagicMock(client_count=1)
        with (
            patch.object(coord_mod.asyncio, "sleep", new_callable=AsyncMock),
            patch.object(coord, "async_stop_video", new_callable=AsyncMock) as stop,
        ):
            await coord._hangup_after_door_open()
        stop.assert_not_awaited()
