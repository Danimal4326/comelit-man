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

        async def _answer(wait_audio=True):
            assert wait_audio is False  # the door path doesn't wait for audio
            coord._inbound_answered = True
            coord._pending_inbound_ring = None
            return True

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

        async def _answer(wait_audio=True):
            coord._inbound_answered = True
            return True

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

    @pytest.mark.asyncio
    async def test_rejected_answer_still_opens_but_does_not_hang_up(self):
        coord = _coord()
        session = MagicMock(active=True)
        session.async_open_door_on_ctpp = AsyncMock()
        coord._video_session = session
        coord._pending_inbound_ring = "00000100"
        with patch.object(coord, "async_answer_inbound", new_callable=AsyncMock, return_value=False):
            await coord.async_open_door(_door())
        session.async_open_door_on_ctpp.assert_awaited_once()
        coord.config_entry.async_create_background_task.assert_not_called()


class TestSetupAckMatchesCapture:
    def test_bytes_equal_homebridge_answer_frame(self):
        """2026-11-24 capture: homebridge-comelit-doorbell's accepted answer on a 6701W."""
        import struct

        from custom_components.comelit_man.protocol import encode_setup_ack

        # DATA conn=9ad1 seq=114 ack=29 body=000c000000000000 addrs=[000000014, 00000001]
        ts = struct.unpack("<I", bytes([0x9A, 0xD1, 114, 29]))[0]
        expected = (
            bytes([0x40, 0x18, 0x9A, 0xD1, 114, 29, 0x00, 0x08])
            + bytes.fromhex("000c000000000000")
            + b"\xff\xff\xff\xff"
            + b"000000014\x00"
            + b"00000001\x00\x00"
        )
        assert encode_setup_ack("000000014", "00000001", ts) == expected


class TestPeerAckTracking:
    def _session(self):
        from custom_components.comelit_man.video_call import VideoCallSession

        s = VideoCallSession.__new__(VideoCallSession)
        s._call_counter = 0x22D6E0B5  # conn b5e0, seq 0xD6, ack 0x22
        return s

    def test_data_frame_sets_peer_ack(self):
        s = self._session()
        s._note_peer_frame(bytes([0x40, 0x18, 0x35, 0xE0, 34, 214, 0x00, 0x08]) + b"x" * 8)
        assert s._peer_ack == 35

    def test_empty_ack_and_syn_frames_ignored(self):
        s = self._session()
        s._note_peer_frame(bytes([0x00, 0x18, 0x35, 0xE0, 40, 214, 0x00, 0x00]))
        s._note_peer_frame(bytes([0xC0, 0x18, 0x11, 0x22, 50, 9, 0x00, 0x28]) + b"x" * 40)
        assert s._peer_ack is None

    def test_next_counter_advances_seq_and_stamps_ack(self):
        s = self._session()
        s._peer_ack = 35
        c = s._next_counter()
        assert c == (0x22D6E0B5 + 0x00010000) & 0x00FFFFFF | (35 << 24)
        assert s._call_counter == c

    def test_next_counter_without_peer_ack_keeps_ack_byte(self):
        s = self._session()
        assert s._next_counter() == (0x22D6E0B5 + 0x00010000) & 0xFFFFFFFF


class TestAnswerAcceptance:
    def _session(self):
        from custom_components.comelit_man.video_call import VideoCallSession

        s = VideoCallSession.__new__(VideoCallSession)
        s._active = True
        s._device_rtpc_req_id = 0
        s._rtp_receiver = MagicMock()
        s._client = MagicMock()
        s._client.get_channel = MagicMock(return_value=MagicMock())
        s._client.send_binary = AsyncMock()
        s._config = MagicMock(apt_address="00000001", apt_subaddress=3)
        s._ctpp_lock = asyncio.Lock()
        s._call_counter = 0x00100000
        s._inbound_device_rtpc = None
        s._answer_handoff = None
        return s

    @pytest.mark.asyncio
    async def test_door_path_returns_on_first_reply(self):
        import struct

        s = self._session()

        async def inject():
            while s._answer_handoff is None:
                await asyncio.sleep(0)
            await s._answer_handoff.put(struct.pack("<HI", 0x1840, 0x1D7203D1) + struct.pack(">H", 0x000A))

        task = asyncio.create_task(inject())
        assert await s.answer_inbound(wait_audio=False) is True
        await task
        s._rtp_receiver.start_audio_sender.assert_not_called()

    @pytest.mark.asyncio
    async def test_silent_device_means_not_accepted(self):
        s = self._session()
        with patch("custom_components.comelit_man.video_call.VIDEO_RESPONSE_TIMEOUT", 0.02):
            assert await s.answer_inbound(wait_audio=False) is False

    @pytest.mark.asyncio
    async def test_drain_without_handoff_queue_is_false(self):
        s = self._session()
        assert await s._drain_answer_reply(MagicMock(), "a", "b", need_both=False) is False


class TestInboundRelease:
    def _session(self, inbound=True):
        from custom_components.comelit_man.video_call import VideoCallSession

        s = VideoCallSession.__new__(VideoCallSession)
        s._inbound = inbound
        s._on_release = MagicMock()
        return s

    @staticmethod
    def _release(cause, flags=0x60):
        import struct

        return bytes([flags, 0x18, 0x35, 0xE0, 35, 218]) + struct.pack(">HH", 0x0003, 0x000E) + bytes([cause])

    @pytest.mark.asyncio
    async def test_answered_elsewhere_is_reported(self):
        s = self._session()
        client = MagicMock(send_binary=AsyncMock())
        assert await s._handle_inbound_release(client, MagicMock(), self._release(10), "000000013", "00000100")
        s._on_release.assert_called_once_with(10)
        client.send_binary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_not_inbound_or_not_release_is_ignored(self):
        client = MagicMock(send_binary=AsyncMock())
        assert not await self._session(inbound=False)._handle_inbound_release(
            client, MagicMock(), self._release(10), "a", "b"
        )
        s = self._session()
        other = bytes([0x40, 0x18, 0x35, 0xE0, 35, 218, 0x00, 0x0A, 0x00, 0x11, 0x00])
        assert not await s._handle_inbound_release(client, MagicMock(), other, "a", "b")
        assert not await s._handle_inbound_release(client, MagicMock(), self._release(10, flags=0x00), "a", "b")
        client.send_binary.assert_not_awaited()


class TestCoordinatorAnswerAndRelease:
    @pytest.mark.asyncio
    async def test_rejected_answer_leaves_ring_unanswered(self):
        coord = _coord()
        coord._video_session = MagicMock(active=True)
        coord._video_session.answer_inbound = AsyncMock(return_value=False)
        coord._pending_inbound_ring = "00000100"
        assert await coord.async_answer_inbound() is False
        assert coord._inbound_answered is False
        assert coord._pending_inbound_ring == "00000100"

    @pytest.mark.asyncio
    async def test_second_answer_after_success_is_a_noop(self):
        coord = _coord()
        coord._video_session = MagicMock(active=True)
        coord._video_session.answer_inbound = AsyncMock(return_value=True)
        assert await coord.async_answer_inbound() is True
        assert await coord.async_answer_inbound() is True
        coord._video_session.answer_inbound.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_session_cannot_answer(self):
        coord = _coord()
        assert await coord.async_answer_inbound() is False

    def test_released_answered_elsewhere_is_not_a_missed_call(self):
        coord = _coord()
        coord._pending_inbound_ring = "00000100"
        coord._last_ring_mono = coord_mod.time.monotonic()
        coord._on_inbound_released(10)
        assert coord._pending_inbound_ring is None
        assert coord._last_ring_mono is None
        assert coord._video_stopped_by_user is True
        coord.config_entry.async_create_background_task.assert_called_once()

    def test_released_by_caller_keeps_missed_call_path(self):
        coord = _coord()
        coord._pending_inbound_ring = "00000100"
        coord._on_inbound_released(0)
        assert coord._pending_inbound_ring == "00000100"
