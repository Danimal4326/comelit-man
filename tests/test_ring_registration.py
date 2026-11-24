"""Tests for the VIP ring registration: REGISTER framing, renewal ACKs, lease refresh.

Behaviour mirrors homebridge-comelit-doorbell, which receives rings on the same
6701W/6741W firmware where the old registration (fixed 0x18C2 conn id, blind
ACK/FIN pair, ACKs from init_ts + 0x01010000, no refresh) received none.
"""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.comelit_man import coordinator as coord_mod
from custom_components.comelit_man.ctpp import ctpp_register, ctpp_reregister
from custom_components.comelit_man.models import DeviceConfig, PushEvent
from custom_components.comelit_man.protocol import ack_word, encode_ctpp_init, fin_word
from custom_components.comelit_man.video_call import _transform_device_ts
from custom_components.comelit_man.vip_listener import VipEventListener

APT = "SB000006"
SUB = 1
OUR = "SB0000061"


def _frame(prefix: int, ts: int, action: int, tail: bytes = b"") -> bytes:
    return struct.pack("<HI", prefix, ts) + struct.pack(">H", action) + tail


def _client_with(responses: list[bytes | type[BaseException]]) -> tuple[MagicMock, MagicMock, list[bytes]]:
    client = MagicMock()
    sent: list[bytes] = []

    async def _send(_ch, data):
        sent.append(data)

    client.send_binary = AsyncMock(side_effect=_send)
    client.read_response = AsyncMock(side_effect=responses)
    channel = MagicMock()
    channel.response_queue = asyncio.Queue()
    return client, channel, sent


# ---------------------------------------------------------------------------
# ack_word / fin_word
# ---------------------------------------------------------------------------


class TestAckWord:
    def test_flips_conn_bit_and_swaps_counters(self):
        # bytes: conn0=0x98 conn1=0xC2 seq=0x05 ack=0x07
        ts = struct.unpack("<I", bytes([0x98, 0xC2, 0x05, 0x07]))[0]
        out = struct.pack("<I", ack_word(ts))
        assert out == bytes([0x18, 0xC2, 0x07, 0x06])

    def test_seq_wraps(self):
        ts = struct.unpack("<I", bytes([0x10, 0x20, 0xFF, 0x00]))[0]
        assert struct.pack("<I", ack_word(ts)) == bytes([0x90, 0x20, 0x00, 0x00])

    def test_fin_word_does_not_advance_seq(self):
        ts = struct.unpack("<I", bytes([0x98, 0xC2, 0x05, 0x07]))[0]
        assert struct.pack("<I", fin_word(ts)) == bytes([0x18, 0xC2, 0x07, 0x05])

    def test_equals_transform_for_device_initiated_connections(self):
        # Device-opened connections have the top bit clear: XOR == OR, so the
        # listener's switch from _transform_device_ts changes nothing there.
        for b0 in (0x00, 0x12, 0x7F):
            ts = struct.unpack("<I", bytes([b0, 0x34, 0x56, 0x78]))[0]
            assert ack_word(ts) == _transform_device_ts(ts)


# ---------------------------------------------------------------------------
# encode_ctpp_init local_id
# ---------------------------------------------------------------------------


class TestEncodeCtppInitLocalId:
    def test_default_keeps_fixed_18c2(self):
        payload = encode_ctpp_init(APT, SUB, 0x11223344)
        assert payload[8:12] == b"\x00\x40\x18\xc2"

    def test_local_id_written_little_endian(self):
        payload = encode_ctpp_init(APT, SUB, 0x11223344, local_id=0x3344)
        # header conn id bytes (2-3) and REGISTER local id agree
        assert payload[2:4] == b"\x44\x33"
        assert payload[8:12] == b"\x00\x40\x44\x33"

    def test_same_length_as_legacy(self):
        assert len(encode_ctpp_init(APT, SUB, 1, local_id=1)) == len(encode_ctpp_init(APT, SUB, 1))


# ---------------------------------------------------------------------------
# ctpp_register
# ---------------------------------------------------------------------------


class TestCtppRegister:
    @pytest.mark.asyncio
    async def test_no_blind_ack_pair_when_no_renewal(self):
        ack = _frame(0x1800, 0x00000000, 0)
        client, channel, sent = _client_with([ack, ack])
        answers = await ctpp_register(client, channel, APT, SUB, OUR, 0xAABBCCDD)
        assert answers == 2
        assert len(sent) == 1  # only the REGISTER
        assert sent[0][8:12] == b"\x00\x40\xdd\xcc"

    @pytest.mark.asyncio
    async def test_acks_device_renewal_with_ack_word(self):
        renewal_ts = struct.unpack("<I", bytes([0xDD, 0xCC, 0x00, 0x01]))[0]
        client, channel, sent = _client_with([_frame(0x1800, 0, 0), _frame(0x1860, renewal_ts, 0x0010)])
        await ctpp_register(client, channel, APT, SUB, OUR, 0xAABBCCDD)
        assert len(sent) == 3
        prefixes = [struct.unpack_from("<H", s, 0)[0] for s in sent[1:]]
        assert prefixes == [0x1800, 0x1820]
        for s in sent[1:]:
            assert struct.unpack_from("<I", s, 2)[0] == ack_word(renewal_ts)

    @pytest.mark.asyncio
    async def test_ring_during_registration_is_requeued(self):
        ring = _frame(0x18C0, 0x00001234, 0x0028, b"x" * 10)
        client, channel, _sent = _client_with([ring, _frame(0x1800, 0, 0), _frame(0x1800, 0, 0)])
        answers = await ctpp_register(client, channel, APT, SUB, OUR, 0x01020304)
        assert answers == 2
        assert channel.response_queue.get_nowait() == ring

    @pytest.mark.asyncio
    async def test_silent_device_warns(self, caplog):
        client, channel, sent = _client_with([TimeoutError()])
        answers = await ctpp_register(client, channel, APT, SUB, OUR, 1, response_timeout=0.01)
        assert answers == 0
        assert len(sent) == 1
        assert "did not answer the registration" in caplog.text

    @pytest.mark.asyncio
    async def test_short_frames_skipped(self):
        client, channel, _ = _client_with([b"\x00", _frame(0x1800, 0, 0), _frame(0x1800, 0, 0)])
        assert await ctpp_register(client, channel, APT, SUB, OUR, 1) == 2

    @pytest.mark.asyncio
    async def test_reregister_sends_only_register(self):
        client, channel, sent = _client_with([])
        await ctpp_reregister(client, channel, APT, SUB, 0x00005678)
        assert len(sent) == 1
        assert sent[0][8:12] == b"\x00\x40\x78\x56"
        client.read_response.assert_not_called()


# ---------------------------------------------------------------------------
# VipEventListener ACKs
# ---------------------------------------------------------------------------


def _listener(on_inbound_ring=None) -> tuple[VipEventListener, list[bytes]]:
    config = MagicMock(spec=DeviceConfig)
    config.apt_address = APT
    config.apt_subaddress = SUB
    client = MagicMock()
    sent: list[bytes] = []

    async def _send(_ch, data):
        sent.append(data)

    client.send_binary = AsyncMock(side_effect=_send)
    listener = VipEventListener(client, config, MagicMock(), init_ts=0x12000000, on_inbound_ring=on_inbound_ring)
    listener._channel = MagicMock()
    return listener, sent


class TestListenerAcks:
    @pytest.mark.asyncio
    async def test_every_renewal_acked_from_its_own_counters(self):
        listener, sent = _listener()
        for ts in (0x0102C298, 0x0203C298):  # later renewals advance seq/ack
            sent.clear()
            await listener._process_message(_frame(0x1860, ts, 0x0010, b"\xff\xff\xff\xffSB0000061\x00SB000006\x00"))
            assert [struct.unpack_from("<I", s, 2)[0] for s in sent] == [ack_word(ts), ack_word(ts)]
            assert [struct.unpack_from("<H", s, 0)[0] for s in sent] == [0x1800, 0x1820]

    @pytest.mark.asyncio
    async def test_empty_vip_frame_gets_ack_then_fin(self):
        listener, sent = _listener()
        ts = 0x0405C298
        await listener._process_message(_frame(0x1860, ts, 0x0000))
        assert [struct.unpack_from("<H", s, 0)[0] for s in sent] == [0x1800, 0x1820]
        assert struct.unpack_from("<I", sent[1], 2)[0] == fin_word(ts)

    @pytest.mark.asyncio
    async def test_invite_with_inbound_hook_not_acked_and_hook_called(self):
        hook = MagicMock()
        listener, sent = _listener(on_inbound_ring=hook)
        await listener._process_message(
            _frame(0x18C0, 0x00003412, 0x0028, b"\xff\xff\xff\xffSB100001\x00SB0000061\x00")
        )
        assert sent == []
        hook.assert_called_once()


# ---------------------------------------------------------------------------
# Coordinator: immediate ring + lease loop
# ---------------------------------------------------------------------------


def _coordinator() -> coord_mod.ComelitLocalCoordinator:
    from tests.test_coordinator import _make_coordinator

    return _make_coordinator(with_client=True)


class TestImmediateRing:
    def test_ring_fires_before_video(self):
        coord = _coordinator()
        events: list[PushEvent] = []
        coord._push_callbacks = {events.append: None}
        coord._on_inbound_ring("SB100001", 0x1234)
        assert [e.event_type for e in events] == ["ring"]
        coord.config_entry.async_create_background_task.assert_called_once()

    def test_retransmit_does_not_refire(self):
        coord = _coordinator()
        events: list[PushEvent] = []
        coord._push_callbacks = {events.append: None}
        coord._on_inbound_ring("SB100001", 0x1234)
        coord._on_inbound_ring("SB100001", 0x1234)
        assert len(events) == 1


class TestLeaseLoop:
    @pytest.mark.asyncio
    async def test_reregisters_when_idle(self):
        coord = _coordinator()
        coord._config = MagicMock(apt_address=APT, apt_subaddress=SUB)
        coord._vip_listener = MagicMock()
        coord._connected_mono = coord_mod.time.monotonic()
        ctpp = MagicMock()
        coord._client.get_channel = MagicMock(return_value=ctpp)
        calls = 0

        async def fake_sleep(_s):
            nonlocal calls
            calls += 1
            if calls > 2:
                raise asyncio.CancelledError

        with (
            patch.object(coord_mod.asyncio, "sleep", side_effect=fake_sleep),
            patch.object(coord_mod, "ctpp_reregister", new_callable=AsyncMock) as rereg,
            pytest.raises(asyncio.CancelledError),
        ):
            await coord._lease_loop()
        assert rereg.await_count == 2
        assert coord._ctpp_init_ts != 0

    @pytest.mark.asyncio
    async def test_skips_while_video_active(self):
        coord = _coordinator()
        coord._vip_listener = MagicMock()
        coord._video_session = MagicMock()
        calls = 0

        async def fake_sleep(_s):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise asyncio.CancelledError

        with (
            patch.object(coord_mod.asyncio, "sleep", side_effect=fake_sleep),
            patch.object(coord_mod, "ctpp_reregister", new_callable=AsyncMock) as rereg,
            pytest.raises(asyncio.CancelledError),
        ):
            await coord._lease_loop()
        rereg.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recycles_old_connection(self):
        coord = _coordinator()
        coord._vip_listener = MagicMock()
        coord._connected_mono = coord_mod.time.monotonic() - coord_mod.LEASE_RECYCLE_AFTER - 1
        with (
            patch.object(coord_mod.asyncio, "sleep", new_callable=AsyncMock),
            patch.object(coord_mod, "ctpp_reregister", new_callable=AsyncMock) as rereg,
        ):
            await coord._lease_loop()
        rereg.assert_not_awaited()
        coord.config_entry.async_create_background_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_stops_when_disconnected(self):
        coord = _coordinator()
        coord._client.connected = False
        with patch.object(coord_mod.asyncio, "sleep", new_callable=AsyncMock):
            await coord._lease_loop()

    @pytest.mark.asyncio
    async def test_refresh_failure_is_swallowed(self):
        coord = _coordinator()
        coord._config = MagicMock(apt_address=APT, apt_subaddress=SUB)
        coord._vip_listener = MagicMock()
        coord._connected_mono = coord_mod.time.monotonic()
        coord._client.get_channel = MagicMock(return_value=MagicMock())
        calls = 0

        async def fake_sleep(_s):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise asyncio.CancelledError

        with (
            patch.object(coord_mod.asyncio, "sleep", side_effect=fake_sleep),
            patch.object(coord_mod, "ctpp_reregister", new_callable=AsyncMock, side_effect=RuntimeError("x")),
            pytest.raises(asyncio.CancelledError),
        ):
            await coord._lease_loop()

    @pytest.mark.asyncio
    async def test_recycle_reconnects_when_idle(self):
        coord = _coordinator()
        with patch.object(coord, "_reconnect", new_callable=AsyncMock) as rc:
            await coord._recycle_connection()
        rc.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recycle_deferred_during_video(self):
        coord = _coordinator()
        coord._video_session = MagicMock()
        with (
            patch.object(coord, "_reconnect", new_callable=AsyncMock) as rc,
            patch.object(coord, "_start_keepalive") as sk,
        ):
            await coord._recycle_connection()
        rc.assert_not_awaited()
        sk.assert_called_once()

    @pytest.mark.asyncio
    async def test_recycle_failure_logged(self, caplog):
        coord = _coordinator()
        with patch.object(coord, "_reconnect", new_callable=AsyncMock, side_effect=RuntimeError("down")):
            await coord._recycle_connection()
        assert "lease recycle failed" in caplog.text
