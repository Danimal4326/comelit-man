"""CTPP channel helpers: shared init/handshake sequence.

All CTPP consumers (VIP listener, video session, standalone door open) use
ctpp_init_sequence() so the registration handshake is implemented exactly once.
"""

from __future__ import annotations

import asyncio
import logging
import struct

from .channels import Channel
from .client import IconaBridgeClient
from .const import RENEWAL_ACK_INCREMENT
from .protocol import ack_word, encode_call_response_ack, encode_ctpp_init

_LOGGER = logging.getLogger(__name__)

# Both sub-counters increment by 1 (bytes[4] and bytes[5] of the CTPP body).
# Used to compute the renewal ACK timestamp offset from the init timestamp.
# Sourced from const.RENEWAL_ACK_INCREMENT so firmware variants have one knob;
# the alias is kept because callers import this name.
_CTR_INCR_BOTH = RENEWAL_ACK_INCREMENT

# Minimum response length: prefix(2) + timestamp(4) + action(2) = 8 bytes.
_CTPP_RESPONSE_MIN_LEN = 8


async def ctpp_init_sequence(
    client: IconaBridgeClient,
    channel: Channel,
    apt_addr: str,
    apt_sub: int,
    our_addr: str,
    timestamp: int | None = None,
    response_timeout: float = 5.0,
    send_ack: bool = True,
) -> None:
    """CTPP handshake: init → drain 2 responses → optionally send ACK pair.

    The ACK pair (0x1800 + 0x1820) is required for VIP listener and video
    sessions but must NOT be sent for standalone door opens — the original
    door open flow never sent it.

    Args:
        client: the shared ICONA Bridge client.
        channel: the already-open CTPP channel.
        apt_addr: apartment address without subaddress (e.g. "SB000006").
        apt_sub: apartment subaddress integer (e.g. 1).
        our_addr: full address including subaddress (e.g. "SB0000061").
        timestamp: LE32 timestamp to embed in the init message.
        response_timeout: seconds to wait for each device response.
        send_ack: send the ACK pair after draining responses (default True).
    """
    init_payload = encode_ctpp_init(apt_addr, apt_sub, timestamp)
    await client.send_binary(channel, init_payload)

    await read_response_ctpp(client, channel, response_timeout)

    if send_ack:
        assert timestamp is not None, "timestamp required when send_ack=True"
        ack_ts = (timestamp + _CTR_INCR_BOTH) & 0xFFFFFFFF
        await client.send_binary(channel, encode_call_response_ack(our_addr, apt_addr, ack_ts))
        await client.send_binary(channel, encode_call_response_ack(our_addr, apt_addr, ack_ts, prefix=0x1820))
        _LOGGER.debug(
            "CTPP ACK pair sent (init_ts=0x%08X ack_ts=0x%08X)",
            timestamp,
            ack_ts,
        )


PREFIX_VIP_EVENT = 0x1860
PREFIX_CALL_INIT = 0x18C0
ACTION_REGISTRATION_RENEWAL = 0x0010
# Reads at most this many frames while waiting for the two registration answers
# (a ring arriving mid-registration doesn't count as one of them).
_REGISTER_MAX_READS = 4


async def ctpp_register(
    client: IconaBridgeClient,
    channel: Channel,
    apt_addr: str,
    apt_sub: int,
    our_addr: str,
    timestamp: int,
    response_timeout: float = 2.0,
) -> int:
    """Register for incoming calls on an open CTPP channel (VIP listener path).

    Differs from ctpp_init_sequence() in the three ways that keep rings flowing
    on 6701W/6741W firmware (matching homebridge-comelit-doorbell):

    * the REGISTER body carries the header's own connection id, not 0x18C2;
    * no ACK pair is sent blind right after the init — its 0x1820 (FIN) is taken
      by the device as a hang-up of the registration;
    * instead the device's own 0x1860/0x0010 renewal check is ACKed (0x1800 +
      0x1820) with ack_word() of *its* frame.

    A ring (0x18C0) arriving during the handshake is put back on the channel's
    queue for the listener instead of being swallowed.

    Returns the number of registration answers received (0 = device silent).
    """
    await client.send_binary(channel, encode_ctpp_init(apt_addr, apt_sub, timestamp, local_id=timestamp & 0xFFFF))
    held: list[bytes] = []
    answers = 0
    reads = 0
    while answers < 2 and reads < _REGISTER_MAX_READS:
        reads += 1
        try:
            async with asyncio.timeout(response_timeout):
                resp = await client.read_response(channel)
        except TimeoutError:
            _LOGGER.debug("CTPP register: no answer %d (timeout)", answers + 1)
            break
        if len(resp) < _CTPP_RESPONSE_MIN_LEN:
            continue
        prefix = struct.unpack_from("<H", resp, 0)[0]
        resp_ts = struct.unpack_from("<I", resp, 2)[0]
        action = struct.unpack_from(">H", resp, 6)[0]
        if prefix == PREFIX_CALL_INIT:
            _LOGGER.debug("CTPP register: call arrived during registration — keeping it for the listener")
            held.append(resp)
            continue
        answers += 1
        _LOGGER.debug(
            "CTPP register answer %d: prefix=0x%04X ts=0x%08X action=0x%04X",
            answers,
            prefix,
            resp_ts,
            action,
        )
        if prefix == PREFIX_VIP_EVENT and action == ACTION_REGISTRATION_RENEWAL:
            word = ack_word(resp_ts)
            await client.send_binary(channel, encode_call_response_ack(our_addr, apt_addr, word))
            await client.send_binary(channel, encode_call_response_ack(our_addr, apt_addr, word, prefix=0x1820))
            _LOGGER.debug("CTPP register: renewal ACK pair sent (ack=0x%08X)", word)
    for frame in held:
        channel.response_queue.put_nowait(frame)
    if answers == 0 and not held:
        _LOGGER.warning("CTPP register: device did not answer the registration — rings may not arrive")
    return answers


async def ctpp_reregister(
    client: IconaBridgeClient,
    channel: Channel,
    apt_addr: str,
    apt_sub: int,
    timestamp: int,
) -> None:
    """Re-send the registration in place; the running listener ACKs the answers."""
    await client.send_binary(channel, encode_ctpp_init(apt_addr, apt_sub, timestamp, local_id=timestamp & 0xFFFF))


async def read_response_ctpp(
    client: IconaBridgeClient,
    channel: Channel,
    response_timeout: float = 5.0,
) -> None:
    # Drain device's two responses (0x1800 ACK + 0x1860/0x0010 renewal request).
    # We don't use the device's timestamp to compute our ACK — see docstring.
    for i in range(2):
        try:
            async with asyncio.timeout(response_timeout):
                resp = await client.read_response(channel)
        except TimeoutError:
            _LOGGER.debug("CTPP init response %d: no response (timeout)", i + 1)
            continue
        if len(resp) >= _CTPP_RESPONSE_MIN_LEN:
            prefix = struct.unpack_from("<H", resp, 0)[0]
            resp_ts = struct.unpack_from("<I", resp, 2)[0]
            action = struct.unpack_from(">H", resp, 6)[0]
            _LOGGER.debug(
                "CTPP init response %d: %d bytes, prefix=0x%04X ts=0x%08X action=0x%04X",
                i + 1,
                len(resp),
                prefix,
                resp_ts,
                action,
            )
