"""
image_builder.py

Takes the stream of ParsedPacket objects belonging to ONE flow and turns
them into a picture the CNN can look at.

How a flow becomes an image (base paper Section III-B, Fig. 4):
  - The image has P rows and Q columns and 3 colour channels (R, G, B).
  - P = 9 (we use the base paper's "9 packets is enough to detect an
    attack with high confidence" finding — also exactly what your PPT's
    early-alert slide promises: "Threat Classification Confirmed at
    Packet 9").
  - Q = 1486 (the length of one packet's feature vector from packet_parser.py)
  - Row i of the image = packet i's feature vector.
  - If packet i travelled FORWARD (client -> server), its bytes go in
    the RED channel of row i, and GREEN/BLUE stay 0 for that row.
  - If packet i travelled BACKWARD (server -> client), its bytes go in
    the GREEN channel instead.
  - BLUE is always 0 (the base paper deliberately leaves it unused —
    there's no third "direction" in a TCP conversation, so a 3rd colour
    channel would just be dead weight).
  - If the flow hasn't produced 9 packets yet, the remaining rows stay
    all-zero (zero-padded) — this is what lets us produce a *partial*
    image after only 1, 4, or however many packets have arrived so far,
    which is exactly what the early-alert progress bar needs.

This file is used in TWO places:
  1. Offline, by data_prep/pcap_to_dataset.py, to build training images.
  2. Live, by app.py, fed one packet at a time as the sniffer sees them,
     to build the image the live dashboard predicts on.
"""

import numpy as np
from feature_extraction.packet_parser import ParsedPacket, N_FEATURES

P_PACKETS = 9          # image height — matches the paper's "9 packets" finding
Q_FEATURES = N_FEATURES  # image width — one packet's feature vector length
EARLY_ALERT_CHECKPOINTS = (1, 4, P_PACKETS)  # packet counts the dashboard reacts to

# Delta time is clipped/scaled into a single byte (0-255) so it fits in
# the same uint8 image as everything else. 2.0 seconds covers the vast
# majority of CICIDS2017 inter-packet gaps within one flow; anything
# slower just saturates at 255 rather than erroring out.
_DELTA_TIME_CLIP_SECONDS = 2.0


def _delta_time_byte(delta_seconds: float) -> int:
    if delta_seconds < 0:
        delta_seconds = 0.0
    scaled = (delta_seconds / _DELTA_TIME_CLIP_SECONDS) * 255.0
    return int(min(255, max(0, round(scaled))))


class FlowImageBuilder:
    """
    One instance per *active flow*. Call add_packet() as packets arrive;
    call get_image() any time to get the current (possibly partial) image.
    """

    def __init__(self):
        self.packets: list[ParsedPacket] = []
        self.last_timestamp: float | None = None

    def add_packet(self, parsed: ParsedPacket):
        """
        Feed one parsed packet into this flow's buffer. Fills in the
        delta-time byte (index 0 of the feature vector) — this is the
        one piece packet_parser.py couldn't know on its own, since it
        needs the *previous* packet's timestamp.
        Only the first P_PACKETS packets of a flow matter — everything
        after that doesn't change the image (paper: performance plateaus
        at 9 packets, so there is no benefit to keeping more in memory).
        """
        if len(self.packets) >= P_PACKETS:
            return  # image is already "full" for this flow

        delta = 0.0 if self.last_timestamp is None else (
            parsed.timestamp - self.last_timestamp
        )
        self.last_timestamp = parsed.timestamp
        parsed.feature_vector[0] = _delta_time_byte(delta)

        self.packets.append(parsed)

    def packet_count(self) -> int:
        return len(self.packets)

    def get_image(self) -> np.ndarray:
        """
        Returns the current PxQx3 uint8 image, zero-padded for any rows
        not yet filled.
        """
        img = np.zeros((P_PACKETS, Q_FEATURES, 3), dtype=np.uint8)
        for i, pkt in enumerate(self.packets):
            channel = 0 if pkt.is_forward else 1  # 0=R (forward), 1=G (backward)
            img[i, :, channel] = pkt.feature_vector
        return img

    def is_at_checkpoint(self) -> bool:
        """True right after a packet count the dashboard should react to
        (1, 4, 9 packets — see EARLY_ALERT_CHECKPOINTS)."""
        return self.packet_count() in EARLY_ALERT_CHECKPOINTS


if __name__ == "__main__":
    # Self-test: reuse the same synthetic packets as packet_parser.py's
    # self-test, push them through a builder, and sanity-check the image.
    import dpkt
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    import tempfile, os
    from feature_extraction.packet_parser import parse_packet

    pkts = [
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5555, dport=80, flags="S"),
        Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=5555, flags="SA"),
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5555, dport=80, flags="A") / Raw(b"GET / HTTP/1.1\r\n\r\n"),
    ]
    tmp_path = os.path.join(tempfile.gettempdir(), "self_test2.pcap")
    wrpcap(tmp_path, pkts)

    builder = FlowImageBuilder()
    with open(tmp_path, "rb") as f:
        for ts, buf in dpkt.pcap.Reader(f):
            parsed = parse_packet(buf, ts)
            assert parsed is not None
            builder.add_packet(parsed)
            img = builder.get_image()
            print(f"after {builder.packet_count()} pkt(s): image shape={img.shape}, "
                  f"red_rows_used={np.count_nonzero(img[:, :, 0].sum(axis=1))}, "
                  f"green_rows_used={np.count_nonzero(img[:, :, 1].sum(axis=1))}, "
                  f"at_checkpoint={builder.is_at_checkpoint()}")

    final_img = builder.get_image()
    assert final_img.shape == (P_PACKETS, Q_FEATURES, 3)
    # packet 0 (forward) -> row 0 red channel nonzero
    assert final_img[0, :, 0].sum() > 0
    # packet 1 (backward) -> row 1 green channel nonzero
    assert final_img[1, :, 1].sum() > 0
    # row 2 onward (never received) -> fully zero
    assert final_img[3:, :, :].sum() == 0
    print("image_builder.py self-test passed.")
