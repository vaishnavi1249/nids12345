"""
packet_parser.py

Turns one raw network packet into:
  1. a fixed-length "packet feature vector" (what the CNN eventually sees)
  2. a small dict of "auxiliary" info (used only to group packets into
     flows and to tell forward/backward apart — never shown to the model)

WHY we remove IPs, ports, and header options (per the base paper, Section III-A):
  If we leave IP addresses or port numbers in, the CNN starts memorising
  "this IP = attacker" instead of learning the actual attack *pattern*.
  That's exactly the "existing systems get confused on a new network"
  problem your problem statement calls out (PPT slide 3) — so stripping
  these out is *why* this approach is supposed to generalise to traffic
  it has never seen.

Feature vector layout (N = 1486 bytes total):
  byte 0        -> delta time (time since the previous packet in this flow,
                   scaled into 0-255 and clipped)
  bytes 1..     -> remaining IP header bytes (minus version/DSCP/proto/IPs),
                   remaining TCP/UDP header bytes (minus ports),
                   then payload bytes, in order
  rest          -> zero-padded if the packet is short

This mirrors Fig. 2/3/Algorithm 1 of the base paper. A few simplifications
were made to keep this buildable in a college-project timeframe:
  - IP/TCP *options* are simply skipped (not reconstructed byte-for-byte),
    since most CICIDS2017 traffic doesn't use them anyway.
  - delta time occupies 1 byte here (paper doesn't pin down an exact
    byte-width for it) instead of a wider encoding.
"""

import dpkt
import numpy as np

# Total length of the per-packet feature vector fed to the image builder.
N_FEATURES = 1486

# Bytes removed from a standard (no-options) IPv4 header once version,
# DSCP/ToS, protocol, source IP and destination IP are stripped out.
# A standard IPv4 header is 20 bytes: we keep TTL, flags/frag, checksum,
# total length, identification -> everything except the 7 bytes named above.
_IP_STRIP_FIELDS = {"v", "tos", "p", "src", "dst"}


class ParsedPacket:
    """Container for one packet's parsed result."""

    __slots__ = ("feature_vector", "src_ip", "dst_ip", "src_port",
                 "dst_port", "protocol", "timestamp")

    def __init__(self, feature_vector, src_ip, dst_ip, src_port,
                 dst_port, protocol, timestamp):
        self.feature_vector = feature_vector
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.timestamp = timestamp

    @property
    def flow_key(self):
        """
        Direction-normalised 5-tuple so request and response packets of the
        same conversation map to the same flow, regardless of who sent this
        particular packet. Mirrors sniffer.py's existing logic.
        """
        a = (self.src_ip, self.src_port)
        b = (self.dst_ip, self.dst_port)
        if a <= b:
            return (self.src_ip, self.dst_ip, self.src_port,
                     self.dst_port, self.protocol)
        return (self.dst_ip, self.src_ip, self.dst_port,
                 self.src_port, self.protocol)

    @property
    def is_forward(self):
        """True if this packet is travelling in the flow_key's 'forward'
        (first-named source) direction."""
        a = (self.src_ip, self.src_port)
        b = (self.dst_ip, self.dst_port)
        return a <= b


def _ip_header_bytes_minus_bias(ip_pkt: dpkt.ip.IP) -> bytes:
    """
    Returns the fixed (non-options) IPv4 header bytes with version, DSCP,
    protocol, source IP and destination IP zeroed out of the byte stream
    (we don't literally zero them — we just never include those byte
    ranges). IP options (if any) are dropped entirely.
    """
    # dpkt gives us hlen in 32-bit words; the fixed part we care about
    # (ignoring options) is always the first 20 bytes of a v4 header.
    raw = bytes(ip_pkt)
    fixed = raw[:20]
    # Layout of a standard IPv4 header (byte offsets):
    #   0:       version(4b)+IHL(4b)
    #   1:       DSCP/ToS
    #   2-3:     total length        <- keep
    #   4-5:     identification      <- keep
    #   6-7:     flags + frag offset <- keep
    #   8:       TTL                 <- keep
    #   9:       protocol
    #   10-11:   header checksum     <- keep
    #   12-15:   source IP
    #   16-19:   destination IP
    keep = fixed[2:9] + fixed[10:12]
    return keep  # 9 bytes kept from the fixed IP header


def _l4_header_bytes_minus_ports(transport_pkt, protocol: int) -> bytes:
    """
    For TCP: fixed 20-byte header minus the 4 port bytes, options dropped.
    For UDP: fixed 8-byte header minus the 4 port bytes.
    """
    raw = bytes(transport_pkt)
    if protocol == dpkt.ip.IP_PROTO_TCP:
        fixed = raw[:20]
        # 0-1 src port, 2-3 dst port -> drop; keep seq/ack/flags/window/etc.
        return fixed[4:20]
    elif protocol == dpkt.ip.IP_PROTO_UDP:
        fixed = raw[:8]
        # 0-1 src port, 2-3 dst port -> drop; keep length + checksum
        return fixed[4:8]
    return b""


def parse_packet(raw_bytes: bytes, timestamp: float):
    """
    Parse one raw Ethernet frame (as captured by scapy/dpkt) into a
    ParsedPacket, or return None if it isn't IPv4 TCP/UDP (nothing for
    this NIDS to analyse — mirrors sniffer.py's existing filtering).
    """
    try:
        eth = dpkt.ethernet.Ethernet(raw_bytes)
        if not isinstance(eth.data, dpkt.ip.IP):
            return None
        ip_pkt = eth.data
        protocol = ip_pkt.p

        if protocol == dpkt.ip.IP_PROTO_TCP:
            transport = ip_pkt.data
            src_port, dst_port = transport.sport, transport.dport
            payload = bytes(transport.data)
        elif protocol == dpkt.ip.IP_PROTO_UDP:
            transport = ip_pkt.data
            src_port, dst_port = transport.sport, transport.dport
            payload = bytes(transport.data)
        else:
            return None  # only TCP/UDP, same as sniffer.py

        header_bytes = (
            _ip_header_bytes_minus_bias(ip_pkt)
            + _l4_header_bytes_minus_ports(transport, protocol)
        )

        # bytes 1..N-1 of the feature vector: header remainder + payload,
        # truncated/zero-padded to fit (N_FEATURES - 1 bytes reserved for
        # delta time at index 0).
        body = header_bytes + payload
        body = body[: N_FEATURES - 1]

        vec = np.zeros(N_FEATURES, dtype=np.uint8)
        vec[1 : 1 + len(body)] = np.frombuffer(body, dtype=np.uint8)
        # index 0 (delta time) is filled in later by the image builder,
        # which is the only thing that knows the previous packet's time.

        import socket
        src_ip = socket.inet_ntoa(ip_pkt.src)
        dst_ip = socket.inet_ntoa(ip_pkt.dst)

        return ParsedPacket(
            feature_vector=vec,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            timestamp=timestamp,
        )
    except Exception:
        # Malformed/truncated packet — skip it rather than crash the parser.
        return None


if __name__ == "__main__":
    # Quick self-test using a handful of synthetic packets (no real pcap
    # needed) just to prove the byte-stripping logic runs end-to-end.
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    import tempfile, os

    # Ether() wrapper matters: scapy writes a "raw IP" linklayer (no
    # Ethernet header) if you skip it, which dpkt would then misparse.
    # Real captures off a NIC (what sniffer.py sees) always include it.
    pkts = [
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5555, dport=80, flags="S"),
        Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=5555, flags="SA"),
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5555, dport=80, flags="A") / Raw(b"GET / HTTP/1.1\r\n\r\n"),
    ]
    tmp_path = os.path.join(tempfile.gettempdir(), "self_test.pcap")
    wrpcap(tmp_path, pkts)

    with open(tmp_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for i, (ts, buf) in enumerate(reader):
            p = parse_packet(buf, ts)
            assert p is not None, f"packet {i} failed to parse"
            print(f"pkt {i}: flow_key={p.flow_key} forward={p.is_forward} "
                  f"vec.shape={p.feature_vector.shape} "
                  f"nonzero_bytes={np.count_nonzero(p.feature_vector)}")
    print("packet_parser.py self-test passed.")
