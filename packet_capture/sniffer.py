import threading
import time

from scapy.all import sniff, IP, IPv6, TCP, UDP, get_if_addr

from feature_extraction.packet_parser import parse_packet
from feature_extraction.image_builder import FlowImageBuilder


class LivePacketSniffer:
    """
    Live packet capture engine for the CNN-based NIDS.

    Flow:
        Scapy packet
            ↓
        parse_packet()
            ↓
        ParsedPacket
            ↓
        FlowImageBuilder
            ↓
        callback(image, metadata)

    Only IPv4 TCP/UDP packets are sent through the CNN pipeline because
    packet_parser.py currently supports IPv4 TCP/UDP traffic.
    """

    # Remove a flow builder if no packet has arrived for this long.
    FLOW_TIMEOUT = 5.0

    def __init__(self, interface=None, callback=None):
        # Validate interface if a specific one is supplied.
        if interface and interface != "Default":
            if get_if_addr(interface) == "0.0.0.0":
                raise ValueError(f"Invalid network interface: {interface}")

        self.interface = interface
        self.callback = callback

        self.flow_builders = {}
        self.flow_last_seen = {}
        self.predicted_checkpoints = {}

        self.packet_count = 0
        self.flow_count = 0

        self.running = False
        self.thread = None
        self.cleanup_thread = None

        self.lock = threading.Lock()

    def _packet_handler(self, packet):
        """
        Receives one packet from Scapy and sends it through the CNN
        packet-parser/image-builder pipeline.
        """
        try:
            # The CNN packet parser currently supports IPv4 TCP/UDP.
            if IP not in packet:
                return

            if TCP not in packet and UDP not in packet:
                return

            # Convert the Scapy packet into the raw Ethernet bytes expected
            # by packet_parser.py.
            raw_bytes = bytes(packet)

            # Scapy provides the packet capture timestamp.
            timestamp = float(packet.time)

            # Convert raw packet → ParsedPacket.
            parsed = parse_packet(raw_bytes, timestamp)

            if parsed is None:
                return

            flow_key = parsed.flow_key
            current_time = time.time()

            with self.lock:

                # Create a builder for a new flow.
                if flow_key not in self.flow_builders:
                    self.flow_builders[flow_key] = FlowImageBuilder()
                    self.predicted_checkpoints[flow_key] = set()
                    self.flow_count += 1

                builder = self.flow_builders[flow_key]
                self.packet_count += 1

                # Add this packet to the flow's 9-packet image.
                builder.add_packet(parsed)

                # Update activity timestamp.
                self.flow_last_seen[flow_key] = current_time

                packet_count = builder.packet_count()

                # Copy metadata needed by the dashboard.
                flow_metadata = {
    "src_ip": parsed.src_ip,
    "dst_ip": parsed.dst_ip,
    "src_port": int(parsed.src_port),
    "dst_port": int(parsed.dst_port),
    "protocol": int(parsed.protocol),
    "packet_count": packet_count,
    "flow_key": flow_key,

    # Real live-system statistics
    "total_packets_processed": self.packet_count,
    "total_flows_analyzed": self.flow_count,
}

                # Only trigger prediction at the intended checkpoints.
                predicted = self.predicted_checkpoints[flow_key]

                should_predict = (
                    builder.is_at_checkpoint()
                    and packet_count not in predicted
                )

                if should_predict:
                    predicted.add(packet_count)
                    image = builder.get_image()

            # Do not call the application callback while holding the lock.
            if should_predict and self.callback:
                try:
                    self.callback(image, flow_metadata)
                except Exception as e:
                    print(
                        f"[-] Error in CNN pipeline callback: {e}"
                    )

        except Exception as e:
            print(f"[!] Packet parsing error: {e}")

    def _sniff_loop(self):
        """
        Main Scapy packet-capture loop.
        """
        print(
            f"[*] Sniffer thread starting monitoring on interface: "
            f"{self.interface or 'Default'}"
        )

        try:
            while self.running:
                sniff(
                    iface=self.interface,
                    prn=self._packet_handler,
                    store=0,
                    timeout=1.0,
                )

        except PermissionError:
            print(
                "[!] Permission denied: Run application with "
                "administrator privileges."
            )

        except Exception as e:
            print(f"[!] Sniffer core error loop: {e}")

    def _cleanup_loop(self):
        """
        Removes inactive flow builders so a long-running NIDS does not
        retain completed/stale flows forever.
        """
        while self.running:
            time.sleep(1.0)

            current_time = time.time()

            with self.lock:
                expired_flows = [
                    flow_key
                    for flow_key, last_seen in self.flow_last_seen.items()
                    if current_time - last_seen >= self.FLOW_TIMEOUT
                ]

                for flow_key in expired_flows:
                    self.flow_builders.pop(flow_key, None)
                    self.flow_last_seen.pop(flow_key, None)
                    self.predicted_checkpoints.pop(flow_key, None)

    def start(self):
        """
        Start packet capture and flow cleanup workers.
        """
        if self.running:
            print("[!] Sniffer engine is already operational.")
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._sniff_loop,
            daemon=True,
        )

        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
        )

        self.thread.start()
        self.cleanup_thread.start()

        time.sleep(0.1)

        if (
            not self.thread.is_alive()
            or not self.cleanup_thread.is_alive()
        ):
            print(
                "[!] Warning: One or more background engine threads "
                "failed initialization checks."
            )
        else:
            print(
                "[+] Network capture and CNN flow engines "
                "successfully initialized."
            )

    def stop(self):
        """
        Stop packet capture and cleanup workers.
        """
        self.running = False

        if self.thread:
            self.thread.join(timeout=2.0)

        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=2.0)

        print("[*] Sniffer sub-engine spun down safely.")