import threading
import time
from scapy.all import sniff, IP, IPv6 , TCP, UDP, get_if_addr
from feature_extraction.extractor import MicroFlowExtractor

class LivePacketSniffer:
    def __init__(self, interface=None, callback=None):
        # Validate interface before creating object if a specific one is targeted
        if interface and interface != "Default":
                if get_if_addr(interface) == '0.0.0.0' :
                    raise ValueError(f"Invalid network interface: {interface}")
        
        self.interface = interface
        self.callback = callback 
        self.extractor = MicroFlowExtractor()
        self.running = False
        self.thread = None
        self.agg_thread = None
        self.lock = threading.Lock()

    # def _packet_handler(self, packet):
    #     try:
    #         if IP in packet:
    #             ip_layer = packet[IP]
    #             src_ip = ip_layer.src
    #             dst_ip = ip_layer.dst
    #             protocol = ip_layer.proto
                
    #             src_port = 0
    #             dst_port = 0
    #             flags = ''

    #             if TCP in packet:
    #                 src_port = packet[TCP].sport
    #                 dst_port = packet[TCP].dport
    #                 flags = str(packet[TCP].flags)
    #             elif UDP in packet:
    #                 src_port = packet[UDP].sport
    #                 dst_port = packet[UDP].dport

    #             pkt_info = {
    #                 'src_ip': src_ip,
    #                 'dst_ip': dst_ip,
    #                 'src_port': src_port,
    #                 'dst_port': dst_port,
    #                 'protocol': protocol,
    #                 'size': len(packet),
    #                 'flags': flags
    #             }
                
    #             with self.lock:
    #                 self.extractor.process_packet(pkt_info)
    #     except Exception as e:
    #         print(f"[!] Packet parsing error: {e}")

    def _packet_handler(self, packet):
        try:
            if IP in packet:
                ip_layer = packet[IP]
                protocol = ip_layer.proto
            elif IPv6 in packet:
                ip_layer = packet[IPv6]
                protocol = ip_layer.nh
            else:
                return  # Not IPv4 or IPv6 (e.g. ARP) - nothing for the NIDS to analyze

            src_ip = ip_layer.src
            dst_ip = ip_layer.dst

            src_port = 0
            dst_port = 0
            flags = ''

            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                flags = str(packet[TCP].flags)
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport

            pkt_info = {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'size': len(packet),
                'flags': flags
            }
        
            with self.lock:
                self.extractor.process_packet(pkt_info)
        except Exception as e:
            print(f"[!] Packet parsing error: {e}") 

    def _sniff_loop(self):
        print(f"[*] Sniffer thread starting monitoring on interface: {self.interface or 'Default'}")
        try:
            while self.running:
                sniff(
                    iface=self.interface, 
                    prn=self._packet_handler, 
                    store=0, 
                    timeout=1.0
                )
        except PermissionError:
            print("[!] Permission denied: Run application with sudo/administrator privileges.")
        except Exception as e:
            print(f"[!] Sniffer core error loop: {e}")

    def _aggregation_loop(self):
        while self.running:
            time.sleep(1.0)
            with self.lock:
                completed_flows = self.extractor.get_and_flush_flows(window_expiry=2.0)
            
            if completed_flows and self.callback:
                for flow in completed_flows:
                    try:
                        # Fixed thread exhaustion: Execute directly or pipe to orchestration queue
                        self.callback(flow)
                    except Exception as e:
                        print(f"[-] Error in live pipeline callback processing: {e}")

    def start(self):
        if self.running:
            print("[!] Sniffer engine is already operational.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.thread.start()

        self.agg_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.agg_thread.start()
        
        # Minor adjustment: Give the OS thread manager a brief window to schedule execution status cleanly
        time.sleep(0.1)
        if not self.thread.is_alive() or not self.agg_thread.is_alive():
            print("[!] Warning: One or more background engine threads failed initialization checks.")
        else:
            print("[+] Network capture engines successfully initialized.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.agg_thread:
            self.agg_thread.join(timeout=2.0)
        print("[*] Sniffer sub-engine spun down safely.")