import time
import numpy as np

class MicroFlowExtractor:
    def __init__(self):
        # Dictionary tracking unique network micro-flows
        self.flows = {}

    def process_packet(self, pkt_info):
        """
        Groups packets into a bidirectional flow key and updates running statistics.
        """
        current_time = time.time()
        
        # Unpack structural identifiers
        src_ip = pkt_info['src_ip']
        dst_ip = pkt_info['dst_ip']
        src_port = pkt_info['src_port']
        dst_port = pkt_info['dst_port']
        protocol = pkt_info['protocol']
        packet_size = pkt_info['size']
        flags = pkt_info['flags']

        # Normalize direction so request and response packets of the
        # same conversation map to the same flow_key, regardless of
        # which side sent this particular packet.
        if (src_ip, src_port) <= (dst_ip, dst_port):
            flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            is_forward = True
        else:
            flow_key = (dst_ip, src_ip, dst_port, src_port, protocol)
            is_forward = False 
        
        #print(f"[DEBUG] key={flow_key} exists={flow_key in self.flows}")  # TEMPORARY DEBUG LINE

        if flow_key not in self.flows:
            # Initialize a new flow window
            self.flows[flow_key] = {
                'start_time': current_time,
                'last_time': current_time,
                'packet_count': 1,
                'byte_count': packet_size,
                'flags_count': 1 if any(f in flags for f in ['S', 'A', 'F', 'R']) else 0,
                'protocol': protocol,
                'src_port': src_port if is_forward else dst_port,
                'dst_port': dst_port if is_forward else src_port
            }
        else:
            # Update an existing flow window metrics
            flow = self.flows[flow_key]

    def get_and_flush_flows(self, window_expiry=5.0):
        """
        Aggregates metrics matching the ML model feature layout 
        and removes expired flows from memory.
        """
        current_time = time.time()
        extracted_features = []
        keys_to_remove = []

        for key, flow in self.flows.items():
            duration = flow['last_time'] - flow['start_time']
            if duration <= 0:
                duration = 0.001 # Prevent zero division errors
            
            # Formulate the feature dict exactly mapping to our ML shape
            features = {
                'src_ip': key[0],
                'dst_ip': key[1],
                'Flow Duration': int(duration * 1000000), # Convert seconds to microseconds
                'Packet Count': flow['packet_count'],
                'Bytes/sec': float(flow['byte_count'] / duration),
                'Packets/sec': float(flow['packet_count'] / duration),
                'Protocol Type': float(flow['protocol']),
                'SYN/ACK/FIN/RST Counts': float(flow['flags_count']),
                'Source Port': float(flow['src_port']),
                'Destination Port': float(flow['dst_port'])
            }
            
            # If the flow has staleness or reached calculation window threshold
            if (current_time - flow['last_time']) >= window_expiry or duration >= window_expiry:
                extracted_features.append(features)
                keys_to_remove.append(key)

        # Clear processed keys to avoid memory leaks
        for key in keys_to_remove:
            del self.flows[key]

        return extracted_features