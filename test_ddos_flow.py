import dpkt
import socket
import datetime

PCAP = r"dataset\pcap\pcap\Friday-WorkingHours.pcap"

TARGETS = {
    ("172.16.0.1", "192.168.10.50", 51684, 80, 6),
("192.168.10.50", "172.16.0.1", 80, 51684, 6),
}

with open(PCAP, "rb") as f:
    reader = dpkt.pcapng.Reader(f)

    for ts, buf in reader:
        try:
            eth = dpkt.ethernet.Ethernet(buf)

            if not isinstance(eth.data, dpkt.ip.IP):
                continue

            ip = eth.data

            if ip.p != dpkt.ip.IP_PROTO_TCP:
                continue

            tcp = ip.data

            src = socket.inet_ntoa(ip.src)
            dst = socket.inet_ntoa(ip.dst)

            key = (src, dst, tcp.sport, tcp.dport, ip.p)

            if key in TARGETS:
                print("FOUND!")
                print("Time:", datetime.datetime.fromtimestamp(ts))
                print("Flow:", key)
                break

        except Exception:
            continue
    else:
        print("NOT FOUND")
