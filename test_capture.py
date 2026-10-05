from scapy.all import sniff, IP, IPv6, TCP

def show(p):
    if IP in p and TCP in p:
        print(f"{p[IP].src}:{p[TCP].sport} -> {p[IP].dst}:{p[TCP].dport} flags={p[TCP].flags}")
    elif IPv6 in p and TCP in p:
        print(f"{p[IPv6].src}:{p[TCP].sport} -> {p[IPv6].dst}:{p[TCP].dport} flags={p[TCP].flags}")

sniff(iface=9, filter='tcp port 443', prn=show, count=20)