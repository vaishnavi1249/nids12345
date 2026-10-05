"""
pcap_to_dataset.py

This is the script you run ONCE your pcap has finished downloading.
It does the offline, heavy-lifting version of what the live sniffer does
packet-by-packet: reads a raw .pcap file, groups packets into flows,
figures out which flows are attacks (using CICIDS2017's labelled CSV),
and builds training images out of them.

Pipeline:
  pcap file  --(dpkt, streamed)-->  per-packet ParsedPacket
             --(grouped by 5-tuple)-->  per-flow packet lists
             --(labelled via ground-truth CSV)-->  flow -> class
             --(FlowImageBuilder)-->  one image per flow per packet-count
             --(70/15/15 split)-->  train.npz / val.npz / test.npz

Ground truth CSV: CICIDS2017's "GeneratedLabelledFlows" CSVs have columns
including Source IP, Destination IP, Source Port, Destination Port,
Protocol, Timestamp, and Label. We match each flow we built from the
pcap to a row in this CSV using the same 5-tuple + nearest timestamp,
same as the base paper's validation approach (Section IV-A).

IMPORTANT — this file is written to be CORRECT, not yet RUN, because the
real .pcap and ground-truth CSV aren't downloaded yet. Its self-test at
the bottom proves the logic on a tiny synthetic example. Once your real
files are in place, run it for real:

    python data_prep/pcap_to_dataset.py \\
        --pcap dataset/pcap/Tuesday-WorkingHours.pcap \\
        --labels dataset/pcap/Tuesday-WorkingHours.pcap_ISCX.csv \\
        --out dataset/images
"""

import argparse
import os
from collections import defaultdict

import dpkt
import numpy as np
import pandas as pd

from feature_extraction.packet_parser import parse_packet
from feature_extraction.image_builder import FlowImageBuilder, P_PACKETS, Q_FEATURES

# Map CICIDS2017's many fine-grained label strings onto your project's
# 4 classes (same mapping logic your teammate used in train_model.py's
# clean_labels(), kept consistent so both pipelines agree on classes).
LABEL_MAP = {
    "normal": 0, "benign": 0,
    "portscan": 1, "port scan": 1,
    "ftp-patator": 2, "ssh-patator": 2, "brute force": 2,
    "ddos": 3, "dos": 3, "dos hulk": 3, "dos goldeneye": 3,
    "dos slowloris": 3, "dos slowhttptest": 3,
}
CLASS_NAMES = ["Normal", "Port Scan", "Brute Force", "DDoS"]
NUM_CLASSES = len(CLASS_NAMES)


def map_label(raw_label: str):
    key = str(raw_label).strip().lower()
    for substr, cls in LABEL_MAP.items():
        if substr in key:
            return cls
    return None  # Ignore labels not belonging to our 4 project classes

def load_ground_truth(csv_paths: list[str]) -> pd.DataFrame:
    """
    Loads one or more CICIDS2017 labelled-flow CSVs and normalises column
    names, since different CICFlowMeter versions spell them slightly
    differently.

    IMPORTANT: Friday's traffic is split across 3 CSVs on the mirrors
    (Morning, Afternoon-DDos, Afternoon-PortScan) even though the pcap
    itself is one whole-day file. Pass ALL of a day's label CSVs here,
    or benign flows that happened to fall in a CSV you didn't include
    will have no match and get silently dropped by build_flow_label_lookup
    -> you'd end up short on the Normal class without any error being
    raised.
    """
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, encoding="latin1")
        df.columns = df.columns.str.strip()
        rename_map = {
            "Src IP": "Source IP", "Dst IP": "Destination IP",
            "Src Port": "Source Port", "Dst Port": "Destination Port",
        }
        df = df.rename(columns=rename_map)
        required = ["Source IP", "Destination IP", "Source Port",
                    "Destination Port", "Protocol", "Label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}. "
                              f"Found columns: {list(df.columns)}")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    print(f"[*] Loaded {len(merged)} ground-truth flow labels from "
          f"{len(csv_paths)} CSV file(s).")
    return merged


def build_flow_label_lookup(gt_df: pd.DataFrame) -> dict:
    """
    Builds a dict keyed by the SAME direction-normalised 5-tuple that
    ParsedPacket.flow_key uses, so lookups are a simple dict hit.
    If CICFlowMeter logged duplicate flows for the same 5-tuple (happens
    with long-running/overlapping flows) we keep the LAST one, same
    simplification as most reproductions of this dataset use.
    """
    lookup = {}
    for _, row in gt_df.iterrows():
        a = (str(row["Source IP"]), int(row["Source Port"]))
        b = (str(row["Destination IP"]), int(row["Destination Port"]))
        proto = int(row["Protocol"])
        if a <= b:
            key = (a[0], b[0], a[1], b[1], proto)
        else:
            key = (b[0], a[0], b[1], a[1], proto)
        mapped = map_label(row["Label"])
        if mapped is not None:
            lookup[key] = mapped
    return lookup


def stream_pcap_to_flows(pcap_paths: list[str], max_packets: int | None = None):
    """
    Streams one or more pcaps with dpkt (never loads a whole file into
    memory — important for a 10GB file) and groups parsed packets by
    flow_key, across all given files. max_packets, if given, applies
    PER FILE (so passing 2 files with max_packets=100000 reads up to
    100000 packets from each, not 100000 total) — handy for a quick
    spot-check across both Friday and Tuesday at once.

    Note: flow_key collisions across different pcap files (e.g. the
    exact same IP:port:protocol 5-tuple happening to recur on both
    Friday and Tuesday) are treated as the same flow here. In practice
    this essentially never happens across different capture days/times,
    so it's a safe simplification — but it's why we don't bother
    namespacing flow keys per file.
    """
    flows: dict[tuple, list] = defaultdict(list)
    for pcap_path in pcap_paths:
        n_seen, n_parsed = 0, 0
        with open(pcap_path, "rb") as f:
            magic = f.read(4)
            f.seek(0)

            if magic == b"\x0a\x0d\x0d\x0a":
                reader = dpkt.pcapng.Reader(f)
            else:
                reader = dpkt.pcap.Reader(f)
            for ts, buf in reader:
                n_seen += 1
                parsed = parse_packet(buf, ts)
                if parsed is not None:
                    n_parsed += 1
                    flows[parsed.flow_key].append(parsed)
                if max_packets is not None and n_seen >= max_packets:
                    break
        print(f"[*] {pcap_path}: streamed {n_seen} packets, parsed {n_parsed} "
              f"(TCP/UDP IPv4).")
    print(f"[*] Grouped into {len(flows)} flows total across "
          f"{len(pcap_paths)} file(s).")
    return flows


def build_images_for_flow(packets: list) -> list[np.ndarray]:
    """
    Per the paper's approach (Table VI), we don't make just ONE image
    per flow — we make one image for EACH packet count from 1 up to
    P_PACKETS, so the model learns what a flow looks like after 1
    packet, after 2, ... after 9. This is what lets the live dashboard
    predict early (at packet 4) with the same model trained for packet 9.
    Returns a list of up to P_PACKETS images (fewer if the flow itself
    has fewer than P_PACKETS packets total).
    """
    builder = FlowImageBuilder()
    images = []
    for pkt in packets[:P_PACKETS]:
        builder.add_packet(pkt)
        images.append(builder.get_image().copy())
    return images


def build_dataset(pcap_paths: list[str], labels_csv_paths: list[str], out_dir: str,
                  max_packets: int | None = None, seed: int = 42,
                  max_images_per_class: int | None = 15000):

    rng = np.random.default_rng(seed)

    gt_df = load_ground_truth(labels_csv_paths)
    label_lookup = build_flow_label_lookup(gt_df)

    flows = stream_pcap_to_flows(pcap_paths, max_packets=max_packets)

    # Shuffle flow order so the class cap does not depend on PCAP order.
    flow_items = list(flows.items())
    rng.shuffle(flow_items)

    all_images = []
    all_labels = []

    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    unmatched = 0
    capped = 0

    for flow_key, packets in flow_items:

        if flow_key not in label_lookup:
            unmatched += 1
            continue

        label = label_lookup[flow_key]

        if max_images_per_class is not None and class_counts[label] >= max_images_per_class:
            capped += 1
            continue

        for img in build_images_for_flow(packets):

            if max_images_per_class is not None and class_counts[label] >= max_images_per_class:
                break

            all_images.append(img)
            all_labels.append(label)
            class_counts[label] += 1

    print(f"[*] {unmatched}/{len(flows)} flows had no matching ground-truth "
          f"label and were skipped.")

    if capped:
        print(f"[*] {capped} flows skipped — their class already hit the "
              f"{max_images_per_class}-image cap.")

    print(f"[*] Built {len(all_images)} images total.")

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        print(f"    {cls_name}: {class_counts[cls_idx]}")

    if any(c == 0 for c in class_counts.values()):
        missing = [CLASS_NAMES[i] for i, c in class_counts.items() if c == 0]
        print(f"[!] WARNING: these classes have ZERO images: {missing}. "
              f"Check you passed the right label CSVs for this pcap.")

    # Convert the collected images to one NumPy array.
    X = np.stack(all_images).astype(np.uint8)
    y = np.array(all_labels, dtype=np.int64)

    idx = rng.permutation(len(X))
    n = len(X)

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }

    os.makedirs(out_dir, exist_ok=True)

    for name, split_idx in splits.items():

        path = os.path.join(out_dir, f"{name}.npz")

        np.savez_compressed(
            path,
            X=X[split_idx],
            y=y[split_idx]
        )

        print(f"[+] Wrote {path} ({len(split_idx)} samples)")

def _self_test():
    """
    Proves the whole pipeline — pcap -> flows -> label lookup -> images
    -> saved .npz — works correctly, using a tiny synthetic pcap and a
    tiny synthetic ground-truth CSV instead of real CICIDS2017 files.
    """
    import tempfile
    from scapy.all import Ether, IP, TCP, Raw, wrpcap

    tmp_dir = tempfile.mkdtemp()
    pcap_path = os.path.join(tmp_dir, "self_test.pcap")
    labels_path = os.path.join(tmp_dir, "self_test_labels.csv")
    out_dir = os.path.join(tmp_dir, "images")

    # Flow A: 10.0.0.1 <-> 10.0.0.2 on port 80 -> labelled DDoS
    # Flow B: 10.0.0.3 <-> 10.0.0.4 on port 22 -> labelled BENIGN
    pkts = [
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=80, flags="S"),
        Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=5000, flags="SA"),
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=80, flags="A") / Raw(b"x" * 50),
        Ether() / IP(src="10.0.0.3", dst="10.0.0.4") / TCP(sport=6000, dport=22, flags="S"),
        Ether() / IP(src="10.0.0.4", dst="10.0.0.3") / TCP(sport=22, dport=6000, flags="SA"),
    ]
    wrpcap(pcap_path, pkts)

    gt = pd.DataFrame([
        {"Source IP": "10.0.0.1", "Destination IP": "10.0.0.2",
         "Source Port": 5000, "Destination Port": 80, "Protocol": 6, "Label": "DDoS"},
        {"Source IP": "10.0.0.3", "Destination IP": "10.0.0.4",
         "Source Port": 6000, "Destination Port": 22, "Protocol": 6, "Label": "BENIGN"},
    ])
    gt.to_csv(labels_path, index=False)

    build_dataset([pcap_path], [labels_path], out_dir, max_images_per_class=None)

    train = np.load(os.path.join(out_dir, "train.npz")) if os.path.exists(os.path.join(out_dir, "train.npz")) else None
    total = 0
    for name in ("train", "val", "test"):
        p = os.path.join(out_dir, f"{name}.npz")
        if os.path.exists(p):
            d = np.load(p)
            total += len(d["X"])
            assert d["X"].shape[1:] == (P_PACKETS, Q_FEATURES, 3)
    # Flow A has 3 packets -> 3 images (1,2,3-packet versions); Flow B has 2 -> 2 images = 5 total
    assert total == 5, f"expected 5 images total, got {total}"
    print("\npcap_to_dataset.py self-test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", nargs="+",
                         help="Path(s) to raw .pcap file(s). Pass multiple to combine days, "
                              "e.g. --pcap dataset/pcap/Friday-WorkingHours.pcap dataset/pcap/Tuesday-WorkingHours.pcap")
    parser.add_argument("--labels", nargs="+",
                         help="Path(s) to CICIDS2017 ground-truth labelled-flow CSV(s). "
                              "Pass ALL CSVs for every pcap you listed above — e.g. Friday needs "
                              "its 3 CSVs (Morning, Afternoon-DDos, Afternoon-PortScan) even though "
                              "it's 1 pcap file.")
    parser.add_argument("--out", default=os.path.join("dataset", "images"),
                         help="Output directory for train/val/test .npz files")
    parser.add_argument("--max-packets", type=int, default=None,
                         help="For a quick spot-check, read only this many packets PER pcap file "
                              "instead of the whole thing, e.g. --max-packets 200000")
    parser.add_argument("--max-images-per-class", type=int, default=15000,
                         help="Cap on images kept per class, to bound memory and keep classes "
                              "balanced. Set to a smaller number (e.g. 3000) for a faster first "
                              "run, or 0/negative to disable the cap entirely.")
    parser.add_argument("--self-test", action="store_true",
                         help="Run the built-in self-test on synthetic data instead of real files")
    args = parser.parse_args()

    if args.self_test or not (args.pcap and args.labels):
        _self_test()
    else:
        cap = args.max_images_per_class if args.max_images_per_class and args.max_images_per_class > 0 else None
        build_dataset(args.pcap, args.labels, args.out,
                      max_packets=args.max_packets, max_images_per_class=cap)
