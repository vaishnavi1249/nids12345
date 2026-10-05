"""
download_dataset.py

Run this. If your connection drops, just run it again — hf_hub_download
resumes from the last completed chunk instead of starting over.

Downloads, in order (smallest/most useful first):
  1. Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv     (~77 MB)  -> labels for DDoS
  2. Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv (~77 MB)  -> labels for Port Scan
  3. Friday-WorkingHours.pcap                             (~8.8 GB) -> raw traffic, covers both above
  4. Tuesday-WorkingHours.pcap_ISCX.csv                    (~135 MB) -> labels for Brute Force
  5. Tuesday-WorkingHours.pcap                             (~10 GB)  -> raw traffic for Brute Force

Everything lands directly in NIDS-main/dataset/pcap/ — run this script
from inside your NIDS-main folder.
"""

from huggingface_hub import hf_hub_download

DEST = "dataset/pcap"

DOWNLOADS = [
    # (repo_id, repo_type, filename)
    ("c01dsnap/CIC-IDS2017", "dataset", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    ("c01dsnap/CIC-IDS2017", "dataset", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("bvsam/cic-ids-2017",   "dataset", "pcap/Friday-WorkingHours.pcap"),
    ("c01dsnap/CIC-IDS2017", "dataset", "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("bvsam/cic-ids-2017",   "dataset", "pcap/Tuesday-WorkingHours.pcap"),
]

for i, (repo_id, repo_type, filename) in enumerate(DOWNLOADS, 1):
    print(f"\n[{i}/{len(DOWNLOADS)}] Downloading {filename} from {repo_id} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=DEST,
    )
    print(f"    -> saved to {path}")

print("\nAll done. Check NIDS-main/dataset/pcap/ for the files.")
