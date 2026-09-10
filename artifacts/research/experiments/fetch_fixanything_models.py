"""Resume FixAnything artifact downloads through the local Motrix RPC service."""

import argparse
import fnmatch
import json
import urllib.request
from pathlib import Path


def rpc(endpoint, method, params):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {"jsonrpc": "2.0", "id": "fixanything", "method": method, "params": params}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    response = json.load(urllib.request.urlopen(request, timeout=30))
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fixanything"))
    parser.add_argument("--rpc", default="http://127.0.0.1:16800/jsonrpc")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    manifest_path = root / "download-manifest.json"
    if args.status:
        records = json.loads(manifest_path.read_text())
        results = [rpc(args.rpc, "aria2.tellStatus", [r["gid"]]) for r in records]
        print(
            json.dumps(
                {
                    "complete": sum(r["status"] == "complete" for r in results),
                    "files": len(records),
                    "completed_bytes": sum(int(r["completedLength"]) for r in results),
                    "expected_bytes": sum(r["bytes"] for r in records),
                    "bytes_per_second": sum(
                        int(r.get("downloadSpeed", 0)) for r in results
                    ),
                    "errors": [
                        r.get("errorMessage") for r in results if r["status"] == "error"
                    ],
                }
            )
        )
        if all(r["status"] == "complete" for r in results):
            # Only Motrix completion after checksum verification produces this receipt.
            receipt = {
                "files": [
                    {**entry, "motrix_status": result["status"]}
                    for entry, result in zip(records, results, strict=True)
                ]
            }
            (root / "download-completion.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        return
    if manifest_path.exists():
        raise ValueError(
            "Existing download manifest found; use --status to monitor its resumable tasks"
        )
    patterns = [
        "diffusion_pytorch_model*",
        "models_t5*",
        "Wan2.1_VAE.pth",
        "models_clip*",
        "google/*",
        "config.json",
        "LICENSE*",
    ]
    records = []
    for repository in ["Wan-AI/Wan2.1-I2V-14B-480P", "kvuong2711/fix-anything"]:
        entries = json.load(
            urllib.request.urlopen(
                f"https://huggingface.co/api/models/{repository}/tree/main?recursive=true&expand=false",
                timeout=30,
            )
        )
        for entry in entries:
            name = entry["path"]
            if entry["type"] != "file":
                continue
            base = repository.startswith("Wan-")
            if (
                base and not any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
            ) or (not base and name != "fixanything_lora.safetensors"):
                continue
            path = root / repository / name if base else root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            checksum = entry.get("lfs", {}).get("oid")
            options = {
                "dir": str(path.parent),
                "out": path.name,
                "continue": "true",
                "allow-overwrite": "false",
                "auto-file-renaming": "false",
                "split": "4",
                "max-connection-per-server": "4",
            }
            if checksum:
                options["checksum"] = "sha-256=" + checksum
            gid = rpc(
                args.rpc,
                "aria2.addUri",
                [[f"https://huggingface.co/{repository}/resolve/main/{name}"], options],
            )
            records.append(
                {
                    "repo": repository,
                    "path": str(path),
                    "bytes": entry["size"],
                    "sha256": checksum,
                    "gid": gid,
                }
            )
    manifest_path.write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": len(records),
                "bytes": sum(r["bytes"] for r in records),
                "manifest": str(manifest_path),
            }
        )
    )


if __name__ == "__main__":
    main()
