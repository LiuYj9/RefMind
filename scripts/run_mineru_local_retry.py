"""One isolated full-PDF MinerU run with local models and persistent live logs.

No RefMind imports, dotenv loads, production index writes, cache changes or model
downloads. Python network connections are limited to the CLI's loopback API.
Only the subprocess tree created by this script is stopped at the deadline.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from prepare_mineru_local_profile import REQUIRED

NETWORK_GUARD = r'''# Isolated MinerU child guard: loopback API allowed; remote sockets denied.
import ipaddress
import socket
def _allowed(host):
    if isinstance(host, bytes): host = host.decode("ascii")
    if str(host).lower() == "localhost": return True
    try: return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError: return False
_connect, _connect_ex, _getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _allowed(address[0]):
        raise OSError("offline_guard: non-loopback network blocked")
    return _connect(self, address)
def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _allowed(address[0]):
        raise OSError("offline_guard: non-loopback network blocked")
    return _connect_ex(self, address)
def getaddrinfo(host, *args, **kwargs):
    if host is not None and not _allowed(host):
        raise OSError("offline_guard: remote DNS blocked")
    return _getaddrinfo(host, *args, **kwargs)
socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo
'''


def write_record(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def stop_owned_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 900:
        raise ValueError("timeout must be within 1..900 seconds")
    root, source, output = args.model_root.resolve(), args.source.resolve(), args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == root or root in output.parents or output == source.parent:
        raise ValueError("Output must be isolated from source and model cache")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; prior evidence is never overwritten")
    missing = [relative for relative in REQUIRED if not (root / relative).is_file() or (root / relative).stat().st_size <= 0]
    if missing:
        raise FileNotFoundError("Missing cached models: " + ", ".join(missing))
    # Fail before starting any subprocess when the output directory is unwritable.
    output.mkdir(parents=True, exist_ok=True)
    config = output / "mineru-local.json"
    write_record(config, {"models-dir": {"pipeline": str(root)}})
    guard = output / "offline-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(NETWORK_GUARD, encoding="utf-8")
    env = os.environ.copy()
    overrides = {"MINERU_MODEL_SOURCE": "local", "MINERU_TOOLS_CONFIG_JSON": str(config),
        "MINERU_DEVICE_MODE": "cpu", "MINERU_FORMULA_CH_SUPPORT": "false",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
    env.update(overrides)
    env["PYTHONPATH"] = str(guard) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # No inherited proxy can turn the CLI's loopback requests into remote calls.
    for key in tuple(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            env.pop(key)
    env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    command = [sys.executable, "-B", "-m", "mineru.cli.client", "-p", str(source),
        "-o", str(output / "parser-output"), "-b", "pipeline", "-m", "auto", "-l", "ch_lite", "-f", "true", "-t", "true"]
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "starting", "source": str(source),
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "parser_version": importlib.metadata.version("mineru"),
        "torch_distribution_version": importlib.metadata.version("torch"), "device": "cpu", "command": command,
        "environment_overrides": overrides, "network_policy": "Python non-loopback sockets and DNS denied; loopback CLI API allowed",
        "model_cache": str(root), "model_cache_check": "required_files_present_not_hash_verified",
        "backend": "pipeline", "method": "auto", "page_scope": "complete_pdf", "timeout_seconds": args.timeout,
        "writes_production_index": False, "loads_original_dotenv": False,
        "stdout_log": str(output / "stdout.log"), "stderr_log": str(output / "stderr.log")}
    record_path = output / "run.json"
    write_record(record_path, record)
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    started = time.perf_counter()
    process = None
    pumps = []
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=output, **kwargs)
        record.update(status="running", pid=process.pid)
        write_record(record_path, record)
        def pump(stream, destination):
            with destination.open("wb") as handle:
                while data := stream.read1(4096):
                    handle.write(data)
                    handle.flush()
        for stream, filename in ((process.stdout, "stdout.log"), (process.stderr, "stderr.log")):
            thread = threading.Thread(target=pump, args=(stream, output / filename), daemon=True)
            thread.start()
            pumps.append(thread)
        last_update = 0.
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed >= args.timeout:
                record["status"] = "timeout"
                stop_owned_tree(process)
                break
            if elapsed - last_update >= 15:
                record["elapsed_seconds"] = round(elapsed, 2)
                write_record(record_path, record)
                print(json.dumps({"status": "running", "elapsed_seconds": record["elapsed_seconds"],
                    "stdout_bytes": (output / "stdout.log").stat().st_size,
                    "stderr_bytes": (output / "stderr.log").stat().st_size}), flush=True)
                last_update = elapsed
            time.sleep(.25)
        if record["status"] != "timeout":
            record["status"] = "success" if process.returncode == 0 else "failed"
        record["returncode"] = process.returncode
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__)
        if process is not None:
            stop_owned_tree(process)
    finally:
        for thread in pumps:
            thread.join(timeout=10)
        record["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["output_files"] = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted((output / "parser-output").rglob("*")) if path.is_file()]
        if record["status"] == "success" and not any(path["path"].endswith("_middle.json") for path in record["output_files"]):
            record.update(status="failed", error_type="NoMiddleJsonOutput")
        write_record(record_path, record)
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"],
        "output_files": len(record["output_files"]), "record": str(record_path)}, ensure_ascii=False), flush=True)
    return 0 if record["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
