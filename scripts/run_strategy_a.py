"""
Strategy A Harness Runner
=========================

Spawns 1 server + 2 client subprocesses that communicate over a real MQTT
broker (mosquitto on localhost:1883) and stream UrbanSound8K samples for
Strategy A (Centralized) comparison.

Prerequisites:
    - mosquitto broker running locally (`brew services start mosquitto`
      or run `mosquitto -v` in another terminal).
    - paho-mqtt installed in .venv.
    - UrbanSound8K downloaded to data/urbansound8k/UrbanSound8K/.
    - data/urbansound8k/cache.npz present (produced by prepare_urbansound8k.py).

Usage:
    # Smoke test — ~5 min wall time
    .venv/bin/python scripts/run_strategy_a.py \\
        --max-samples 100 \\
        --stream-rate-ms 100 \\
        --num-rounds 2 \\
        --buffer-size 50 \\
        --train-trigger-samples 100

    # Real run — ~30-45 min wall time
    .venv/bin/python scripts/run_strategy_a.py \\
        --max-samples 5000 \\
        --stream-rate-ms 500 \\
        --num-rounds 10 \\
        --buffer-size 500 \\
        --train-trigger-samples 1000

Outputs:
    experiments/strategy_a/{timestamp}/
        client_A.json          — client A metrics
        client_B.json          — client B metrics
        server.json            — server round-by-round metrics
        server.log             — server stdout/stderr
        client_A.log           — client A stdout/stderr
        client_B.log           — client B stdout/stderr
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Samples per client")
    p.add_argument("--stream-rate-ms", type=int, default=500,
                   help="Sleep between samples (500ms = real sensor cadence)")
    p.add_argument("--buffer-size", type=int, default=500,
                   help="Client-side batch size before MQTT publish")
    p.add_argument("--num-rounds", type=int, default=10,
                   help="Number of server training rounds to run")
    p.add_argument("--train-trigger-samples", type=int, default=1000,
                   help="Server-side buffer size that triggers a training round")
    p.add_argument("--epochs-per-round", type=int, default=1)
    p.add_argument("--server-ready-timeout", type=float, default=60.0,
                   help="Max seconds to wait for server to subscribe before starting clients")
    p.add_argument("--client-timeout", type=float, default=None,
                   help="Kill clients after N seconds (safety net; default = no timeout)")
    return p.parse_args()


def check_broker(host: str, port: int, timeout: float = 3.0) -> bool:
    """Try a TCP connect to the broker to make sure mosquitto is up."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def spawn(name: str, cmd: list, log_path: Path) -> subprocess.Popen:
    """Spawn a subprocess with stdout+stderr redirected to a log file."""
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    proc._log_fh = log_fh  # keep ref so it doesn't get GC'd
    print(f"  → spawned {name} (pid={proc.pid}), logging to {log_path}")
    return proc


def main() -> int:
    args = parse_args()

    # Preflight
    if not VENV_PYTHON.exists():
        print(f"ERROR: venv python not found at {VENV_PYTHON}", file=sys.stderr)
        return 1

    if not check_broker(args.broker, args.port):
        print(
            f"ERROR: cannot reach MQTT broker at {args.broker}:{args.port}.\n"
            f"Start mosquitto first:\n"
            f"    brew services start mosquitto\n"
            f"  OR\n"
            f"    mosquitto -v  (in another terminal, to see traffic)",
            file=sys.stderr,
        )
        return 1

    # Run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "experiments" / "strategy_a" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Write the invocation for reproducibility
    with open(run_dir / "invocation.txt", "w") as f:
        f.write("argv: " + " ".join(sys.argv) + "\n")
        f.write(f"timestamp: {ts}\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # 1) Server first (so it's subscribed before clients start publishing)
    print("\nStarting server...")
    server_cmd = [
        str(VENV_PYTHON), "-m", "src.experiments.strategy_a_server",
        "--broker", args.broker,
        "--port", str(args.port),
        "--num-rounds", str(args.num_rounds),
        "--train-trigger-samples", str(args.train_trigger_samples),
        "--epochs-per-round", str(args.epochs_per_round),
        "--run-dir", str(run_dir),
    ]
    server_proc = spawn("server", server_cmd, run_dir / "server.log")

    # Wait for server to actually subscribe to the data topic before starting
    # clients — otherwise QoS 1 messages sent before subscription are silently
    # dropped (broker doesn't retain unless retain=True is set).
    ready_marker = "Server ready. Waiting for edge batches"
    server_log = run_dir / "server.log"
    print(f"  waiting up to {args.server_ready_timeout}s for server to become ready...")
    deadline = time.time() + args.server_ready_timeout
    ready = False
    while time.time() < deadline:
        if server_proc.poll() is not None:
            print(f"ERROR: server exited (code {server_proc.returncode})."
                  f" Check {server_log}", file=sys.stderr)
            return 1
        if server_log.exists():
            with open(server_log, "rb") as f:
                if ready_marker.encode() in f.read():
                    ready = True
                    break
        time.sleep(0.5)

    if not ready:
        print(f"ERROR: server did not become ready within "
              f"{args.server_ready_timeout}s. Check {server_log}", file=sys.stderr)
        server_proc.terminate()
        return 1

    print("  server ready ✓")

    # 2) Clients
    print("\nStarting clients...")
    client_procs = []
    for node_id in ["A", "B"]:
        cmd = [
            str(VENV_PYTHON), "-m", "src.experiments.strategy_a_client",
            "--node-id", node_id,
            "--broker", args.broker,
            "--port", str(args.port),
            "--max-samples", str(args.max_samples),
            "--stream-rate-ms", str(args.stream_rate_ms),
            "--buffer-size", str(args.buffer_size),
            "--run-dir", str(run_dir),
        ]
        proc = spawn(f"client-{node_id}", cmd, run_dir / f"client_{node_id}.log")
        client_procs.append((node_id, proc))
        time.sleep(0.5)  # stagger client startup slightly

    # 3) Wait for clients to finish
    print("\nWaiting for clients to finish streaming...")
    start = time.time()
    for node_id, proc in client_procs:
        rc = proc.wait(timeout=args.client_timeout)
        elapsed = time.time() - start
        print(f"  client-{node_id} exited (code={rc}) after {elapsed:.1f}s")

    # 4) Give server a moment to process the final batch, then signal it to stop
    print("\nDraining server (10s)...")
    time.sleep(10.0)
    if server_proc.poll() is None:
        print("  sending SIGTERM to server")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            print("  server didn't exit in 15s, killing")
            server_proc.kill()
            server_proc.wait()

    print(f"\nDone. Artifacts in {run_dir}")
    # Print quick summary if server.json exists
    server_json = run_dir / "server.json"
    if server_json.exists():
        import json
        with open(server_json) as f:
            data = json.load(f)
        rounds = data.get("rounds", [])
        print(f"\nServer completed {len(rounds)}/{data.get('num_rounds_target')} rounds")
        for r in rounds:
            print(
                f"  round {r['round']:2d}: buffer={r['buffer_samples']:5d}, "
                f"test_acc={r['test_accuracy']:.3f}, "
                f"train_time={r['train_time_seconds']:.1f}s"
            )
        print(f"\nBytes broadcast: {data.get('bytes_broadcast_total', 0)/1024:.1f} KB")
        for node, bytes_up in data.get("per_client_bytes_total", {}).items():
            print(f"Bytes received from {node}: {bytes_up/1024/1024:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
