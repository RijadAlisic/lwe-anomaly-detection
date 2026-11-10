"""
aero2_plant_bridge.py -- runs NATIVELY on the machine physically wired to
the real Aero2 (i.e. the remote lab PC, hardware=1). Zero crypto/lattice
dependencies -- only pal, numpy, and the stdlib (socket/json/argparse) --
so it installs cleanly wherever the Quanser HIL/USB driver stack lives
(native Windows), independent of whether charm-crypto/blaster (Linux-only)
are available there or not.

WHY THIS EXISTS
----------------
Aero2(hardware=1) needs the native Quanser HIL/USB driver stack to talk to
the physical board. charm-crypto and blaster (this project's LWE/lattice
deps) are Linux-only and don't have a real Windows build path. Those two
requirements don't reliably coexist in one process/OS, so the real-hardware
deployment splits into two processes instead of one:

    aero2_plant_bridge.py (THIS FILE)          aero2_lwe_run.py
    -----------------------------------        --------------------------
    runs natively on the lab PC                runs wherever charm/blaster
    owns the Aero2(hardware=1) object          actually work (e.g. WSL on
    zero crypto deps                           the same box)
    speaks the protocol below over TCP  <---->  aero2_tools.RemotePlantProxy
                                                 is a drop-in myAero2 stand-in
                                                 that speaks the client side
                                                 of this exact protocol

If it turns out charm/blaster/pal can all coexist in one environment on
the lab PC after all (e.g. real USB passthrough into WSL works), this file
just isn't needed -- run aero2_lwe_run.py directly with PLANT_BACKEND =
"local" in its USER CONFIG block, same as against the virtual QLab sim.

PROTOCOL
--------
Newline-delimited JSON over TCP, one blocking request/reply round trip per
message (client sends a line, server replies with a line before the next
request is sent -- strictly synchronous, no pipelining):

  {"cmd": "zero"}                     -> {"ok": true}
      Zero the encoders in software (same effect as the physical reset
      button) -- called once at startup by the client in place of the
      direct-Aero2 backend's myAero2.card.set_encoder_counts(...) call.

  {"cmd": "read"}                     -> {"pitch": .., "yaw": .., "pitch_rate": .., "yaw_rate": ..}
      Radians / rad-s, straight from myAero2.pitchAngle etc. -- same units
      as the direct-Aero2 backend, no conversion needed on the client side.

  {"cmd": "write", "v": [v_pitch, v_yaw]}   -> {"ok": true}
      Commands voltage. Clipped to +/-VOLTAGE_LIMIT and subject to the
      pitch safety cutout below REGARDLESS of what the client asked for --
      this is the last line of defense against a bug/runaway on the
      remote crypto side, so it's enforced here, not trusted from the wire.

  {"cmd": "led", "rgb": [r, g, b]}    -> {"ok": true}
      Best-effort/cosmetic only.

  {"cmd": "stop"}                     -> {"ok": true}, then the server
      closes this session and waits for the next connection.

SAFETY (enforced HERE, independent of the client -- see above)
----------------------------------------------------------------
  - voltage clipped to +/-VOLTAGE_LIMIT every "write"
  - pitch cutout: |pitch| > PITCH_LIMIT_DEG -> voltage forced to 0 this step
  - watchdog: if no message arrives within WATCHDOG_TIMEOUT_S, voltage is
    forced to 0 and the session is dropped after WATCHDOG_GIVEUP_S total

USAGE
-----
    python aero2_plant_bridge.py --host 0.0.0.0 --port 8765 --hardware 1

Defaults to --hardware 1 (real hardware) since that's this file's whole
reason to exist; pass --hardware 0 to point it at the virtual QLab sim
instead, e.g. to test the bridge plumbing itself before trusting it with
the real unit.
"""
import argparse
import json
import signal
import socket
import time

import numpy as np
from pal.products.aero2 import Aero2

# ── Safety limits -- edit to match the real unit's rated limits ───────────
VOLTAGE_LIMIT       = 10.0   # V, per channel -- clipped every "write" regardless of client
PITCH_LIMIT_DEG     = 40.0   # deg -- voltage forced to 0 if exceeded (mirrors the QA2C reference
                              # plant.py's own cutout)
WATCHDOG_TIMEOUT_S  = 1.0    # s  -- per-recv timeout (10x the 100ms/step budget at 10Hz)
WATCHDOG_GIVEUP_S   = 5.0    # s  -- drop the session entirely if silence persists this long

_stop = False


def _sig_handler(*_a):
    global _stop
    _stop = True


signal.signal(signal.SIGINT, _sig_handler)


def _recv_line(conn, buf):
    """Read one newline-terminated message from conn, using buf (bytearray) as a
    persistent partial-read buffer across calls. Returns (line_bytes_or_None, buf)."""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            return None, buf
        buf += chunk
    line, _, rest = buf.partition(b"\n")
    return line, rest


def serve(host, port, hardware, frequency):
    print(f"[bridge] opening Aero2(hardware={hardware}, frequency={frequency})...", flush=True)
    myAero2 = Aero2(id=0, hardware=hardware, readMode=0, frequency=frequency, oneDOF=0)
    print("[bridge] Aero2 opened.", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[bridge] listening on {host}:{port} -- waiting for controller to connect...", flush=True)
    print(f"[bridge] safety: VOLTAGE_LIMIT=+/-{VOLTAGE_LIMIT}V  PITCH_LIMIT={PITCH_LIMIT_DEG}deg  "
          f"WATCHDOG_TIMEOUT={WATCHDOG_TIMEOUT_S}s", flush=True)

    try:
        while not _stop:
            srv.settimeout(1.0)
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[bridge] controller connected from {addr}", flush=True)
            _handle_session(conn, myAero2)
            print("[bridge] controller disconnected -- waiting for next connection...", flush=True)
    finally:
        myAero2.write_voltage(0.0, 0.0)
        print("[bridge] voltage zeroed, shutting down.", flush=True)
        srv.close()


def _handle_session(conn, myAero2):
    buf = bytearray()
    conn.settimeout(WATCHDOG_TIMEOUT_S)
    last_msg_t = time.perf_counter()
    last_pitch_deg = 0.0

    try:
        while not _stop:
            try:
                line, buf = _recv_line(conn, buf)
            except socket.timeout:
                myAero2.write_voltage(0.0, 0.0)   # watchdog: no message in time -- cut power
                if time.perf_counter() - last_msg_t > WATCHDOG_GIVEUP_S:
                    print(f"[bridge] watchdog: no client message for {WATCHDOG_GIVEUP_S}s, "
                          f"closing session.", flush=True)
                    break
                continue

            if line is None:
                break   # client closed the socket

            last_msg_t = time.perf_counter()
            try:
                msg = json.loads(line.decode())
            except Exception:
                print(f"[bridge] bad message, ignoring: {line[:200]!r}", flush=True)
                continue

            cmd = msg.get("cmd")

            if cmd == "zero":
                myAero2.card.set_encoder_counts(
                    myAero2.READ_ENCODER_CHANNELS, len(myAero2.READ_ENCODER_CHANNELS),
                    np.array([0, 0, 0, 0], dtype=np.float64))
                conn.sendall((json.dumps({"ok": True}) + "\n").encode())

            elif cmd == "read":
                myAero2.read_analog_encoder_other_channels()
                last_pitch_deg = float(np.rad2deg(myAero2.pitchAngle))
                reply = {"pitch": float(myAero2.pitchAngle), "yaw": float(myAero2.yawAngle),
                         "pitch_rate": float(myAero2.pitchRate), "yaw_rate": float(myAero2.yawRate)}
                conn.sendall((json.dumps(reply) + "\n").encode())

            elif cmd == "write":
                v = msg.get("v", [0.0, 0.0])
                v_pitch = float(np.clip(v[0], -VOLTAGE_LIMIT, VOLTAGE_LIMIT))
                v_yaw   = float(np.clip(v[1], -VOLTAGE_LIMIT, VOLTAGE_LIMIT))
                if abs(last_pitch_deg) > PITCH_LIMIT_DEG:
                    v_pitch, v_yaw = 0.0, 0.0
                    print(f"[bridge] SAFETY CUTOUT: |pitch|={last_pitch_deg:.1f}deg > "
                          f"{PITCH_LIMIT_DEG}deg -- voltage forced to 0", flush=True)
                myAero2.write_voltage(v_pitch, v_yaw)
                conn.sendall((json.dumps({"ok": True}) + "\n").encode())

            elif cmd == "led":
                rgb = msg.get("rgb", [0, 0, 0])
                try:
                    myAero2.write_led(np.array(rgb, dtype=np.float64))
                except Exception:
                    pass
                conn.sendall((json.dumps({"ok": True}) + "\n").encode())

            elif cmd == "stop":
                conn.sendall((json.dumps({"ok": True}) + "\n").encode())
                break

            else:
                print(f"[bridge] unknown cmd {cmd!r}, ignoring", flush=True)
    finally:
        myAero2.write_voltage(0.0, 0.0)
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0",
                     help="interface to listen on (default: 0.0.0.0, all interfaces)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hardware", type=int, default=1, choices=[0, 1],
                     help="0=virtual QLab, 1=real hardware (default: 1)")
    ap.add_argument("--frequency", type=int, default=500,
                     help="passed through to Aero2(); readMode=0 (immediate I/O) is used here, "
                          "so this mostly just matches the direct-Aero2 backend's own call for "
                          "parity, not a hard timing constraint")
    args = ap.parse_args()
    serve(args.host, args.port, args.hardware, args.frequency)
