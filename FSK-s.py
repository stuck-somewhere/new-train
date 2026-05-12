"""
Send data over audio using 4-FSK. Pairs with fsk_recv.py.

Wire format:
    [chirp preamble][silence gap][2B length BE][payload][4B CRC32 BE]

Each byte is split into 4 symbols of 2 bits each (MSB first).
Each symbol picks one of 4 tones, played for SYMBOL_S seconds.

Usage:
    python fsk_send.py "your message here"
    python fsk_send.py < some_file.txt
"""
import sys
import zlib
import numpy as np
import sounddevice as sd

SR        = 44100                           # sample rate (Hz)
SYMBOL_S  = 0.05                            # seconds per symbol
FREQS     = [1000, 1500, 2000, 2500]        # 4 tones -> 2 bits/symbol
CHIRP_S   = 0.5                             # preamble duration
GAP_S     = 0.1                             # silence between chirp and data
AMP       = 0.5                             # output amplitude (0..1)


def make_chirp() -> np.ndarray:
    """Linear sweep 500 -> 3500 Hz. Used as a sync marker."""
    n = int(SR * CHIRP_S)
    t = np.arange(n) / SR
    # phase = 2π * (f0*t + (f1-f0)/(2T) * t^2)
    return AMP * np.sin(2 * np.pi * (500 * t + (3000 / (2 * CHIRP_S)) * t ** 2))


def bytes_to_symbols(data: bytes) -> list[int]:
    """Each byte -> 4 symbols of 2 bits, MSB first."""
    syms = []
    for b in data:
        for shift in (6, 4, 2, 0):
            syms.append((b >> shift) & 0b11)
    return syms


def symbols_to_audio(syms: list[int]) -> np.ndarray:
    """Each symbol -> one windowed tone of SYMBOL_S seconds."""
    n = int(SR * SYMBOL_S)
    t = np.arange(n) / SR
    win = np.hanning(n)                     # avoids clicks at symbol edges
    out = np.zeros(n * len(syms))
    for i, s in enumerate(syms):
        out[i * n : (i + 1) * n] = AMP * win * np.sin(2 * np.pi * FREQS[s] * t)
    return out


def frame(data: bytes) -> bytes:
    """Wrap payload with 2B length and 4B CRC32."""
    if len(data) > 65535:
        raise ValueError("payload too large for 2-byte length field")
    return len(data).to_bytes(2, "big") + data + zlib.crc32(data).to_bytes(4, "big")


def send(data: bytes) -> None:
    framed = frame(data)
    audio = np.concatenate([
        make_chirp(),
        np.zeros(int(SR * GAP_S)),
        symbols_to_audio(bytes_to_symbols(framed)),
    ]).astype(np.float32)

    duration = len(audio) / SR
    print(f"Sending {len(data)} bytes in {duration:.1f}s "
          f"({len(data) / duration:.1f} B/s effective)")
    sd.play(audio, SR)
    sd.wait()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        payload = sys.argv[1].encode()
    else:
        payload = sys.stdin.buffer.read()
    send(payload)
