# A worker ring holds this multiple of the per-cycle target so the trainer can
# fall a little behind without the worker dropping fresh transitions.
DEFAULT_RING_CAPACITY_MULTIPLIER = 4

# Smallest ring a worker is ever given, regardless of the per-cycle target.
MIN_RING_CAPACITY = 64

# Bytes reserved per slot for a serialized info dict. Empty infos use zero of
# it; non-empty infos (episode boundaries) must serialize within this cap.
INFO_BLOB_CAPACITY = 4096

# How long a worker waits between retries while its ring is full (backpressure).
RING_FULL_POLL_SECONDS = 0.001
