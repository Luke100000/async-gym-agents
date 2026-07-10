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

TRANSPORT_CONSUMER_BUFFERED_ROWS_KEY = "consumer_buffered_rows"
TRANSPORT_CONSUMER_MAX_BUFFERED_ROWS_KEY = "consumer_max_buffered_rows"
TRANSPORT_CONSUMED_ROWS_KEY = "consumed_rows"
TRANSPORT_DROPPED_ROWS_KEY = "dropped_rows"
TRANSPORT_PENDING_ROWS_KEY = "pending_rows"
TRANSPORT_PRODUCED_ROWS_KEY = "produced_rows"
TRANSPORT_RING_ALLOCATED_BYTES_KEY = "ring_allocated_bytes"
TRANSPORT_RING_CAPACITY_ROWS_KEY = "ring_capacity_rows"
TRANSPORT_RING_UTILIZATION_KEY = "ring_utilization"
TRANSPORT_TRAIN_ALLOCATED_BYTES_KEY = "train_allocated_bytes"
