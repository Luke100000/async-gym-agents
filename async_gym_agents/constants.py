import struct

NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000
BYTES_PER_MEBIBYTE = 1_048_576

TRANSPORT_ACQUIRE_RETRY_TIMEOUT_SECONDS = 0.1
ASSEMBLER_RECEIVE_TIMEOUT_SECONDS = 0.1
ON_POLICY_ROLLOUT_ASSEMBLER_THREAD_NAME = "on-policy-rollout-assembler"
EPISODE_FEEDER_THREAD_NAME_PREFIX = "episode-feeder"
ASYNC_AGENT_WORKER_NAME_PREFIX = "async-agent-worker"
SHARED_COUNTER_TYPE_CODE = "q"
SHARED_SLOT_INDEX_TYPE_CODE = "i"
SHARED_SIZE_TYPE_CODE = "Q"
POLICY_SNAPSHOT_SLOT_COUNT = 2
POLICY_INITIAL_SLOT_INDEX = 0
POLICY_UNPUBLISHED_VERSION = -1
EPISODE_PACKET_HEADER = struct.Struct("!BqBQQ")
EPISODE_KIND_ON_POLICY_CODE = 0
EPISODE_KIND_OFF_POLICY_CODE = 1

CALLBACK_PROFILE_HOOK_NAMES = (
    "on_training_start",
    "on_rollout_start",
    "on_step",
    "on_rollout_end",
    "on_training_end",
)
CALLBACK_PROFILER_REPORT_KEY = "callbacks"
EPISODE_ACTIONS_FIELD = "actions"
EPISODE_DONES_FIELD = "dones"
EPISODE_LAST_DONES_FIELD = "last_dones"
EPISODE_LAST_OBSERVATION_FIELD = "last_obs"
EPISODE_LOG_PROBABILITIES_FIELD = "log_probs"
EPISODE_NEW_OBSERVATION_FIELD = "new_obs"
EPISODE_REWARDS_FIELD = "rewards"
EPISODE_VALUES_FIELD = "values"

PROFILE_PHASE_POLICY_BROADCAST = "policy_broadcast"
PROFILE_PHASE_POLICY_LOADING = "policy_loading"
PROFILE_PHASE_POLICY_SNAPSHOT_COPY = "policy_snapshot_copy"
PROFILE_PHASE_POLICY_SNAPSHOT_RETRY = "policy_snapshot_retry"
PROFILE_PHASE_POLICY_SERIALIZATION = "policy_serialization"
PROFILE_PHASE_EPISODE_DESERIALIZATION = "episode_deserialization"
PROFILE_PHASE_EPISODE_PACKING = "episode_packing"
PROFILE_PHASE_EPISODE_SERIALIZATION = "episode_serialization"
PROFILE_PHASE_ASSEMBLER_TRANSPORT = "assembler_transport"
PROFILE_PHASE_ASSEMBLER_WAITING = "assembler_waiting"
PROFILE_PHASE_ASSEMBLER_ACQUIRE = "assembler_acquire"
PROFILE_PHASE_CALLBACK_PROCESSING = "callback_processing"
PROFILE_PHASE_ROLLOUT_BUFFER_BUILDING = "rollout_buffer_building"
PROFILE_PHASE_PROFILER_REPORTING = "profiler_reporting"
PROFILE_PHASE_LOGGER_DUMP = "logger_dump"
PROFILE_PHASE_TRANSPORT = "transport"
PROFILE_PHASE_TRANSITION_RECONSTRUCTION = "transition_reconstruction"
PROFILE_PHASE_WAITING = "waiting"

BUFFER_AVG_POLICY_LAG_KEY = "avg_policy_lag"
BUFFER_AVG_PUSH_TIME_SECONDS_KEY = "avg_push_time_seconds"
BUFFER_AVG_PUSH_WAIT_SECONDS_KEY = "avg_push_wait_seconds"
BUFFER_MAX_POLICY_LAG_KEY = "max_policy_lag"

POLICY_PUBLISHED_VERSION_KEY = "published_version"
POLICY_PAYLOAD_BYTES_KEY = "payload_bytes"
POLICY_SLOT_CAPACITY_BYTES_KEY = "slot_capacity_bytes"
POLICY_PUBLICATION_COUNT_KEY = "publication_count"
POLICY_PUBLICATION_FAILURES_KEY = "publication_failures"

POLICY_CLOSED_STORE_ERROR = "Cannot publish to a closed shared policy store"
POLICY_CLOSED_READER_ERROR = "Cannot read from a closed shared policy reader"
POLICY_MISSING_INITIAL_SNAPSHOT_ERROR = (
    "Workers require an initial shared policy snapshot"
)
POLICY_OVERSIZED_PAYLOAD_ERROR = (
    "Policy payload size {payload_size} exceeds shared policy slot capacity "
    "{slot_capacity}"
)
POLICY_VERSION_ORDER_ERROR = (
    "Policy version {version} must be newer than published version {published_version}"
)
SHARED_POLICY_STARTUP_LOG = (
    "Shared policy initialized: payload_bytes={payload_bytes}, "
    "slot_capacity_bytes={slot_capacity_bytes}"
)
WORKER_EXCEPTION_LOG = "Async worker {worker_index} failed"
WORKER_FAILURE_ERROR = "Async workers failed: {failures}"
WORKER_FAILURE_DETAIL = "worker {worker_index} exited with {exit_reason}"
WORKER_EXIT_CODE_REASON = "exit code {exit_code}"
WORKER_UNKNOWN_SIGNAL_REASON = "signal {signal_number}"

PROFILER_LOG_PREFIX = "profiler"
PROFILER_EXCLUDED_OUTPUT_FORMATS = ("stdout", "log")

TRANSPORT_PENDING_EPISODES_KEY = "pending_episodes"
TRANSPORT_MAX_PENDING_EPISODES_KEY = "max_pending_episodes"
TRANSPORT_CAPACITY_EPISODES_KEY = "capacity_episodes"
TRANSPORT_UTILIZATION_KEY = "utilization"
TRANSPORT_PENDING_BYTES_KEY = "pending_bytes"
TRANSPORT_MAX_PENDING_BYTES_KEY = "max_pending_bytes"
TRANSPORT_SENT_EPISODES_KEY = "sent_episodes"
TRANSPORT_SENT_BYTES_KEY = "sent_bytes"
TRANSPORT_RECEIVED_EPISODES_KEY = "received_episodes"
TRANSPORT_RECEIVED_BYTES_KEY = "received_bytes"
TRANSPORT_RECEIVE_ATTEMPTS_KEY = "receive_attempts"
TRANSPORT_RECEIVE_TIMEOUTS_KEY = "receive_timeouts"
TRANSPORT_RECEIVE_TIMEOUT_FRACTION_KEY = "receive_timeout_fraction"
TRANSPORT_RECEIVE_TIMEOUTS_WITH_PENDING_KEY = "receive_timeouts_with_pending"
TRANSPORT_RECEIVE_TIMEOUTS_WITH_PENDING_FRACTION_KEY = (
    "receive_timeouts_with_pending_fraction"
)
TRANSPORT_READINESS_WAIT_PROFILE_KEY = "readiness_wait"
TRANSPORT_READINESS_TIMEOUT_PROFILE_KEY = "readiness_timeout"
TRANSPORT_PAYLOAD_RECEIVE_PROFILE_KEY = "payload_receive"
TRANSPORT_PAYLOAD_RECEIVE_TIMEOUT_PROFILE_KEY = "payload_receive_timeout"
TRANSPORT_PAYLOAD_MEBIBYTES_PER_SECOND_KEY = "payload_mib_per_second"
TRANSPORT_PIPE_LATENCY_PROFILE_KEY = "pipe_latency"

ASSEMBLY_TARGET_TRANSITIONS_KEY = "target_transitions"
ASSEMBLY_FILLING_TRANSITIONS_KEY = "filling_transitions"
ASSEMBLY_COMPLETED_BUFFERS_KEY = "completed_buffers"
ASSEMBLY_LAST_TRANSITIONS_KEY = "last_transitions"
ASSEMBLY_MAX_TRANSITIONS_KEY = "max_transitions"
ASSEMBLY_LAST_PAYLOAD_BYTES_KEY = "last_payload_bytes"
ASSEMBLY_MAX_PAYLOAD_BYTES_KEY = "max_payload_bytes"
