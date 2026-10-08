"""Own queue protocol names without importing a model or tensor library.

Inputs: none. Logic: define the one canonical token/job environment names,
startup-event prefix and launch protocol version. Consumers import these names.
Outputs: constant strings only. This module does not launch work or own records.
"""
PREFIX = 'AVATAR_QUEUE_EVENT '
TOKEN_ENV = 'ONESTEP_AVATAR_QUEUE_TOKEN'
JOB_ENV = 'ONESTEP_AVATAR_QUEUE_JOB_SHA256'
LAUNCH_ENV = 'ONESTEP_AVATAR_QUEUE_LAUNCH'
LAUNCH_PROTOCOL = 'registered_child_v1'
