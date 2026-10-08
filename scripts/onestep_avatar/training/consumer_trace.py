"""Optional observations at actual LTX/PEFT consumers; see doc/training/consumer_trace.md."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

IDENTITY_FIELDS = {"rank", "world_size", "queue_job_sha256", "queue_attempt_token", "launch_sha256"}
HASH_FIELDS = ("queue_job_sha256", "launch_sha256")


def _binding(binding: dict) -> None:
    if (
        not isinstance(binding, dict)
        or not binding.keys() >= IDENTITY_FIELDS
        or type(binding["rank"]) is not int
        or type(binding["world_size"]) is not int
        or not 0 <= binding["rank"] < binding["world_size"]
    ):
        raise ValueError("consumer trace requires explicit rank/world and launch identity fields")
    for key in HASH_FIELDS:
        value = binding[key]
        if value is not None and (
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"consumer trace {key} must be an exact digest or null")
    token = binding["queue_attempt_token"]
    if token is not None and (not isinstance(token, str) or not token):
        raise ValueError("consumer trace attempt token must be explicit or null")
    json.dumps(binding, allow_nan=False)


def metadata(tensor: torch.Tensor | None) -> dict | None:
    """Read metadata without retaining or copying an activation."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("consumer input is not a tensor")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "numel": tensor.numel(),
    }


def _tensor(tensor: torch.Tensor, limit: int) -> dict:
    result = metadata(tensor)
    if tensor.numel() > limit:
        raise ValueError(f"conditioning tensor has {tensor.numel()} elements, limit is {limit}")
    host = tensor.detach().to(device="cpu").contiguous()
    result["sha256"] = hashlib.sha256(host.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    flat = host.reshape(-1).to(torch.float64)
    finite = torch.isfinite(flat)
    result["finite_elements"] = int(finite.sum())
    values = flat[finite]
    result["minimum"] = None if not values.numel() else float(values.min())
    result["maximum"] = None if not values.numel() else float(values.max())
    # Values aid small controls; the full byte digest remains the comparison authority.
    result["sample_values"] = [float(x) if torch.isfinite(x) else None for x in flat[:16]]
    return result


def _transformed(tensor: torch.Tensor, limit: int) -> dict:
    """Copy an optional transformed condition only within the same declared bound."""
    if tensor.numel() > limit:
        return {**metadata(tensor), "sha256": None, "hash_scope": "not_copied_exceeds_limit"}
    return {**_tensor(tensor, limit), "hash_scope": "complete_tensor"}


def _argument(args: tuple, kwargs: dict, key: str, index: int):  # noqa: ANN202
    return kwargs[key] if key in kwargs else args[index] if len(args) > index else None


class Trace:
    """Install bounded observational hooks only during explicit sample scopes."""

    def __init__(
        self,
        model: torch.nn.Module,
        binding: dict,
        *,
        max_events: int = 32768,
        max_tensor_elements: int = 1048576,
        failure_path: Path | None = None,
    ) -> None:
        _binding(binding)
        if (
            type(max_events) is not int
            or max_events < 1
            or type(max_tensor_elements) is not int
            or max_tensor_elements < 1
        ):
            raise ValueError("consumer trace limits must be positive integers")
        self.model = model
        self.binding = json.loads(json.dumps(binding))
        self.max_events = max_events
        self.max_tensor_elements = max_tensor_elements
        self.failure_path = None if failure_path is None else Path(failure_path)
        self.events: list[dict] = []
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.dropped_events = 0
        self.adapter_storage: list[dict] = []
        self._handles = []
        self._sample: dict | None = None
        self._call: dict | None = None
        self._block_stack: list[dict] = []
        self._block_counts: dict[str, int] = {}
        self._started = False
        self._closed = False
        self._checkpointing = False
        self._operation_failure: str | None = None

    def _emit(self, kind: str, module: str, observe) -> None:  # noqa: ANN001
        if self._sample is None:
            return
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        try:
            event = {
                "event": len(self.events),
                "kind": kind,
                "module": module,
                "sample": self._sample["sample"],
                "grad_enabled": torch.is_grad_enabled(),
                "model_call": None if self._call is None else self._call["model_call"],
                "role": None if self._call is None else self._call["role"],
                "block_pass": None if not self._block_stack else self._block_stack[-1]["pass"],
            }
            event.update(observe())
            self.events.append(event)
        except Exception as error:  # Observation failures must not change the scientific operation.
            if len(self.errors) < 32:
                self.errors.append(f"{kind}:{module}: {type(error).__name__}: {error}"[:1024])

    def _model_pre(self, name: str, args: tuple, kwargs: dict) -> None:
        if self._sample is None:
            return
        video = _argument(args, kwargs, "video", 0)
        call = self._sample["model_calls"]
        role = (
            "prime"
            if self._sample["mode"] == "causal" and call == 0
            else "refresh"
            if self._sample["mode"] == "causal" and video is not None and video.kv_write
            else "denoise"
        )
        self._sample["model_calls"] += 1
        self._call = {"model_call": call, "role": role}
        self._block_counts.clear()
        self._block_stack.clear()

        def observe() -> dict:
            if video is None:
                raise ValueError("avatar consumer has no video modality")
            return {
                "sigma": _tensor(video.sigma, self.max_tensor_elements),
                "timesteps": _tensor(video.timesteps, self.max_tensor_elements),
                "positions": _tensor(video.positions, self.max_tensor_elements),
                "latent": metadata(video.latent),
                "context": metadata(video.context),
                "kv_write": video.kv_write,
                "kv_present": video.kv_caches is not None,
                "kv_start": video.kv_start,
            }

        self._emit("ltx_input", name, observe)

    def _block_pre(self, name: str, args: tuple, kwargs: dict) -> None:
        if self._sample is None:
            return
        count = self._block_counts.get(name, 0) + 1
        self._block_counts[name] = count
        block_pass = "forward" if count == 1 else "checkpoint_recompute" if self._checkpointing else "repeat"
        self._block_stack.append({"module": name, "pass": block_pass})
        video = _argument(args, kwargs, "video", 0)

        def observe() -> dict:
            if video is None:
                raise ValueError("avatar block has no video arguments")
            return {
                "invocation": count,
                "x": metadata(video.x),
                "context": metadata(video.context),
                "timesteps": _transformed(video.timesteps, self.max_tensor_elements),
                "embedded_timestep": _transformed(video.embedded_timestep, self.max_tensor_elements),
                "positional_embeddings": [
                    _transformed(x, self.max_tensor_elements) for x in video.positional_embeddings
                ],
                "kv_write": video.kv_write,
                "kv_present": video.self_attn_kv_cache is not None,
                "kv_start": video.kv_start,
            }

        self._emit("block_input", name, observe)

    def _block_post(self, name: str) -> None:
        if self._block_stack and self._block_stack[-1]["module"] == name:
            self._block_stack.pop()

    def start(self) -> Trace:
        if self._started or self._closed:
            raise ValueError("consumer trace cannot be started twice or reopened")
        modules = list(self.model.named_modules())
        consumers = [(name, module) for name, module in modules if isinstance(module, LTXModel)]
        if len(consumers) != 1:
            raise ValueError("consumer trace requires exactly one actual LTXModel")
        self._checkpointing = consumers[0][1]._enable_gradient_checkpointing  # Installed LTX behavior, not CLI intent.
        self.adapter_storage = [
            {"name": name, **metadata(parameter)}
            for name, parameter in self.model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        ]
        self._started = True
        try:
            for name, module in modules:
                if isinstance(module, LTXModel):
                    self._handles.append(
                        module.register_forward_pre_hook(
                            lambda _mod, args, kwargs, name=name: self._model_pre(name, args, kwargs), with_kwargs=True
                        )
                    )
                elif isinstance(module, BasicAVTransformerBlock):
                    self._handles.append(
                        module.register_forward_pre_hook(
                            lambda _mod, args, kwargs, name=name: self._block_pre(name, args, kwargs), with_kwargs=True
                        )
                    )
                    self._handles.append(
                        module.register_forward_hook(
                            lambda _mod, _args, _kwargs, _output, name=name: self._block_post(name),
                            with_kwargs=True,
                            always_call=True,
                        )
                    )
                elif isinstance(module, AdaLayerNormSingle):
                    self._handles.append(
                        module.register_forward_pre_hook(
                            lambda _mod, args, kwargs, name=name: self._emit(
                                "adaln_input",
                                name,
                                lambda: {
                                    "scaled_timestep": _tensor(
                                        _argument(args, kwargs, "timestep", 0), self.max_tensor_elements
                                    ),
                                    "hidden_dtype": str(_argument(args, kwargs, "hidden_dtype", 1)),
                                },
                            ),
                            with_kwargs=True,
                        )
                    )
                elif isinstance(module, torch.nn.Linear) and (".lora_A." in name or ".lora_B." in name):
                    self._handles.append(
                        module.register_forward_pre_hook(
                            lambda mod, args, kwargs, name=name: self._emit(
                                "adapter_input",
                                name,
                                lambda: {
                                    "input": metadata(_argument(args, kwargs, "input", 0)),
                                    "weight": _transformed(mod.weight, self.max_tensor_elements),
                                },
                            ),
                            with_kwargs=True,
                        )
                    )
                    self._handles.append(
                        module.register_forward_hook(
                            lambda _mod, _args, _kwargs, output, name=name: self._emit(
                                "adapter_output", name, lambda: {"output": metadata(output)}
                            ),
                            with_kwargs=True,
                        )
                    )
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True

    def __enter__(self) -> Trace:
        return self.start()

    def __exit__(self, _exc_type, error, _traceback) -> None:  # noqa: ANN001 -- context protocol
        try:
            if error is not None:
                self._operation_failure = f"{type(error).__name__}: {error}"[:1024]
            if error is not None and self.failure_path is not None and self.failure_path.parent.is_dir():
                try:
                    self.write(self.failure_path, require_complete=False)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed consumer trace publication; preserve operation failure"
                    )
        finally:
            self.close()

    @contextlib.contextmanager
    def sample(self, *, mode: str, step: int, slot: int, index: int) -> Iterator[None]:
        if not self._started or self._closed or self._sample is not None:
            raise ValueError("consumer sample requires one active, non-nested trace")
        if mode not in ("bidirectional", "causal") or any(type(x) is not int or x < 0 for x in (step, slot, index)):
            raise ValueError("consumer sample mode or visit identity is invalid")
        if len(self.samples) >= self.max_events:
            self.dropped_events += 1
            yield
            return
        sample = {
            "sample": len(self.samples),
            "mode": mode,
            "step": step,
            "slot": slot,
            "index": index,
            "model_calls": 0,
            "state": "active",
        }
        self.samples.append(sample)
        self._sample = sample
        self._call = None
        try:
            yield
            sample["state"] = "passed"
        except BaseException as error:
            sample["state"] = "failed"
            sample["error"] = type(error).__name__
            raise
        finally:
            self._sample = None
            self._call = None
            self._block_stack.clear()

    def record(self) -> dict:
        record = {
            "schema_version": 1,
            "binding": self.binding,
            "limits": {"max_events": self.max_events, "max_tensor_elements": self.max_tensor_elements},
            "adapter_storage_window": "outside_forward_after_prepare",
            "adapter_storage": self.adapter_storage,
            "gradient_checkpointing": self._checkpointing,
            "samples": self.samples,
            "events": self.events,
            "dropped_events": self.dropped_events,
            "observation_errors": self.errors,
            "operation_failure": self._operation_failure,
            "complete": self._started
            and self._sample is None
            and not self.errors
            and not self.dropped_events
            and self._operation_failure is None
            and all(s["state"] == "passed" and s["model_calls"] > 0 for s in self.samples),
        }
        return json.loads(json.dumps(record, allow_nan=False))

    def write(self, path: Path, *, require_complete: bool = True) -> dict:
        """Publish a snapshot exclusively; no existing evidence is overwritten."""
        record = self.record()
        validate(record, require_complete=require_complete)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            Path(temporary).unlink()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "events": len(record["events"]),
            "complete": record["complete"],
        }


def validate(record: dict, binding: dict | None = None, *, require_complete: bool = True) -> None:  # noqa: PLR0912
    """Check evidence structure and identity; this does not accept native numerical results."""
    if (
        not isinstance(record, dict)
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or type(record.get("complete")) is not bool
    ):
        raise ValueError("consumer trace record schema is invalid")
    _binding(record.get("binding"))
    if binding is not None and record["binding"] != binding:
        raise ValueError("consumer trace launch binding differs")
    if (
        not isinstance(record.get("events"), list)
        or not isinstance(record.get("samples"), list)
        or not isinstance(record.get("observation_errors"), list)
        or type(record.get("dropped_events")) is not int
        or record["dropped_events"] < 0
    ):
        raise ValueError("consumer trace inventory is malformed")
    limits = record.get("limits")
    if (
        not isinstance(limits, dict)
        or set(limits) != {"max_events", "max_tensor_elements"}
        or any(type(x) is not int or x < 1 for x in limits.values())
        or len(record["events"]) > limits["max_events"]
        or len(record["samples"]) > limits["max_events"]
    ):
        raise ValueError("consumer trace limits are invalid")
    complete = (
        not record["observation_errors"]
        and not record["dropped_events"]
        and record.get("operation_failure") is None
        and all(
            isinstance(s, dict)
            and s.get("state") == "passed"
            and type(s.get("model_calls")) is int
            and s["model_calls"] > 0
            for s in record["samples"]
        )
    )
    if record["complete"] and not complete:
        raise ValueError("consumer trace claims completion with missing observations")
    if require_complete and not record["complete"]:
        raise ValueError("consumer trace is incomplete")
    call_inventory = [[] for _ in record["samples"]]
    for index, sample in enumerate(record["samples"]):
        if (
            not isinstance(sample, dict)
            or sample.get("sample") != index
            or sample.get("mode") not in ("bidirectional", "causal")
            or sample.get("state") not in ("active", "passed", "failed")
            or any(type(sample.get(key)) is not int or sample[key] < 0 for key in ("step", "slot", "index"))
            or type(sample.get("model_calls")) is not int
            or sample["model_calls"] < 0
        ):
            raise ValueError("consumer trace sample identity is malformed")
    for index, event in enumerate(record["events"]):
        if (
            not isinstance(event, dict)
            or event.get("event") != index
            or type(event.get("sample")) is not int
            or not 0 <= event["sample"] < len(record["samples"])
            or event.get("kind") not in ("ltx_input", "block_input", "adaln_input", "adapter_input", "adapter_output")
            or not isinstance(event.get("module"), str)
            or type(event.get("grad_enabled")) is not bool
            or type(event.get("model_call")) is not int
            or not 0 <= event["model_call"] < record["samples"][event["sample"]]["model_calls"]
            or event.get("role") not in ("prime", "denoise", "refresh")
            or event.get("block_pass") not in (None, "forward", "checkpoint_recompute", "repeat")
        ):
            raise ValueError("consumer trace event inventory is invalid")
        if event.get("kind") == "ltx_input":
            call_inventory[event["sample"]].append(event["model_call"])
            for key in ("sigma", "timesteps", "positions"):
                tensor = event.get(key)
                if (
                    not isinstance(tensor, dict)
                    or not isinstance(tensor.get("sha256"), str)
                    or len(tensor["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in tensor["sha256"])
                    or type(tensor.get("numel")) is not int
                    or not 0 <= tensor["numel"] <= limits["max_tensor_elements"]
                    or not isinstance(tensor.get("dtype"), str)
                    or not isinstance(tensor.get("shape"), list)
                    or any(type(x) is not int or x < 0 for x in tensor["shape"])
                    or not isinstance(tensor.get("sample_values"), list)
                    or len(tensor["sample_values"]) != min(16, tensor["numel"])
                    or type(tensor.get("finite_elements")) is not int
                    or not 0 <= tensor["finite_elements"] <= tensor["numel"]
                ):
                    raise ValueError("consumer conditioning evidence is not an exact bounded tensor hash")
    if record["complete"]:
        for sample, calls in zip(record["samples"], call_inventory, strict=True):
            if calls != list(range(sample["model_calls"])):
                raise ValueError("consumer trace is missing or duplicates actual LTX inputs")
