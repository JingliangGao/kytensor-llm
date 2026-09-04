#!/usr/bin/env python3
"""llama.cpp cross-backend profiler analysis tool.

Usage:
    python -m tools.profiler.profiler profile.json
    python -m tools.profiler.profiler profile.json --chrome-trace trace.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


OP_EVENT = 0
COPY_EVENT = 1

TYPE_NAMES = {0: "OP", 1: "COPY"}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    # 4, 5 removed
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    # 31-33 removed
    34: "TQ1_0", 35: "TQ2_0",
    # 36-38 removed
    39: "MXFP4", 40: "NVFP4",
}

GGML_UNARY_OP_NAMES = {
    0: "ABS", 1: "SGN", 2: "NEG", 3: "STEP",
    4: "TANH", 5: "ELU", 6: "RELU", 7: "SIGMOID",
    8: "GELU", 9: "GELU_QUICK", 10: "SILU", 11: "HARDSWISH",
    12: "HARDSIGMOID", 13: "EXP", 14: "EXPM1", 15: "SOFTPLUS",
    16: "GELU_ERF", 17: "XIELU", 18: "FLOOR", 19: "CEIL",
    20: "ROUND", 21: "TRUNC",
}

GGML_GLU_OP_NAMES = {
    0: "REGLU", 1: "GEGLU", 2: "SWIGLU", 3: "GEGLU_ERF",
    4: "GEGLU_QUICK", 5: "SWIGLU_OAI",
}

GGML_OP_NAMES = {
    0: "NONE", 1: "DUP", 2: "ADD", 3: "ADD_ID", 4: "ADD1",
    5: "ACC", 6: "SUB", 7: "MUL", 8: "DIV", 9: "SQR",
    10: "SQRT", 11: "LOG", 12: "SIN", 13: "COS", 14: "SUM",
    15: "SUM_ROWS", 16: "CUMSUM", 17: "MEAN", 18: "ARGMAX",
    19: "COUNT_EQUAL", 20: "REPEAT", 21: "REPEAT_BACK", 22: "CONCAT",
    23: "SILU_BACK", 24: "NORM", 25: "RMS_NORM", 26: "RMS_NORM_BACK",
    27: "GROUP_NORM", 28: "L2_NORM", 29: "MUL_MAT", 30: "MUL_MAT_ID",
    31: "OUT_PROD", 32: "SCALE", 33: "SET", 34: "CPY", 35: "CONT",
    36: "RESHAPE", 37: "VIEW", 38: "PERMUTE", 39: "TRANSPOSE",
    40: "GET_ROWS", 41: "GET_ROWS_BACK", 42: "SET_ROWS", 43: "DIAG",
    44: "DIAG_MASK_INF", 45: "DIAG_MASK_ZERO", 46: "SOFT_MAX",
    47: "SOFT_MAX_BACK", 48: "ROPE", 49: "ROPE_BACK", 50: "CLAMP",
    51: "CONV_TRANSPOSE_1D", 52: "IM2COL", 53: "IM2COL_BACK", 54: "IM2COL_3D",
    55: "CONV_2D", 56: "CONV_3D", 57: "CONV_2D_DW", 58: "CONV_TRANSPOSE_2D",
    59: "POOL_1D", 60: "POOL_2D", 61: "POOL_2D_BACK", 62: "UPSCALE",
    63: "PAD", 64: "PAD_REFLECT_1D", 65: "ROLL", 66: "ARANGE",
    67: "TIMESTEP_EMBEDDING", 68: "ARGSORT", 69: "TOP_K", 70: "LEAKY_RELU",
    71: "TRI", 72: "FILL", 73: "FLASH_ATTN_EXT", 74: "FLASH_ATTN_BACK",
    75: "SSM_CONV", 76: "SSM_SCAN", 77: "WIN_PART", 78: "WIN_UNPART",
    79: "GET_REL_POS", 80: "ADD_REL_POS", 81: "RWKV_WKV6",
    82: "GATED_LINEAR_ATTN", 83: "RWKV_WKV7", 84: "SOLVE_TRI",
    85: "GATED_DELTA_NET", 86: "UNARY", 87: "MAP_CUSTOM1",
    88: "MAP_CUSTOM2", 89: "MAP_CUSTOM3", 90: "CUSTOM",
    91: "CROSS_ENTROPY_LOSS", 92: "CROSS_ENTROPY_LOSS_BACK",
    93: "OPT_STEP_ADAMW", 94: "OPT_STEP_SGD", 95: "GLU",
    96: "COUNT",
}

GGML_TYPE_NAMES_TO_ID = {v: k for k, v in GGML_TYPE_NAMES.items()}

GGML_OP_NAMES_TO_ID = {v: k for k, v in GGML_OP_NAMES.items()}


_EXPORT_SKIP_OPS = frozenset({
    33,  # SET
    34,  # CPY
    35,  # CONT
    36,  # RESHAPE
    37,  # VIEW
    38,  # PERMUTE
    39,  # TRANSPOSE
    41,  # GET_ROWS_BACK
    42,  # SET_ROWS
    43,  # DIAG
    44,  # DIAG_MASK_INF
    45,  # DIAG_MASK_ZERO
    47,  # SOFT_MAX_BACK
    49,  # ROPE_BACK
    51,  # CONV_TRANSPOSE_1D
    52,  # IM2COL
    53,  # IM2COL_BACK
    54,  # IM2COL_3D
    58,  # CONV_TRANSPOSE_2D
    61,  # POOL_2D_BACK
    63,  # PAD
    64,  # PAD_REFLECT_1D
    65,  # ROLL
    66,  # ARANGE
    70,  # LEAKY_RELU (covered by UNARY)
    71,  # TRI
    72,  # FILL
    77,  # WIN_PART
    78,  # WIN_UNPART
    92,  # CROSS_ENTROPY_LOSS_BACK
    93,  # OPT_STEP_ADAMW
    94,  # OPT_STEP_SGD
    96,  # COUNT
})


def _compute_output_ne(op_id: int, ne0: list, ne1: list, ne2: list) -> list | None:
    if op_id == 29:  # MUL_MAT
        return [ne0[1], ne1[1], max(ne0[2], ne1[2]), max(ne0[3], ne1[3])]
    if op_id == 30:  # MUL_MAT_ID
        return [ne0[1], ne2[0], ne1[2], 1]
    if op_id in (2, 7, 8, 32):  # ADD, MUL, DIV, SCALE
        return [max(ne0[i], ne1[i]) for i in range(4)]
    if op_id == 4:  # ADD1
        return list(ne0)
    if op_id in (9, 86):  # SQR, UNARY
        return list(ne0)
    if op_id in (46, 25):  # SOFT_MAX, RMS_NORM
        return list(ne0)
    if op_id == 73:  # FLASH_ATTN_EXT
        # Per ggml_flash_attn_ext: result.ne = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }
        # When V was not captured (legacy records), fall back to hsk == hsv (q->ne[0]).
        hsv = ne2[0] if (ne2 and ne2[0] > 0) else ne0[0]
        return [hsv, ne0[2], ne0[1], ne0[3]]
    if op_id == 40:  # GET_ROWS
        return [ne0[0], ne1[1], ne1[2], ne1[3]]
    if op_id == 41:  # GET_ROWS_BACK
        return [ne0[0], ne1[1], ne1[2], ne1[3]]
    if op_id == 42:  # SET_ROWS
        return list(ne0)
    if op_id == 31:  # OUT_PROD
        return [ne0[0], ne1[0], max(ne0[2], ne1[2]), max(ne0[3], ne1[3])]
    if op_id == 22:  # CONCAT
        return [ne0[0] + ne1[0], max(ne0[1], ne1[1]),
                max(ne0[2], ne1[2]), max(ne0[3], ne1[3])]
    if op_id in (34, 35, 50, 60, 53, 68):  # CPY, CONT, CLAMP, POOL_2D, IM2COL_BACK, ARGSORT
        return list(ne0)
    return None


GGML_MAX_SRC = 10
GGML_MAX_OP_PARAMS_I32 = 16  # 64 bytes / sizeof(int32)


@dataclass
class ProfileRecord:
    type: int
    name: str
    backend_id: int
    split_id: int
    start_ns: int
    duration_ns: int
    bytes: int
    extra: Optional[str]
    # Output tensor info
    tensor_name: Optional[str] = None
    ne: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    out_type: int = -1
    # Source tensors (variable length, up to GGML_MAX_SRC)
    ne_src: list[list[int]] = field(default_factory=list)
    nb_src: list[list[int]] = field(default_factory=list)
    type_src: list[int] = field(default_factory=list)
    # Operation parameters (16 int32, raw from ggml_tensor::op_params)
    op_params: list[int] = field(default_factory=lambda: [0] * GGML_MAX_OP_PARAMS_I32)
    sub_op: int = -1

    # --- Convenience accessors mirroring the old API ---
    @property
    def ne_src0(self) -> list[int]:
        return self.ne_src[0] if len(self.ne_src) > 0 else [0, 0, 0, 0]

    @property
    def ne_src1(self) -> list[int]:
        return self.ne_src[1] if len(self.ne_src) > 1 else [0, 0, 0, 0]

    @property
    def ne_src2(self) -> list[int]:
        return self.ne_src[2] if len(self.ne_src) > 2 else [0, 0, 0, 0]

    @property
    def type_src0(self) -> int:
        return self.type_src[0] if len(self.type_src) > 0 else -1

    @property
    def type_src1(self) -> int:
        return self.type_src[1] if len(self.type_src) > 1 else -1

    @property
    def type_src2(self) -> int:
        return self.type_src[2] if len(self.type_src) > 2 else -1

    @property
    def sub_op_name(self) -> str:
        if self.sub_op < 0:
            return ""
        if self.name == "UNARY":
            return GGML_UNARY_OP_NAMES.get(self.sub_op, f"UNARY_OP({self.sub_op})")
        if self.name == "GLU":
            return GGML_GLU_OP_NAMES.get(self.sub_op, f"GLU_OP({self.sub_op})")
        return str(self.sub_op)

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type, f"UNKNOWN({self.type})")

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0

    @property
    def bandwidth_gbps(self) -> float:
        """Bandwidth in GB/s."""
        if self.duration_ns == 0 or self.bytes == 0:
            return 0.0
        return self.bytes / self.duration_ns

    @staticmethod
    def _fmt_ne(ne: list[int]) -> str:
        dims = [n for n in ne if n > 0]
        if not dims:
            return ""
        return "[" + ", ".join(str(d) for d in dims) + "]"

    @property
    def shape_str(self) -> str:
        """Human-readable tensor shapes, e.g. '[4096, 4096] x [4096, 1] x [8, 1]'."""
        parts = []
        for ne, gt in zip(self.ne_src, self.type_src):
            s = self._fmt_ne(ne)
            if s:
                type_name = GGML_TYPE_NAMES.get(gt, None)
                if type_name:
                    s = f"{s} ({type_name})"
                parts.append(s)
        result = " x ".join(parts)
        if self.sub_op_name:
            result = f"[{self.sub_op_name}] {result}"
        return result

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "backend_id": self.backend_id,
            "split_id": self.split_id,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "bytes": self.bytes,
            "extra": self.extra,
            "tensor_name": self.tensor_name,
            "ne": self.ne,
            "out_type": self.out_type,
            "n_src": len(self.ne_src),
            "ne_src": self.ne_src,
            "nb_src": self.nb_src,
            "type_src": self.type_src,
            "op_params": self.op_params,
            "sub_op": self.sub_op,
        }


@dataclass
class OpStats:
    name: str
    event_type: int
    backend_id: int
    count: int = 0
    total_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0
    total_bytes: int = 0
    representative_ne: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @property
    def avg_ns(self) -> float:
        return self.total_ns / self.count if self.count > 0 else 0

    @property
    def avg_us(self) -> float:
        return self.avg_ns / 1000.0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0

    @property
    def min_us(self) -> float:
        return self.min_ns / 1000.0

    @property
    def max_us(self) -> float:
        return self.max_ns / 1000.0

    @property
    def bandwidth_gbps(self) -> float:
        if self.total_ns == 0 or self.total_bytes == 0:
            return 0.0
        return self.total_bytes / self.total_ns

    @property
    def time_per_byte_ns(self) -> float:
        """Time per byte (lower = more efficient)."""
        if self.total_bytes == 0:
            return float("inf")
        return self.total_ns / self.total_bytes

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.event_type, f"UNKNOWN({self.event_type})")


@dataclass
class FnRecord:
    """Function-level span (v4+ "fn_records"): one timed scope of a C/C++ function."""
    name: str
    start_ns: int
    end_ns: int
    pid: int
    tid: int

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


@dataclass
class FnStats:
    """Aggregated function-level statistics (nested totals include children)."""
    name: str
    count: int = 0
    total_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0

    @property
    def avg_ns(self) -> float:
        return self.total_ns / self.count if self.count > 0 else 0

    @property
    def avg_us(self) -> float:
        return self.avg_ns / 1000.0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0

    @property
    def min_us(self) -> float:
        return self.min_ns / 1000.0

    @property
    def max_us(self) -> float:
        return self.max_ns / 1000.0


class ProfileData:
    def __init__(self, records: list[ProfileRecord], metadata: dict,
                 fn_records: list[FnRecord] | None = None,
                 fn_threads: dict[int, str] | None = None):
        self.records = records
        self.metadata = metadata
        self.fn_records = fn_records if fn_records is not None else []
        self.fn_threads = fn_threads if fn_threads is not None else {}

    @classmethod
    def load(cls, filepath: str | Path) -> ProfileData:
        """Load a profiler JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        if data.get("profiler") != "ggml":
            print(f"Warning: file may not be a ggml profiler output (profiler={data.get('profiler')})")

        records = []
        def _pad_ne(v):
            if isinstance(v, list) and len(v) < 4:
                return list(v) + [0] * (4 - len(v))
            if not isinstance(v, list):
                return [0, 0, 0, 0]
            return list(v)

        def _load_sources(r) -> tuple[list[list[int]], list[list[int]], list[int]]:
            """Read source tensor arrays, supporting both v3+ (arrays) and v2 (ne_src0/1/2) JSON."""
            ne_list_raw = r.get("ne_src")
            if isinstance(ne_list_raw, list):
                # v3+ format
                ne_src = [_pad_ne(x) for x in ne_list_raw]
                nb_raw = r.get("nb_src", [])
                if isinstance(nb_raw, list):
                    nb_src = [_pad_ne(x) for x in nb_raw]
                else:
                    nb_src = []
                while len(nb_src) < len(ne_src):
                    nb_src.append([0, 0, 0, 0])
                type_raw = r.get("type_src", [])
                if isinstance(type_raw, list):
                    type_src = [int(t) for t in type_raw]
                else:
                    type_src = []
                while len(type_src) < len(ne_src):
                    type_src.append(-1)
                return ne_src, nb_src[:len(ne_src)], type_src[:len(ne_src)]

            # Legacy v2 fallback
            ne_src: list[list[int]] = []
            type_src: list[int] = []
            for i in range(3):
                key_ne = f"ne_src{i}"
                key_type = f"type_src{i}"
                ne_v = r.get(key_ne)
                if ne_v is None and i == 0:
                    ne_v = r.get("ne")
                if ne_v is None:
                    break
                ne_padded = _pad_ne(ne_v)
                if all(v == 0 for v in ne_padded) and i > 0:
                    break
                ne_src.append(ne_padded)
                type_src.append(int(r.get(key_type, -1)))
            nb_src = [[0, 0, 0, 0] for _ in ne_src]
            return ne_src, nb_src, type_src

        def _load_op_params(r) -> list[int]:
            raw = r.get("op_params")
            if isinstance(raw, list):
                ops = [int(x) for x in raw[:GGML_MAX_OP_PARAMS_I32]]
                while len(ops) < GGML_MAX_OP_PARAMS_I32:
                    ops.append(0)
                return ops
            return [0] * GGML_MAX_OP_PARAMS_I32

        for r in data.get("records", []):
            ne_src, nb_src, type_src = _load_sources(r)

            # v3+ records have a real "ne" (output shape).  Legacy v2 records did not —
            # leave zero so export_graph_ops falls back to op-specific shape inference.
            ne_raw = r.get("ne") if "ne_src" in r else None
            ne_out = _pad_ne(ne_raw) if isinstance(ne_raw, list) and ne_raw else [0, 0, 0, 0]

            records.append(ProfileRecord(
                type=r.get("type", 0),
                name=r.get("name", "unknown"),
                backend_id=r.get("backend_id", 0),
                split_id=r.get("split_id", 0),
                start_ns=r.get("start_ns", 0),
                duration_ns=r.get("duration_ns", 0),
                bytes=r.get("bytes", 0),
                extra=r.get("extra"),
                tensor_name=r.get("tensor_name"),
                ne=ne_out,
                out_type=int(r.get("out_type", -1)),
                ne_src=ne_src,
                nb_src=nb_src,
                type_src=type_src,
                op_params=_load_op_params(r),
                sub_op=int(r.get("sub_op", -1)),
            ))

        backends_raw = data.get("backends", [])
        backends = []
        for b in backends_raw:
            backends.append({
                "id": b.get("id", 0),
                "name": b.get("name", "unknown"),
                "device": b.get("device", "unknown"),
                "device_type": b.get("device_type", 0),
            })

        metadata = {
            "version": data.get("version", 0),
            "total_records": data.get("total_records", len(records)),
            "total_ns": data.get("total_ns", sum(r.duration_ns for r in records)),
            "backends": backends,
        }

        # Function-level (call-stack) spans, embedded since v4
        fn_records = []
        for r in data.get("fn_records", []):
            fn_records.append(FnRecord(
                name=r.get("name", "unknown"),
                start_ns=int(r.get("start_ns", 0)),
                end_ns=int(r.get("end_ns", 0)),
                pid=int(r.get("pid", 0)),
                tid=int(r.get("tid", 0)),
            ))

        fn_threads: dict[int, str] = {}
        for t in data.get("fn_threads", []):
            fn_threads[int(t.get("tid", 0))] = t.get("name", "")

        return cls(records, metadata, fn_records, fn_threads)

    @property
    def total_ns(self) -> int:
        return sum(r.duration_ns for r in self.records)

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0

    def stats(self) -> list[OpStats]:
        """Aggregate stats grouped by (name, type, backend_id)."""
        groups: dict[tuple, OpStats] = {}
        for rec in self.records:
            key = (rec.name, rec.type, rec.backend_id)
            if key not in groups:
                groups[key] = OpStats(
                    name=rec.name,
                    event_type=rec.type,
                    backend_id=rec.backend_id,
                    min_ns=rec.duration_ns,
                    max_ns=rec.duration_ns,
                    representative_ne=list(rec.ne_src0),
                )
            s = groups[key]
            s.count += 1
            s.total_ns += rec.duration_ns
            s.min_ns = min(s.min_ns, rec.duration_ns)
            s.max_ns = max(s.max_ns, rec.duration_ns)
            s.total_bytes += rec.bytes

            # Track the ne from the longest individual call
            if rec.duration_ns >= s.max_ns:
                s.representative_ne = list(rec.ne_src0)

        return sorted(groups.values(), key=lambda s: s.total_ns, reverse=True)

    def top_operations(self, n: int = 10) -> list[OpStats]:
        """Return the N most time-consuming operations (aggregated)."""
        return self.stats()[:n]

    def top_kernels(self, n: int = 10) -> list[ProfileRecord]:
        """Return the N longest individual kernel executions."""
        return sorted(self.records, key=lambda r: r.duration_ns, reverse=True)[:n]

    def by_backend(self) -> dict[int, list[ProfileRecord]]:
        """Group records by backend ID."""
        groups: dict[int, list[ProfileRecord]] = {}
        for rec in self.records:
            groups.setdefault(rec.backend_id, []).append(rec)
        return dict(sorted(groups.items()))

    def timeline(self) -> list[ProfileRecord]:
        """Return records sorted by start_ns for timeline visualization."""
        return sorted(self.records, key=lambda r: r.start_ns)

    def inefficiency_ranking(self, n: int = 10) -> list[OpStats]:
        """Rank operations by time per byte (inefficiency). Lower is better."""
        all_stats = [s for s in self.stats() if s.total_bytes > 0 and s.event_type == OP_EVENT]
        return sorted(all_stats, key=lambda s: s.time_per_byte_ns, reverse=True)[:n]

    #
    # Function-level (call-stack) analysis
    #

    def fn_stats(self) -> list[FnStats]:
        """Aggregate function-level spans by name, sorted by total time.

        Note: totals of nested scopes include their children (a llama_decode
        span covers the whole call tree underneath it).
        """
        groups: dict[str, FnStats] = {}
        for rec in self.fn_records:
            s = groups.get(rec.name)
            if s is None:
                groups[rec.name] = FnStats(
                    name=rec.name,
                    count=1,
                    total_ns=rec.duration_ns,
                    min_ns=rec.duration_ns,
                    max_ns=rec.duration_ns,
                )
                continue
            s.count += 1
            s.total_ns += rec.duration_ns
            s.min_ns = min(s.min_ns, rec.duration_ns)
            s.max_ns = max(s.max_ns, rec.duration_ns)

        return sorted(groups.values(), key=lambda s: s.total_ns, reverse=True)

    def top_fns(self, n: int = 10) -> list[FnStats]:
        """Return the N most time-consuming functions (aggregated)."""
        return self.fn_stats()[:n]

    def fn_total_ns(self) -> int:
        """Grand total of the top-level (non-nested) spans.

        Computed with a per-thread sweep-line over the real spans, so nested
        scopes are not double counted. Usable as a reference for percentages.
        """
        from collections import defaultdict

        spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for rec in self.fn_records:
            spans[rec.tid].append((rec.start_ns, rec.end_ns))

        total = 0
        for _, iv in spans.items():
            iv.sort()
            cur_s, cur_e = iv[0]
            for s, e in iv[1:]:
                if s > cur_e:
                    total += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            total += cur_e - cur_s
        return total

    def fn_summary(self) -> None:
        """Print the function-level summary table to stdout."""
        if not self.fn_records:
            return

        print(f"  {'='*76}")
        print(f"  Function-level Profiling Summary ({len(self.fn_records)} spans)")
        print(f"  note: totals of nested scopes include their children")
        print(f"  {'='*76}\n")

        total_ns = self.fn_total_ns()
        print(f"  {'Function':<44} {'%Time':>7}  {'Count':>7}  {'Total':>10}  "
              f"{'Avg':>10}  {'Min':>10}  {'Max':>10}")
        print(f"  {'':->44} {'':->7}  {'':->7}  {'(ms)':>10}  "
              f"{'(us)':>10}  {'(us)':>10}  {'(us)':>10}")

        for s in self.fn_stats():
            pct = 100.0 * s.total_ns / total_ns if total_ns > 0 else 0.0
            print(f"  {s.name:<44} {pct:>6.1f}%  {s.count:>7}  {s.total_ms:>10.2f}  "
                  f"{s.avg_us:>10.2f}  {s.min_us:>10.2f}  {s.max_us:>10.2f}")
        print()

    def summary(self) -> None:
        """Print a formatted summary table to stdout."""
        print(f"\n{'='*80}")
        print(f"  ggml Profiler Summary")
        print(f"{'='*80}")
        print(f"  Total records: {len(self.records)}")
        print(f"  Total time:    {self.total_ms:.2f} ms")
        print(f"  Unique ops:    {len(set((r.name, r.type, r.backend_id) for r in self.records))}")
        print(f"{'='*80}\n")

        stats = self.stats()
        if not stats:
            print("  No profiling data.\n")
            return

        print(f"  {'TYPE':<5} {'BKND':>4}  {'Operation':<28} {'%Time':>7}  {'Count':>6}  "
              f"{'Total':>10}  {'Avg':>10}  {'Min':>10}  {'Max':>10}  {'Bandwidth':>12}")
        print(f"  {'':->5} {'':->4}  {'':->28} {'':->7}  {'':->6}  "
              f"{'(ms)':>10}  {'(us)':>10}  {'(us)':>10}  {'(us)':>10}  {'':->12}")

        for s in stats:
            pct = 100.0 * s.total_ns / self.total_ns if self.total_ns > 0 else 0

            line = (f"  {s.type_name:<5} {s.backend_id:>4}  {s.name:<28} {pct:>6.1f}%  "
                    f"{s.count:>6}  {s.total_ms:>10.2f}  {s.avg_us:>10.2f}  "
                    f"{s.min_us:>10.2f}  {s.max_us:>10.2f}")

            if s.total_bytes > 0 and s.total_ns > 0:
                bw = s.bandwidth_gbps
                if bw >= 1000.0:
                    line += f"  {bw / 1000.0:>9.2f} TB/s"
                else:
                    line += f"  {bw:>9.2f} GB/s"
            else:
                line += f"  {'':>12}"

            # Tensor shape from longest call
            shape_dims = [n for n in s.representative_ne if n > 0]
            if shape_dims:
                line += f"  [{', '.join(str(d) for d in shape_dims)}]"

            print(line)

        backend_groups = self.by_backend()
        if len(backend_groups) > 1:
            print(f"\n  --- By Backend ---")
            for bid, recs in sorted(backend_groups.items()):
                bk_total = sum(r.duration_ns for r in recs)
                bk_pct = 100.0 * bk_total / self.total_ns if self.total_ns > 0 else 0
                print(f"  Backend {bid}: {bk_total / 1e6:.2f} ms ({bk_pct:.1f}%) — {len(recs)} records")

        inef = self.inefficiency_ranking(5)
        if inef:
            print(f"\n  --- Top 5 Inefficient Operations (time/byte) ---")
            for s in inef:
                print(f"  {s.name:<28} {s.time_per_byte_ns / 1000:.2f} us/byte  "
                      f"({s.count} calls, {s.total_bytes / 1e6:.1f} MB)")

        top_k = self.top_kernels(5)
        print(f"\n  --- Top 5 Longest Kernels ---")
        for rec in top_k:
            shape = f" {rec.shape_str}" if rec.shape_str else ""
            tname = f" '{rec.tensor_name}'" if rec.tensor_name else ""
            print(f"  {rec.type_name:<5} {rec.name:<28}{tname} {rec.duration_us:>10.2f} us{shape}  "
                  f"(split={rec.split_id}, backend={rec.backend_id})")

        print()

        self.fn_summary()

    def export_chrome_trace(self, filepath: str | Path) -> None:
        """Export as Chrome Trace Event format for chrome://tracing."""
        events = []

        # Build backend name mapping and remap to non-negative PIDs
        # (Chrome cannot handle negative PIDs)
        backend_ids = sorted(set(rec.backend_id for rec in self.records))
        backend_names: dict[int, str] = {}
        pid_map: dict[int, int] = {}

        # Use metadata from JSON if available
        metadata_backends = self.metadata.get("backends", [])
        backend_by_id: dict[int, dict] = {b["id"]: b for b in metadata_backends}

        device_type_names = {0: "CPU", 1: "GPU", 2: "ACCEL"}
        # When function-level spans are present, reserve pid 0 for them so that
        # viewers which order processes by pid (ignoring sort_index) also show
        # the function-level lanes at the very top.
        pid_base = 1 if self.fn_records else 0
        for idx, bid in enumerate(backend_ids):
            pid_map[bid] = idx + pid_base
            if bid in backend_by_id:
                binfo = backend_by_id[bid]
                dev_type = binfo.get("device_type", 0)
                dev_name = binfo.get("device", "")
                type_name = device_type_names.get(dev_type, "Device")
                if dev_name and dev_name != "unknown":
                    backend_names[bid] = f"{type_name}: {dev_name}"
                else:
                    backend_names[bid] = f"{type_name}: {binfo.get('name', f'Backend {bid}')}"
            else:
                backend_names[bid] = f"Backend {bid}"

        # Process metadata events
        for i, bid in enumerate(backend_ids):
            pid = pid_map[bid]
            events.append({
                "ph": "M",  # metadata
                "pid": pid,
                "name": "process_name",
                "args": {"name": backend_names[bid]},
            })
            # Viewers order processes by sort_index ascending (smallest on top).
            events.append({
                "ph": "M",
                "pid": pid,
                "name": "process_sort_index",
                "args": {"sortIndex": i},
            })

        # Use real timestamps, but prevent overlaps within each track.
        # GPU kernels are launched rapidly (small start_ns gaps) but have long
        # durations, so naive real timestamps overlap.  Sweep-line per track:
        # sort by start_ns, then place each event at max(start, prev_end).
        from collections import defaultdict
        tracks: dict[tuple, list[ProfileRecord]] = defaultdict(list)
        for rec in self.records:
            tracks[(rec.backend_id, rec.split_id)].append(rec)

        for key in tracks:
            tracks[key].sort(key=lambda r: r.start_ns)

        for key, recs in tracks.items():
            pid = pid_map[key[0]]
            tid = f"split_{key[1]}"
            cursor = 0.0
            for rec in recs:
                ts = max(rec.start_ns / 1000.0, cursor)
                dur = rec.duration_ns / 1000.0
                cat = "copy" if rec.type == COPY_EVENT else "compute"
                events.append({
                    "ph": "X",  # complete event
                    "pid": pid,
                    "tid": tid,
                    "name": rec.name,
                    "ts": ts,
                    "dur": dur,
                    "cat": cat,
                    "args": {
                        "bytes": rec.bytes,
                        "duration_us": dur,
                        "shape": rec.shape_str,
                    },
                })
                cursor = ts + dur

        # Function-level spans (v4+): rendered as a separate process on the
        # same time axis (timestamps share the ggml_profiler_time_ns epoch).
        # Nested scopes on the same thread render as a call-stack flame view.
        # sort_index keeps this lane pinned ABOVE all op-level backends in
        # viewers that honor it; pid 0 covers viewers that sort by pid.
        if self.fn_records:
            fn_pid = 0
            events.append({
                "ph": "M",
                "pid": fn_pid,
                "name": "process_name",
                "args": {"name": "llama.cpp (function-level)"},
            })
            events.append({
                "ph": "M",
                "pid": fn_pid,
                "name": "process_sort_index",
                "args": {"sortIndex": -1000},
            })
            for i, (tid, tname) in enumerate(sorted(self.fn_threads.items())):
                events.append({
                    "ph": "M",
                    "pid": fn_pid,
                    "tid": tid,
                    "name": "thread_name",
                    "args": {"name": tname if tname else f"tid-{tid}"},
                })
                events.append({
                    "ph": "M",
                    "pid": fn_pid,
                    "tid": tid,
                    "name": "thread_sort_index",
                    "args": {"sortIndex": i},
                })
            for rec in self.fn_records:
                events.append({
                    "ph": "X",
                    "pid": fn_pid,
                    "tid": rec.tid,
                    "name": rec.name,
                    "ts": rec.start_ns / 1000.0,
                    "dur": rec.duration_ns / 1000.0,
                    "cat": "function",
                })

        trace = {"traceEvents": events}
        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2)

        print(f"Chrome trace exported to: {filepath}")
        print(f"Open chrome://tracing in Chrome/Edge and load this file.")

    def export_graph_ops(self, filepath: str | Path) -> None:
        """Export operations in export-graph-ops format for test-backend-ops --test-file.

        Output line layout matches tests/export-graph-ops.cpp:
            <op> <out_type> <ne[0..3]> <n_op_params> <op_params...> <n_sources>
            (<src_type> <src_ne[0..3]> <src_nb[0..3]>)*  <name|-> [<backend>]
        """
        seen: set[tuple] = set()
        lines: list[str] = []

        backend_by_id: dict[int, dict] = {}
        for b in self.metadata.get("backends", []):
            backend_by_id[b["id"]] = b

        for rec in self.records:
            if rec.type != OP_EVENT:
                continue

            op_id = GGML_OP_NAMES_TO_ID.get(rec.name, -1)
            if op_id < 0:
                continue

            if op_id in _EXPORT_SKIP_OPS:
                continue

            # --- Build the source list directly from the captured arrays ---
            sources: list[tuple[int, list[int], list[int]]] = []
            for i in range(len(rec.ne_src)):
                ne_i = rec.ne_src[i]
                if not any(v != 0 for v in ne_i):
                    continue
                src_type = rec.type_src[i] if i < len(rec.type_src) and rec.type_src[i] >= 0 else 0
                nb_i = rec.nb_src[i] if i < len(rec.nb_src) else [0, 0, 0, 0]
                sources.append((src_type, list(ne_i), list(nb_i)))

            # MUL_MAT_ID needs the ids tensor as src[2]; synthesize one if missing.
            if op_id == 30 and len(sources) == 2:
                sources.append((24, [sources[1][1][1], 1, 1, 1], [0, 0, 0, 0]))  # I32

            if not sources:
                continue

            # --- Output shape ---
            if any(v != 0 for v in rec.ne):
                ne_out = list(rec.ne)
            else:
                # Legacy records without captured output ne: fall back to op-specific formula.
                src_ne0 = sources[0][1] if len(sources) > 0 else [0, 0, 0, 0]
                src_ne1 = sources[1][1] if len(sources) > 1 else [0, 0, 0, 0]
                src_ne2 = sources[2][1] if len(sources) > 2 else [0, 0, 0, 0]
                computed = _compute_output_ne(op_id, src_ne0, src_ne1, src_ne2)
                if computed is None:
                    continue
                ne_out = computed

            # --- Output type ---
            out_type = rec.out_type if rec.out_type >= 0 else 0

            # --- op_params: emit the full 16-int32 block when captured. ---
            if any(p != 0 for p in rec.op_params):
                op_params = list(rec.op_params)
            else:
                # Legacy fallback synthesis (best-effort for v2 JSON files).
                op_params = []
                if op_id == 30 and len(sources) >= 2:  # MUL_MAT_ID
                    op_params.append(sources[1][1][1])
                elif op_id in (86, 95) and rec.sub_op >= 0:  # UNARY, GLU
                    op_params.append(rec.sub_op)

            bname = ""
            if rec.backend_id in backend_by_id:
                bname = backend_by_id[rec.backend_id].get("device", "")
                if not bname or bname == "unknown":
                    bname = backend_by_id[rec.backend_id].get("name", "")

            key = (op_id, out_type, tuple(ne_out), tuple(op_params),
                   tuple((s[0], tuple(s[1]), tuple(s[2])) for s in sources), bname)
            if key in seen:
                continue
            seen.add(key)

            parts: list[str] = [str(op_id), str(out_type),
                                str(ne_out[0]), str(ne_out[1]), str(ne_out[2]), str(ne_out[3])]
            parts.append(str(len(op_params)))
            parts.extend(str(p) for p in op_params)
            parts.append(str(len(sources)))
            for src_type, src_ne, src_nb in sources:
                parts.append(str(src_type))
                parts.extend(str(v) for v in src_ne)
                parts.extend(str(v) for v in src_nb)
            parts.append(rec.name if rec.name else "-")
            if bname:
                parts.append(bname)

            lines.append(" ".join(parts) + "\n")

        with open(filepath, "w") as f:
            f.writelines(lines)

        print(f"Exported {len(lines)} unique ops to: {filepath}")

    def export_html_viewer(self, filepath: str | Path, max_records: int = 0) -> None:
        """Export a self-contained interactive HTML timeline viewer using Canvas."""
        import json as json_mod

        metadata_backends = self.metadata.get("backends", [])
        backend_by_id: dict[int, dict] = {b["id"]: b for b in metadata_backends}

        backend_names: dict[int, str] = {}
        for bid in sorted(set(rec.backend_id for rec in self.records)):
            binfo = backend_by_id.get(bid, {})
            name = binfo.get("name", f"Backend {bid}")
            device = binfo.get("device", "")
            backend_names[bid] = device if device and device != "unknown" else name

        events: list[dict] = []
        cum_us = 0.0
        for rec in self.records:
            dur_us = rec.duration_ns / 1000.0
            events.append({
                "n": rec.name,
                "d": dur_us,
                "s": rec.shape_str,
                "b": rec.bytes,
                "t": rec.type,
                "bid": rec.backend_id,
                "start": cum_us,
            })
            cum_us += dur_us
        total_us = cum_us

        if max_records > 0 and len(events) > max_records:
            stride = len(events) // max_records
            events = events[::stride][:max_records]

        if total_us == 0:
            print("No profiling data to export.")
            return

        # Op lanes use a packed cumulative axis (sum of durations).  Stats
        # percentages must keep that denominator even when function-level
        # spans extend the visible time range.
        stats_total_us = total_us

        # Function-level spans (v4+): separate top lanes on a wall-clock axis
        # (rebased so the first span starts at 0), plus their own stats tab.
        fn_events_js: list[dict] = []
        fn_threads_js: dict[str, str] = {}
        fn_stats_js: list[dict] = []
        fn_span_us = 0.0
        if self.fn_records:
            for tid, tname in sorted(self.fn_threads.items()):
                fn_threads_js[str(tid)] = tname if tname else f"tid-{tid}"

            # Merge coverage intervals across threads, then compress idle gaps
            # (e.g. interactive waits) so they do not squeeze the op lanes into
            # a sliver.  Time inside coverage keeps a 1:1 scale.
            iv: list[list[int]] = []
            for s, e in sorted((r.start_ns, r.end_ns) for r in self.fn_records):
                if iv and s <= iv[-1][1]:
                    iv[-1][1] = max(iv[-1][1], e)
                else:
                    iv.append([s, e])
            coverage_ns = sum(e - s for s, e in iv)
            GAP_SQUEEZE_NS = 1_000_000  # gaps > 1 ms are shrunk to 1 ms
            offsets: list[int] = []
            acc = 0
            for i, (s, e) in enumerate(iv):
                offsets.append(acc - s)  # compress(t) = t + offset of its interval
                acc += e - s
                if i + 1 < len(iv):
                    acc += min(iv[i + 1][0] - e, GAP_SQUEEZE_NS)

            def compress(t_ns: int) -> int:
                lo, hi = 0, len(iv) - 1
                while lo < hi:
                    m = (lo + hi + 1) // 2
                    if iv[m][0] <= t_ns:
                        lo = m
                    else:
                        hi = m - 1
                return t_ns + offsets[lo]

            t0 = compress(min(r.start_ns for r in self.fn_records))
            for rec in self.fn_records:
                fn_events_js.append({
                    "n": rec.name,
                    "start": (compress(rec.start_ns) - t0) / 1000.0,
                    "d": rec.duration_ns / 1000.0,
                    "tid": rec.tid,
                })
            fn_span_us = coverage_ns / 1000.0
            total_us = max(total_us, (compress(max(r.end_ns for r in self.fn_records)) - t0) / 1000.0)

            fn_total_ns = self.fn_total_ns()
            for st in self.fn_stats():  # already sorted by total desc
                fn_stats_js.append({
                    "n": st.name,
                    "d": st.total_ns / 1000.0,
                    "c": st.count,
                    "pct": (st.total_ns / fn_total_ns * 100.0) if fn_total_ns > 0 else 0.0,
                    "min": st.min_us,
                    "max": st.max_us,
                })

        header_stats = str(len(events)) + ' events | ' + f'{stats_total_us/1000:.1f}' + ' ms'
        if fn_events_js:
            header_stats += f' | functions: {len(fn_events_js)} spans, {fn_span_us/1000:.1f} ms covered'

        # Build backend name map with string keys for JSON
        bn_str = {str(k): v for k, v in backend_names.items()}

        # --- HTML ---
        html = (
            '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
            '<title>Kylin Profiler</title>\n<style>\n'
            '*{margin:0;padding:0;box-sizing:border-box}\n'
            'body{font-family:system-ui,sans-serif;background:#f5f6f8;color:#222;'
            'display:flex;flex-direction:column;height:100vh;overflow:hidden}\n'
            '#hd{background:#ffffff;padding:8px 16px;display:flex;align-items:center;'
            'gap:16px;border-bottom:1px solid #d8dee9;flex-shrink:0}\n'
            '#hd h1{font-size:15px;color:#e94560}\n'
            '#hd .st{font-size:11px;color:#888}\n'
            '#tb{background:#ffffff;padding:6px 16px;border-bottom:1px solid #d8dee9;'
            'display:flex;align-items:center;gap:6px;flex-shrink:0}\n'
            '#tb button{background:#e8ebf0;color:#333;border:none;padding:5px 12px;'
            'cursor:pointer;border-radius:3px;font-size:11px}\n'
            '#tb button:hover{background:#e94560}\n'
            '#vi{font-size:10px;color:#888;margin-left:auto}\n'
            '#main{flex:1;display:flex;flex-direction:column;overflow:hidden}\n'
            '#cw{flex-shrink:0;overflow:hidden;position:relative}\n'
            '#c{display:block}\n'
            '#stats{flex:1;display:flex;flex-direction:column;background:#ffffff;'
            'border-top:1px solid #d8dee9;min-height:0}\n'
            '#sttabs{padding:4px 8px;background:#f0f2f6;border-bottom:1px solid #d8dee9;'
            'flex-shrink:0}\n'
            '.stab{border:1px solid #d0d6e0;background:#fff;color:#555;font-size:11px;'
            'padding:2px 10px;margin-right:4px;cursor:pointer;border-radius:3px}\n'
            '.stab.act{background:#e94560;color:#fff;border-color:#e94560}\n'
            '#stbody{flex:1;overflow-y:auto}\n'
            '#stats table{width:100%;border-collapse:collapse;font-size:11px}\n'
            '#stats thead{position:sticky;top:0;z-index:1}\n'
            '#stats th{text-align:left;padding:6px 10px;color:#888;background:#f0f2f6;'
            'border-bottom:1px solid #d8dee9;font-weight:normal;font-size:10px;'
            'text-transform:uppercase;letter-spacing:0.5px}\n'
            '#stats th.r{text-align:right}\n'
            '#stats td{padding:4px 10px;border-bottom:1px solid rgba(0,0,0,0.06)}\n'
            '#stats td.r{text-align:right;font-variant-numeric:tabular-nums;font-family:monospace,system-ui}\n'
            '#stats .l0 td{background:rgba(0,0,0,0.025)}\n'
            '#stats .l0:hover td{background:rgba(0,0,0,0.07)}\n'
            '#stats .l1:hover td,.l2:hover td{background:rgba(0,0,0,0.045)}\n'
            '#stats .tog{cursor:pointer;user-select:none;color:#666;'
            'width:16px;display:inline-block;text-align:center;font-size:9px}\n'
            '#stats .tog:hover{color:#e94560}\n'
            '#stats .pct-cell{position:relative}\n'
            '#stats .pct-bg{position:absolute;left:0;top:1px;bottom:1px;border-radius:2px;pointer-events:none}\n'
            '#stats .pct-tx{position:relative}\n'
            '#tt{position:fixed;background:#ffffff;border:1px solid #e94560;'
            'padding:10px;border-radius:5px;font-size:11px;display:none;'
            'z-index:100;pointer-events:none;max-width:280px;line-height:1.6;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.15)}\n'
            '#lg{background:#ffffff;padding:6px 16px;border-top:1px solid #d8dee9;'
            'font-size:10px;flex-shrink:0;color:#555}\n'
            '</style></head><body>\n'
            '<div id="hd"><h1>Kylin Profiler Timeline</h1>'
            '<span class="st">' + header_stats + '</span></div>\n'
            '<div id="tb">'
            '<button onclick="fitAll()">Fit</button>'
            '<button onclick="zoomTo(1000000)">1s</button>'
            '<button onclick="zoomTo(100000)">100ms</button>'
            '<button onclick="zoomTo(10000)">10ms</button>'
            '<button onclick="zoomTo(1000)">1ms</button>'
            '<button onclick="zoomTo(100)">100\u03bcs</button>'
            '<span id="vi"></span></div>\n'
            '<div id="main">\n'
            '<div id="cw"><canvas id="c"></canvas></div>\n'
            '<div id="stats"><div id="sttabs">'
            '<button class="stab act" onclick="setStTab(\'ops\',this)">OP-level</button>'
            + ('<button class="stab" onclick="setStTab(\'fn\',this)">Function-level</button>'
               if fn_events_js else '')
            + '</div><div id="stbody"></div></div>\n'
            '</div>\n'
            '<div id="tt"></div>\n'
            '<div id="lg"></div>\n'
            '<script>\n'
        )

        # --- Inject data ---
        html += 'var EVENTS=' + json_mod.dumps(events, separators=(',', ':')) + ';\n'
        html += 'var BACKENDS=' + json_mod.dumps(bn_str, separators=(',', ':')) + ';\n'
        html += 'var TOTAL_US=' + repr(total_us) + ';\n'
        html += 'var STAT_TOTAL_US=' + repr(stats_total_us) + ';\n'
        html += 'var FN_EVENTS=' + json_mod.dumps(fn_events_js, separators=(',', ':')) + ';\n'
        html += 'var FN_THREADS=' + json_mod.dumps(fn_threads_js, separators=(',', ':')) + ';\n'
        html += 'var FN_STATS=' + json_mod.dumps(fn_stats_js, separators=(',', ':')) + ';\n'
        html += 'var FN_SPAN_US=' + repr(fn_span_us) + ';\n'

        # --- JavaScript (plain string, no f-strings) ---
        js = r"""
// Pre-process: group events by lane
var LANE_IDS=[],seen={};
for(var i=0;i<EVENTS.length;i++){var b=EVENTS[i].bid;if(!(b in seen)){LANE_IDS.push(b);seen[b]=true;}}
LANE_IDS.sort(function(a,b){return a-b;});
var LANE_EVENTS={};
for(var i=0;i<LANE_IDS.length;i++)LANE_EVENTS[LANE_IDS[i]]=[];
for(var i=0;i<EVENTS.length;i++)LANE_EVENTS[EVENTS[i].bid].push(EVENTS[i]);

// Function-level lanes: always pinned to the TOP, wall-clock axis (rebased
// to 0 and idle-gap-compressed on the Python side).  One lane per thread,
// kept separate from the op-level lanes.  Sorted by start asc / dur desc so
// nested children paint over their parents (flame-like stacking).
var FN_LANE_IDS=[];
function isFnLane(id){return typeof id==='string'&&id.substring(0,3)==='fn_';}
function laneLabel(id){
  if(isFnLane(id)){
    var t=id.substring(3);
    return 'fn \u00b7 '+(FN_THREADS[t]||('tid-'+t));
  }
  return BACKENDS[id]||('B'+id);
}
if(FN_EVENTS.length>0){
  var byTid={};
  for(var i=0;i<FN_EVENTS.length;i++){var e=FN_EVENTS[i];if(!(e.tid in byTid))byTid[e.tid]=[];byTid[e.tid].push(e);}
  var tids=[];for(var t in byTid)tids.push(t);
  tids.sort(function(a,b){return a-b;});
  for(var i=0;i<tids.length;i++){
    var arr=byTid[tids[i]];
    arr.sort(function(a,b){return (a.start-b.start)||(b.d-a.d);});
    var id='fn_'+tids[i];
    FN_LANE_IDS.push(id);
    LANE_EVENTS[id]=arr;
  }
}
LANE_IDS=FN_LANE_IDS.concat(LANE_IDS);
for(var li=0;li<LANE_IDS.length;li++){
  var _evts=LANE_EVENTS[LANE_IDS[li]];
  for(var i=0;i<_evts.length;i++)_evts[i].lane=LANE_IDS[li];
}

// Constants
var LANE_H=32,LABEL_W=150,MINIMAP_H=28,AXIS_H=18;
var TOP_PAD=MINIMAP_H+AXIS_H;
var BAR_PAD=3,COPY_PAD=8;

// Colors
var OP_COL={'MUL_MAT':'#4285f4','FLASH_ATTN_EXT':'#e879a0','ADD':'#81c784',
'ROPE':'#ce93d8','GET_ROWS':'#ffab91','CPY':'#b0bec5','CONCAT':'#90caf9',
'SCALE':'#80deea','MUL':'#a5d6a7','SOFT_MAX':'#fff176','RMS_NORM':'#ffcc80',
'SILU':'#ef9a9a','CONT':'#80cbc4','RESHAPE':'#9fa8da','VIEW':'#a1887f',
'PERMUTE':'#90a4ae','TRANSPOSE':'#c5e1a5','UNARY':'#f48fb1'};
function hash(s){var h=0;for(var i=0;i<s.length;i++)h=((h<<5)-h)+s.charCodeAt(i);return Math.abs(h);}
function col(n){return OP_COL[n]||('hsl('+hash(n)%360+',60%,55%)');}
function fmtT(us){if(us>=1e6)return(us/1e6).toFixed(2)+'s';if(us>=1e3)return(us/1e3).toFixed(2)+'ms';return us.toFixed(1)+'\u03bcs';}
function fmtB(b){if(!b)return'';if(b>=1e9)return(b/1e9).toFixed(1)+'GB';if(b>=1e6)return(b/1e6).toFixed(1)+'MB';if(b>=1e3)return(b/1e3).toFixed(1)+'KB';return b+'B';}
function fmtSh(s){if(!s)return'';return s.replace(/[\[\],]| x /g,function(m){return'<span style="color:#e8a040">'+m+'</span>';});}

// Canvas state
var canvas=document.getElementById('c');
var ctx,canvasW,canvasH,viewW;
var scale=1,offsetUs=0;
var hoveredEv=null,isDragging=false,dragStartX,dragStartOff;

function setup(){
  var dpr=window.devicePixelRatio||1;
  canvasW=canvas.parentElement.clientWidth;
  canvasH=Math.max(200,LANE_IDS.length*LANE_H+TOP_PAD+4);
  canvas.width=Math.round(canvasW*dpr);
  canvas.height=Math.round(canvasH*dpr);
  canvas.style.width=canvasW+'px';
  canvas.style.height=canvasH+'px';
  ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  viewW=canvasW-LABEL_W;
  document.getElementById('cw').style.height=canvasH+'px';
}

// Binary search: first event where start+d >= t
function bsFirst(evts,t){
  var lo=0,hi=evts.length;
  while(lo<hi){var m=(lo+hi)>>1;if(evts[m].start+evts[m].d<t)lo=m+1;else hi=m;}
  return lo;
}
// Binary search: find event containing time t
function bsHit(evts,t){
  var lo=0,hi=evts.length-1;
  while(lo<=hi){
    var m=(lo+hi)>>1;var ev=evts[m];
    if(t<ev.start)hi=m-1;
    else if(t>ev.start+ev.d)lo=m+1;
    else return ev;
  }
  return null;
}
// Function-level lanes contain nested (overlapping) spans, so binary search
// does not apply.  Linear scan for the innermost (shortest) containing span;
// fn lanes are small, so this is cheap.
function fnHit(evts,t){
  var best=null;
  for(var i=0;i<evts.length;i++){
    var e=evts[i];
    if(t>=e.start&&t<=e.start+e.d){
      if(!best||e.d<best.d)best=e;
    }
  }
  return best;
}

// Pre-render minimap to offscreen canvas
var mmCanvas;
function buildMinimap(){
  var dpr=window.devicePixelRatio||1;
  mmCanvas=document.createElement('canvas');
  mmCanvas.width=Math.round(canvasW*dpr);
  mmCanvas.height=Math.round(MINIMAP_H*dpr);
  var mc=mmCanvas.getContext('2d');
  mc.scale(dpr,dpr);
  mc.fillStyle='#e8ebf0';
  mc.fillRect(0,0,canvasW,MINIMAP_H);
  var mmScale=canvasW/TOTAL_US;
  for(var li=0;li<LANE_IDS.length;li++){
    var evts=LANE_EVENTS[LANE_IDS[li]];
    var step=Math.max(1,Math.floor(evts.length/(canvasW*2)));
    mc.globalAlpha=0.6;
    for(var i=0;i<evts.length;i+=step){
      var ev=evts[i];
      mc.fillStyle=col(ev.n);
      mc.fillRect(ev.start*mmScale,2,Math.max(0.5,ev.d*mmScale),MINIMAP_H-4);
    }
  }
  mc.globalAlpha=1;
}

function clampOffset(){
  var maxOff=TOTAL_US-viewW/scale;
  if(maxOff<0)maxOff=0;
  if(offsetUs<0)offsetUs=0;
  if(offsetUs>maxOff)offsetUs=maxOff;
}

function render(){
  ctx.clearRect(0,0,canvasW,canvasH);
  var visStart=offsetUs,visEnd=offsetUs+viewW/scale;

  // Minimap
  ctx.drawImage(mmCanvas,0,0,canvasW,MINIMAP_H);
  var vpX=offsetUs/TOTAL_US*canvasW,vpW=viewW/scale/TOTAL_US*canvasW;
  ctx.strokeStyle='#e94560';ctx.lineWidth=2;
  ctx.strokeRect(vpX,1,Math.max(2,vpW),MINIMAP_H-2);
  ctx.fillStyle='rgba(233,69,96,0.15)';
  ctx.fillRect(vpX,1,Math.max(2,vpW),MINIMAP_H-2);

  // Time axis background
  ctx.fillStyle='#f0f2f6';
  ctx.fillRect(LABEL_W,MINIMAP_H,viewW,AXIS_H);

  // Time axis ticks
  var rangeUs=visEnd-visStart;
  if(rangeUs>0){
    var raw=rangeUs/8;
    var mag=Math.pow(10,Math.floor(Math.log10(raw)));
    var iv;if(raw/mag<2)iv=2*mag;else if(raw/mag<5)iv=5*mag;else iv=10*mag;
    var firstTick=Math.ceil(visStart/iv)*iv;
    ctx.fillStyle='#555';ctx.font='9px monospace';
    ctx.strokeStyle='rgba(0,0,0,0.06)';ctx.lineWidth=1;
    for(var t=firstTick;t<=visEnd;t+=iv){
      var tx=LABEL_W+(t-offsetUs)*scale;
      ctx.beginPath();ctx.moveTo(tx,TOP_PAD);ctx.lineTo(tx,canvasH);ctx.stroke();
      ctx.fillText(fmtT(t),tx+3,MINIMAP_H+AXIS_H-4);
    }
  }

  // Lanes
  for(var li=0;li<LANE_IDS.length;li++){
    var bid=LANE_IDS[li];
    var y=TOP_PAD+li*LANE_H;
    var fnLane=isFnLane(bid);

    // Background (function-level lanes get a distinct tint, kept separate
    // from the op-level backend lanes)
    if(fnLane){
      ctx.fillStyle=li%2===0?'#fff7f9':'#fdeef2';
    }else{
      ctx.fillStyle=li%2===0?'#ffffff':'#f3f5f8';
    }
    ctx.fillRect(LABEL_W,y,viewW,LANE_H);

    // Events (clipped to event area)
    ctx.save();
    ctx.beginPath();ctx.rect(LABEL_W,y,viewW,LANE_H);ctx.clip();
    var evts=LANE_EVENTS[bid];
    if(evts&&evts.length>0){
      var si=bsFirst(evts,visStart);
      for(var i=si;i<evts.length;i++){
        var ev=evts[i];
        if(ev.start>visEnd)break;
        var x=LABEL_W+(ev.start-offsetUs)*scale;
        var w=ev.d*scale;
        ctx.fillStyle=col(ev.n);
        if(ev.t===1){
          ctx.globalAlpha=0.7;
          ctx.fillRect(x,y+COPY_PAD,Math.max(0.5,w),LANE_H-2*COPY_PAD);
          ctx.globalAlpha=1;
        }else{
          ctx.fillRect(x,y+BAR_PAD,Math.max(0.5,w),LANE_H-2*BAR_PAD);
        }
        if(w>50){
          ctx.fillStyle='#fff';ctx.font='10px system-ui';
          ctx.fillText(ev.n,x+3,y+LANE_H/2+3,w-6);
        }
      }
    }
    ctx.restore();

    // Hover highlight
    if(hoveredEv&&hoveredEv.lane===bid){
      var hx=LABEL_W+(hoveredEv.start-offsetUs)*scale;
      var hw=hoveredEv.d*scale;
      ctx.save();
      ctx.beginPath();ctx.rect(LABEL_W,y,viewW,LANE_H);ctx.clip();
      ctx.strokeStyle='#e94560';ctx.lineWidth=2;
      ctx.strokeRect(hx-1,y+2,Math.max(3,hw+2),LANE_H-4);
      ctx.restore();
    }

    // Lane separator (thicker accent line between the function-level group
    // and the op-level backend group)
    var groupEdge=fnLane&&li===FN_LANE_IDS.length-1;
    ctx.strokeStyle=groupEdge?'#e94560':'#d8dee9';
    ctx.lineWidth=groupEdge?1.5:0.5;
    ctx.beginPath();ctx.moveTo(0,y+LANE_H-0.5);ctx.lineTo(canvasW,y+LANE_H-0.5);ctx.stroke();

    // Label background + text
    ctx.fillStyle=fnLane?'#fbe4ea':'#f0f2f6';ctx.fillRect(0,y,LABEL_W,LANE_H);
    ctx.fillStyle=fnLane?'#a3455e':'#444';ctx.font='11px system-ui';
    ctx.fillText(laneLabel(bid),8,y+LANE_H/2+4);
  }

  // Axis label area background (covers labels column in axis row)
  ctx.fillStyle='#f0f2f6';ctx.fillRect(0,MINIMAP_H,LABEL_W,AXIS_H);
  ctx.fillStyle='#666';ctx.font='9px monospace';ctx.fillText('Time',8,MINIMAP_H+AXIS_H-4);

  // View info
  document.getElementById('vi').textContent=fmtT(visStart)+' \u2014 '+fmtT(visEnd)+' ('+fmtT(rangeUs)+' visible)';
}

// --- Zoom / Pan ---
function fitAll(){scale=viewW/TOTAL_US;offsetUs=0;render();}
function zoomTo(us){scale=viewW/us;render();}

canvas.addEventListener('wheel',function(e){
  e.preventDefault();
  var r=canvas.getBoundingClientRect();
  var mx=e.clientX-r.left-LABEL_W;
  if(mx<0)return;
  var mu=offsetUs+mx/scale;
  scale*=(e.deltaY>0?0.8:1.25);
  var minScale=viewW/TOTAL_US*0.5;
  if(scale<minScale)scale=minScale;
  offsetUs=mu-mx/scale;
  clampOffset();render();
},{passive:false});

canvas.addEventListener('mousedown',function(e){
  var r=canvas.getBoundingClientRect();
  var my=e.clientY-r.top;
  if(my<MINIMAP_H){
    var frac=(e.clientX-r.left)/canvasW;
    offsetUs=frac*TOTAL_US-viewW/scale/2;
    clampOffset();render();return;
  }
  isDragging=true;dragStartX=e.clientX;dragStartOff=offsetUs;
  canvas.style.cursor='grabbing';
});
document.addEventListener('mousemove',function(e){
  if(!isDragging)return;
  offsetUs=dragStartOff-(e.clientX-dragStartX)/scale;
  clampOffset();render();
});
document.addEventListener('mouseup',function(){
  if(isDragging){isDragging=false;canvas.style.cursor='default';}
});

// --- Tooltip ---
var tip=document.getElementById('tt');
canvas.addEventListener('mousemove',function(e){
  if(isDragging)return;
  var r=canvas.getBoundingClientRect();
  var mx=e.clientX-r.left,my=e.clientY-r.top;
  var li=Math.floor((my-TOP_PAD)/LANE_H);
  if(li<0||li>=LANE_IDS.length||mx<LABEL_W){
    if(hoveredEv){hoveredEv=null;render();}
    tip.style.display='none';return;
  }
  var bid=LANE_IDS[li];
  var mu=offsetUs+(mx-LABEL_W)/scale;
  var ev=isFnLane(bid)?fnHit(LANE_EVENTS[bid],mu):bsHit(LANE_EVENTS[bid],mu);
  if(ev){
    if(hoveredEv!==ev){hoveredEv=ev;render();}
    var h='<b style="color:#e94560">'+ev.n+'</b><br>'+fmtT(ev.d)+' | '+laneLabel(ev.lane);
    if(ev.s)h+='<br>Shape: '+fmtSh(ev.s);
    if(ev.b)h+='<br>Bytes: '+fmtB(ev.b);
    tip.innerHTML=h;tip.style.display='block';
    tip.style.left=Math.min(e.clientX+15,window.innerWidth-280)+'px';
    tip.style.top=Math.min(e.clientY+15,window.innerHeight-100)+'px';
  }else{
    if(hoveredEv){hoveredEv=null;render();}
    tip.style.display='none';
  }
});
canvas.addEventListener('mouseleave',function(){
  if(hoveredEv){hoveredEv=null;render();}
  tip.style.display='none';
});

// --- Keyboard ---
document.addEventListener('keydown',function(e){
  var step=viewW/scale*0.2;
  if(e.key==='ArrowLeft'){offsetUs-=step;clampOffset();render();}
  else if(e.key==='ArrowRight'){offsetUs+=step;clampOffset();render();}
  else if(e.key==='+'||e.key==='='){scale*=1.5;render();}
  else if(e.key==='-'){scale/=1.5;var mn=viewW/TOTAL_US*0.5;if(scale<mn)scale=mn;render();}
  else if(e.key==='Home'){fitAll();}
});

// --- Resize ---
window.addEventListener('resize',function(){setup();buildMinimap();render();});

// --- Legend ---
function buildLegend(){
  var counts={};
  for(var i=0;i<EVENTS.length;i++){var n=EVENTS[i].n;counts[n]=(counts[n]||0)+1;}
  var entries=[];for(var n in counts)entries.push([n,counts[n]]);
  entries.sort(function(a,b){return b[1]-a[1];});
  var top=entries.slice(0,12);
  var h='';
  for(var i=0;i<top.length;i++){
    h+='<span style="display:inline-block;margin:0 8px"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:'+col(top[i][0])+';margin-right:4px;vertical-align:middle"></span>'+top[i][0]+'</span>';
  }
  document.getElementById('lg').innerHTML=h;
}

// --- Stats tree-table ---
function buildStats(){
  var ops={};
  for(var i=0;i<EVENTS.length;i++){
    var ev=EVENTS[i];
    if(!ops[ev.n])ops[ev.n]={name:ev.n,d:0,count:0,min:Infinity,max:0,bytes:0,backends:{}};
    var op=ops[ev.n];
    op.d+=ev.d;op.count++;op.bytes+=(ev.b||0);
    if(ev.d<op.min)op.min=ev.d;if(ev.d>op.max)op.max=ev.d;
    var bk=String(ev.bid);
    if(!op.backends[bk])op.backends[bk]={bid:ev.bid,d:0,count:0,min:Infinity,max:0,bytes:0,shapes:{}};
    var b=op.backends[bk];
    b.d+=ev.d;b.count++;b.bytes+=(ev.b||0);
    if(ev.d<b.min)b.min=ev.d;if(ev.d>b.max)b.max=ev.d;
    var sh=ev.s||'\u2014';
    if(!b.shapes[sh])b.shapes[sh]={d:0,count:0,min:Infinity,max:0,bytes:0};
    var s=b.shapes[sh];
    s.d+=ev.d;s.count++;s.bytes+=(ev.b||0);
    if(ev.d<s.min)s.min=ev.d;if(ev.d>s.max)s.max=ev.d;
  }
  var sorted=[];for(var k in ops)sorted.push(ops[k]);
  sorted.sort(function(a,b){return b.d-a.d;});

  // Build flat row list
  var rows=[],rid=0;
  for(var oi=0;oi<sorted.length;oi++){
    var op=sorted[oi];
    var opId=rid++;
    var bkeys=[];for(var bk in op.backends)bkeys.push(bk);
    bkeys.sort(function(a,b){return op.backends[b].d-op.backends[a].d;});
    rows.push({id:opId,p:-1,lv:0,name:op.name,d:op.d,count:op.count,bytes:op.bytes,
      min:op.min,max:op.max,pct:op.d/STAT_TOTAL_US*100,ch:bkeys.length>0});
    for(var bi=0;bi<bkeys.length;bi++){
      var bdata=op.backends[bkeys[bi]];
      var bId=rid++;
      var bname=BACKENDS[bdata.bid]||('B'+bdata.bid);
      var skeys=[];for(var sk in bdata.shapes)skeys.push(sk);
      skeys.sort(function(a,b){return bdata.shapes[b].d-bdata.shapes[a].d;});
      rows.push({id:bId,p:opId,lv:1,name:bname,d:bdata.d,count:bdata.count,bytes:bdata.bytes,
        min:bdata.min,max:bdata.max,pct:bdata.d/STAT_TOTAL_US*100,ch:skeys.length>0});
      for(var si=0;si<skeys.length;si++){
        var sdata=bdata.shapes[skeys[si]];
        var sId=rid++;
        rows.push({id:sId,p:bId,lv:2,name:skeys[si],d:sdata.d,count:sdata.count,bytes:sdata.bytes,
          min:sdata.min,max:sdata.max,pct:sdata.d/STAT_TOTAL_US*100,ch:false});
      }
    }
  }

  // Render
  function fmtBW(bytes,d_us){
    if(!bytes||!d_us)return'\u2014';
    var gbps=bytes/(d_us*1000);
    if(gbps>=1000)return(gbps/1000).toFixed(2)+' TB/s';
    if(gbps>=1)return gbps.toFixed(2)+' GB/s';
    return(gbps*1000).toFixed(1)+' MB/s';
  }
  var h='<table><thead><tr><th style="width:26%">Operation</th>'
    +'<th class="r" style="width:10%">% Time</th>'
    +'<th class="r" style="width:10%">Total</th>'
    +'<th class="r" style="width:8%">Count</th>'
    +'<th class="r" style="width:10%">Avg</th>'
    +'<th class="r" style="width:10%">Min</th>'
    +'<th class="r" style="width:10%">Max</th>'
    +'<th class="r" style="width:10%">Bandwidth</th>'
    +'</tr></thead><tbody>';

  for(var ri=0;ri<rows.length;ri++){
    var r=rows[ri];
    var indent=8+r.lv*20;
    var vis=r.lv===0?'':'display:none';
    var tog=r.ch?'<span class="tog" onclick="togRow('+r.id+',this)">\u25b6</span>'
      :'<span style="width:16px;display:inline-block"></span>';
    var nc;
    if(r.lv===0)nc='color:'+col(r.name)+';font-weight:bold';
    else if(r.lv===1)nc='color:#ccc';
    else nc='color:#888';
    var barC=r.lv===0?col(r.name):'rgba(100,140,200,0.3)';
    var barO=r.lv===0?'0.25':'0.2';

    h+='<tr class="l'+r.lv+'" data-id="'+r.id+'" data-p="'+r.p+'" style="'+vis+'">';
    var dn=r.lv===2?fmtSh(r.name):r.name;
    h+='<td style="padding-left:'+indent+'px">'+tog+'<span style="'+nc+'">'+dn+'</span></td>';
    h+='<td class="r pct-cell"><div class="pct-bg" style="width:'+Math.max(0.5,r.pct)+'%;background:'+barC+';opacity:'+barO+'"></div><span class="pct-tx">'+r.pct.toFixed(1)+'%</span></td>';
    h+='<td class="r">'+fmtT(r.d)+'</td>';
    h+='<td class="r">'+r.count.toLocaleString()+'</td>';
    h+='<td class="r">'+fmtT(r.d/r.count)+'</td>';
    h+='<td class="r">'+fmtT(r.min)+'</td>';
    h+='<td class="r">'+fmtT(r.max)+'</td>';
    h+='<td class="r">'+fmtBW(r.bytes,r.d)+'</td>';
    h+='</tr>';
  }
  h+='</tbody></table>';
  document.getElementById('stbody').innerHTML=h;
}

// --- Stats tab switching (op-level vs function-level, kept separate) ---
function setStTab(tab,el){
  var bs=document.querySelectorAll('.stab');
  for(var i=0;i<bs.length;i++)bs[i].classList.remove('act');
  if(el)el.classList.add('act');
  if(tab==='fn')buildFnStats();else buildStats();
}

// --- Function-level stats table ---
function buildFnStats(){
  var h='<div style="padding:6px 10px;font-size:10px;color:#888">'
    +'Function-level spans (covered wall time: '+fmtT(FN_SPAN_US)+'). '
    +'Nested scopes include their children, so totals can exceed 100%.</div>';
  h+='<table><thead><tr><th style="width:30%">Function</th>'
    +'<th class="r" style="width:12%">% Time</th>'
    +'<th class="r" style="width:14%">Total</th>'
    +'<th class="r" style="width:10%">Count</th>'
    +'<th class="r" style="width:12%">Avg</th>'
    +'<th class="r" style="width:11%">Min</th>'
    +'<th class="r" style="width:11%">Max</th>'
    +'</tr></thead><tbody>';
  for(var i=0;i<FN_STATS.length;i++){
    var r=FN_STATS[i];
    h+='<tr class="l0">'
      +'<td style="padding-left:8px"><span style="color:#a3455e;font-weight:bold">'+r.n+'</span></td>'
      +'<td class="r pct-cell"><div class="pct-bg" style="width:'+Math.min(100,Math.max(0.5,r.pct))+'%;background:#e8879c;opacity:0.3"></div><span class="pct-tx">'+r.pct.toFixed(1)+'%</span></td>'
      +'<td class="r">'+fmtT(r.d)+'</td>'
      +'<td class="r">'+r.c.toLocaleString()+'</td>'
      +'<td class="r">'+fmtT(r.d/r.c)+'</td>'
      +'<td class="r">'+fmtT(r.min)+'</td>'
      +'<td class="r">'+fmtT(r.max)+'</td>'
      +'</tr>';
  }
  h+='</tbody></table>';
  document.getElementById('stbody').innerHTML=h;
}

function togRow(pid,el){
  var exp=el.textContent==='\u25bc';
  el.textContent=exp?'\u25b6':'\u25bc';
  var children=document.querySelectorAll('#stats tr[data-p="'+pid+'"]');
  for(var i=0;i<children.length;i++){
    children[i].style.display=exp?'none':'';
    if(exp){
      // Collapse grandchildren too
      var cid=children[i].getAttribute('data-id');
      var ctog=children[i].querySelector('.tog');
      if(ctog)ctog.textContent='\u25b6';
      var gc=document.querySelectorAll('#stats tr[data-p="'+cid+'"]');
      for(var j=0;j<gc.length;j++)gc[j].style.display='none';
    }
  }
}

// --- Init ---
setup();buildMinimap();buildLegend();buildStats();fitAll();
"""

        html += js + '\n</script></body></html>'

        with open(filepath, "w") as f:
            f.write(html)

        print(f"HTML viewer exported to: {filepath}")
        print(f"Open in browser: file://{Path(filepath).resolve()}")


def load(filepath: str | Path) -> ProfileData:
    """Load a profiler JSON file."""
    return ProfileData.load(filepath)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="llama.cpp profiler analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.profiler.profiler profile.json
  python -m tools.profiler.profiler profile.json --chrome-trace trace.json
  python -m tools.profiler.profiler profile.json --top-ops 20
  python -m tools.profiler.profiler profile.json --export-ops ops.txt
        """,
    )
    parser.add_argument("profile", help="Path to profiler JSON file")
    parser.add_argument("--chrome-trace", metavar="FILE",
                        help="Export as Chrome Trace Event format")
    parser.add_argument("--html-viewer", metavar="FILE",
                        help="Export as interactive HTML timeline viewer")
    parser.add_argument("--export-ops", metavar="FILE",
                        help="Export ops in export-graph-ops format (for test-backend-ops --test-file)")
    parser.add_argument("--html-max-records", type=int, default=0,
                        help="Max records in HTML viewer (0=unlimited, set to downsample for huge traces)")
    parser.add_argument("--top-ops", type=int, default=0,
                        help="Show top N operations (0 = show summary)")
    parser.add_argument("--top-fns", type=int, default=0,
                        help="Show top N functions (function-level spans, v4+ files)")
    parser.add_argument("--top-kernels", type=int, default=0,
                        help="Show top N longest kernels")
    parser.add_argument("--inefficiency", action="store_true",
                        help="Show inefficiency ranking")

    args = parser.parse_args()

    data = load(args.profile)

    if args.chrome_trace:
        data.export_chrome_trace(args.chrome_trace)

    if args.html_viewer:
        data.export_html_viewer(args.html_viewer, max_records=args.html_max_records)

    if args.export_ops:
        data.export_graph_ops(args.export_ops)

    if args.top_ops > 0:
        print(f"\nTop {args.top_ops} operations by total time:\n")
        for s in data.top_operations(args.top_ops):
            pct = 100.0 * s.total_ns / data.total_ns if data.total_ns > 0 else 0
            print(f"  {s.type_name:<5} {s.backend_id:>4}  {s.name:<28} {pct:>6.1f}%  "
                  f"{s.count:>6}x  {s.total_ms:>10.2f} ms  avg={s.avg_us:.2f} us")
        print()

    if args.top_fns > 0:
        if not data.fn_records:
            print("\nNo function-level data in this file (v4+ profiler output required).\n")
        else:
            total_ns = data.fn_total_ns()
            print(f"\nTop {args.top_fns} functions by total time "
                  f"(nested totals include children):\n")
            for s in data.top_fns(args.top_fns):
                pct = 100.0 * s.total_ns / total_ns if total_ns > 0 else 0
                print(f"  {s.name:<44} {pct:>6.1f}%  {s.count:>7}x  "
                      f"{s.total_ms:>10.2f} ms  avg={s.avg_us:.2f} us")
            print()

    if args.top_kernels > 0:
        print(f"\nTop {args.top_kernels} longest kernels:\n")
        for rec in data.top_kernels(args.top_kernels):
            print(f"  {rec.type_name:<5} {rec.backend_id:>4}  {rec.name:<28} "
                  f"{rec.duration_us:>10.2f} us  split={rec.split_id}")
        print()

    if args.inefficiency:
        print("\nInefficiency ranking (time/byte for operations with data):\n")
        for s in data.inefficiency_ranking(10):
            print(f"  {s.name:<28} {s.time_per_byte_ns / 1000:>10.2f} us/byte  "
                  f"{s.count:>6} calls  {s.total_bytes / 1e6:.1f} MB")
        print()

    if args.top_ops == 0 and args.top_kernels == 0 and args.top_fns == 0 and not args.inefficiency and not args.chrome_trace and not args.html_viewer and not args.export_ops:
        data.summary()


if __name__ == "__main__":
    main()
