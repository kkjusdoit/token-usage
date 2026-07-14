#!/usr/bin/env python3
"""Summarize local Antigravity, Claude Code, and Codex usage by calendar day.

The script reads only usage metadata from local JSONL session logs. It does not
send data anywhere and never prints prompt or response content.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = os.environ.get("TOKEN_USAGE_TIMEZONE", "Asia/Shanghai")


@dataclass
class Usage:
    calls: int = 0
    uncached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    other_tokens: int = 0
    total_tokens: int = 0

    def add(
        self,
        *,
        uncached_input: int = 0,
        cache_write: int = 0,
        cache_read: int = 0,
        output: int = 0,
        reasoning_output: int = 0,
        other: int = 0,
        total: int = 0,
    ) -> None:
        self.calls += 1
        self.uncached_input_tokens += uncached_input
        self.cache_write_tokens += cache_write
        self.cache_read_tokens += cache_read
        self.output_tokens += output
        self.reasoning_output_tokens += reasoning_output
        self.other_tokens += other
        self.total_tokens += total

    def merge(self, other: "Usage") -> None:
        for field in asdict(self):
            setattr(self, field, getattr(self, field) + getattr(other, field))


@dataclass
class Channel:
    channel: str
    root: str
    status: str
    files_scanned: int
    malformed_lines: int
    usage: Usage


def integer(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def discover_jsonl(
    roots: Iterable[Path], start_epoch: float, full_scan: bool
) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            candidates = root.rglob("*.jsonl")
            for path in candidates:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                # A log containing an event from the requested day must have
                # been modified no earlier than that event. The small margin
                # avoids filesystem timestamp precision edge cases.
                if not full_scan and stat.st_mtime + 2 < start_epoch:
                    continue
                files.append(path)
        except OSError:
            continue
    return sorted(files)


def read_jsonl(path: Path) -> Iterator[tuple[dict[str, Any] | None, bool]]:
    """Stream (record, malformed) pairs without retaining conversation logs."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    yield None, True
                    continue
                if isinstance(record, dict):
                    yield record, False
    except OSError:
        yield None, True


def summarize_claude(
    root: Path, start: datetime, end: datetime, full_scan: bool
) -> Channel:
    if not root.is_dir():
        return Channel("Claude Code", str(root), "missing", 0, 0, Usage())

    files = discover_jsonl([root], start.timestamp(), full_scan)
    # Claude Code writes one streaming response several times (thinking, text,
    # and tool-use blocks) with the same message.id. Only the latest record has
    # the final output token count, so retain exactly one record per message.
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}
    malformed = 0

    for path in files:
        for record, bad in read_jsonl(path):
            malformed += int(bad)
            if record is None:
                continue
            message = record.get("message")
            if record.get("type") != "assistant" or not isinstance(message, dict):
                continue
            token_data = message.get("usage")
            if not isinstance(token_data, dict):
                continue
            timestamp = parse_timestamp(record.get("timestamp"))
            if timestamp is None or timestamp < start:
                continue

            message_id = message.get("id")
            if isinstance(message_id, str) and message_id:
                key = f"message:{message_id}"
            else:
                request_id = record.get("requestId")
                session_id = record.get("sessionId")
                if request_id:
                    key = f"request:{session_id}:{request_id}"
                else:
                    # Old logs normally have message.id. This fallback avoids
                    # silently dropping an otherwise valid usage record.
                    key = f"record:{path}:{record.get('uuid', timestamp.isoformat())}"

            previous = latest.get(key)
            if previous is None or timestamp >= previous[0]:
                latest[key] = (timestamp, token_data)

    usage = Usage()
    for timestamp, token_data in latest.values():
        if not start <= timestamp < end:
            continue
        uncached = integer(token_data.get("input_tokens"))
        cache_write = integer(token_data.get("cache_creation_input_tokens"))
        cache_read = integer(token_data.get("cache_read_input_tokens"))
        output = integer(token_data.get("output_tokens"))
        total = uncached + cache_write + cache_read + output
        usage.add(
            uncached_input=uncached,
            cache_write=cache_write,
            cache_read=cache_read,
            output=output,
            total=total,
        )

    return Channel("Claude Code", str(root), "ok", len(files), malformed, usage)


def pb_varint(blob: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(blob) and shift < 70:
        byte = blob[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def pb_fields(blob: bytes | None) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield protobuf wire fields without needing Antigravity's private schema."""
    if not isinstance(blob, bytes):
        return
    offset = 0
    while offset < len(blob):
        try:
            key, offset = pb_varint(blob, offset)
            field_number, wire_type = key >> 3, key & 7
            if field_number == 0:
                return
            if wire_type == 0:
                value, offset = pb_varint(blob, offset)
            elif wire_type == 1:
                value = blob[offset : offset + 8]
                offset += 8
            elif wire_type == 2:
                length, offset = pb_varint(blob, offset)
                value = blob[offset : offset + length]
                offset += length
            elif wire_type == 5:
                value = blob[offset : offset + 4]
                offset += 4
            else:
                return
        except (IndexError, ValueError):
            return
        if offset > len(blob):
            return
        yield field_number, wire_type, value


def pb_uint(blob: bytes | None, field_number: int) -> int:
    for number_, wire_type, value in pb_fields(blob):
        if number_ == field_number and wire_type == 0 and isinstance(value, int):
            return value
    return 0


def pb_uints(blob: bytes | None, field_number: int) -> list[int]:
    values: list[int] = []
    for number_, wire_type, value in pb_fields(blob):
        if number_ != field_number:
            continue
        if wire_type == 0 and isinstance(value, int):
            values.append(value)
        elif wire_type == 2 and isinstance(value, bytes):
            offset = 0
            try:
                while offset < len(value):
                    item, offset = pb_varint(value, offset)
                    values.append(item)
            except ValueError:
                continue
    return values


def pb_messages(blob: bytes | None, field_number: int) -> Iterator[bytes]:
    for number_, wire_type, value in pb_fields(blob):
        if number_ == field_number and wire_type == 2 and isinstance(value, bytes):
            yield value


def pb_message(blob: bytes | None, field_number: int) -> bytes | None:
    return next(pb_messages(blob, field_number), None)


def pb_text(blob: bytes | None, field_number: int) -> str:
    value = pb_message(blob, field_number)
    if value is None:
        return ""
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def pb_timestamp(blob: bytes | None) -> datetime | None:
    if blob is None:
        return None
    try:
        seconds = pb_uint(blob, 1)
        nanos = pb_uint(blob, 2)
        return datetime.fromtimestamp(seconds + nanos / 1_000_000_000, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def antigravity_usage(blob: bytes) -> dict[str, int | str]:
    """Decode exa.codeium_common_pb.ModelUsageStats."""
    return {
        "model": pb_uint(blob, 1),
        "input": pb_uint(blob, 2),
        "output": pb_uint(blob, 3),
        "cache_write": pb_uint(blob, 4),
        "cache_read": pb_uint(blob, 5),
        "provider": pb_uint(blob, 6),
        "message_id": pb_text(blob, 7),
        "thinking_output": pb_uint(blob, 9),
        "response_id": pb_text(blob, 11),
        "provider_message_id": pb_text(blob, 12),
    }


def summarize_antigravity(
    root: Path, start: datetime, end: datetime, full_scan: bool
) -> Channel:
    name = "Antigravity"
    if not root.is_dir():
        return Channel(name, str(root), "missing", 0, 0, Usage())

    files: list[Path] = []
    for path in root.glob("*.db"):
        try:
            activity_mtime = path.stat().st_mtime
            wal_path = Path(f"{path}-wal")
            if wal_path.exists():
                activity_mtime = max(activity_mtime, wal_path.stat().st_mtime)
            if full_scan or activity_mtime + 2 >= start.timestamp():
                files.append(path)
        except OSError:
            continue

    # Antigravity stores the same ModelUsageStats in generator metadata, step
    # metadata, and retry metadata. A provider/message/response ID identifies
    # the API call across all three copies.
    calls: dict[
        tuple[str, ...], tuple[datetime, dict[str, int | str]]
    ] = {}
    errors = 0

    def remember(
        token_blob: bytes | None,
        timestamp: datetime | None,
        fallback: tuple[str, ...],
    ) -> None:
        if token_blob is None or timestamp is None:
            return
        decoded = antigravity_usage(token_blob)
        identifier = next(
            (
                str(decoded[field])
                for field in ("message_id", "response_id", "provider_message_id")
                if decoded[field]
            ),
            "",
        )
        if identifier:
            key = ("id", identifier)
        else:
            signature = tuple(
                str(decoded[field])
                for field in (
                    "model",
                    "provider",
                    "input",
                    "cache_write",
                    "cache_read",
                    "output",
                )
            )
            key = ("fallback", *fallback, *signature)

        previous = calls.get(key)
        if previous is None:
            calls[key] = (timestamp, decoded)
            return

        previous_time, previous_usage = previous
        previous_total = sum(
            int(previous_usage[field])
            for field in ("input", "cache_write", "cache_read", "output")
        )
        decoded_total = sum(
            int(decoded[field])
            for field in ("input", "cache_write", "cache_read", "output")
        )
        calls[key] = (
            min(previous_time, timestamp),
            decoded if decoded_total >= previous_total else previous_usage,
        )

    for path in sorted(files):
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.execute("PRAGMA query_only = ON")
            step_times: dict[int, datetime] = {}

            for step_index, metadata in connection.execute(
                "SELECT idx, metadata FROM steps WHERE metadata IS NOT NULL"
            ):
                created_at = pb_timestamp(pb_message(metadata, 1))
                if created_at is not None:
                    step_times[int(step_index)] = created_at
                remember(
                    pb_message(metadata, 9),
                    created_at,
                    (path.name, "step", str(step_index)),
                )
                for retry_index, retry in enumerate(pb_messages(metadata, 28)):
                    remember(
                        pb_message(retry, 2),
                        created_at,
                        (path.name, "step-retry", str(step_index), str(retry_index)),
                    )

            for generation_index, metadata in connection.execute(
                "SELECT idx, data FROM gen_metadata WHERE data IS NOT NULL"
            ):
                generated_steps = pb_uints(metadata, 2)
                timestamps = [
                    step_times[index]
                    for index in generated_steps
                    if index in step_times
                ]
                generated_at = min(timestamps) if timestamps else None
                chat_metadata = pb_message(metadata, 1)
                remember(
                    pb_message(chat_metadata, 4),
                    generated_at,
                    (path.name, "generation", str(generation_index)),
                )
                for retry_index, retry in enumerate(
                    pb_messages(chat_metadata, 17)
                ):
                    remember(
                        pb_message(retry, 2),
                        generated_at,
                        (
                            path.name,
                            "generation-retry",
                            str(generation_index),
                            str(retry_index),
                        ),
                    )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            errors += 1
        finally:
            try:
                if connection is not None:
                    connection.close()
            except sqlite3.Error:
                errors += 1

    usage = Usage()
    for timestamp, decoded in calls.values():
        if not start <= timestamp < end:
            continue
        uncached = int(decoded["input"])
        cache_write = int(decoded["cache_write"])
        cache_read = int(decoded["cache_read"])
        output = int(decoded["output"])
        total = uncached + cache_write + cache_read + output
        usage.add(
            uncached_input=uncached,
            cache_write=cache_write,
            cache_read=cache_read,
            output=output,
            reasoning_output=int(decoded["thinking_output"]),
            total=total,
        )

    return Channel(name, str(root), "ok", len(files), errors, usage)


def summarize_opencode_db(
    name: str, db_path: Path, start: datetime, end: datetime, full_scan: bool
) -> Channel:
    if not db_path.is_file():
        return Channel(name, str(db_path), "missing", 0, 0, Usage())

    # Opencode and its forks (mimo) store one row per message in `message`;
    # the `data` JSON on assistant messages carries the per-call token counts.
    # Their total already sums reasoning alongside input/output/cache, so we
    # map reasoning into the Other column to preserve the table identity
    # UncachedIn + CacheWrite + CacheRead + Output + Other = Total.
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    usage = Usage()
    errors = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        for (data,) in connection.execute(
            "SELECT data FROM message "
            "WHERE time_created >= ? AND time_created < ?",
            (start_ms, end_ms),
        ):
            if not isinstance(data, str):
                continue
            try:
                record = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                errors += 1
                continue
            if not isinstance(record, dict) or record.get("role") != "assistant":
                continue
            tokens = record.get("tokens")
            if not isinstance(tokens, dict):
                continue
            cache = tokens.get("cache")
            cache = cache if isinstance(cache, dict) else {}
            uncached = integer(tokens.get("input"))
            output = integer(tokens.get("output"))
            reasoning = integer(tokens.get("reasoning"))
            cache_write = integer(cache.get("write"))
            cache_read = integer(cache.get("read"))
            total = uncached + cache_write + cache_read + output + reasoning
            if total == 0:
                continue
            usage.add(
                uncached_input=uncached,
                cache_write=cache_write,
                cache_read=cache_read,
                output=output,
                reasoning_output=reasoning,
                other=reasoning,
                total=total,
            )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        errors += 1
    finally:
        try:
            if connection is not None:
                connection.close()
        except sqlite3.Error:
            errors += 1

    return Channel(name, str(db_path), "ok", 1, errors, usage)


def summarize_gemini(
    root: Path, start: datetime, end: datetime, full_scan: bool
) -> Channel:
    name = "Gemini CLI"
    if not root.is_dir():
        return Channel(name, str(root), "missing", 0, 0, Usage())

    # Gemini CLI writes one JSON session file per chat under tmp/*/chats/.
    # Each `gemini` message carries tokens {input, output, cached, thoughts,
    # tool, total} with total == the sum of the five parts. cached and
    # thoughts are counted alongside input/output (not subsets), so we place
    # cached in CacheRead and fold thoughts+tool into Other to preserve the
    # table identity UncachedIn + CacheWrite + CacheRead + Output + Other = Total.
    files: list[Path] = []
    for path in root.glob("*/chats/session-*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if not full_scan and stat.st_mtime + 2 < start.timestamp():
            continue
        files.append(path)

    usage = Usage()
    malformed = 0
    for path in sorted(files):
        try:
            document = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError, ValueError):
            malformed += 1
            continue
        if not isinstance(document, dict):
            continue
        messages = document.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict) or message.get("type") != "gemini":
                continue
            tokens = message.get("tokens")
            if not isinstance(tokens, dict):
                continue
            timestamp = parse_timestamp(message.get("timestamp"))
            if timestamp is None or not start <= timestamp < end:
                continue
            uncached = integer(tokens.get("input"))
            output = integer(tokens.get("output"))
            cached = integer(tokens.get("cached"))
            thoughts = integer(tokens.get("thoughts"))
            tool = integer(tokens.get("tool"))
            total = uncached + output + cached + thoughts + tool
            if total == 0:
                continue
            usage.add(
                uncached_input=uncached,
                cache_read=cached,
                output=output,
                reasoning_output=thoughts,
                other=thoughts + tool,
                total=total,
            )

    return Channel(name, str(root), "ok", len(files), malformed, usage)


def choose_codex_files(
    root: Path, start_epoch: float, full_scan: bool
) -> list[Path]:
    candidates = discover_jsonl(
        [root / "sessions", root / "archived_sessions"], start_epoch, full_scan
    )
    # A session can briefly exist in both locations while being archived.
    # Its rollout filename is stable, so choose the largest/newest copy.
    chosen: dict[str, Path] = {}
    for path in candidates:
        previous = chosen.get(path.name)
        if previous is None:
            chosen[path.name] = path
            continue
        try:
            current_rank = (path.stat().st_size, path.stat().st_mtime)
            previous_rank = (previous.stat().st_size, previous.stat().st_mtime)
        except OSError:
            continue
        if current_rank > previous_rank:
            chosen[path.name] = path
    return sorted(chosen.values())


def summarize_codex(
    profile: int, root: Path, start: datetime, end: datetime, full_scan: bool
) -> Channel:
    name = f"Codex profile {profile}"
    if not root.is_dir():
        return Channel(name, str(root), "missing", 0, 0, Usage())

    files = choose_codex_files(root, start.timestamp(), full_scan)
    malformed = 0
    usage = Usage()
    seen_events: set[tuple[str, str, str]] = set()

    for path in files:
        for record, bad in read_jsonl(path):
            malformed += int(bad)
            if record is None:
                continue
            payload = record.get("payload")
            if record.get("type") != "event_msg" or not isinstance(payload, dict):
                continue
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            token_data = info.get("last_token_usage")
            if not isinstance(token_data, dict):
                continue
            timestamp = parse_timestamp(record.get("timestamp"))
            if timestamp is None or not start <= timestamp < end:
                continue

            signature = json.dumps(token_data, sort_keys=True, separators=(",", ":"))
            event_key = (path.name, timestamp.isoformat(), signature)
            if event_key in seen_events:
                continue
            seen_events.add(event_key)

            input_tokens = integer(token_data.get("input_tokens"))
            cached_input = integer(token_data.get("cached_input_tokens"))
            output = integer(token_data.get("output_tokens"))
            reasoning = integer(token_data.get("reasoning_output_tokens"))
            recorded_total = integer(token_data.get("total_tokens"))
            component_total = input_tokens + output
            total = recorded_total if recorded_total > 0 else component_total

            # For Codex, cached_input_tokens is a subset of input_tokens.
            uncached = max(0, input_tokens - cached_input)
            other = total - component_total
            usage.add(
                uncached_input=uncached,
                cache_read=cached_input,
                output=output,
                reasoning_output=reasoning,
                other=other,
                total=total,
            )

    return Channel(name, str(root), "ok", len(files), malformed, usage)


def codex_roots(home: Path) -> dict[int, Path]:
    profile_zero = home / ".codex-profiles" / "profile-0"
    if not profile_zero.is_dir():
        profile_zero = home / ".codex"
    return {
        0: profile_zero,
        1: home / ".codex-profiles" / "profile-1",
        2: home / ".codex-profiles" / "profile-2",
        3: home / ".codex-profiles" / "profile-3",
    }


def number(value: int) -> str:
    return f"{value:,}"


def tokens_m(value: int) -> str:
    # Token columns are reported in millions (M). Sub-million values still show
    # meaningful precision; large totals stay compact.
    millions = value / 1_000_000
    if millions == 0:
        return "0"
    if abs(millions) < 100:
        return f"{millions:,.2f}"
    return f"{millions:,.1f}"


def render_table(channels: list[Channel], total: Usage) -> str:
    headers = [
        "Channel",
        "Calls",
        "UncachedIn/M",
        "CacheWrite/M",
        "CacheRead/M",
        "Output/M",
        "Other/M",
        "Total/M",
    ]
    rows: list[list[str]] = []
    for channel in channels:
        label = channel.channel
        if channel.status != "ok":
            label += " (missing)"
        u = channel.usage
        rows.append(
            [
                label,
                number(u.calls),
                tokens_m(u.uncached_input_tokens),
                tokens_m(u.cache_write_tokens),
                tokens_m(u.cache_read_tokens),
                tokens_m(u.output_tokens),
                tokens_m(u.other_tokens),
                tokens_m(u.total_tokens),
            ]
        )
    rows.append(
        [
            "ALL CHANNELS",
            number(total.calls),
            tokens_m(total.uncached_input_tokens),
            tokens_m(total.cache_write_tokens),
            tokens_m(total.cache_read_tokens),
            tokens_m(total.output_tokens),
            tokens_m(total.other_tokens),
            tokens_m(total.total_tokens),
        ]
    )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def line(row: list[str], first_left: bool = True) -> str:
        cells = []
        for index, value in enumerate(row):
            cells.append(value.ljust(widths[index]) if index == 0 and first_left else value.rjust(widths[index]))
        return "  ".join(cells)

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按本地日期汇总 Claude Code、Antigravity、MimoCode、OpenCode、"
            "Gemini CLI 与 Codex profile 0–3 的 token 用量。"
        )
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD", help="统计日期；默认是指定时区的今天"
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"IANA 时区名（默认：{DEFAULT_TIMEZONE}）",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=Path.home(),
        help="用户主目录；主要用于测试或自定义日志位置",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="忽略文件修改时间优化，扫描全部 JSONL（恢复过旧日志时使用）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="统计有史以来的全部用量（忽略 --date，自动开启全量扫描）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        tz = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError:
        print(f"错误：未知时区 {args.timezone!r}", file=sys.stderr)
        return 2

    if args.all and args.date:
        print("错误：--all 与 --date 不能同时使用", file=sys.stderr)
        return 2

    if args.all:
        # Cover every recorded call: from the epoch to just after now. --all
        # implies a full scan so mtime pruning never hides an old log.
        target = None
        start = datetime.fromtimestamp(0, tz)
        end = datetime.now(tz) + timedelta(days=1)
        full_scan = True
    else:
        if args.date:
            try:
                target = date.fromisoformat(args.date)
            except ValueError:
                print("错误：--date 必须是 YYYY-MM-DD 格式", file=sys.stderr)
                return 2
        else:
            target = datetime.now(tz).date()

        start = datetime.combine(target, time.min, tzinfo=tz)
        end = datetime.combine(target + timedelta(days=1), time.min, tzinfo=tz)
        full_scan = args.full_scan

    home = args.home.expanduser().resolve()

    channels = [
        summarize_claude(home / ".claude" / "projects", start, end, full_scan),
        summarize_antigravity(
            home / ".gemini" / "antigravity" / "conversations",
            start,
            end,
            full_scan,
        ),
        summarize_opencode_db(
            "MimoCode",
            home / ".local" / "share" / "mimocode" / "mimocode.db",
            start,
            end,
            full_scan,
        ),
        summarize_opencode_db(
            "OpenCode",
            home / ".local" / "share" / "opencode" / "opencode.db",
            start,
            end,
            full_scan,
        ),
        summarize_gemini(home / ".gemini" / "tmp", start, end, full_scan),
    ]
    for profile, root in codex_roots(home).items():
        channels.append(summarize_codex(profile, root, start, end, full_scan))

    total = Usage()
    for channel in channels:
        total.merge(channel.usage)

    generated_at = datetime.now(tz)
    date_label = "all-time" if target is None else target.isoformat()
    if args.json:
        payload = {
            "date": date_label,
            "timezone": args.timezone,
            "generated_at": generated_at.isoformat(timespec="seconds"),
            "channels": [
                {
                    "channel": channel.channel,
                    "root": channel.root,
                    "status": channel.status,
                    "files_scanned": channel.files_scanned,
                    "malformed_lines": channel.malformed_lines,
                    **asdict(channel.usage),
                }
                for channel in channels
            ],
            "all_channels": asdict(total),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"Grand total: {tokens_m(total.total_tokens)}M tokens "
            f"({number(total.total_tokens)})"
        )
        print(
            f"Date: {date_label}  Timezone: {args.timezone}  "
            f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print()
        print(render_table(channels, total))
        print()
        print(
            "口径：UncachedIn + CacheWrite + CacheRead + Output + Other = Total，"
            "token 列以百万（M）为单位。"
        )
        print(
            "Claude 流式记录按 message.id 去重；Codex CacheRead 是 Input 的子集。"
        )
        print(
            "Antigravity 按响应 ID 去重；Other 是只记录总量、未提供细分的 token。"
        )
        print(
            "OpenCode/MimoCode 的 reasoning、Gemini 的 thoughts+tool 计入 Other。"
        )
        print(
            "JSON 中的 ReasoningOutput 是 Output 的子集，不会重复计入 Total。"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
