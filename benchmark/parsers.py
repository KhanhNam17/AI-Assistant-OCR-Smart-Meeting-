"""Parsers for transcript files in the "timestamp - speaker - text" format.

Supported inputs:
  - JSON: a list of objects, each with start/end times, a speaker label and
    text. Several common key spellings are accepted (start/start_time/begin,
    end/end_time, speaker/speaker_id/spk, text/content/transcript).
  - Plain text lines such as:
        [00:00:12.500 --> 00:00:15.300] Speaker 1: noi dung...
        00:00:12.500 - 00:00:15.300  Speaker 1: noi dung...
        12.5\t15.3\tSpeaker 1\tnoi dung...
        12.5 - 15.3 | Speaker 1 | noi dung...
  - Plain text lines with a single timestamp (turn start only), e.g.:
        [00:05] Speaker 1: noi dung...
        [01:22] Speaker 2: noi dung...
    For this style, each turn's end time is inferred as the start time of
    the next turn (and, for the last turn, as start + the previous turn's
    duration).

Each parsed line/object becomes a Segment(start, end, speaker, text).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


_TIME_RE = r"(\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:\.\d+)?"

_LINE_PATTERNS = [
    # [00:00:12.500 --> 00:00:15.300] Speaker 1: text
    re.compile(
        r"^\s*\[?(?P<start>" + _TIME_RE + r")\]?\s*(?:-->|->|-|–|to)\s*"
        r"\[?(?P<end>" + _TIME_RE + r")\]?\s*\]?\s*[:\-|]?\s*"
        r"(?P<speaker>[^:|\t]+?)\s*[:|]\s*(?P<text>.+)$"
    ),
    # 12.5\t15.3\tSpeaker 1\ttext  (tab/pipe separated, no colon)
    re.compile(
        r"^\s*(?P<start>" + _TIME_RE + r")\s*[\t|]\s*"
        r"(?P<end>" + _TIME_RE + r")\s*[\t|]\s*"
        r"(?P<speaker>[^\t|]+?)\s*[\t|]\s*(?P<text>.+)$"
    ),
]

# [00:05] Speaker 1: text  (single timestamp = turn start; end is inferred)
_SINGLE_TIME_PATTERN = re.compile(
    r"^\s*\[(?P<start>" + _TIME_RE + r")\]\s*(?P<speaker>[^:]+?):\s*(?P<text>.+)$"
)


def _parse_timestamp(value: str) -> float:
    """Convert a timestamp string to seconds. Accepts HH:MM:SS(.mmm), MM:SS(.mmm) or raw seconds."""
    value = value.strip().replace(",", ".")
    if ":" not in value:
        return float(value)
    parts = value.split(":")
    parts = [float(p) for p in parts]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


_KEY_ALIASES = {
    "start": ["start", "start_time", "begin", "from", "ts_start", "t0"],
    "end": ["end", "end_time", "stop", "to", "ts_end", "t1"],
    "speaker": ["speaker", "speaker_id", "spk", "speaker_label", "label"],
    "text": ["text", "content", "transcript", "utterance", "words"],
}


def _get_key(obj: dict, field: str):
    for key in _KEY_ALIASES[field]:
        if key in obj:
            return obj[key]
    return None


def _parse_json(content: str) -> list[Segment]:
    data = json.loads(content)
    if isinstance(data, dict):
        # allow {"segments": [...]} wrappers
        for key in ("segments", "utterances", "data", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON transcript must be a list of segments")

    segments: list[Segment] = []
    for obj in data:
        start = _get_key(obj, "start")
        end = _get_key(obj, "end")
        speaker = _get_key(obj, "speaker")
        text = _get_key(obj, "text")
        if start is None or end is None or speaker is None or text is None:
            raise ValueError(f"Missing required field in JSON segment: {obj}")
        segments.append(
            Segment(
                start=_parse_timestamp(str(start)),
                end=_parse_timestamp(str(end)),
                speaker=str(speaker).strip(),
                text=str(text).strip(),
            )
        )
    return segments


def _parse_text(content: str) -> list[Segment]:
    segments: list[Segment] = []
    # Indices of segments parsed from the single-timestamp style, whose
    # `end` still needs to be inferred from the following turn's start.
    single_ts_indices: list[int] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = None
        for pattern in _LINE_PATTERNS:
            match = pattern.match(line)
            if match:
                matched = match
                break
        if matched is not None:
            groups = matched.groupdict()
            segments.append(
                Segment(
                    start=_parse_timestamp(groups["start"]),
                    end=_parse_timestamp(groups["end"]),
                    speaker=groups["speaker"].strip(),
                    text=groups["text"].strip(),
                )
            )
            continue

        single_match = _SINGLE_TIME_PATTERN.match(line)
        if single_match is None:
            raise ValueError(f"Could not parse transcript line: {line!r}")
        groups = single_match.groupdict()
        start = _parse_timestamp(groups["start"])
        single_ts_indices.append(len(segments))
        segments.append(
            Segment(
                start=start,
                end=start,  # placeholder, inferred below
                speaker=groups["speaker"].strip(),
                text=groups["text"].strip(),
            )
        )

    # Infer end times for single-timestamp turns: end = next turn's start,
    # and for the final turn, end = start + previous turn's duration.
    for idx in single_ts_indices:
        if idx + 1 < len(segments):
            segments[idx].end = segments[idx + 1].start
        elif idx > 0:
            prev_duration = segments[idx].start - segments[idx - 1].start
            segments[idx].end = segments[idx].start + max(prev_duration, 0.0)

    return segments


def parse_transcript(content: str) -> list[Segment]:
    """Parse transcript content (JSON or plain text) into a list of Segments."""
    content = content.strip()
    if not content:
        return []
    if content[0] in "[{":
        try:
            return _parse_json(content)
        except (json.JSONDecodeError, ValueError):
            pass
    return _parse_text(content)


def segments_to_text(segments: list[Segment]) -> str:
    """Concatenate the spoken text of all segments, in time order, for WER/CER scoring."""
    ordered = sorted(segments, key=lambda s: s.start)
    return " ".join(seg.text for seg in ordered if seg.text.strip())
