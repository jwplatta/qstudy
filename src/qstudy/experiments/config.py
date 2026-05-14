from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qstudy.experiments.errors import ConfigError

CONFIG_FILENAME = ".qstudy.toml"


@dataclass(frozen=True)
class StudiesConfig:
    studies_root: Path
    data_root: Path | None
    source: Path | None


def load_studies_config(
    cwd: Path | None = None,
    home: Path | None = None,
) -> StudiesConfig:
    cwd = Path.cwd() if cwd is None else Path(cwd).resolve()
    home = Path.home() if home is None else Path(home).resolve()

    local_config = cwd / CONFIG_FILENAME
    if local_config.exists():
        config_values = _read_config(local_config)
        return StudiesConfig(
            studies_root=config_values["studies_dir"],
            data_root=config_values["data_dir"],
            source=local_config,
        )

    global_config = home / CONFIG_FILENAME
    if global_config.exists():
        config_values = _read_config(global_config)
        return StudiesConfig(
            studies_root=config_values["studies_dir"],
            data_root=config_values["data_dir"],
            source=global_config,
        )

    return StudiesConfig(studies_root=cwd, data_root=None, source=None)


def _read_config(config_path: Path) -> dict[str, Path | None]:
    raw = _parse_config_text(config_path.read_text(encoding="utf-8"), config_path)
    return {
        "studies_dir": _resolve_config_path(raw["studies_dir"], config_path),
        "data_dir": _resolve_config_path(raw["data_dir"], config_path),
    }


def _resolve_config_path(raw: str | None, config_path: Path) -> Path | None:
    if raw is None:
        return None
    resolved = Path(raw)
    if not resolved.is_absolute():
        resolved = (config_path.parent / resolved).resolve()
    return resolved


def _parse_config_text(text: str, config_path: Path) -> dict[str, str | None]:
    studies_dir: str | None = None
    data_dir: str | None = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if "=" not in line:
            raise ConfigError(f"Invalid config in {config_path}:{lineno}: expected key = value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in {"studies_dir", "data_dir"}:
            raise ConfigError(
                f"Invalid config in {config_path}:{lineno}: unsupported key {key!r}"
            )
        if key == "studies_dir":
            if studies_dir is not None:
                raise ConfigError(
                    f"Invalid config in {config_path}:{lineno}: duplicate studies_dir"
                )
            studies_dir = _parse_toml_string(value, config_path, lineno, key)
            continue

        if data_dir is not None:
            raise ConfigError(f"Invalid config in {config_path}:{lineno}: duplicate data_dir")
        data_dir = _parse_toml_string(value, config_path, lineno, key)

    if studies_dir is None:
        raise ConfigError(f"Invalid config in {config_path}: missing studies_dir")
    return {"studies_dir": studies_dir, "data_dir": data_dir}


def _strip_toml_comment(line: str) -> str:
    in_quote: str | None = None
    escaped = False
    out: list[str] = []
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\" and in_quote is not None:
            out.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            if in_quote is None:
                in_quote = char
            elif in_quote == char:
                in_quote = None
            out.append(char)
            continue
        if char == "#" and in_quote is None:
            break
        out.append(char)
    return "".join(out)


def _parse_toml_string(value: str, config_path: Path, lineno: int, key: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    raise ConfigError(
        f"Invalid config in {config_path}:{lineno}: {key} must be a quoted string"
    )
