"""Decodificación y diagnóstico numérico WINTEM F300 sin interfaz gráfica."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_LEVEL = "F300"
KT_TO_M_S = 0.5
DEGREE_TO_M = 110_000.0
GRAVITY_M_S2 = 9.8
OMEGA_RAD_S = 7.2e-5
EQUATORIAL_MASK_DEG = 5.0

LATITUDE_RE = re.compile(r"^(\d{3})([NS])$")
LONGITUDE_RE = re.compile(r"^(\d{4})([EW])$")
BULLETIN_RE = re.compile(r"^(FB\w+)\s+KWBC\b")
GROUP_RE = re.compile(
    r"^(?P<direction>\d{2})(?P<speed>\d{3})(?P<minus>M?)(?P<temperature>\d{2})$"
)


@dataclass(frozen=True, slots=True)
class WintemPoint:
    bulletin: str
    latitude: float
    longitude: float
    direction_deg: int
    speed_kt: int
    temperature_c: int

    @property
    def speed_m_s(self) -> float:
        return self.speed_kt * KT_TO_M_S


@dataclass(frozen=True, slots=True)
class Bulletin:
    name: str
    points: tuple[WintemPoint, ...]


@dataclass(frozen=True, slots=True)
class ParseResult:
    bulletins: dict[str, Bulletin]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisGrid:
    latitudes: np.ndarray
    longitudes: np.ndarray
    temperature: np.ndarray
    speed_m_s: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray
    d2t_dx2: np.ndarray
    d2t_dy2: np.ndarray
    laplacian: np.ndarray
    coriolis: np.ndarray
    eta_temperature: np.ndarray
    dx_m: float
    dy_m: float

    @property
    def valid_eta_count(self) -> int:
        return int(np.isfinite(self.eta_temperature).sum())


@dataclass(frozen=True, slots=True)
class BulletinSummary:
    bulletin: str
    total_points: int
    valid_points: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeError(f"No fue posible decodificar el archivo: {path}")


def _coordinate(token: str) -> float:
    token = token.strip().upper()
    pattern = LATITUDE_RE if token.endswith(("N", "S")) else LONGITUDE_RE
    match = pattern.fullmatch(token)
    if match is None:
        raise ValueError(f"Coordenada no válida: {token!r}")
    value = int(match.group(1)) / 10.0
    return -value if match.group(2) in {"S", "W"} else value


def _group(token: str) -> tuple[int, int, int]:
    match = GROUP_RE.fullmatch(token.strip().upper())
    if match is None:
        raise ValueError(f"Grupo F300 no válido: {token!r}")
    temperature = int(match.group("temperature"))
    if match.group("minus"):
        temperature *= -1
    return int(match.group("direction")) * 10, int(match.group("speed")), temperature


def parse_wintem(path: Path, target_level: str = TARGET_LEVEL) -> ParseResult:
    points: dict[str, list[WintemPoint]] = {}
    warnings: list[str] = []
    bulletin: str | None = None
    latitude: float | None = None
    longitudes: list[float] = []

    for line_number, raw_line in enumerate(_read_text(path).splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        header = BULLETIN_RE.match(line)
        if header:
            bulletin = header.group(1)
            points.setdefault(bulletin, [])
            latitude, longitudes = None, []
            continue
        tokens = line.split()
        first = tokens[0].upper()
        if LATITUDE_RE.fullmatch(first):
            if bulletin is None:
                raise ValueError(f"Línea {line_number}: coordenada sin encabezado de boletín.")
            latitude = _coordinate(first)
            if len(tokens) > 1:
                longitudes = [_coordinate(token) for token in tokens[1:]]
            if not longitudes:
                raise ValueError(f"Línea {line_number}: faltan longitudes.")
            continue
        if first != target_level:
            continue
        if bulletin is None or latitude is None or not longitudes:
            raise ValueError(f"Línea {line_number}: {target_level} fuera de un bloque válido.")
        groups = tokens[1:]
        if len(groups) != len(longitudes):
            warnings.append(
                f"Línea {line_number}: {len(groups)} grupos para {len(longitudes)} longitudes."
            )
        for longitude, encoded in zip(longitudes, groups):
            try:
                direction, speed, temperature = _group(encoded)
            except ValueError as error:
                warnings.append(f"Línea {line_number}: {error}")
                continue
            points[bulletin].append(
                WintemPoint(bulletin, latitude, longitude, direction, speed, temperature)
            )

    result = {
        name: Bulletin(name, tuple(sorted(values, key=lambda p: (-p.latitude, p.longitude))))
        for name, values in points.items()
        if values
    }
    if not result:
        raise ValueError(f"No se encontraron observaciones {target_level} decodificables.")
    return ParseResult(result, tuple(warnings))


def meteorological_components(point: WintemPoint) -> tuple[float, float]:
    angle = np.deg2rad(point.direction_deg)
    return (
        -point.speed_m_s * float(np.sin(angle)),
        -point.speed_m_s * float(np.cos(angle)),
    )


def _spacing(values: np.ndarray, name: str) -> float:
    differences = np.diff(values)
    if differences.size == 0 or np.any(differences <= 0):
        raise ValueError(f"La coordenada {name} no es estrictamente creciente.")
    if not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-8):
        raise ValueError(f"La coordenada {name} no forma una malla regular.")
    return float(differences[0])


def build_analysis_grid(
    bulletins: dict[str, Bulletin], equatorial_mask_deg: float = EQUATORIAL_MASK_DEG
) -> AnalysisGrid:
    points = [point for item in bulletins.values() for point in item.points]
    latitudes = np.array(sorted({point.latitude for point in points}), dtype=float)
    longitudes = np.array(sorted({point.longitude for point in points}), dtype=float)
    if len(latitudes) < 3 or len(longitudes) < 3:
        raise ValueError("Se requieren al menos tres latitudes y tres longitudes.")
    dx_m = _spacing(longitudes, "longitud") * DEGREE_TO_M
    dy_m = _spacing(latitudes, "latitud") * DEGREE_TO_M
    shape = len(latitudes), len(longitudes)
    temperature, speed, u, v = (np.full(shape, np.nan) for _ in range(4))
    rows = {value: index for index, value in enumerate(latitudes)}
    columns = {value: index for index, value in enumerate(longitudes)}
    for point in points:
        row, column = rows[point.latitude], columns[point.longitude]
        previous = temperature[row, column]
        if np.isfinite(previous) and not np.isclose(previous, point.temperature_c):
            raise ValueError(f"Temperaturas contradictorias en ({point.latitude}, {point.longitude}).")
        temperature[row, column] = point.temperature_c
        speed[row, column] = point.speed_m_s
        u[row, column], v[row, column] = meteorological_components(point)
    d2x, d2y = np.full(shape, np.nan), np.full(shape, np.nan)
    d2x[:, 1:-1] = (temperature[:, 2:] - 2 * temperature[:, 1:-1] + temperature[:, :-2]) / dx_m**2
    d2y[1:-1, :] = (temperature[2:, :] - 2 * temperature[1:-1, :] + temperature[:-2, :]) / dy_m**2
    laplacian = d2x + d2y
    coriolis = 2 * OMEGA_RAD_S * np.sin(np.deg2rad(latitudes))
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = GRAVITY_M_S2 / coriolis[:, np.newaxis] * laplacian
    eta[np.abs(latitudes) <= equatorial_mask_deg, :] = np.nan
    return AnalysisGrid(latitudes, longitudes, temperature, speed, u, v, d2x, d2y, laplacian, coriolis, eta, dx_m, dy_m)


def point_diagnostic(point: WintemPoint, grid: AnalysisGrid) -> tuple[float, ...]:
    row = int(np.flatnonzero(np.isclose(grid.latitudes, point.latitude))[0])
    column = int(np.flatnonzero(np.isclose(grid.longitudes, point.longitude))[0])
    return grid.d2t_dx2[row, column], grid.d2t_dy2[row, column], grid.laplacian[row, column], grid.coriolis[row], grid.eta_temperature[row, column]


def summarize_bulletins(bulletins: dict[str, Bulletin], grid: AnalysisGrid) -> tuple[BulletinSummary, ...]:
    summaries = []
    for bulletin in bulletins.values():
        values = np.array([point_diagnostic(point, grid)[-1] for point in bulletin.points])
        finite = values[np.isfinite(values)]
        summaries.append(BulletinSummary(
            bulletin.name, len(values), len(finite),
            float(np.mean(finite)) if finite.size else np.nan,
            float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
            float(np.min(finite)) if finite.size else np.nan,
            float(np.max(finite)) if finite.size else np.nan,
        ))
    return tuple(summaries)
