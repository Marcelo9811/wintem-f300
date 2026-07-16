"""Genera un sitio estático WINTEM (HTML, PNG y CSV) listo para GitHub Pages."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from wintem_core import (
    EQUATORIAL_MASK_DEG, AnalysisGrid, BulletinSummary, build_analysis_grid,
    meteorological_components, parse_wintem, point_diagnostic, summarize_bulletins,
)

PRODUCTS = (
    ("wind_map.png", "Viento y velocidad", "Velocidad regional con vectores de viento."),
    ("temperature_map.png", "Temperatura", "Campo térmico regional en F300."),
    ("d2x_map.png", "Segunda derivada zonal", "Curvatura térmica en la dirección x."),
    ("d2y_map.png", "Segunda derivada meridional", "Curvatura térmica en la dirección y."),
    ("laplacian_map.png", "Laplaciano", "Suma de las dos segundas derivadas."),
    ("eta_map.png", "Vorticidad térmica", "Diagnóstico ηT; banda ecuatorial enmascarada."),
    ("bulletin_comparison.png", "Comparación por boletín", "Media y dispersión de ηT."),
)


def _finite(value: float, scale: float = 1.0, decimals: int = 4) -> str:
    return "—" if not np.isfinite(value) else f"{value * scale:.{decimals}f}"


def _axis(axis, grid: AnalysisGrid, title: str) -> None:
    axis.set(title=title, xlabel="Longitud (°)", ylabel="Latitud (°)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#64748b", alpha=.25, linewidth=.6)
    axis.set_xlim(grid.longitudes.min() - 2.5, grid.longitudes.max() + 2.5)
    axis.set_ylim(grid.latitudes.min() - 2.5, grid.latitudes.max() + 2.5)


def _scalar(grid: AnalysisGrid, field: np.ndarray, path: Path, title: str, label: str, cmap: str, centered: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    finite = field[np.isfinite(field)]
    norm = None
    if centered and finite.size and np.max(np.abs(finite)) > 0:
        limit = float(np.max(np.abs(finite)))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = axis.pcolormesh(grid.longitudes, grid.latitudes, np.ma.masked_invalid(field), shading="nearest", cmap=cmap, norm=norm)
    _axis(axis, grid, title)
    figure.colorbar(image, ax=axis, label=label, shrink=.82)
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _wind(grid: AnalysisGrid, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    image = axis.pcolormesh(grid.longitudes, grid.latitudes, np.ma.masked_invalid(grid.speed_m_s), shading="nearest", cmap="turbo")
    lon, lat = np.meshgrid(grid.longitudes, grid.latitudes)
    mask = np.isfinite(grid.wind_u) & np.isfinite(grid.wind_v)
    axis.quiver(lon[mask], lat[mask], grid.wind_u[mask], grid.wind_v[mask], color="#0f172a", scale=700, width=.0018)
    _axis(axis, grid, "Viento y velocidad regional — F300")
    figure.colorbar(image, ax=axis, label="Velocidad (m/s)", shrink=.82)
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _comparison(summaries: tuple[BulletinSummary, ...], path: Path) -> None:
    labels = [item.bulletin for item in summaries]
    means = np.array([item.mean for item in summaries]) * 1e6
    deviations = np.nan_to_num(np.array([item.standard_deviation for item in summaries]) * 1e6)
    figure, axis = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    axis.bar(labels, means, yerr=deviations, capsize=4, color="#2563eb")
    axis.axhline(0, color="#0f172a", linewidth=.8)
    axis.set(title="Comparación de ηT por boletín", ylabel="ηT × 10⁶", xlabel="Boletín")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=.25)
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _csv_files(output: Path, bulletins, grid: AnalysisGrid, summaries) -> None:
    with (output / "resultados_eta_f300.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("bulletin", "latitude", "longitude", "direction_deg", "speed_kt", "speed_m_s", "temperature_c", "d2T_dx2", "d2T_dy2", "laplacian", "coriolis_s-1", "eta_temperature"))
        for bulletin in bulletins.values():
            for point in bulletin.points:
                writer.writerow((bulletin.name, point.latitude, point.longitude, point.direction_deg, point.speed_kt, point.speed_m_s, point.temperature_c, *point_diagnostic(point, grid)))
    with (output / "resumen_eta_por_boletin.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("bulletin", "total_points", "valid_points", "mean", "std", "minimum", "maximum"))
        for item in summaries:
            writer.writerow((item.bulletin, item.total_points, item.valid_points, item.mean, item.standard_deviation, item.minimum, item.maximum))


def _html(output: Path, source: Path, bulletins, grid: AnalysisGrid, summaries, warnings: tuple[str, ...]) -> None:
    rows = "".join(f"<tr><td>{html.escape(s.bulletin)}</td><td>{s.total_points}</td><td>{s.valid_points}</td><td>{_finite(s.mean, 1e6)}</td><td>{_finite(s.standard_deviation, 1e6)}</td><td>{_finite(s.minimum, 1e6)}</td><td>{_finite(s.maximum, 1e6)}</td></tr>" for s in summaries)
    cards = "".join(f'<article class="product"><a href="assets/{name}"><img src="assets/{name}" alt="{title}" loading="lazy"></a><div><h3>{title}</h3><p>{description}</p></div></article>' for name, title, description in PRODUCTS)
    bulletin_tables = []
    for bulletin in bulletins.values():
        point_rows = []
        for point in bulletin.points:
            d2x, d2y, laplacian, coriolis, eta = point_diagnostic(point, grid)
            point_rows.append(
                "<tr>"
                f"<td>{point.latitude:g}°</td>"
                f"<td>{point.longitude:g}°</td>"
                f"<td>{point.direction_deg:03d}°</td>"
                f"<td>{point.speed_kt}</td>"
                f"<td>{point.speed_m_s:.1f}</td>"
                f"<td>{point.temperature_c}</td>"
                f"<td>{_finite(d2x, 1e11)}</td>"
                f"<td>{_finite(d2y, 1e11)}</td>"
                f"<td>{_finite(laplacian, 1e11)}</td>"
                f"<td>{_finite(coriolis, 1e5)}</td>"
                f"<td>{_finite(eta, 1e6)}</td>"
                "</tr>"
            )
        bulletin_tables.append(
            f'<article class="bulletin"><div class="bulletin-title">'
            f'<h3>{html.escape(bulletin.name)}</h3><span>{len(bulletin.points)} puntos F300</span>'
            '</div><div class="table"><table><thead><tr>'
            '<th>Latitud</th><th>Longitud</th><th>Dirección</th><th>KT</th>'
            '<th>m/s</th><th>T (°C)</th><th>d²T/dx² ×10¹¹</th>'
            '<th>d²T/dy² ×10¹¹</th><th>∇²T ×10¹¹</th><th>f ×10⁵</th>'
            '<th>ηT ×10⁶</th></tr></thead><tbody>'
            f'{"".join(point_rows)}</tbody></table></div></article>'
        )
    bulletin_content = "".join(bulletin_tables)
    warning_block = "" if not warnings else f'<aside><strong>Advertencias de decodificación:</strong><ul>{"".join(f"<li>{html.escape(w)}</li>" for w in warnings)}</ul></aside>'
    point_count = sum(len(item.points) for item in bulletins.values())
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    document = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WINTEM F300</title><link rel="stylesheet" href="assets/styles.css"></head><body><header><div class="wrap"><p class="eyebrow">ANÁLISIS METEOROLÓGICO · F300</p><h1>Boletines WINTEM</h1><p>Resultados estáticos generados localmente. No requieren Python, servidor ni conexión a una API.</p><nav><a href="#productos">Mapas</a><a href="#resumen">Resumen</a><a href="#boletines">Todos los boletines</a><a href="#metodo">Método</a></nav></div></header><main class="wrap"><section class="stats"><div><b>{len(bulletins)}</b><span>boletines</span></div><div><b>{point_count}</b><span>puntos F300</span></div><div><b>{len(grid.latitudes)} × {len(grid.longitudes)}</b><span>malla regional</span></div><div><b>{grid.valid_eta_count}</b><span>ηT válidos</span></div></section>{warning_block}<section id="productos"><div class="heading"><h2>Productos gráficos</h2><p>Seleccione una imagen para verla a resolución completa.</p></div><div class="gallery">{cards}</div></section><section id="resumen"><div class="heading"><h2>Resumen por boletín</h2><p>Valores de ηT expresados × 10⁶.</p></div><div class="table"><table><thead><tr><th>Boletín</th><th>Puntos</th><th>Válidos</th><th>Media</th><th>Desv.</th><th>Mín.</th><th>Máx.</th></tr></thead><tbody>{rows}</tbody></table></div></section><section id="boletines"><div class="heading"><h2>Todos los boletines</h2><p>Datos F300 y diagnósticos completos para los {point_count} puntos.</p></div><div class="bulletins">{bulletin_content}</div></section><section id="metodo"><div class="heading"><h2>Metodología</h2></div><div class="method"><p>Se extrae exclusivamente F300 y se ensamblan todos los boletines en una malla regional regular. Se usa 1° = 110 km y 1 KT = 0.5 m/s.</p><p>Las segundas derivadas se calculan con diferencias centradas. El diagnóstico es ηT = (g/f) · (∂²T/∂x² + ∂²T/∂y²), con g = 9.8 m/s² y f = 2Ω·sin(φ). Se excluye |latitud| ≤ {EQUATORIAL_MASK_DEG:g}° y los bordes o puntos sin vecindad completa quedan como NaN.</p><p>ηT es un diagnóstico térmico operativo; no equivale a la vorticidad cinemática del viento ni demuestra por sí solo la presencia de frentes, jets o ciclogénesis.</p></div></section></main><footer><div class="wrap">Fuente: {html.escape(source.name)} · Generado {generated}</div></footer></body></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")


def generate(source: Path, output: Path) -> None:
    result = parse_wintem(source)
    grid = build_analysis_grid(result.bulletins)
    summaries = summarize_bulletins(result.bulletins, grid)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    _wind(grid, assets / "wind_map.png")
    _scalar(grid, grid.temperature, assets / "temperature_map.png", "Temperatura regional — F300", "Temperatura (°C)", "coolwarm")
    _scalar(grid, grid.d2t_dx2 * 1e11, assets / "d2x_map.png", "Segunda derivada zonal", "10⁻¹¹ K m⁻²", "PuOr_r", True)
    _scalar(grid, grid.d2t_dy2 * 1e11, assets / "d2y_map.png", "Segunda derivada meridional", "10⁻¹¹ K m⁻²", "PuOr_r", True)
    _scalar(grid, grid.laplacian * 1e11, assets / "laplacian_map.png", "Laplaciano horizontal de temperatura", "10⁻¹¹ K m⁻²", "BrBG_r", True)
    _scalar(grid, grid.eta_temperature * 1e6, assets / "eta_map.png", "Vorticidad relativa de la temperatura ηT", "ηT × 10⁶", "RdBu_r", True)
    _comparison(summaries, assets / "bulletin_comparison.png")
    _csv_files(output, result.bulletins, grid, summaries)
    shutil.copyfile(Path(__file__).with_name("styles.css"), assets / "styles.css")
    (output / ".nojekyll").touch()
    _html(output, source, result.bulletins, grid, summaries, result.warnings)
    print(f"Sitio generado en: {output.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Archivo de boletines WINTEM")
    parser.add_argument("--output", "-o", type=Path, default=Path("docs"), help="Carpeta de salida (predeterminado: docs)")
    args = parser.parse_args()
    generate(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
