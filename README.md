# WINTEM F300 estático

Este proyecto convierte un archivo WINTEM local en un sitio compuesto únicamente por HTML, PNG y CSV. No usa Tkinter, Basemap, servidor Python ni código de cálculo en el navegador.

## Generar el sitio

```powershell
python -m pip install -r requirements.txt
python generate_static_site.py "C:\ruta\al\boletin.txt" --output docs
python -m http.server 8000 --directory docs
```

Abra `http://localhost:8000`. El servidor local es solo para previsualizar; GitHub Pages sirve los archivos estáticos directamente.

Cada nueva corrida reemplaza los archivos de igual nombre dentro de `docs`. Después se versionan y publican:

```powershell
git add docs
git commit -m "Actualizar análisis WINTEM F300"
git push
```

## Activar GitHub Pages

En el repositorio de GitHub abra **Settings → Pages**. En **Build and deployment**, seleccione **Deploy from a branch**, la rama `main` y la carpeta `/docs`. Guarde. GitHub mostrará la URL pública cuando termine el primer despliegue.

## Contenido generado

- `docs/index.html`: informe navegable y adaptable a móviles.
- `docs/assets/*.png`: siete productos gráficos.
- `docs/resultados_eta_f300.csv`: datos y diagnósticos por punto.
- `docs/resumen_eta_por_boletin.csv`: estadísticas por boletín.
- `docs/.nojekyll`: indica a Pages que publique los archivos sin procesamiento Jekyll.

El cálculo conserva las constantes del programa de escritorio: 1° = 110 km, 1 KT = 0.5 m/s, Ω = 7.2×10⁻⁵ rad/s, g = 9.8 m/s² y máscara ecuatorial |φ| ≤ 5°.
