# Imagen del servicio IA -- este repo no había containerizado el
# FastAPI (docker-compose.yml solo levantaba Redis) porque hasta ahora
# todo corría con `uv run` directo. Se agrega para poder invocar
# scripts/compare.py (y el resto del servicio) también dentro de un
# contenedor, no solo vía `uv run` en el host.
FROM python:3.13-slim

# Copia el binario oficial de uv en vez de instalarlo con pip -- evita
# resolver una dependencia más y mantiene la imagen chica. Versión
# fijada para que el build sea reproducible (misma que la usada en
# desarrollo local).
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

WORKDIR /app

# Las dependencias se instalan ANTES de copiar el código: así Docker
# cachea esta capa y un cambio en app/ no reinstala todo uv.lock de
# nuevo en cada build.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# `uv sync` instala en /app/.venv, no en el Python de sistema. Sin esto,
# `docker compose exec servicio_ia python scripts/compare.py ...` (sin
# `uv run` delante) ejecuta el Python de sistema, que no tiene ninguna
# dependencia instalada.
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
