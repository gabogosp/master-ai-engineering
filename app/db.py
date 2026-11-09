"""Conexión a Postgres.

Un único engine a nivel de módulo, creado al importar -- no por
request. Crear un engine abre un pool de conexiones; hacerlo por cada
petición agotaría los file descriptors del proceso en minutos bajo
cualquier carga real.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url)

# expire_on_commit=False: después de un commit, los objetos ya
# persistidos (p.ej. un Document recién creado) siguen siendo legibles
# sin que SQLAlchemy dispare una query nueva solo para refrescarlos --
# el handler del endpoint necesita leer el id generado justo después
# del commit para armar la respuesta.
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency de FastAPI: una sesión nueva por request, cerrada al
    terminar el request sin importar si hubo excepción o no."""
    async with AsyncSessionLocal() as session:
        yield session
