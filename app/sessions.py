import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.config import settings


class ProjectMetadata(BaseModel):
    """Hechos conocidos sobre el proyecto en curso, separados del
    historial conversacional. Se inyectan en el system prompt en cada
    turno (ver app.prompts.loader.render_session_prompt) y se actualizan
    después de cada respuesta del LLM -- viven aparte del historial a
    propósito: el historial es "qué se dijo", esto es "qué sabemos"."""

    project_name: str | None = None
    assumed_team_size: int | None = None
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None

    def is_empty(self) -> bool:
        return not (
            self.project_name
            or self.assumed_team_size
            or self.mentioned_technologies
            or self.agreed_scope
        )

    def merge(self, update: "ProjectMetadata") -> "ProjectMetadata":
        """Combina lo nuevo sobre lo existente: un campo no vacío en
        `update` reemplaza al viejo; las tecnologías mencionadas se
        acumulan sin duplicados; un turno que no aporta datos nuevos no
        borra lo que ya se sabía (la extracción puede devolver None para
        un campo simplemente porque ese turno no lo tocó)."""
        merged_technologies = list(self.mentioned_technologies)
        for tech in update.mentioned_technologies:
            if tech not in merged_technologies:
                merged_technologies.append(tech)

        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=merged_technologies,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )


@dataclass
class ConversationHistory:
    """Ventana deslizante de turnos (pares user+assistant), aumentada con
    anclas y resumen acumulativo.

    Tres compartimentos:

    - `messages`: la ventana deslizante reciente (últimos `max_turns` pares).
      La mutan directamente `add_turn` y la compresión (ver app.compression).
    - `anchors`: turnos marcados por `AnchorDetector` como compromisos
      durables (NDA, alcance congelado, compliance...). Viven fuera de la
      ventana y la compresión nunca los evicta.
    - `summary`: texto libre con el resumen acumulado de los turnos
      no-ancla que ya se plegaron fuera de la ventana.

    El system prompt NO vive acá: se regenera fresco en cada turno a partir
    del project_metadata actual, así que nunca queda desactualizado respecto
    a lo que el servicio sabe del proyecto en ese momento.

    `add_turn` es una operación de datos pura -- no recorta ni promueve
    nada. Después de cada turno, el router llama explícitamente a
    `app.compression.apply_compression(history)`, que es la única pieza que
    muta `messages`/`anchors`/`summary` más allá de `add_turn`.
    """

    max_turns: int
    messages: list[dict] = field(default_factory=list)
    anchors: list[dict] = field(default_factory=list)
    summary: str | None = None

    def add_turn(self, user_message: dict, assistant_message: dict) -> None:
        self.messages.append(user_message)
        self.messages.append(assistant_message)

    def to_messages_list(self, system_prompt: str) -> list[dict]:
        """Arma el array `messages` listo para la API del LLM: system
        prompt + [resumen, si hay] + anclas (en orden) + ventana reciente.

        El orden es intencional: el resumen ancla al modelo en el arranque
        de la conversación, las anclas cargan los compromisos irreducibles
        palabra por palabra, y la ventana reciente da el contexto turno a
        turno activo.

        El resumen NO se agrega como un mensaje `user` aparte: como las
        anclas y la ventana siempre empiezan con un turno de usuario, eso
        dejaría dos mensajes `user` consecutivos. Algunos proveedores (la
        API de mensajes de Anthropic, en particular) exigen alternancia
        estricta user/assistant y rechazan esa forma; en la práctica
        también vimos que el propio pipeline de Instructor la fusiona
        silenciosamente. En vez de eso, el resumen se antepone al
        contenido del primer mensaje `user` que exista.
        """
        body = [*self.anchors, *self.messages]
        if self.summary:
            summary_block = (
                "[Resumen de la conversación anterior -- los turnos "
                "recientes abajo son el hilo vivo]\n" + self.summary
            )
            if body:
                first = body[0]
                body = [
                    {"role": "user", "content": f"{summary_block}\n\n{first['content']}"},
                    *body[1:],
                ]
            else:
                body = [{"role": "user", "content": summary_block}]
        return [{"role": "system", "content": system_prompt}, *body]

    def __len__(self) -> int:
        """Cantidad de turnos en la ventana ACTIVA (sin contar anclas ni
        resumen) -- es la cifra que demuestra que la ventana deslizante
        respeta su límite."""
        return len(self.messages) // 2


@dataclass
class Session:
    session_id: str
    history: ConversationHistory
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Cachean la última resolución de tier para que GET /sessions/{id} pueda
    # mostrarla sin tener que re-correr el resolver.
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None


class SessionStore:
    """Las sesiones viven en un diccionario del proceso -- sin base de
    datos ni Redis. Se pierden al reiniciar el servicio.

    Es una volatilidad aceptada a propósito en esta fase: la consigna
    excluye explícitamente la persistencia entre reinicios ("un
    diccionario en memoria del proceso es suficiente para esta fase").
    Si el servicio corre con más de un worker, cada worker tendría su
    propio diccionario -- fuera de alcance acá, se resolvería con un
    store compartido (Redis) cuando haga falta.
    """

    def __init__(self, max_turns: int):
        self._max_turns = max_turns
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            history=ConversationHistory(max_turns=self._max_turns),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)


session_store = SessionStore(max_turns=settings.max_conversation_turns)
