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
    """Ventana deslizante de turnos (pares user+assistant).

    Guarda solo los mensajes de la conversación -- el system prompt NO
    vive acá: se regenera fresco en cada turno a partir del
    project_metadata actual, así que nunca queda desactualizado respecto
    a lo que el servicio sabe del proyecto en ese momento.
    """

    max_turns: int
    _turns: list[tuple[dict, dict]] = field(default_factory=list)

    def add_turn(self, user_message: dict, assistant_message: dict) -> None:
        self._turns.append((user_message, assistant_message))

    def to_messages_list(self, system_prompt: str) -> list[dict]:
        """Arma el array `messages` listo para la API del LLM: el system
        prompt (siempre presente, regenerado por quien llama) + los
        últimos `max_turns` turnos. Los turnos más viejos se descartan
        primero -- ventana deslizante simple, sin resumen ni anclas.
        """
        windowed = self._turns[-self.max_turns :] if self.max_turns > 0 else []
        messages = [{"role": "system", "content": system_prompt}]
        for user_message, assistant_message in windowed:
            messages.append(user_message)
            messages.append(assistant_message)
        return messages

    def __len__(self) -> int:
        return len(self._turns)


@dataclass
class Session:
    session_id: str
    history: ConversationHistory
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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
