"""Tres escenarios sintéticos multi-turno con fact-tracker, para el
stress-test de sesión 6 (Bloque 2 / Paso 2).

Cada escenario es una secuencia FIJA de `ScenarioTurn`. El runner (Bloque
3) los recorre en orden contra una única sesión, así que el turno N ve la
vista ya comprimida de los turnos 1..N-1 -- exactamente el mecanismo que
queremos estresar.

El campo `fact_to_remember` de cada turno declara qué hecho debería poder
encontrar una llamada POSTERIOR (en el historial, las anclas, el resumen,
o el project_metadata). Eso alimenta `MemoryDriftMetric`, que se construye
recién en el Bloque 4 -- acá solo declaramos los hechos a trackear, no los
verificamos todavía.

Los tres perfiles apuntan a un modo de falla distinto del CAG:

- `growing`       -- la sesión acumula requisitos sin parar. Estresa el
                      descarte de la ventana deslizante: para el turno 20,
                      el hecho del turno 1 (nombre del proyecto) tiene que
                      sobrevivir como ancla o dentro del resumen
                      acumulativo -- la ventana cruda ya lo perdió hace
                      rato (MAX_CONVERSATION_TURNS=6 por defecto).
- `pivot`         -- el turno 5 cambia el stack. Prueba si
                      `mentioned_technologies` ACUMULA ambas tecnologías
                      (comportamiento esperado de ProjectMetadata.merge,
                      que es aditivo a propósito) o si se "olvida" de la
                      primera -- drift real, no el acumulado esperado.
- `contradiction` -- el turno 3 dice un presupuesto, el turno 8 lo
                      contradice. Prueba qué versión sobrevive en la
                      metadata/resumen: la regla de negocio de
                      ProjectMetadata.merge es que gana el dato MÁS
                      reciente, así que el valor correcto a esperar es el
                      del turno 8, no el del 3.

Por qué 20 turnos por perfil, no menos: el Bloque 2 corre cada perfil
truncado a N ∈ {1, 3, 6, 10, 20} (ver `turns_up_to`) para graficar costo/
latencia/drift en función de la longitud de la conversación -- el perfil
completo tiene que alcanzar para el N más grande.
"""

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from app.schemas import DetailLevel, OutputFormat, ProjectType
from evals.stress.metrics import FactField


class ScenarioTurn(NamedTuple):
    """Un turno de un escenario guionado.

    `fact_to_remember` es el substring literal que una llamada posterior
    debería poder encontrar (case-insensitive) en algún lugar del snapshot
    de sesión -- `None` si el turno no introduce nada trackeable (relleno
    coherente de alcance, sin un hecho puntual que valga la pena medir).
    `fact_field` acota DÓNDE buscarlo: útil para que el pivot scenario
    pueda exigir específicamente "Flutter en mentioned_technologies", no
    "Flutter en cualquier parte del texto".
    """

    turn_index: int
    transcript: str
    fact_to_remember: str | None = None
    fact_field: FactField = "any"


@dataclass(frozen=True)
class Scenario:
    name: Literal["growing", "pivot", "contradiction"]
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    turns: list[ScenarioTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.turns) < 20:
            raise ValueError(
                f"el escenario {self.name!r} tiene {len(self.turns)} turnos; "
                "el ejercicio pide al menos 20 para poder correr hasta N=20 "
                "y para que la ventana deslizante realmente se estrese"
            )

    def turns_up_to(self, n: int) -> list[ScenarioTurn]:
        """Los primeros `n` turnos -- lo que usa el runner del Bloque 2
        para correr el mismo perfil en N ∈ {1, 3, 6, 10, 20}."""
        return [t for t in self.turns if t.turn_index <= n]


# --- growing: "MediFlow" crece sin parar -------------------------------

_GROWING_TURNS: list[ScenarioTurn] = [
    ScenarioTurn(
        1,
        "Queremos un SaaS llamado MediFlow para una cadena de clínicas de "
        "salud. Stack: React + Postgres. Equipo de 3 personas. Alcance "
        "inicial: agenda de turnos, historia clínica básica, panel de "
        "recepción.",
        "MediFlow",
        "project_name",
    ),
    ScenarioTurn(
        2,
        "Agreguen autenticación con SSO vía Google Workspace para el personal.",
        "SSO con Google Workspace",
        "technologies",
    ),
    ScenarioTurn(
        3,
        "Presupuesto confirmado: 55.000 EUR para la primera entrega. El "
        "equipo se mantiene en 3 personas.",
        "presupuesto 55000",
    ),
    ScenarioTurn(
        4,
        "Necesitamos un registro de auditoría de cada cambio en la historia "
        "clínica, con retención de 10 años por norma.",
        "registro de auditoría",
    ),
    ScenarioTurn(
        5,
        "Agreguen importación de pacientes existentes desde un CSV, "
        "esperamos unos 20.000 registros.",
        "importación CSV",
    ),
    ScenarioTurn(
        6,
        "Multi-tenant: vamos a vender esto a otras 2 cadenas de clínicas. "
        "Aislamiento estricto de datos por tenant.",
        "multi-tenant",
    ),
    ScenarioTurn(
        7,
        "Sumá notificaciones por email y WhatsApp cuando se acerca un turno.",
        "notificaciones WhatsApp",
        "technologies",
    ),
    ScenarioTurn(
        8,
        "API pública de solo lectura para que las obras sociales consulten "
        "disponibilidad. OAuth2 con client credentials.",
        "OAuth2",
    ),
    ScenarioTurn(
        9,
        "Agreguen un módulo de reportes: turnos atendidos por profesional, "
        "exportable a PDF semanalmente.",
        "reportes PDF semanales",
    ),
    ScenarioTurn(
        10,
        "Internacionalización: español, portugués e inglés. La moneda "
        "depende del tenant.",
        "español, portugués, inglés",
    ),
    ScenarioTurn(
        11,
        "UI responsive para que los pacientes reserven desde el celular, "
        "sin necesidad de app nativa todavía.",
        "responsive",
    ),
    ScenarioTurn(
        12,
        "Job en background para reconciliar nocturnamente contra el "
        "sistema de facturación (SAP) vía archivos IDoc.",
        "SAP",
        "technologies",
    ),
    ScenarioTurn(
        13,
        "Doble factor de autenticación para cualquier usuario con permisos "
        "de administración.",
        "doble factor",
    ),
    ScenarioTurn(
        14,
        "Exportación masiva de turnos a XML siguiendo un esquema que nos "
        "pasó la obra social principal.",
        "exportación XML",
    ),
    ScenarioTurn(
        15,
        "Cumplimiento de la ley de protección de datos: flujo de baja de "
        "paciente a pedido, con borrado efectivo.",
        "protección de datos",
    ),
    ScenarioTurn(
        16,
        "Agreguen un feature flag por tenant para un panel experimental de "
        "sugerencias con IA sobre turnos recurrentes.",
        "feature flag",
    ),
    ScenarioTurn(
        17,
        "Portal de autogestión para que los profesionales actualicen su "
        "propia disponibilidad, con aprobación del admin.",
        "autogestión",
    ),
    ScenarioTurn(
        18,
        "SLA de performance: P95 de carga de página bajo 2 segundos con "
        "300 usuarios concurrentes.",
        "P95",
    ),
    ScenarioTurn(
        19,
        "Plan de recuperación ante desastres: RPO de 1 hora, RTO de 4 "
        "horas. Snapshots diarios cross-region.",
        "RPO",
    ),
    ScenarioTurn(
        20,
        "Último pedido: integración con un proveedor de firma digital para "
        "el consentimiento informado, justo después de confirmar el turno.",
        "firma digital",
        "technologies",
    ),
]


# --- pivot: "FitPulse" cambia de stack a mitad de camino ----------------

_PIVOT_TURNS: list[ScenarioTurn] = [
    ScenarioTurn(
        1,
        "Necesitamos una app para los socios de una cadena de gimnasios, "
        "se llama FitPulse. Mobile nativo con React Native para iOS y "
        "Android. Equipo de 4.",
        "FitPulse",
        "project_name",
    ),
    ScenarioTurn(
        2,
        "Reserva de clases grupales: el socio ve el calendario semanal y "
        "reserva cupo.",
        "reserva de clases",
    ),
    ScenarioTurn(
        3,
        "Backend: una API en Node.js sobre la base de datos de socios que "
        "ya tienen en Postgres.",
        "Node.js",
        "technologies",
    ),
    ScenarioTurn(
        4,
        "Notificaciones push cuando se libera un cupo en una clase con "
        "lista de espera.",
        "notificaciones push",
    ),
    ScenarioTurn(
        5,
        "PIVOTE: el cliente nos avisó que van a estandarizar todas sus "
        "apps mobile en Flutter. Cambiamos el stack a Flutter. React "
        "Native queda descartado.",
        "Flutter",
        "technologies",
    ),
    ScenarioTurn(
        6,
        "Reconfirmado: Flutter para iOS y Android, mismo alcance que "
        "veníamos hablando.",
        "Flutter",
        "technologies",
    ),
    ScenarioTurn(
        7,
        "Agreguen pases de Apple Wallet y Google Wallet para la membresía.",
        "Apple Wallet",
    ),
    ScenarioTurn(
        8,
        "Integración con el control de acceso por molinete que ya tienen "
        "instalado (API REST).",
        "control de acceso",
    ),
    ScenarioTurn(
        9,
        "Escaneo de código de barras en las máquinas para mostrar "
        "instrucciones de uso.",
        "código de barras",
    ),
    ScenarioTurn(
        10,
        "Notificaciones geolocalizadas: avisar cuando el socio está cerca "
        "de una sede.",
        "geolocalizadas",
    ),
    ScenarioTurn(11, "Login social con Apple ID y Google.", "Apple ID"),
    ScenarioTurn(
        12,
        "Panel de administración web para que el personal programe las "
        "clases. Para eso usamos Vue.js del lado admin.",
        "Vue.js",
        "technologies",
    ),
    ScenarioTurn(
        13,
        "Analítica de uso: eventos a Segment y de ahí a BigQuery.",
        "Segment",
        "technologies",
    ),
    ScenarioTurn(
        14,
        "Mensajería in-app para soporte, integrada con Zendesk.",
        "Zendesk",
        "technologies",
    ),
    ScenarioTurn(
        15,
        "Funcionalidad: carnet digital que funcione aunque falle el "
        "molinete.",
        "carnet digital",
    ),
    ScenarioTurn(
        16,
        "Localización: solo español y catalán por ahora, inglés queda "
        "para una v2.",
        "español y catalán",
    ),
    ScenarioTurn(
        17,
        "Reporte de errores y performance con Firebase Crashlytics.",
        "Crashlytics",
        "technologies",
    ),
    ScenarioTurn(
        18,
        "Material para las tiendas de apps: capturas y descripciones para "
        "3 tamaños de dispositivo.",
        "capturas de tienda",
    ),
    ScenarioTurn(
        19,
        "Cumplimiento: pantalla de consentimiento de datos en el primer uso.",
        "consentimiento",
    ),
    ScenarioTurn(
        20,
        "Último pedido: modo offline para que el socio vea su carnet y "
        "clases reservadas sin conexión.",
        "modo offline",
    ),
]


# --- contradiction: "Kairos" se contradice sobre el presupuesto ---------

_CONTRADICTION_TURNS: list[ScenarioTurn] = [
    ScenarioTurn(
        1,
        "Necesitamos una herramienta interna llamada Kairos para el área "
        "de finanzas. Flujo de aprobación de facturas de proveedores. "
        "Stack: Django + Postgres.",
        "Kairos",
        "project_name",
    ),
    ScenarioTurn(
        2,
        "El equipo son 2 desarrolladores más una diseñadora part-time.",
        "2 desarrolladores",
    ),
    ScenarioTurn(3, "Presupuesto confirmado: 28.000 EUR para todo el proyecto.", "28000"),
    ScenarioTurn(
        4,
        "Agreguen un registro de auditoría de quién aprobó qué, "
        "exportable a CSV.",
        "registro de auditoría",
    ),
    ScenarioTurn(
        5,
        "Integración con el SSO que ya tienen (Azure AD) -- todo aprobador "
        "debe autenticarse ahí.",
        "Azure AD",
        "technologies",
    ),
    ScenarioTurn(
        6,
        "Panel con las aprobaciones pendientes agrupadas por aprobador.",
        "panel de aprobaciones",
    ),
    ScenarioTurn(
        7,
        "Aprobación en lote de hasta 40 facturas en una sola acción, con "
        "confirmación.",
        "aprobación en lote",
    ),
    ScenarioTurn(
        8,
        "CONTRADICCIÓN: la dirección revisó el presupuesto al alza. El "
        "nuevo presupuesto es 75.000 EUR; la cifra anterior de 28.000 "
        "queda obsoleta.",
        "75000",
    ),
    ScenarioTurn(
        9,
        "Confirmado: el presupuesto nuevo es 75.000 EUR, el de 28.000 era "
        "una cifra preliminar.",
        "75000",
    ),
    ScenarioTurn(
        10,
        "Agreguen matching a tres bandas contra órdenes de compra y remitos.",
        "matching a tres bandas",
    ),
    ScenarioTurn(
        11,
        "Integración con el ERP (SAP) vía su gateway REST.",
        "SAP",
        "technologies",
    ),
    ScenarioTurn(
        12,
        "Enrutamiento de facturas al aprobador correcto según centro de "
        "costos.",
        "centro de costos",
    ),
    ScenarioTurn(
        13,
        "Resumen por email cada mañana con las aprobaciones pendientes "
        "por aprobador.",
        "resumen por email",
    ),
    ScenarioTurn(
        14,
        "Carga asistida por OCR para facturas en papel, usando AWS Textract.",
        "Textract",
        "technologies",
    ),
    ScenarioTurn(
        15,
        "Soporte multi-moneda con tipo de cambio diario del Banco Central.",
        "multi-moneda",
    ),
    ScenarioTurn(
        16,
        "Módulo de reportes: antigüedad de saldos y exposición total por "
        "proveedor.",
        "antigüedad de saldos",
    ),
    ScenarioTurn(
        17,
        "Hilo de comentarios en cada factura para el ida y vuelta entre "
        "aprobadores.",
        "hilo de comentarios",
    ),
    ScenarioTurn(
        18,
        "Performance: soportar 4.000 facturas por día en el pico de fin "
        "de mes.",
        "4000 facturas",
    ),
    ScenarioTurn(
        19,
        "Doble aprobación obligatoria para facturas mayores a 20.000 EUR.",
        "doble aprobación",
    ),
    ScenarioTurn(
        20,
        "Último pedido: exportación nocturna de facturas aprobadas al "
        "data warehouse (Snowflake).",
        "Snowflake",
        "technologies",
    ),
]


_SCENARIOS: dict[str, Scenario] = {
    "growing": Scenario(
        name="growing",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        turns=_GROWING_TURNS,
    ),
    "pivot": Scenario(
        name="pivot",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        turns=_PIVOT_TURNS,
    ),
    "contradiction": Scenario(
        name="contradiction",
        project_type=ProjectType.INTERNAL_TOOL,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        turns=_CONTRADICTION_TURNS,
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return _SCENARIOS[name]
    except KeyError as exc:
        raise ValueError(f"escenario desconocido {name!r}; disponibles: {sorted(_SCENARIOS)}") from exc


def all_scenarios() -> list[Scenario]:
    """Los tres escenarios en orden determinístico."""
    return [_SCENARIOS["growing"], _SCENARIOS["pivot"], _SCENARIOS["contradiction"]]
