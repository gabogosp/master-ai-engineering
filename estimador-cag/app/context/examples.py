ESTIMATION_EXAMPLES = [
    {
        "meeting_summary": (
            "El cliente, una distribuidora con tres almacenes, necesita una plataforma web "
            "de gestión de inventario para sustituir sus hojas de cálculo. Quiere registrar "
            "entradas y salidas de stock, recibir alertas de stock mínimo, tener usuarios con "
            "distintos roles (administrador, encargado de almacén y solo consulta) y un "
            "dashboard con métricas de rotación. El diseño aún no existe. "
            "Plazo deseado: 2 meses."
        ),
        "estimation": """## Estimación: Plataforma de Gestión de Inventario

### Desglose de tareas:
1. Diseño UI/UX: 40 horas
2. Backend API (CRUD inventario): 60 horas
3. Autenticación y roles: 20 horas
4. Dashboard con métricas: 30 horas
5. Testing y QA: 25 horas

**Total estimado: 175 horas**
**Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 6-8 semanas**
**Coste estimado: 175 h x 50 €/h = 8.750 €**

### Supuestos y riesgos:
- Se parte de cero: no hay sistema previo ni migración de datos.
- Las alertas de stock mínimo se envían solo por email.
- Riesgo: la importación inicial de las hojas de cálculo puede requerir horas extra.
""",
    },
        {
        "meeting_summary": (
            "El cliente, una cadena de tres gimnasios, quiere una app móvil (iOS y Android) "
            "para que sus socios reserven clases, vean el horario y paguen bonos. Necesita "
            "recordatorios por notificación push y un panel web para que el personal gestione "
            "clases, aforos y socios. Tienen una identidad de marca pero ningún diseño de "
            "pantallas. Plazo deseado: 3 meses."
        ),
        "estimation": """## Estimación: App Móvil de Reservas para Gimnasios

### Desglose de tareas:
1. Diseño UI/UX (app móvil): 35 horas
2. App móvil multiplataforma (React Native): 90 horas
3. Backend API (reservas, horarios, socios): 70 horas
4. Notificaciones push y recordatorios: 20 horas
5. Integración de pagos (Stripe): 25 horas
6. Panel de administración web: 40 horas
7. Testing y QA en dispositivos: 30 horas

**Total estimado: 310 horas**
**Equipo recomendado: 1 desarrollador móvil + 1 desarrollador backend + 1 diseñador UX (part-time) + 1 QA (part-time)**
**Duración estimada: 10-12 semanas**
**Coste estimado: 310 h x 50 €/h = 15.500 €**

### Supuestos y riesgos:
- Se usa una sola base de código para iOS y Android.
- La publicación en App Store y Google Play está incluida, pero no las tasas de las cuentas de desarrollador.
- Riesgo: la revisión de Apple puede retrasar el lanzamiento entre 1 y 2 semanas.
""",
    },
]