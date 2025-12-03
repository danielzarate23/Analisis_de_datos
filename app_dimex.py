# ============================================
# Dimex – Tablero Regional de Desempeño y Riesgo (Streamlit)
# Archivo: app_dimex.py  (versión integrada, sin filtro de Periodo)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import re
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Si vas a usar el agente IA con OpenAI
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None  # para que no truene si no está instalado

import plotly.io as pio



# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(
    page_title="Dimex | Tablero Regional de Desempeño y Riesgo",
    layout="wide"
)
alt.data_transformers.disable_max_rows()

# Paleta Dimex
DIMEX_GREEN = "#6BFF00"
DIMEX_BLUE = "#001F3F"
DIMEX_ORANGE = "#FF9900"
DIMEX_RED = "#FF4B4B"

# -----------------------------
# CSS de estilo (fondo blanco, tipografías, KPI cards, filtros)
# -----------------------------
st.markdown(
    f"""
    <style>
    .main {{
        background-color: #ffffff;
        color: {DIMEX_BLUE};
        font-family: "Segoe UI", sans-serif;
    }}

    /* Títulos principales */
    h1, h2, h3, h4, h5, h6 {{
        color: {DIMEX_GREEN};
        font-weight: 700;
    }}

    /* Tabs: Director / Cobrador / Agente virtual */
    .stTabs [data-baseweb="tab"] p {{
        color: {DIMEX_BLUE};
        font-weight: 600;
    }}

    /* Tarjetas KPI */
    div[data-testid="stMetric"] {{
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid #d0defc;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {DIMEX_BLUE};
        font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        color: {DIMEX_GREEN};
        font-size: 1.6rem;
        font-weight: 700;
    }}

    /* Selectboxes: etiqueta azul */
    .stSelectbox label {{
        color: {DIMEX_BLUE} !important;
        font-weight: 600;
    }}

    /* TextInput label */
    .stTextInput label {{
        color: {DIMEX_BLUE} !important;
        font-weight: 600;
    }}

    /* Botones primarios */
    .stButton>button {{
        background-color: {DIMEX_BLUE};
        color: white;
        border-radius: 8px;
        border: 1px solid {DIMEX_GREEN};
        font-weight: 600;
    }}
    .stButton>button:hover {{
        background-color: {DIMEX_GREEN};
        color: {DIMEX_BLUE};
    }}

    /* Recomendaciones (alerts) – texto azul */
    .stAlert p {{
        color: {DIMEX_BLUE};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
pio.templates["dimex_global"] = pio.templates["plotly_white"]

pio.templates["dimex_global"].layout.update(
    font=dict(
        family="Segoe UI",
        size=49,           # 👈 MASTER FONT SIZE FOR ALL GRAPHS
        color="#001F3F"    # 👈 DIMEX_BLUE
    )
)

pio.templates.default = "dimex_global"

# -----------------------------
# Logo Dimex
# -----------------------------
logo_path = "dimex_logo.jpg"  # asegúrate que el JPG está en la misma carpeta
if os.path.exists(logo_path):
    cols_logo = st.columns([0.15, 0.85])
    with cols_logo[0]:
        st.image(logo_path, use_container_width=True)
    with cols_logo[1]:
        st.title("Desempeño Sucursales – Dimex")
else:
    st.title("Desempeño Sucursales – Dimex")

# -----------------------------
# Utilidades
# -----------------------------
@st.cache_data
def load_excel(path_or_buffer, sheet_name=None):
    return pd.read_excel(path_or_buffer, sheet_name=sheet_name)


def percent(x, digits=1):
    try:
        return f"{100 * x:.{digits}f}%"
    except Exception:
        return "-"


def money(x, digits=1):
    try:
        return f"${x / 1_000_000:.{digits}f}M"
    except Exception:
        return "-"


def format_money_columns(df, cols, digits=0):
    """Formatea columnas numéricas en formato pesos con comas."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].apply(
                lambda v: f"${v:,.{digits}f}" if pd.notnull(v) else "-"
            )
    return df


def guess_period_cols(df: pd.DataFrame):
    """Detecta columnas de series por periodo (Actual, T-xx)."""
    patron = r"(.*)\s(T-\d+|Actual)$"
    cols_temporales = [c for c in df.columns if re.search(patron, c)]
    return patron, cols_temporales


def melt_long(df, id_cols=("Región", "Vendedor", "Sucursal")):
    """Convierte a formato largo todas las métricas con sufijo temporal."""
    patron, cols_temporales = guess_period_cols(df)
    if not cols_temporales:
        return None

    df_long = df.melt(
        id_vars=[c for c in id_cols if c in df.columns],
        value_vars=cols_temporales,
        var_name="VariablePeriodo",
        value_name="Valor"
    )
    extra = df_long["VariablePeriodo"].str.extract(patron)
    extra.columns = ["Variable", "Periodo"]
    df_long = pd.concat([df_long.drop(columns=["VariablePeriodo"]), extra], axis=1)
    return df_long


def ensure_business_columns(df: pd.DataFrame):
    """Crea/asegura columnas clave para KPIs del tablero."""
    need = [
        "Saldo Insoluto Actual",
        "Saldo Insoluto Vencido Actual",
        "Saldo Insoluto 30-89  Actual",
        "Capital Dispersado Actual",
        "Capital Liquidado Actual",
        "Quitas Actual",
        "Castigos Actual",
    ]
    for c in need:
        if c not in df.columns:
            df[c] = 0.0

    # ICV (si no existe)
    if "ICV" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ICV"] = (
                df["Saldo Insoluto Vencido Actual"]
                / df["Saldo Insoluto Actual"]
            ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # IMORA (si no existe)
    if "IMORA" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (
                df["Saldo Insoluto Vencido Actual"]
                + df["Quitas Actual"]
                + df["Castigos Actual"]
            )
            den = (
                df["Saldo Insoluto Actual"]
                + df["Quitas Actual"]
                + df["Castigos Actual"]
            )
            df["IMORA"] = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0)

    # EBITDA proxy simple (ilustrativo)
    if "EBITDA" not in df.columns:
        tasa_interes_mensual = 0.65 / 12
        costo_fondeo_mensual = 0.11 / 12
        comision_mensual = 0.12 / 12

        saldo_vigente = df["Saldo Insoluto Actual"] - df["Saldo Insoluto Vencido Actual"]
        intereses = saldo_vigente * tasa_interes_mensual
        costo = df["Saldo Insoluto Actual"] * costo_fondeo_mensual
        comisiones = df["Capital Dispersado Actual"] * comision_mensual
        perdidas = df["Quitas Actual"] + df["Castigos Actual"]

        df["EBITDA"] = (intereses - costo - comisiones - perdidas).clip(lower=0)

    # Crecimiento trimestral (Actual vs T-12) si existe T-12
    if "Saldo Insoluto T-12" in df.columns:
        base_12 = pd.to_numeric(df["Saldo Insoluto T-12"], errors="coerce")
        base_12 = base_12.replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            crec = df["Saldo Insoluto Actual"] / base_12 - 1
        df["Crecimiento_trimestral"] = crec.replace([np.inf, -np.inf], np.nan)
    else:
        df["Crecimiento_trimestral"] = np.nan

    # Region_codigo como numérico si existe
    if "Region_codigo" in df.columns:
        df["Region_codigo"] = pd.to_numeric(df["Region_codigo"], errors="coerce")
    else:
        df["Region_codigo"] = np.nan

    return df


def segment_recommendation(row):
    """Regla simple de recomendación por sucursal (para tabla de detalle)."""
    icv = row.get("ICV", 0)
    cast = row.get("Castigos Actual", 0)
    quitas = row.get("Quitas Actual", 0)
    crec = row.get("Crecimiento_trimestral", np.nan)

    if icv >= 0.12 or cast > quitas:
        return "Control de riesgo"
    if pd.notnull(crec) and crec > 0.10 and icv < 0.06:
        return "Fidelización"
    if pd.notnull(crec) and crec < 0.02 and icv < 0.06:
        return "Campaña de crecimiento"
    return "Monitoreo"


def cluster_action_priority(cluster_id):
    """Acción prioritaria por clúster."""
    mapping = {
        0: "Mantener políticas y buenas prácticas.",
        1: "Impulsar colocación segura (microcréditos).",
        2: "Control de riesgo y cobranza intensiva.",
        3: "Revisión de cobranza y reestructuras.",
    }
    try:
        return mapping.get(int(cluster_id), "Monitoreo ejecutivo.")
    except Exception:
        return "Monitoreo ejecutivo."


def cluster_summary_text(cluster_id):
    """Descripción corta del clúster en panel de recomendaciones."""
    mapping = {
        0: "Clúster estable: operación sana y eficiente.",
        1: "Clúster con poco dinamismo: cartera sana, baja penetración.",
        2: "Clúster con crecimiento + riesgo: revisar originación.",
        3: "Clúster con presión por castigos: activar alertas tempranas.",
    }
    try:
        return mapping.get(int(cluster_id), "Clúster sin descripción.")
    except Exception:
        return "Clúster sin descripción."


# =============================
# Función para series de tiempo (ICV / IMORA)
# =============================
def build_icv_imora_timeseries(df: pd.DataFrame):
    """
    Construye una serie de tiempo mensual de ICV e IMORA
    usando las columnas temporales (Actual, T-xx).

    Mantiene: T-12 ... T-02 y Actual
    Elimina:  T-01 / T-1
    """
    df_long = melt_long(df, id_cols=("Región", "Sucursal", "Vendedor"))
    if df_long is None:
        return None

    vars_needed = [
        "Saldo Insoluto",
        "Saldo Insoluto Vencido",
        "Quitas",
        "Castigos",
    ]
    df_long = df_long[df_long["Variable"].isin(vars_needed)].copy()
    if df_long.empty:
        return None

    grp = (
        df_long.groupby(["Periodo", "Variable"], as_index=False)["Valor"]
        .sum()
    )

    pivot = grp.pivot_table(
        index="Periodo",
        columns="Variable",
        values="Valor",
        aggfunc="sum"
    ).reset_index()

    for col in ["Saldo Insoluto", "Saldo Insoluto Vencido", "Quitas", "Castigos"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    # lag numérico (T-01, T-02, ..., T-12)
    def get_lag(p):
        m = re.search(r"T-(\d+)", str(p))
        return int(m.group(1)) if m else None

    pivot["lag"] = pivot["Periodo"].apply(get_lag)

    # Solo Actual y T-12..T-02 (quitamos lag==1)
    mask_actual = pivot["Periodo"].str.lower().eq("actual")
    mask_12_2 = pivot["lag"].notna() & (pivot["lag"] <= 12) & (pivot["lag"] >= 2)
    pivot = pivot[mask_actual | mask_12_2].copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        icv = pivot["Saldo Insoluto Vencido"] / pivot["Saldo Insoluto"]
        num_imora = (
            pivot["Saldo Insoluto Vencido"]
            + pivot["Quitas"]
            + pivot["Castigos"]
        )
        den_imora = (
            pivot["Saldo Insoluto"]
            + pivot["Quitas"]
            + pivot["Castigos"]
        )
        imora = num_imora / den_imora

    pivot["ICV_pct"] = (icv.replace([np.inf, -np.inf], np.nan).fillna(0)) * 100
    pivot["IMORA_pct"] = (imora.replace([np.inf, -np.inf], np.nan).fillna(0)) * 100

    # orden: T-12 ... T-02 ... Actual
    def periodo_order(row):
        periodo = str(row["Periodo"])
        lag = row["lag"]
        if periodo.strip().lower() == "actual":
            return 13
        if pd.notnull(lag):
            return 13 - int(lag)  # T-12 -> 1, T-02 -> 11
        return 0

    pivot["order"] = pivot.apply(periodo_order, axis=1)
    pivot = pivot.sort_values("order").drop(columns=["order", "lag"])

    return pivot[["Periodo", "ICV_pct", "IMORA_pct"]]



# =============================
# Agente IA – función LLM
# =============================
def generate_agent_answer_llm(region, agg_row: pd.Series, question: str) -> str:
    """
    Llama a OpenAI para generar la respuesta del agente virtual.
    """
    if client is None:
        return "⚠️ No se pudo inicializar el cliente de OpenAI. Verifica tu entorno."

    cluster = int(agg_row.get("Cluster", -1))

    kpis_text = []
    for k, v in agg_row.items():
        if k in {"ICV", "IMORA"}:
            kpis_text.append(f"{k} = {v*100:.2f}%")
        elif k in {
            "Saldo Insoluto Actual",
            "Saldo Insoluto Vencido Actual",
            "Capital Dispersado Actual",
            "EBITDA",
            "Quitas Actual",
            "Castigos Actual",
        }:
            kpis_text.append(f"{k} = ${float(v):,.0f}")
        else:
            kpis_text.append(f"{k} = {v}")

    kpis_str = "\n".join(f"- {t}" for t in kpis_text)

    system_prompt = f"""
Eres un Analista Senior de Inteligencia Comercial especializado en sucursales.
Tu objetivo es evaluar sucursales/zonas, explicar su cluster y recomendar acciones ejecutivas.

CONTEXTO:
- Región / zona: {region}
- Cluster asignado: {cluster}
- Principales KPIs numéricos:
{kpis_str}

FORMATO DE RESPUESTA:
1. Insight clave
2. Justificación
3. Riesgos
4. Acciones sugeridas
5. Pregunta de seguimiento

Escribe en tono ejecutivo, claro y conciso en español.
"""

    user_content = f"Región: {region}\nPregunta del usuario: {question}"

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    return chat.choices[0].message.content


# -----------------------------
# 0) Login sencillo
# -----------------------------
st.subheader("Inicio de sesión")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

col_user, col_pass = st.columns(2)
with col_user:
    username = st.text_input("Nombre de usuario", "")
with col_pass:
    password = st.text_input("Contraseña", "", type="password")

login_btn = st.button("Entrar")

# puedes cambiar aquí las contraseñas
VALID_USERS = {"admin": "dimex2024", "director": "cobranza123"}

if login_btn:
    if username in VALID_USERS and password == VALID_USERS[username]:
        st.session_state.logged_in = True
        st.session_state.current_user = username
        st.success(f"Bienvenido, {username}. Cargando dashboard...")
    else:
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.error("Usuario o contraseña incorrectos.")

st.caption(f"👤 Usuario conectado: {st.session_state.current_user or 'None'}")

st.markdown("---")

if not st.session_state.logged_in:
    st.stop()

# -----------------------------
# 0) Carga de datos
# -----------------------------
st.sidebar.title("Dimex")
st.sidebar.caption("Tablero Regional de Desempeño y Riesgo")

uploaded = st.sidebar.file_uploader(
    "Sube el archivo de Excel (o deja vacío para leer el default)",
    type=["xlsx"],
)

try:
    if uploaded:
        base = load_excel(uploaded, sheet_name=None)
    else:
        base = load_excel("MasterSucursalestemporal (1).xlsx", sheet_name=None)
except Exception as e:
    st.error(f"No se pudo leer el archivo. Detalle: {e}")
    st.stop()

# Detecta la hoja que contiene los indicadores
target_sheet = None
for name in base.keys():
    if "indicadores" in name.lower() or "comercial" in name.lower():
        target_sheet = name
        break
if target_sheet is None:
    target_sheet = list(base.keys())[0]

df = base[target_sheet].copy()
df.columns = [str(c).strip() for c in df.columns]

# Eliminar fila "Total" si existe
if "Región" in df.columns:
    df = df[df["Región"].astype(str).str.strip().str.lower() != "total"]

# Asegura columnas negocio (incluye IMORA)
df = ensure_business_columns(df)

# Normaliza tipos
for col in ["Región", "Vendedor", "Sucursal"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# =============================
# 🔹 Recalcular clústers (4)
# =============================
cluster_features = [
    c for c in [
        "Saldo Insoluto Actual",
        "Capital Dispersado Actual",
        "ICV",
        "IMORA",
        "Quitas Actual",
        "Castigos Actual",
        "Region_codigo",
    ]
    if c in df.columns
]

if len(cluster_features) >= 2:
    X = df[cluster_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
else:
    df["Cluster"] = 0

df["Recomendación"] = df.apply(segment_recommendation, axis=1)

# -----------------------------
# 1) Filtros principales (multi selección + dependientes)
# -----------------------------
st.subheader("Tablero Dimex – Desempeño y Riesgo por Región")

# Opciones de cluster (sin "Todos", usamos multiselect con default = todos)
if "Cluster" in df.columns:
    cluster_options = (
        df["Cluster"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
else:
    cluster_options = []

# Layout de filtros
f1, f2, f3 = st.columns([1.2, 1, 1])

# 1) Filtro de Cluster (se selecciona primero)
with f2:
    sel_clusters = st.multiselect(
        "Cluster",
        options=cluster_options,
        default=cluster_options,   # por defecto todos
        help="Selecciona uno o varios clústeres",
    )

# 2) Filtro de Región (depende de lo seleccionado en Cluster)
#    Si hay clusters seleccionados, solo mostramos las regiones de esos clusters
if sel_clusters and "Cluster" in df.columns and "Región" in df.columns:
    df_regiones_base = df[df["Cluster"].astype(int).isin(sel_clusters)]
else:
    df_regiones_base = df

if "Región" in df_regiones_base.columns:
    region_options = (
        df_regiones_base["Región"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
else:
    region_options = []

with f1:
    sel_regiones = st.multiselect(
        "Región",
        options=region_options,
        default=region_options,   # por defecto todas las disponibles
        help="Selecciona una o varias regiones",
    )


# -----------------------------
# Aplicar filtros a dff
# -----------------------------
dff = df.copy()

# Filtro por cluster (si hay alguno seleccionado)
if sel_clusters and "Cluster" in dff.columns:
    dff = dff[dff["Cluster"].astype(int).isin(sel_clusters)]

# Filtro por región (si hay alguna seleccionada)
if sel_regiones and "Región" in dff.columns:
    dff = dff[dff["Región"].isin(sel_regiones)]


st.markdown("---")

# =============================
# TABS: Director / Cobrador / Agente virtual
# =============================
tab_dir, tab_cob, tab_agent = st.tabs(
    ["📘 Director de Cobranza", "🧍 Cobrador", "◎ Agente virtual"]
)

# =============================
# VISTA 1: DIRECTOR DE COBRANZA
# =============================
with tab_dir:
    st.subheader("Vista Director de Cobranza – Visión Ejecutiva")

    # FILA 1 – KPIs ejecutivos
    icv_prom = dff["ICV"].mean()
    imora_prom = dff["IMORA"].mean()
    cartera_vencida = dff["Saldo Insoluto Vencido Actual"].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("ICV promedio", percent(icv_prom if pd.notnull(icv_prom) else 0))
    with c2:
        st.metric("IMORA promedio", percent(imora_prom if pd.notnull(imora_prom) else 0))
    with c3:
        st.metric("Cartera vencida total", money(cartera_vencida))

    # FILA 2 – Línea de tiempo + burbujas
    col_ts, col_bubble = st.columns([1.4, 1])

    with col_ts:
        st.markdown("#### Evolución mensual de ICV e IMORA")
        ts = build_icv_imora_timeseries(df)
        if ts is not None and not ts.empty:
            fig_ts = px.line(
                ts,
                x="Periodo",
                y=["ICV_pct", "IMORA_pct"],
                markers=True,
                title="ICV e IMORA por periodo",
                labels={"value": "Porcentaje", "variable": "Indicador"},
                color_discrete_map={
                    "ICV_pct": DIMEX_GREEN,
                    "IMORA_pct": DIMEX_ORANGE,
                },
            )
            # Etiquetas
            fig_ts.update_traces(
                mode="lines+markers+text",
                texttemplate="%{y:.1f}",
                textposition="top center",
            )
            fig_ts.update_layout(
                xaxis_title="Periodo (meses)",
                yaxis_title="Porcentaje",
                template="plotly_white",
                font=dict(color=DIMEX_BLUE),
                legend_title="Indicador",
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No se encontraron series mensuales para calcular la evolución de ICV/IMORA.")

    with col_bubble:
        st.markdown("#### ICV vs IMORA por Región (tamaño = cartera vencida)")
        if {"Región", "ICV", "IMORA", "Saldo Insoluto Vencido Actual", "Cluster"}.issubset(dff.columns):
            df_bubble_reg = (
                dff.groupby(["Región", "Cluster"], as_index=False)
                .agg({
                    "ICV": "mean",
                    "IMORA": "mean",
                    "Saldo Insoluto Vencido Actual": "sum",
                })
            )
            df_bubble_reg["ICV_pct"] = df_bubble_reg["ICV"] * 100
            df_bubble_reg["IMORA_pct"] = df_bubble_reg["IMORA"] * 100

            fig_bubble = px.scatter(
                df_bubble_reg,
                x="ICV_pct",
                y="IMORA_pct",
                size="Saldo Insoluto Vencido Actual",
                color="Cluster",
                hover_name="Región",
                hover_data=["Saldo Insoluto Vencido Actual"],
                title="ICV vs IMORA por Región y Clúster",
                labels={
                    "ICV_pct": "ICV (%)",
                    "IMORA_pct": "IMORA (%)",
                    "Saldo Insoluto Vencido Actual": "Cartera vencida",
                },
                color_discrete_sequence=[DIMEX_GREEN, DIMEX_BLUE, DIMEX_ORANGE, DIMEX_RED],
            )
            fig_bubble.update_traces(
                text=df_bubble_reg["Región"],
                textposition="top center"
            )
            fig_bubble.update_layout(
                xaxis_title="ICV (%)",
                yaxis_title="IMORA (%)",
                template="plotly_white",
                font=dict(color=DIMEX_BLUE),
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("Faltan columnas para graficar el mapa de burbujas por región.")

    # FILA 3 – Top 10 ICV vs Cartera vencida
    col_icv, col_venc = st.columns(2)

    with col_icv:
        st.markdown("#### Top 10 Regiones por ICV (%)")
        if {"Región", "ICV"}.issubset(df.columns):
            df_icv_region = (
                df.groupby("Región", as_index=False)["ICV"]
                .mean()
            )
            df_icv_region["ICV_pct"] = df_icv_region["ICV"] * 100
            df_icv_region = df_icv_region.sort_values("ICV_pct", ascending=False).head(10)

            fig_icv_region = px.bar(
                df_icv_region,
                x="Región",
                y="ICV_pct",
                title="Top 10 Regiones por ICV (%)",
                labels={"ICV_pct": "ICV (%)"},
                color_discrete_sequence=[DIMEX_GREEN],
                text_auto=".1f"
            )
            fig_icv_region.update_layout(
                xaxis_title="Región",
                yaxis_title="ICV (%)",
                template="plotly_white",
                font=dict(color=DIMEX_BLUE),
            )
            st.plotly_chart(fig_icv_region, use_container_width=True)
        else:
            st.info("No se encontraron las columnas necesarias para calcular el ICV por región.")

    with col_venc:
        st.markdown("#### Top 10 Regiones por Cartera Vencida")
        if {"Región", "Saldo Insoluto Vencido Actual"}.issubset(df.columns):
            df_venc = (
                df.groupby("Región", as_index=False)["Saldo Insoluto Vencido Actual"]
                .sum()
            )
            df_venc = df_venc.sort_values("Saldo Insoluto Vencido Actual", ascending=False).head(10)

            fig_venc = px.bar(
                df_venc,
                x="Región",
                y="Saldo Insoluto Vencido Actual",
                title="Top 10 Regiones por Cartera Vencida",
                labels={"Saldo Insoluto Vencido Actual": "Cartera vencida"},
                color_discrete_sequence=[DIMEX_ORANGE],
                text_auto=",.0f"
            )
            fig_venc.update_layout(
                xaxis_title="Región",
                yaxis_title="Monto",
                template="plotly_white",
                font=dict(color=DIMEX_BLUE),
            )
            st.plotly_chart(fig_venc, use_container_width=True)
        else:
            st.info("No se encontró información suficiente para cartera vencida por región.")

    # FILA 4 – Bottom 20 crecimiento + Boxplot ICV
    col_crec, col_box = st.columns(2)

    with col_crec:
        st.markdown("#### Top 20 vendedores por crecimiento trimestral de cartera")
        if "Crecimiento_trimestral" in dff.columns and dff["Crecimiento_trimestral"].notna().any():

            df_crec = dff.dropna(subset=["Crecimiento_trimestral"]).copy()
            df_crec["Crecimiento_trimestral_pct"] = (df_crec["Crecimiento_trimestral"] * 100).round(1)

            if "Sucursal" in df_crec.columns:
                key_crec = "Sucursal"
            elif "Vendedor" in df_crec.columns:
                key_crec = "Vendedor"
            else:
                key_crec = "Región"

            df_crec_group = (
                df_crec.groupby(key_crec, as_index=False)["Crecimiento_trimestral_pct"]
                .mean()
            )

            df_crec_bottom = df_crec_group.sort_values(
                "Crecimiento_trimestral_pct", ascending=True
            ).head(20)

            fig_crec = px.bar(
                df_crec_bottom,
                x="Crecimiento_trimestral_pct",
                y=key_crec,
                orientation="h",
                title=f"Top 20 {key_crec} por crecimiento trimestral de cartera",
                labels={"Crecimiento_trimestral_pct": "Crecimiento trimestral (%)", key_crec: ""},
                color_discrete_sequence=[DIMEX_BLUE],
                text_auto=".1f"
            )
            fig_crec.update_layout(
                xaxis_title="Crecimiento trimestral (%)",
                yaxis_title="",
                template="plotly_white",
                font=dict(color=DIMEX_BLUE),
            )
            st.plotly_chart(fig_crec, use_container_width=True)
        else:
            st.info("No hay información suficiente para mostrar el crecimiento trimestral.")

    with col_box:
        st.markdown("#### Distribución de ICV por Clúster")
        if {"Cluster", "ICV"}.issubset(df.columns):
            df_box = df.copy()
            df_box["ICV_pct"] = df_box["ICV"] * 100
            fig_box = px.box(
                df_box,
                x="Cluster",
                y="ICV_pct",
                points="outliers",
                title="Distribución de ICV (%) por Clúster",
                labels={"ICV_pct": "ICV (%)"},
                color_discrete_sequence=[DIMEX_GREEN],
            )
            fig_box.update_layout(
                xaxis_title="Clúster",
                yaxis_title="ICV (%)",
                template="plotly_white",
                font=dict(color=DIMEX_BLUE),
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No se encontró información suficiente para el boxplot de ICV por clúster.")

    # FILA 5 – Tabla ejecutiva + recomendaciones
    st.markdown("#### Tabla Ejecutiva por Región")
    df_exec = df.copy()
    df_exec["ICV (%)"] = (df_exec["ICV"] * 100).round(1)
    df_exec["IMORA (%)"] = (df_exec["IMORA"] * 100).round(1)

    cols_exec = [c for c in [
        "Región",
        "Cluster",
        "ICV (%)",
        "IMORA (%)",
        "Saldo Insoluto Actual",
        "Saldo Insoluto Vencido Actual",
        "Quitas Actual",
        "Castigos Actual",
        "EBITDA",
        "Recomendación",
    ] if c in df_exec.columns]

    df_exec = format_money_columns(
        df_exec,
        ["Saldo Insoluto Actual",
         "Saldo Insoluto Vencido Actual",
         "Quitas Actual",
         "Castigos Actual",
         "EBITDA"],
        digits=0
    )

    df_exec = df_exec[cols_exec].sort_values(
        by=["ICV (%)"] if "ICV (%)" in df_exec.columns else "Saldo Insoluto Vencido Actual",
        ascending=False
    )

    st.dataframe(df_exec, use_container_width=True)

    st.markdown("#### Recomendaciones por Clúster")
    colr = st.columns(4)
    for i, col in enumerate(colr):
        with col:
            st.subheader(f"Cluster {i}")
            st.caption(cluster_summary_text(i))
            st.success(cluster_action_priority(i))

# =============================
# VISTA 2: COBRADOR
# =============================
with tab_cob:
    st.subheader("Vista de Cobrador – Enfoque Operativo")

    icv_prom = dff["ICV"].mean()
    imora_prom = dff["IMORA"].mean()
    cartera_vencida = dff["Saldo Insoluto Vencido Actual"].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("ICV promedio (filtro)", percent(icv_prom if pd.notnull(icv_prom) else 0))
    with c2:
        st.metric("IMORA promedio (filtro)", percent(imora_prom if pd.notnull(imora_prom) else 0))
    with c3:
        st.metric("Cartera vencida en gestión", money(cartera_vencida))

    # Intensidad de crédito – SOLO en vista Cobrador
    st.markdown("#### Intensidad de crédito por Región (Capital dispersado vs Saldo insoluto)")
    if {"Región", "Capital Dispersado Actual", "Saldo Insoluto Actual"}.issubset(df.columns):
        df_cap = df.groupby("Región", as_index=False)[
            ["Capital Dispersado Actual", "Saldo Insoluto Actual"]
        ].sum()

        df_cap["ratio"] = (
            df_cap["Capital Dispersado Actual"]
            / df_cap["Saldo Insoluto Actual"].replace(0, np.nan)
        ) * 100

        df_cap = df_cap.sort_values("ratio", ascending=True).tail(20)

        fig_int = px.bar(
            df_cap,
            x="ratio",
            y="Región",
            orientation="h",
            title="Top 20 regiones por intensidad de crédito (Capital dispersado / Saldo insoluto)",
            labels={"ratio": "Capital dispersado / Saldo insoluto (%)", "Región": ""},
            color="ratio",
            color_continuous_scale="Blues",
            text_auto=".1f",
        )
        fig_int.update_layout(
            xaxis_title="Capital dispersado / Saldo insoluto (%)",
            yaxis_title="",
            template="plotly_white",
            font=dict(color=DIMEX_BLUE),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_int, use_container_width=True)
    else:
        st.info("No se encontraron columnas necesarias para comparar capital y saldo.")

    # Prioridad de Gestión de Cobranza – tabla
    st.markdown("#### Prioridad de Gestión de Cobranza")
    df_cob = dff.copy()
    if "ICV" in df_cob.columns:
        df_cob["ICV (%)"] = (df_cob["ICV"] * 100).round(1)
    if "IMORA" in df_cob.columns:
        df_cob["IMORA (%)"] = (df_cob["IMORA"] * 100).round(1)

    cols_cobrador = [c for c in [
        "Región",
        "Cluster",
        "Vendedor",
        "ICV (%)" if "ICV (%)" in df_cob.columns else None,
        "IMORA (%)" if "IMORA (%)" in df_cob.columns else None,
        "Saldo Insoluto Actual",
        "Saldo Insoluto Vencido Actual",
        "Quitas Actual",
        "Castigos Actual",
        "Recomendación",
    ] if c is not None and c in df_cob.columns]

    df_cob = df_cob[cols_cobrador].sort_values(
        by=["ICV (%)"] if "ICV (%)" in df_cob.columns else "Saldo Insoluto Vencido Actual",
        ascending=False
    )

    df_cob = format_money_columns(
        df_cob,
        ["Saldo Insoluto Actual",
         "Saldo Insoluto Vencido Actual",
         "Quitas Actual",
         "Castigos Actual"],
        digits=0
    )

    st.dataframe(df_cob.head(50), use_container_width=True)

    # Desempeño por Sucursal (ICV vs IMORA)
    st.markdown("#### Desempeño por Sucursal (ICV vs IMORA)")
    df_suc = dff.copy()
    key_suc = "Sucursal" if "Sucursal" in df_suc.columns else "Vendedor"

    if {"ICV", "IMORA"}.issubset(df_suc.columns):
        df_suc["ICV_pct"] = df_suc["ICV"] * 100
        df_suc["IMORA_pct"] = df_suc["IMORA"] * 100

        df_suc_group = (
            df_suc.groupby(key_suc, as_index=False)[["ICV_pct", "IMORA_pct"]].mean()
        )

        df_suc_top = df_suc_group.sort_values("ICV_pct", ascending=False).head(20)

        df_suc_long = df_suc_top.melt(
            id_vars=key_suc,
            value_vars=["ICV_pct", "IMORA_pct"],
            var_name="Indicador",
            value_name="Valor"
        )

        fig_suc = px.bar(
            df_suc_long,
            x="Valor",
            y=key_suc,
            orientation="h",
            color="Indicador",
            facet_col="Indicador",
            facet_col_spacing=0.03,
            title=f"ICV e IMORA por {key_suc} (Top 20)",
            labels={"Valor": "Porcentaje", key_suc: ""},
            color_discrete_map={"ICV_pct": DIMEX_GREEN, "IMORA_pct": DIMEX_ORANGE},
            text_auto=".1f",
        )

        fig_suc.update_layout(
            template="plotly_white",
            font=dict(color=DIMEX_BLUE),
            showlegend=False,
        )

        fig_suc.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1].replace("_pct", ""))
        )

        st.plotly_chart(fig_suc, use_container_width=True)
    else:
        st.info("No hay información suficiente para el gráfico de desempeño por sucursal.")

# =============================
# VISTA 3: AGENTE VIRTUAL
# =============================
with tab_agent:
    st.subheader("Agente Virtual – Analista Senior de Inteligencia Comercial (IA)")

    # Filtramos por REGION para el agente virtual
    key_ag = "Región"
    regiones_agent = sorted(df[key_ag].dropna().unique().tolist())

    sel_region_ag = st.selectbox("Selecciona región a analizar", regiones_agent)

    user_question = st.text_input(
        "Pregunta para el agente (en lenguaje natural)",
        value="Ejemplo: ¿por qué esta región está en este cluster y qué debería priorizar en cobranza?"
    )

    if st.button("Generar análisis con IA"):
        df_agent = df.copy()
        row = df_agent[df_agent[key_ag] == sel_region_ag].copy()
        if row.empty:
            st.error("No se encontró información para la región seleccionada.")
        else:
            agg_dict = {}

            if "ICV" in row.columns:
                agg_dict["ICV"] = row["ICV"].mean()
            if "IMORA" in row.columns:
                agg_dict["IMORA"] = row["IMORA"].mean()
            if "Saldo Insoluto Actual" in row.columns:
                agg_dict["Saldo Insoluto Actual"] = row["Saldo Insoluto Actual"].sum()
            if "Saldo Insoluto Vencido Actual" in row.columns:
                agg_dict["Saldo Insoluto Vencido Actual"] = row["Saldo Insoluto Vencido Actual"].sum()
            if "Capital Dispersado Actual" in row.columns:
                agg_dict["Capital Dispersado Actual"] = row["Capital Dispersado Actual"].sum()
            if "EBITDA" in row.columns:
                agg_dict["EBITDA"] = row["EBITDA"].sum()
            if "Quitas Actual" in row.columns:
                agg_dict["Quitas Actual"] = row["Quitas Actual"].sum()
            if "Castigos Actual" in row.columns:
                agg_dict["Castigos Actual"] = row["Castigos Actual"].sum()

            if "Cluster" in row.columns and not row["Cluster"].isna().all():
                agg_dict["Cluster"] = int(row["Cluster"].iloc[0])
            else:
                agg_dict["Cluster"] = -1

            agg_row = pd.Series(agg_dict)

            respuesta_markdown = generate_agent_answer_llm(sel_region_ag, agg_row, user_question)

            st.markdown(
                f"""
                <div style="background-color:#d0d3d6; padding:0.8rem 1rem; border-radius:8px; margin-top:1rem;">
                    <span style="font-weight:bold; color:{DIMEX_BLUE};">Consulta sobre {sel_region_ag}</span><br>
                    <span style="color:{DIMEX_BLUE};">Pregunta: {user_question}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="margin-top:1rem; color:{DIMEX_BLUE};">
                {respuesta_markdown}
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------
# Pie de página
# -----------------------------
st.markdown("---")
st.caption("Dimex • Tablero analítico (prototipo) – Datos históricos, no en tiempo real.")
