# ============================================
# Dimex – Tablero Regional de Desempeño y Riesgo (Streamlit)
# Archivo: app_dimex.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import re

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(
    page_title="Dimex | Tablero Regional de Desempeño y Riesgo",
    layout="wide"
)
alt.data_transformers.disable_max_rows()

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

def guess_period_cols(df: pd.DataFrame):
    """Detecta columnas de series por periodo (Actual, T-xx)."""
    patron = r"(.*)\s(T-\d+|Actual)$"
    cols_temporales = [c for c in df.columns if re.search(patron, c)]
    return patron, cols_temporales

def melt_long(df, id_cols=("Región", "Vendedor")):
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

    # EBITDA proxy simple
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
        base_12 = base_12.replace(0, np.nan)   # evitar división entre 0

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
        # 👉 archivo por defecto
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

# 🔥 QUITAMOS EL RENGLÓN "TOTAL" PARA QUE NO ENTRE A KPIs NI GRÁFICOS
if "Región" in df.columns:
    df["Región"] = df["Región"].astype(str).str.strip()
    df = df[df["Región"].str.lower() != "total"].copy()

# Asegura columnas negocio
df = ensure_business_columns(df)

# Normaliza tipos (ya sin la fila Total)
if "Región" in df.columns:
    df["Región"] = df["Región"].astype(str).str.strip()
if "Vendedor" in df.columns:
    df["Vendedor"] = df["Vendedor"].astype(str).str.strip()

# =============================
# 🔹 Recalcular clústers (4)
# =============================
cluster_features = [
    c for c in [
        "Saldo Insoluto Actual",
        "Capital Dispersado Actual",
        "ICV",
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
    # fallback: todo a cluster 0 si faltan columnas
    df["Cluster"] = 0

# Recomendación por fila
df["Recomendación"] = df.apply(segment_recommendation, axis=1)

# -----------------------------
# 1) Selector de perfil (mockup)
# -----------------------------
with st.container():
    st.markdown("### Seleccionar Perfil")
    colp1, colp2, _ = st.columns([1, 1, 2])
    with colp1:
        st.button("👔 Director de Cobranza", type="secondary")
    with colp2:
        st.button("🧍 Cobrador", type="secondary")

st.markdown("---")

# -----------------------------
# 2) Filtros principales
# -----------------------------
regiones = (
    ["Todas"] + sorted(df["Región"].dropna().unique().tolist())
    if "Región" in df.columns
    else ["Todas"]
)
clusters = ["Todos"] + sorted(df["Cluster"].dropna().unique().astype(int).tolist())
periodos = ["Actual"]

f1, f2, f3 = st.columns([1.2, 1, 1])
with f1:
    sel_region = st.selectbox("Región", regiones, index=0)
with f2:
    sel_cluster = st.selectbox("Cluster", clusters, index=0)
with f3:
    sel_periodo = st.selectbox("Periodo", periodos, index=0)

# Aplica filtros
dff = df.copy()
if sel_region != "Todas" and "Región" in dff.columns:
    dff = dff[dff["Región"] == sel_region]
if sel_cluster != "Todos":
    dff = dff[dff["Cluster"] == int(sel_cluster)]

# -----------------------------
# 3) KPI cards (Margen Bruto proxy, EBITDA, %Crecimiento)
# -----------------------------
margen_bruto = (1 - dff["ICV"]).replace([np.inf, -np.inf], np.nan).mean()
ebitda_total = dff["EBITDA"].sum()
crec = dff["Crecimiento_trimestral"].mean()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Margen Bruto", percent(margen_bruto if pd.notnull(margen_bruto) else 0))
with c2:
    st.metric("EBITDA", money(ebitda_total))
with c3:
    st.metric(
        "% Crecimiento (T vs T-12)",
        percent(crec if pd.notnull(crec) else 0),
    )

# -----------------------------
# 4) Tendencia trimestral
# -----------------------------
st.markdown("#### Tendencia de KPIs por Trimestre")
df_long = melt_long(df)
if df_long is not None:
    base_var = "Capital Dispersado"
    dfl = df_long[df_long["Variable"].str.contains(base_var, na=False)].copy()
    if sel_region != "Todas" and "Región" in dfl.columns:
        dfl = dfl[dfl["Región"] == sel_region]

    def periodo_key(p):
        if p == "Actual":
            return 999
        try:
            return -int(p.split("-")[1])
        except Exception:
            return 0

    dfl["orden"] = dfl["Periodo"].apply(periodo_key)
    dfl = dfl.sort_values("orden")

    chart = (
        alt.Chart(dfl)
        .mark_line(point=True)
        .encode(
            x=alt.X("Periodo:N", sort=list(dfl["Periodo"].unique())),
            y=alt.Y("sum(Valor):Q", title="Capital Dispersado"),
            tooltip=["Periodo:N", "sum(Valor):Q"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info(
        "No se detectaron columnas por periodo (Actual / T-xx) para graficar tendencia."
    )

# -----------------------------
# 5) Tabla de sucursales con recomendación
# -----------------------------
st.markdown("#### Desempeño por Sucursal")
cols_table = []
for c in [
    "Región",
    "Cluster",
    "Margen Bruto",
    "EBITDA",
    "ICV",
    "Capital Dispersado Actual",
    "Saldo Insoluto Actual",
    "Quitas Actual",
    "Castigos Actual",
    "Recomendación",
]:
    if c in dff.columns:
        cols_table.append(c)

if "Margen Bruto" not in dff.columns:
    dff["Margen Bruto"] = (1 - dff["ICV"]).clip(lower=0, upper=1)

st.dataframe(
    dff[cols_table]
    .rename(
        columns={
            "Capital Dispersado Actual": "Capital Dispersado",
            "Saldo Insoluto Actual": "Saldo Insoluto",
        }
    )
    .sort_values(["Región", "Cluster"]),
    use_container_width=True,
)

# -----------------------------
# 6) Visualizaciones adicionales
# -----------------------------
st.markdown("### Visualizaciones adicionales")

# 6.1 ICV promedio por Región (barras)
st.markdown("#### ICV promedio por Región")
if {"Región", "ICV"}.issubset(df.columns):
    df_icv_region = (
        df.groupby("Región", as_index=False)["ICV"]
        .mean()
    )
    df_icv_region = df_icv_region.sort_values("ICV", ascending=False)

    fig_icv_region = px.bar(
        df_icv_region,
        x="Región",
        y="ICV",
        title="ICV promedio por Región",
    )
    fig_icv_region.update_layout(
        xaxis_title="Región",
        yaxis_title="ICV promedio",
    )
    st.plotly_chart(fig_icv_region, use_container_width=True)
else:
    st.info("No se encontraron las columnas necesarias para calcular el ICV por región.")

# 6.2 Top 10 regiones por cartera vencida (barras horizontales apiladas)
st.markdown("#### Top 10 regiones por cartera vencida (composición vigente / 30-89 / vencido)")
if {
    "Región",
    "Saldo Insoluto Actual",
    "Saldo Insoluto Vencido Actual",
    "Saldo Insoluto 30-89  Actual",
}.issubset(df.columns):
    df_comp = df.copy()
    df_comp["Saldo Vigente"] = (
        df_comp["Saldo Insoluto Actual"]
        - df_comp["Saldo Insoluto Vencido Actual"]
        - df_comp["Saldo Insoluto 30-89  Actual"]
    ).clip(lower=0)

    df_comp_region = (
        df_comp.groupby("Región", as_index=False)[
            ["Saldo Vigente", "Saldo Insoluto 30-89  Actual", "Saldo Insoluto Vencido Actual"]
        ].sum()
    )

    # ordenar por cartera vencida y tomar top 10
    df_comp_region = df_comp_region.sort_values(
        "Saldo Insoluto Vencido Actual", ascending=False
    ).head(10)

    df_comp_long = df_comp_region.melt(
        id_vars="Región",
        value_vars=["Saldo Vigente", "Saldo Insoluto 30-89  Actual", "Saldo Insoluto Vencido Actual"],
        var_name="Tipo",
        value_name="Monto"
    )

    fig_comp = px.bar(
        df_comp_long,
        y="Región",
        x="Monto",
        color="Tipo",
        barmode="stack",
        orientation="h",
        title="Top 10 regiones por cartera vencida – composición de cartera",
    )
    fig_comp.update_layout(
        xaxis_title="Monto total",
        yaxis_title="Región",
    )
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("No se encontraron columnas suficientes para la composición de cartera por región.")

# 6.3 Distribución de ICV por Clúster (boxplot)
st.markdown("#### Distribución de ICV por Clúster")
if {"Cluster", "ICV"}.issubset(df.columns):
    fig_box = px.box(
        df,
        x="Cluster",
        y="ICV",
        points="outliers",
        title="Distribución de ICV por Clúster",
    )
    fig_box.update_layout(
        xaxis_title="Clúster",
        yaxis_title="ICV",
    )
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("No se encontró información suficiente para el boxplot de ICV por clúster.")

# -----------------------------
# 7) Panel de recomendaciones por clúster
# -----------------------------
st.markdown("#### Recomendaciones por Clúster")
colr = st.columns(4)
for i, col in enumerate(colr):
    with col:
        st.subheader(f"Cluster {i}")
        st.caption(cluster_summary_text(i))
        st.success(cluster_action_priority(i))

# -----------------------------
# 8) Pie de página
# -----------------------------
st.markdown("---")
st.caption("Dimex • Tablero analítico (prototipo) – Datos históricos, no en tiempo real.")
