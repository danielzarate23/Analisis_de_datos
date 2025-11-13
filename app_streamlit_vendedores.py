# app_streamlit_vendedores.py

import io
import os
import pandas as pd
import streamlit as st
import altair as alt

# Configuración de página 
st.set_page_config(page_title="Dashboard de Vendedores", page_icon="", layout="wide")
st.title(" Dashboard de Vendedores")
st.caption("Lee 'vendedores.csv' (si existe) o carga un CSV desde la barra lateral.")

def normaliza_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pone nombres de columnas en minúsculas, sin acentos y con guiones bajos.
    Esto nos permite trabajar sin preocuparnos por mayúsculas/acentos/espacios.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
    )
    return df

def busca_columna(cols, candidatos):
    """
    Devuelve el primer nombre de columna que:
    1) coincide exactamente con alguno de 'candidatos', o
    2) contiene esa palabra (match "suave").
    """
    cols = list(cols)
    for c in candidatos:
        if c in cols:
            return c
    for c in candidatos:
        for col in cols:
            if c in col:
                return col
    return None

def a_numerico(serie: pd.Series) -> pd.Series:
    """
    Convierte texto con símbolos a número (ej. '$12,345.67' -> 12345.67).
    Si no puede convertir, devuelve 0.
    """
    return pd.to_numeric(
        serie.astype(str)
             .str.replace(r"[^\d\.\-]", "", regex=True)  
             .str.replace(",", ".", regex=False),        
        errors="coerce"
    ).fillna(0)

@st.cache_data
def lee_csv_local(path: str) -> pd.DataFrame | None:
    """
    Intenta leer un CSV desde disco; si no existe, devuelve None.
    Se prueba UTF-8 y si falla, Latin-1.
    """
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

@st.cache_data
def lee_csv_subido(file) -> pd.DataFrame | None:
    """
    Lee un CSV subido por el usuario mediante el 'file_uploader'.
    """
    if file is None:
        return None
    data = file.read()
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1")

# ------------------ Carga de datos ------------------
ruta = "vendedores.csv"
df_local = lee_csv_local(ruta)

with st.sidebar:
    st.header("Datos")
    if df_local is not None:
        st.success("Se cargó 'vendedores.csv' automáticamente.")
        st.caption("Si prefieres otro archivo, súbelo aquí:")
    else:
        st.info("No encontré 'vendedores.csv'. Súbelo aquí:")
    archivo = st.file_uploader("Subir .csv", type=["csv"])

# Prioridad: usar lo subido; si no, usar archivo local
df_raw = lee_csv_subido(archivo) if archivo else df_local

# Si no hay datos, terminamos temprano
if df_raw is None or df_raw.empty:
    st.error("No hay datos. Verifica tener 'vendedores.csv' o subir un archivo.")
    st.stop()

# Normalizamos nombres de columnas
df_raw = normaliza_columnas(df_raw)


col_vendedor = busca_columna(df_raw.columns, ["vendedor", "seller", "asesor", "representante", "ejecutivo"])
if col_vendedor is None:
    col_nombre = busca_columna(df_raw.columns, ["nombre"])
    col_apellido = busca_columna(df_raw.columns, ["apellido"])
    if col_nombre or col_apellido:
        df_raw["vendedor"] = (
            (df_raw[col_nombre] if col_nombre else "").astype(str).str.strip()
            + " "
            + (df_raw[col_apellido] if col_apellido else "").astype(str).str.strip()
        ).str.strip()
        col_vendedor = "vendedor"
    else:
        col_id = busca_columna(df_raw.columns, ["id"])
        if col_id:
            df_raw["vendedor"] = df_raw[col_id].astype(str)
            col_vendedor = "vendedor"

col_region   = busca_columna(df_raw.columns, ["region", "zona", "area", "territorio"])
col_unidades = busca_columna(df_raw.columns, ["unidades_vendidas", "unidades", "cantidad", "piezas", "items", "qty"])
col_ventas   = busca_columna(df_raw.columns, ["ventas_totales", "ventas", "monto", "importe", "total_ventas", "revenue"])

# Validamos que estén las columnas mínimas
faltantes = [n for n,c in {
    "vendedor": col_vendedor,
    "region": col_region,
    "unidades_vendidas": col_unidades,
    "ventas_totales": col_ventas,
}.items() if c is None]

if faltantes:
    st.error("No pude identificar columnas necesarias: " + ", ".join(faltantes))
    st.write("Columnas detectadas:", list(df_raw.columns))
    st.stop()

# Creamos un DataFrame homogéneo con nombres estándar
df = df_raw[[col_vendedor, col_region, col_unidades, col_ventas]].copy()
df.columns = ["vendedor", "region", "unidades_vendidas", "ventas_totales"]

# Aseguramos tipos numéricos para métricas y gráficas
df["unidades_vendidas"] = a_numerico(df["unidades_vendidas"])
df["ventas_totales"] = a_numerico(df["ventas_totales"])

# Controles y contenedores 
filtros = st.container()   # contenedor para filtros
tabla = st.container()     # contenedor para la tabla
kpis = st.container()      # contenedor para indicadores
graficas = st.container()  # contenedor para charts
detalle = st.container()   # contenedor para detalle de vendedor

# ---------- FILTROS ----------
with filtros:
    st.subheader(" Filtros")
    c1, c2, c3 = st.columns([1.2, 1, 1])

    # Filtro por Región
    with c1:
        regiones = sorted(df["region"].dropna().unique().tolist())
        regiones_sel = st.multiselect("Región", regiones, default=regiones)

    # Filtro y selector de vendedor para ver detalle
    with c2:
        vendedores = ["(Ninguno)"] + sorted(df["vendedor"].dropna().unique().tolist())
        vendedor_sel = st.selectbox("Vendedor (detalle)", vendedores, index=0)

    # Botón para limpiar filtros (resetea a todo)
    with c3:
        if st.button("Limpiar filtros"):
            regiones_sel = regiones
            vendedor_sel = "(Ninguno)"

# Aplicamos el filtro de región
df_f = df[df["region"].isin(regiones_sel)].copy()

# ---------- TABLA ----------
with tabla:
    st.subheader("Tabla filtrada")
    st.dataframe(df_f, use_container_width=True, hide_index=True)

    # Botón para descargar el CSV filtrado
    st.download_button(
        "⬇Descargar CSV filtrado",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="vendedores_filtrado.csv",
        mime="text/csv",
    )

# KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendedores únicos", df_f["vendedor"].nunique())
    c2.metric("Regiones", df_f["region"].nunique())
    c3.metric("Unidades (filtro)", int(df_f["unidades_vendidas"].sum()))
    c4.metric("Ventas (filtro)", f"${df_f['ventas_totales'].sum():,.2f}")

st.divider()

# GRÁFICAS 
with graficas:
    st.subheader("Gráficas")

    # 1) Unidades vendidas por vendedor (barras)
    g_units = (
        df_f.groupby("vendedor", as_index=False)["unidades_vendidas"].sum()
        .sort_values("unidades_vendidas", ascending=False)
    )
    chart_units = (
        alt.Chart(g_units)
        .mark_bar()
        .encode(
            x=alt.X("unidades_vendidas:Q", title="Unidades vendidas"),
            y=alt.Y("vendedor:N", sort="-x", title="Vendedor"),
            tooltip=["vendedor", "unidades_vendidas"],
        )
        .properties(title="Unidades vendidas por vendedor", height=320)
    )
    st.altair_chart(chart_units, use_container_width=True)

    # 2) Ventas totales por vendedor (barras)
    g_sales = (
        df_f.groupby("vendedor", as_index=False)["ventas_totales"].sum()
        .sort_values("ventas_totales", ascending=False)
    )
    chart_sales = (
        alt.Chart(g_sales)
        .mark_bar()
        .encode(
            x=alt.X("ventas_totales:Q", title="Ventas totales ($)"),
            y=alt.Y("vendedor:N", sort="-x", title="Vendedor"),
            tooltip=[
                "vendedor",
                alt.Tooltip("ventas_totales:Q", title="Ventas", format=",.2f"),
            ],
        )
        .properties(title="Ventas totales por vendedor", height=320)
    )
    st.altair_chart(chart_sales, use_container_width=True)

    # 3) Participación de ventas por vendedor 
    total = float(g_sales["ventas_totales"].sum())
    g_share = g_sales.copy()
    g_share["porcentaje"] = (g_share["ventas_totales"] / total * 100) if total else 0

    chart_share = (
        alt.Chart(g_share)
        .mark_arc(innerRadius=80)  
        .encode(
            theta=alt.Theta("ventas_totales:Q", stack=True, title="Ventas"),
            color=alt.Color("vendedor:N", legend=alt.Legend(title="Vendedor")),
            tooltip=[
                "vendedor",
                alt.Tooltip("ventas_totales:Q", title="Ventas", format=",.2f"),
                alt.Tooltip("porcentaje:Q", title="% Ventas", format=".2f"),
            ],
        )
        .properties(title="Participación de ventas (%)", height=360)
    )
    st.altair_chart(chart_share, use_container_width=True)

st.divider()

# DETALLE POR VENDEDOR 
with detalle:
    st.subheader("Detalle por vendedor")
    if vendedor_sel != "(Ninguno)":
        # Filtramos al vendedor seleccionado, respetando la región elegida
        df_v = df_f[df_f["vendedor"] == vendedor_sel].copy()
        if df_v.empty:
            st.warning("Ese vendedor no tiene registros dentro de las regiones seleccionadas.")
        else:
            # Pequeños KPIs del vendedor
            c1, c2, c3, c4 = st.columns(4)
            unidades_v = int(df_v["unidades_vendidas"].sum())
            ventas_v = float(df_v["ventas_totales"].sum())
            ticket = (ventas_v / unidades_v) if unidades_v else 0.0
            regiones_v = ", ".join(sorted(df_v["region"].unique().tolist()))

            c1.metric("Vendedor", vendedor_sel)
            c2.metric("Unidades", unidades_v)
            c3.metric("Ventas", f"${ventas_v:,.2f}")
            c4.metric("Ticket promedio", f"${ticket:,.2f}")
            st.markdown(f"**Regiones con ventas:** {regiones_v}")

            # Barras por región (unidades y ventas) del vendedor
            by_region = df_v.groupby("region", as_index=False)[["unidades_vendidas", "ventas_totales"]].sum()

            colA, colB = st.columns(2)
            with colA:
                chart_v_units = (
                    alt.Chart(by_region)
                    .mark_bar()
                    .encode(
                        x=alt.X("unidades_vendidas:Q", title="Unidades"),
                        y=alt.Y("region:N", sort="-x", title="Región"),
                        tooltip=["region", "unidades_vendidas"],
                    )
                    .properties(title=f"Unidades de {vendedor_sel} por región", height=280)
                )
                st.altair_chart(chart_v_units, use_container_width=True)

            with colB:
                chart_v_sales = (
                    alt.Chart(by_region)
                    .mark_bar()
                    .encode(
                        x=alt.X("ventas_totales:Q", title="Ventas ($)"),
                        y=alt.Y("region:N", sort="-x", title="Región"),
                        tooltip=[
                            "region",
                            alt.Tooltip("ventas_totales:Q", title="Ventas", format=",.2f"),
                        ],
                    )
                    .properties(title=f"Ventas de {vendedor_sel} por región", height=280)
                )
                st.altair_chart(chart_v_sales, use_container_width=True)
    else:
        st.info("Selecciona un vendedor en el filtro para ver su detalle.")
# Fin de app_streamlit_vendedores.py