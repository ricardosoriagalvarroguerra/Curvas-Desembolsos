import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from datetime import datetime

# Configurar pandas para que no use comas como separadores de miles
pd.options.display.float_format = '{:.0f}'.format

# Logistic model function (modificado según la metodología del BID)
def logistic_model(k, b0, b1, b2):
    return 1 / ((1 + np.exp(b0 - b1 * k)) ** b2)

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    # Convertir columnas de fecha al tipo datetime
    data['fecha_desembolso'] = pd.to_datetime(data['fecha_desembolso'], errors='coerce')
    data['fecha_aprobacion'] = pd.to_datetime(data['fecha_aprobacion'], errors='coerce')
    # Añadir columnas auxiliares
    data['year'] = data['fecha_desembolso'].dt.year.astype(int)  # Asegurar que 'year' es entero
    data['months_since_approval'] = (data['fecha_desembolso'] - data['fecha_aprobacion']).dt.days // 30
    return data

# Diccionario de colores minimalistas para países
country_colors = {
    "bolivia": "#4CAF50",  # Verde
    "argentina": "#2196F3",  # Azul
    "uruguay": "#00BCD4",  # Celeste
    "brasil": "#FFC107",  # Amarillo
    "paraguay": "#B71C1C"  # Rojo oscuro
}

# Diccionario de colores para sectores y tipos de préstamo
default_colors = [
    "#FF5733",  # Naranja
    "#33FF57",  # Verde claro
    "#3357FF",  # Azul fuerte
    "#FFC300",  # Amarillo oro
    "#DAF7A6",  # Verde pastel
    "#FF33F6",  # Rosa fuerte
    "#C70039",  # Rojo fuerte
    "#900C3F",  # Rojo vino
    "#581845",  # Morado oscuro
]

# Diccionario de abreviaturas para sectores
sector_abbreviations = {
    "Infraestructura y Medio Ambiente": "infra",
    "Sector Social": "social",
    "Gobernanza e Instituciones": "gob",
    "Mercado y Competitividad": "merc",
    "Integración Regional": "int"
}

# Streamlit app
st.title("Estimación de la Curva de Desembolsos - FONPLATA")

# Define file path
file_path = "fonplata_bdd.xlsx"  # Especifica el nombre del archivo local
data = load_data(file_path)

# Barra lateral para la selección de categoría
st.sidebar.header("Opciones")
categories = st.sidebar.selectbox(
    "Selecciona una categoría para visualizar las curvas:",
    ["General", "Sectores", "Tipos de Préstamo", "Países"],
    index=0
)

# Mostrar u ocultar observaciones individuales
show_observations = st.sidebar.checkbox("Mostrar observaciones individuales", value=False)

# Agrupar y preparar datos generales
general_data = data.copy()

# Preparar la curva histórica general
general_summary = (
    general_data.groupby(['IDOperacion', 'year'], as_index=False)
    .agg({
        'monto_desembolsado': 'sum',
        'monto_aprobacion': 'first',
        'months_since_approval': 'max'
    })
    .rename(columns={
        'monto_desembolsado': 'cumulative_disbursement_year',
        'monto_aprobacion': 'approval_amount',
        'months_since_approval': 'k'
    })
)

general_summary['cumulative_disbursement_total'] = general_summary.groupby('IDOperacion')['cumulative_disbursement_year'].cumsum()
general_summary['d'] = general_summary['cumulative_disbursement_total'] / general_summary['approval_amount']

# Eliminar filas donde 'd' > 1.0 o 'k' < 0
general_summary = general_summary[
    (general_summary['d'] <= 1.0) & (general_summary['k'] >= 0)
]

# Ajustar modelo logístico general
general_params = [0, 0, 0]
if not general_summary.empty:
    general_params, _ = curve_fit(
        lambda k, b0, b1, b2: logistic_model(k, b0, b1, b2),  # Mapear `x` a `k`
        general_summary['k'],
        general_summary['d'],
        p0=general_params,
        maxfev=2000
    )
    general_summary['hd_k'] = logistic_model(general_summary['k'], *general_params)
    general_summary_sorted = general_summary.sort_values(by='k')

# Crear gráfico
fig = go.Figure()

# Mostrar observaciones individuales si está activado
if show_observations:
    fig.add_trace(go.Scatter(
        x=general_summary['k'],
        y=general_summary['d'],
        mode='markers',
        name='observaciones',
        marker=dict(size=6, color="#888888", opacity=0.7),
        hovertemplate="IDOperacion: %{text}<br>Meses (K): %{x}<br>Proporción: %{y:.2f}",
        text=general_summary['IDOperacion']  # IDOperacion en el tooltip
    ))

# Añadir la curva histórica general después de las específicas para que quede arriba
fig.add_trace(go.Scatter(
    x=general_summary_sorted['k'],
    y=general_summary_sorted['hd_k'],
    mode='lines',
    name='general',
    line=dict(color='white', width=4),  # Más gruesa y de color blanco
    hovertemplate="Meses (K): %{{x}}<br>Proporción General: %{{y:.2f}}"
))

# Personalizar diseño del gráfico
fig.update_layout(
    title="Curvas de Desembolsos - FONPLATA",
    xaxis=dict(
        title='Meses desde la Aprobación (k)',
        titlefont=dict(color='white'),
        tickfont=dict(color='white')
    ),
    yaxis=dict(
        title='Proporción de Desembolsos Acumulados (hd(k))',
        titlefont=dict(color='white'),
        tickfont=dict(color='white')
    ),
    plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
    paper_bgcolor='rgba(0,0,0,0)',  # Fondo de todo el gráfico transparente
    font=dict(color='white'),  # Color del texto en blanco
    height=800,  # Altura del gráfico
    margin=dict(t=150, b=50, l=50, r=50),  # Más espacio superior para la leyenda
    legend=dict(
        orientation="h",  # Leyenda horizontal
        yanchor="top",  # Anclada en la parte superior del margen
        y=1.15,  # Por encima del área del gráfico
        xanchor="center",  # Centrada horizontalmente
        x=0.5,  # En el centro
        bgcolor="rgba(0,0,0,0)",  # Fondo transparente
    )
)

# Mostrar el gráfico
st.plotly_chart(fig, use_container_width=True)
