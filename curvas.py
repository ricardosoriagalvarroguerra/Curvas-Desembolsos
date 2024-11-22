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
@st.cache
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
general_params = [2.0, 0.1, 1.5]
if not general_summary.empty:
    general_params, _ = curve_fit(
        logistic_model,
        general_summary['k'],
        general_summary['d'],
        p0=general_params,
        maxfev=2000
    )
    general_summary['hd_k'] = logistic_model(general_summary['k'], *general_params)
    general_summary_sorted = general_summary.sort_values(by='k')

# Crear gráfico
fig = go.Figure()

# Generar curvas específicas si se selecciona una categoría
group_column = None
if categories == "Sectores":
    group_column = "sector_name"
elif categories == "Tipos de Préstamo":
    group_column = "tipo_prestamo"
elif categories == "Países":
    group_column = "pais"

if group_column:
    grouped_data = data.groupby(group_column)
    color_index = 0  # Para asignar colores diferentes a cada curva

    for group_name, group_df in grouped_data:
        # Convertir nombres de sectores a minúsculas y abreviados si corresponde
        if group_column == "sector_name":
            group_name = sector_abbreviations.get(group_name, group_name).lower()

        # Preparar datos para el modelo
        datamodelo_sumary = (
            group_df.groupby(['IDOperacion', 'year'], as_index=False)
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

        datamodelo_sumary['cumulative_disbursement_total'] = datamodelo_sumary.groupby('IDOperacion')['cumulative_disbursement_year'].cumsum()
        datamodelo_sumary['d'] = datamodelo_sumary['cumulative_disbursement_total'] / datamodelo_sumary['approval_amount']

        # Eliminar filas donde 'd' > 1.0 o 'k' < 0
        datamodelo_sumary = datamodelo_sumary[
            (datamodelo_sumary['d'] <= 1.0) & (datamodelo_sumary['k'] >= 0)
        ]

        # Ajustar modelo logístico solo si hay al menos 3 datos válidos
        if len(datamodelo_sumary) >= 3:
            initial_params = [2.0, 0.1, 1.5]
            params, _ = curve_fit(
                logistic_model,
                datamodelo_sumary['k'],
                datamodelo_sumary['d'],
                p0=initial_params,
                maxfev=2000
            )

            b0_hat, b1_hat, b2_hat = params
            datamodelo_sumary['hd_k'] = logistic_model(datamodelo_sumary['k'], b0_hat, b1_hat, b2_hat)

            # Ordenar los datos por 'k' para graficar correctamente
            datamodelo_sumary_sorted = datamodelo_sumary.sort_values(by='k')

            # Elegir color para el grupo o usar uno del conjunto por defecto
            if group_column == "pais":
                line_color = country_colors.get(group_name.lower(), "#808080")  # Default gris
            else:
                line_color = default_colors[color_index % len(default_colors)]
                color_index += 1

            # Añadir la curva estimada para este grupo
            fig.add_trace(go.Scatter(
                x=datamodelo_sumary_sorted['k'],
                y=datamodelo_sumary_sorted['hd_k'],
                mode='lines',
                name=group_name.lower(),
                line=dict(width=2, color=line_color, dash='dot'),
                hovertemplate=f"{group_column}: {group_name}<br>K (meses): %{{x}}<br>Proporción: %{{y:.2f}}"
            ))

# Añadir la curva histórica general después de las específicas para que quede arriba
fig.add_trace(go.Scatter(
    x=general_summary_sorted['k'],
    y=general_summary_sorted['hd_k'],
    mode='lines',
    name='general',
    line=dict(color='white', width=4),  # Más gruesa y de color blanco
    hovertemplate="K (meses): %{x}<br>Proporción General: %{y:.2f}"
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
    height=700,  # Altura del gráfico más grande
    width=1200,  # Ancho del gráfico más grande
    legend=dict(
        orientation="v",  # Apiladas verticalmente
        yanchor="top",
        y=1,
        xanchor="right",
        x=1.2  # A la derecha del gráfico
    )
)

# Mostrar el gráfico
st.plotly_chart(fig, use_container_width=True)
