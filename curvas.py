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

# Streamlit app
st.title("Estimación de la Curva de Desembolsos - FONPLATA")

# Define file path
file_path = "fonplata_bdd.xlsx"  # Especifica el nombre del archivo local
data = load_data(file_path)

# Opciones para visualizar curvas por categorías
categories = st.radio(
    "Selecciona una categoría para visualizar las curvas:",
    ["Sectores", "Tipos de Préstamo", "Países"]
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

# Curvas específicas por categoría seleccionada
group_column = None
if categories == "Sectores":
    group_column = "sector_name"
elif categories == "Tipos de Préstamo":
    group_column = "tipo_prestamo"
elif categories == "Países":
    group_column = "pais"

# Crear gráfico
fig = go.Figure()

# Añadir la curva histórica general
fig.add_trace(go.Scatter(
    x=general_summary_sorted['k'],
    y=general_summary_sorted['hd_k'],
    mode='lines',
    name='Curva General (hd_k)',
    line=dict(color='red', width=4),
    hovertemplate="K (meses): %{x}<br>Proporción General: %{y:.2f}"
))

# Generar curvas específicas si se selecciona una categoría
if group_column:
    grouped_data = data.groupby(group_column)

    for group_name, group_df in grouped_data:
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

        # Ajustar modelo logístico
        if not datamodelo_sumary.empty:
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

            # Añadir la curva estimada para este grupo
            fig.add_trace(go.Scatter(
                x=datamodelo_sumary_sorted['k'],
                y=datamodelo_sumary_sorted['hd_k'],
                mode='lines',
                name=f"{group_name} - Curva Estimada",
                line=dict(width=2),
                hovertemplate=f"{group_column}: {group_name}<br>K (meses): %{{x}}<br>Proporción: %{{y:.2f}}"
            ))

            # Añadir puntos observados para este grupo
            fig.add_trace(go.Scatter(
                x=datamodelo_sumary['k'],
                y=datamodelo_sumary['d'],
                mode='markers',
                name=f"{group_name} - Datos Observados",
                marker=dict(size=6, opacity=0.8),
                hovertemplate=f"{group_column}: {group_name}<br>K (meses): %{{x}}<br>Proporción: %{{y:.2f}}"
            ))

# Personalizar diseño del gráfico
fig.update_layout(
    title=f"Curvas de Desembolsos (General y por {categories})",
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
    width=1200  # Ancho del gráfico más grande
)

# Mostrar el gráfico
st.plotly_chart(fig, use_container_width=True)
