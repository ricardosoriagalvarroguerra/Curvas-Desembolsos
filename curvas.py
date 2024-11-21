import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar pandas para que no use comas como separadores de miles
pd.options.display.float_format = '{:.0f}'.format

# Logistic model function
def logistic_model(k, b0, b1, b2):
    return b0 + (1 / (1 + b2 * np.exp(-b1 * k)))

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

# Sidebar for filters
st.sidebar.header("Filtros Opcionales")

# Menú desplegable para escoger tipo de filtro
filter_type = st.sidebar.selectbox(
    "Selecciona el tipo de filtro:",
    ["Sin Filtro", "sector_name", "pais", "tipo_prestamo"]
)

selected_values = None

# Opciones dinámicas si no se selecciona "Sin Filtro"
if filter_type != "Sin Filtro":
    unique_values = data[filter_type].unique()
    selected_values = st.sidebar.multiselect(
        f"Selecciona los valores de {filter_type} (todos seleccionados por defecto):",
        unique_values,
        default=unique_values
    )

# Filtro de fechas con control deslizante
min_date = data['fecha_aprobacion'].min().to_pydatetime()
max_date = data['fecha_aprobacion'].max().to_pydatetime()
selected_date_range = st.sidebar.slider(
    "Selecciona el rango de fechas de aprobación:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Aplicar filtros si corresponden
filtered_data = data.copy()

if filter_type != "Sin Filtro" and selected_values:
    filtered_data = filtered_data[filtered_data[filter_type].isin(selected_values)]

filtered_data = filtered_data[
    filtered_data['fecha_aprobacion'].between(selected_date_range[0], selected_date_range[1])
]

# Convertir 'year' a string para evitar comas en la visualización
filtered_data['year'] = filtered_data['year'].astype(str)

# Mostrar datos filtrados
st.subheader("Datos Filtrados")
st.write(filtered_data)

# Agrupar y preparar datos para el modelo
datamodelo_sumary = (
    filtered_data.groupby(['IDOperacion', 'year'], as_index=False)
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

# Aplicar la lógica de filtrado adicional
# Eliminar filas donde 'd' > 1.0 o 'k' < 0
datamodelo_sumary = datamodelo_sumary[
    (datamodelo_sumary['d'] <= 1.0) & (datamodelo_sumary['k'] >= 0)
]

# Ajustar modelo logístico
if not datamodelo_sumary.empty:
    initial_params = [-0.034145, 0.037973, 5.682123]
    params, covariance = curve_fit(
        logistic_model, 
        datamodelo_sumary['k'], 
        datamodelo_sumary['d'], 
        p0=initial_params,
        maxfev=2000
    )

    b0_hat, b1_hat, b2_hat = params
    datamodelo_sumary['hd_k'] = logistic_model(datamodelo_sumary['k'], b0_hat, b1_hat, b2_hat)

    # Graficar la curva
    st.subheader("Curva de Desembolsos Estimada")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(datamodelo_sumary['k'], datamodelo_sumary['hd_k'], label='Curva Estimada (hd_k)', color='red')
    ax.scatter(datamodelo_sumary['k'], datamodelo_sumary['d'], label='Datos Observados (d)', alpha=0.7)
    ax.set_title('Curva Histórica de Desembolsos - FONPLATA')
    ax.set_xlabel('Meses desde la Aprobación (k)')
    ax.set_ylabel('Proporción de Desembolsos Acumulados (hd(k))')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning("No hay datos que coincidan con los filtros seleccionados.")
