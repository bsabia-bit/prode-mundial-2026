import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Prode 2026", layout="wide")

equipos_elo = {
    "Argentina": 2140, "Francia": 2110, "Brasil": 2080, "España": 2040, 
    "Inglaterra": 2030, "Bélgica": 2000, "Países Bajos": 1980, "Alemania": 1970, 
    "Portugal": 1960, "Uruguay": 1940, "Colombia": 1920, "Croacia": 1880, 
    "México": 1850, "Estados Unidos": 1830, "Senegal": 1780, "Marruecos": 1770, 
    "Japón": 1760, "Argelia": 1750, "Ecuador": 1740, "Paraguay": 1700,
    "Corea del Sur": 1690, "Suiza": 1680, "Austria": 1670, "Suecia": 1660,
    "Turquía": 1650, "República Checa": 1640, "Noruega": 1630, "Australia": 1620,
    "Escocia": 1610, "Canadá": 1600, "Túnez": 1590, "Egipto": 1580,
    "Irán": 1570, "Costa de Marfil": 1560, "Uzbekistán": 1550, 
    "Bosnia y Herzegovina": 1540, "Panamá": 1530, "Ghana": 1520, 
    "Arabia Saudita": 1510, "Qatar": 1500, "Nueva Zelanda": 1490, 
    "Sudáfrica": 1480, "Jordania": 1470, "Cabo Verde": 1460, 
    "Irak": 1450, "RD Congo": 1440, "Haití": 1430, "Curazao": 1420
}

grupos = {
    "Grupo A": ["México", "Sudáfrica", "Corea del Sur", "República Checa"],
    "Grupo B": ["Canadá", "Bosnia y Herzegovina", "Qatar", "Suiza"],
    "Grupo C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "Grupo D": ["Estados Unidos", "Paraguay", "Australia", "Turquía"],
    "Grupo E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "Grupo F": ["Países Bajos", "Japón", "Suecia", "Túnez"],
    "Grupo G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "Grupo H": ["España", "Cabo Verde", "Arabia Saudita", "Uruguay"],
    "Grupo I": ["Francia", "Senegal", "Irak", "Noruega"],
    "Grupo J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "Grupo K": ["Portugal", "RD Congo", "Uzbekistán", "Colombia"],
    "Grupo L": ["Inglaterra", "Croacia", "Ghana", "Panamá"]
}

nombres_equipos = sorted(list(equipos_elo.keys()))

def calcular_probabilidades(elo_a, elo_b):
    xg_a = max(0.1, 1.0 + (elo_a - elo_b) / 200)
    xg_b = max(0.1, 1.0 + (elo_b - elo_a) / 200)
    
    prob_a = [poisson.pmf(i, xg_a) for i in range(6)]
    prob_b = [poisson.pmf(i, xg_b) for i in range(6)]
    
    matriz = np.outer(prob_a, prob_b)
    
    gana_a = np.sum(np.tril(matriz, -1))
    empate = np.trace(matriz)
    gana_b = np.sum(np.triu(matriz, 1))
    
    return xg_a, xg_b, gana_a, empate, gana_b, matriz

st.title("Dashboard Analítico - Mundial 2026")

modo_simulacion = st.radio("Seleccioná el Modo:", ["Fase de Grupos", "Simulador Libre (Eliminatorias)"], horizontal=True)

if modo_simulacion == "Fase de Grupos":
    st.subheader("Simular por Grupo")
    grupo_elegido = st.selectbox("Elegí la zona:", list(grupos.keys()))
    equipos_del_grupo = grupos[grupo_elegido]
    
    col1, col2 = st.columns(2)
    with col1:
        equipo_a = st.selectbox("Equipo Local", equipos_del_grupo)
    with col2:
        equipo_b = st.selectbox("Equipo Visitante", [e for e in equipos_del_grupo if e != equipo_a])

else:
    st.subheader("Simulador Libre")
    col1, col2 = st.columns(2)
    with col1:
        equipo_a = st.selectbox("Equipo Local", nombres_equipos)
    with col2:
        equipo_b = st.selectbox("Equipo Visitante", nombres_equipos, index=1)

if equipo_a and equipo_b and equipo_a != equipo_b:
    elo_a, elo_b = equipos_elo[equipo_a], equipos_elo[equipo_b]
    xg_a, xg_b, gana_a, empate, gana_b, matriz = calcular_probabilidades(elo_a, elo_b)

    st.divider()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(f"xG {equipo_a}", f"{xg_a:.2f}")
    kpi2.metric("Diferencia de Nivel", f"{abs(elo_a - elo_b)} pts")
    kpi3.metric(f"xG {equipo_b}", f"{xg_b:.2f}")

    st.write("") 

    col_grafico, col_heatmap = st.columns([1, 1])

    with col_grafico:
        st.subheader("Probabilidades (1X2)")
        df_probs = pd.DataFrame({
            "Resultado": [f"Gana {equipo_a}", "Empate", f"Gana {equipo_b}"],
            "Prob (%)": [gana_a * 100, empate * 100, gana_b * 100]
        })
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Prob (%)", y="Resultado", data=df_probs, palette="mako", ax=ax)
        ax.set_xlim(0, 100)
        ax.set_ylabel("")
        
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 2, p.get_y() + 0.5), va='center')
        
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)

    with col_heatmap:
        st.subheader("Resultados Exactos")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(matriz * 100, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax2)
        ax2.set_xlabel(f"Goles {equipo_b}", fontsize=10)
        ax2.set_ylabel(f"Goles {equipo_a}", fontsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        st.pyplot(fig2)