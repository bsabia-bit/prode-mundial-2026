import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de la página
st.set_page_config(page_title="Dashboard Prode 2026", layout="wide")

# Base de datos simulada con los 48 equipos del Mundial 2026 (Ranking Elo)
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

# Composición de los 12 grupos
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
    
    # Extraemos el resultado exacto más probable de la matriz
    max_idx = np.unravel_index(np.argmax(matriz), matriz.shape)
    goles_a_exacto, goles_b_exacto = max_idx
    prob_exacta = matriz[goles_a_exacto][goles_b_exacto] * 100
    
    return xg_a, xg_b, gana_a, empate, gana_b, matriz, goles_a_exacto, goles_b_exacto, prob_exacta

st.title("Dashboard Analítico - Mundial 2026")

tab_grupos, tab_resumen, tab_libre, tab_llaves = st.tabs([
    "Simulador Grupos", "Resumen Automático", "Simulador Libre", "Llaves Eliminatorias"
])

# --- PESTAÑA 1: SIMULADOR GRUPOS ---
with tab_grupos:
    st.subheader("Simular partido específico por Grupo")
    grupo_elegido = st.selectbox("Elegí la zona:", list(grupos.keys()), key="sel_g1")
    equipos_del_grupo = grupos[grupo_elegido]
    
    col1, col2 = st.columns(2)
    with col1:
        equipo_a_g = st.selectbox("Equipo Local (Grupo)", equipos_del_grupo)
    with col2:
        equipo_b_g = st.selectbox("Equipo Visitante (Grupo)", [e for e in equipos_del_grupo if e != equipo_a_g])

    if equipo_a_g and equipo_b_g:
        elo_a, elo_b = equipos_elo[equipo_a_g], equipos_elo[equipo_b_g]
        xg_a, xg_b, gana_a, empate, gana_b, matriz, g_a, g_b, p_exacta = calcular_probabilidades(elo_a, elo_b)

        st.divider()
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(f"xG {equipo_a_g}", f"{xg_a:.2f}")
        
        # Lógica de Prode para la sugerencia visual
        if abs(elo_a - elo_b) <= 70:
            resultado_prode_g = "1 - 1 (Empate Técnico)"
        else:
            resultado_prode_g = f"{g_a} - {g_b}"
            
        kpi2.metric("Sugerencia Prode", resultado_prode_g)
        kpi3.metric(f"xG {equipo_b_g}", f"{xg_b:.2f}")

        col_grafico, col_heatmap = st.columns([1, 1])

        with col_grafico:
            st.subheader("Probabilidades Agrupadas (1X2)")
            df_probs = pd.DataFrame({
                "Resultado": [f"Gana {equipo_a_g}", "Empate", f"Gana {equipo_b_g}"],
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
            ax2.set_xlabel(f"Goles {equipo_b_g}", fontsize=10)
            ax2.set_ylabel(f"Goles {equipo_a_g}", fontsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            st.pyplot(fig2)

# --- PESTAÑA 2: RESUMEN AUTOMÁTICO ---
with tab_resumen:
    st.subheader("Predicciones Automáticas de la Fase de Grupos")
    grupo_resumen = st.selectbox("Elegí el grupo para ver la tabla completa:", list(grupos.keys()), key="sel_resumen")
    
    equipos = grupos[grupo_resumen]
    resultados_grupo = []
    
    for i in range(len(equipos)):
        for j in range(i + 1, len(equipos)):
            eq_a, eq_b = equipos[i], equipos[j]
            elo_a, elo_b = equipos_elo[eq_a], equipos_elo[eq_b]
            _, _, _, _, _, _, g_a, g_b, p_exacta = calcular_probabilidades(elo_a, elo_b)
            
            # Lógica de Prode: Si la diferencia de Elo es menor a 70 puntos, sugerir empate
            if abs(elo_a - elo_b) <= 70:
                pronostico = "Empate (1-1)"
            elif g_a > g_b:
                pronostico = f"Gana {eq_a} ({g_a}-{g_b})"
            elif g_b > g_a:
                pronostico = f"Gana {eq_b} ({g_a}-{g_b})"
            else:
                pronostico = f"Empate ({g_a}-{g_b})"
            
            resultados_grupo.append({
                "Partido": f"{eq_a} vs {eq_b}",
                "Pronóstico Exacto": pronostico,
                "Prob (Ese resultado)": f"{p_exacta:.1f}%"
            })
    
    st.table(pd.DataFrame(resultados_grupo).set_index("Partido"))

# --- PESTAÑA 3: SIMULADOR LIBRE ---
with tab_libre:
    st.subheader("Simulador Libre (Para cruces manuales)")
    col1, col2 = st.columns(2)
    with col1:
        equipo_a_l = st.selectbox("Equipo Local", nombres_equipos, key="loc_libre")
    with col2:
        equipo_b_l = st.selectbox("Equipo Visitante", nombres_equipos, index=1, key="vis_libre")

    if equipo_a_l and equipo_b_l and equipo_a_l != equipo_b_l:
        elo_a, elo_b = equipos_elo[equipo_a_l], equipos_elo[equipo_b_l]
        xg_a, xg_b, gana_a, empate, gana_b, matriz, g_a, g_b, p_exacta = calcular_probabilidades(elo_a, elo_b)

        st.divider()
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(f"xG {equipo_a_l}", f"{xg_a:.2f}")
        
        # Lógica de Prode para la sugerencia visual
        if abs(elo_a - elo_b) <= 70:
            resultado_prode_l = "1 - 1 (Empate Técnico)"
        else:
            resultado_prode_l = f"{g_a} - {g_b}"
            
        kpi2.metric("Sugerencia Prode", resultado_prode_l)
        kpi3.metric(f"xG {equipo_b_l}", f"{xg_b:.2f}")

        col_grafico, col_heatmap = st.columns([1, 1])

        with col_grafico:
            st.subheader("Probabilidades Agrupadas (1X2)")
            df_probs = pd.DataFrame({
                "Resultado": [f"Gana {equipo_a_l}", "Empate", f"Gana {equipo_b_l}"],
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
            ax2.set_xlabel(f"Goles {equipo_b_l}", fontsize=10)
            ax2.set_ylabel(f"Goles {equipo_a_l}", fontsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            st.pyplot(fig2)

# --- PESTAÑA 4: LLAVES ELIMINATORIAS ---
with tab_llaves:
    st.subheader("Calculadora Visual de Llaves Eliminatorias")
    st.markdown("Armá tu cruce. Acá no hay empates: el modelo evalúa quién avanza de ronda (por victoria en los 90 minutos, alargue o penales).")
    
    col_izq, col_med, col_der = st.columns([2, 1, 2])
    
    with col_izq:
        eq_llave_1 = st.selectbox("Equipo 1", nombres_equipos, index=0, key="llave_1")
    with col_der:
        eq_llave_2 = st.selectbox("Equipo 2", nombres_equipos, index=1, key="llave_2")
        
    if eq_llave_1 and eq_llave_2 and eq_llave_1 != eq_llave_2:
        elo_1, elo_2 = equipos_elo[eq_llave_1], equipos_elo[eq_llave_2]
        
        _, _, gana_1, empate, gana_2, _, _, _, _ = calcular_probabilidades(elo_1, elo_2)
        
        # Lógica de eliminación directa: Quien tiene mayor prob agrupada avanza
        total_victoria = gana_1 + gana_2
        prob_avanza_1 = (gana_1 / total_victoria) * 100
        prob_avanza_2 = (gana_2 / total_victoria) * 100
        
        with col_med:
            st.write("")
            st.write("")
            st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
            
        st.divider()
        st.markdown("### Predicción de Clasificación")
        
        if prob_avanza_1 > prob_avanza_2:
            st.success(f"Avanza {eq_llave_1} (Probabilidad: {prob_avanza_1:.1f}%)")
        else:
            st.success(f"Avanza {eq_llave_2} (Probabilidad: {prob_avanza_2:.1f}%)")
            
        st.progress(int(prob_avanza_1))
        st.caption(f"<- {eq_llave_1} | {eq_llave_2} ->")
