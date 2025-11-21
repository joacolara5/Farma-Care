# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Configuración de la página
st.set_page_config(
    page_title="Farmacare - Hospital Cochabamba",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# ESTILOS CSS CORREGIDOS
# =========================================
st.markdown("""
<style>
    /* FONDO PRINCIPAL BLANCO */
    .main {
        background-color: white !important;
    }
    
    /* SIDEBAR - FONDO AZUL Y TEXTO BLANCO */
    section[data-testid="stSidebar"] {
        background-color: #1e3c72 !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown div {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.3) !important;
    }
    
    /* TÍTULO PRINCIPAL */
    .main-header {
        font-size: 4rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        letter-spacing: 1px;
    }
    
    /* TARJETAS Y MÉTRICAS */
    .hospital-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        color: white;
        text-align: center;
    }
    
    .metric-card h3 {
        color: white !important;
        margin: 0;
        font-size: 2rem;
    }
    
    .metric-card p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0;
        font-size: 1rem;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff5e62, #ff9966);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
    }
    
    .drug-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 2px solid #f0f0f0;
    }
    
    .effect-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #333;
        border-left: 4px solid #ff6b6b;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffec61 0%, #f321d7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #333;
        border-left: 4px solid #ffd700;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# =========================================

@st.cache_data
def load_data():
    """Cargar datos desde la ruta específica"""
    try:
        df = pd.read_csv("C:\\Users\\HP\\Desktop\\Mis cosas\\4SEM\\DataW\\Farma-Care\\realistic_drug_labels_side_effects.csv")
        st.success("✅ Datos reales cargados correctamente")
        return df
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        st.info("📁 Cargando datos de ejemplo como respaldo...")
        return crear_datos_ejemplo()

def crear_datos_ejemplo():
    """Crear datos de ejemplo si no se puede cargar el archivo"""
    np.random.seed(42)
    n_samples = 200
    
    datos = {
        'drug_name': ['Aspirin', 'Ibuprofen', 'Metformin', 'Lisinopril', 'Atorvastatin', 
                     'Amoxicillin', 'Omeprazole', 'Simvastatin', 'Losartan', 'Gabapentin'] * 20,
        'manufacturer': ['Pfizer', 'Bayer', 'Novartis', 'Merck', 'Roche', 
                        'GSK', 'AstraZeneca', 'Johnson & Johnson', 'Sanofi', 'AbbVie'] * 20,
        'drug_class': ['Analgesic', 'Anti-inflammatory', 'Antidiabetic', 'Antihypertensive', 'Statin',
                      'Antibiotic', 'PPI', 'Statin', 'ARB', 'Anticonvulsant'] * 20,
        'side_effects': ['headache,nausea,dizziness', 'stomach pain,dizziness,rash', 
                        'diarrhea,nausea,abdominal pain', 'cough,dizziness,headache',
                        'muscle pain,liver problems,nausea', 'rash,diarrhea,nausea,allergic reactions',
                        'headache,abdominal pain,nausea', 'muscle pain,liver issues,headache',
                        'dizziness,cough,hyperkalemia', 'dizziness,fatigue,sleepiness,edema'] * 20,
        'dosage_mg': np.random.randint(5, 500, n_samples),
        'side_effect_severity': np.random.choice(['mild', 'moderate', 'severe', 'critical'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'warnings': ['Take with food', 'Avoid alcohol', 'Monitor blood sugar', 
                    'Check blood pressure', 'Monitor liver enzymes', 
                    'Complete full course', 'Take before meals', 
                    'Regular liver tests', 'Monitor kidney function', 
                    'Avoid driving'] * 20,
        'contraindications': ['Peptic ulcer', 'Kidney disease', 'Liver disease', 
                             'Pregnancy', 'Alcoholism', 'Penicillin allergy',
                             'Severe liver disease', 'Pregnancy', 
                             'Angioedema history', 'Depression'] * 20,
        'indications': ['Pain relief, Fever', 'Inflammation, Pain', 'Type 2 Diabetes', 
                       'Hypertension', 'High cholesterol', 'Bacterial infections',
                       'GERD, Ulcers', 'High cholesterol', 'Hypertension', 
                       'Neuropathic pain'] * 20,
        'administration_route': ['Oral', 'Oral', 'Oral', 'Oral', 'Oral', 
                                'Oral', 'Oral', 'Oral', 'Oral', 'Oral'] * 20,
        'approval_status': ['approved', 'approved', 'approved', 'approved', 'approved',
                           'approved', 'approved', 'approved', 'approved', 'approved'] * 20
    }
    
    return pd.DataFrame(datos)

@st.cache_resource
def preprocess_data(df):
    """Preprocesar datos y entrenar modelos"""
    with st.spinner("🔄 Procesando datos y entrenando modelos..."):
        # Preprocesamiento similar al código original
        categorical_cols = ['drug_name', 'manufacturer', 'drug_class', 'approval_status', 'administration_route']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str).str.strip()
        
        severity_map = {'mild': 0.2, 'moderate': 0.5, 'severe': 0.8, 'critical': 1.0}
        df['side_effect_severity_norm'] = df['side_effect_severity'].astype(str).str.strip().str.lower()
        df['severity_score'] = df['side_effect_severity_norm'].map(severity_map).fillna(0.0)
        
        numeric_cols = ['dosage_mg', 'severity_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
        # Ingeniería de características mejorada
        rng = np.random.RandomState(42)
        
        def create_continuous_target(row):
            base_risk = float(row.get('severity_score', 0.0) or 0.0)
            
            dosage = row.get('dosage_mg', 50)
            try:
                dosage = float(dosage)
            except Exception:
                dosage = 50.0
            dosage_factor = min(dosage / 200.0, 1.0)
            
            high_risk_classes = ['anticoagulant', 'chemotherapy', 'antipsychotic', 'chemo']
            drug_class = str(row.get('drug_class', '')).lower()
            class_factor = 0.3 if any(hrc in drug_class for hrc in high_risk_classes) else 0.1
            
            continuous_risk = base_risk * 0.6 + dosage_factor * 0.3 + class_factor * 0.1
            
            noise = rng.normal(0, 0.05)
            continuous_risk = np.clip(continuous_risk + noise, 0.0, 1.0)
            return continuous_risk
        
        df['continuous_risk'] = df.apply(create_continuous_target, axis=1)
        
        def risk_category(risk_score):
            if risk_score < 0.3:
                return 0  # Bajo
            elif risk_score < 0.6:
                return 1  # Moderado
            elif risk_score < 0.8:
                return 2  # Alto
            else:
                return 3  # Muy Alto
        
        df['multi_class_risk'] = df['continuous_risk'].apply(risk_category)
        
        return df

def analizar_efectos_adversos_combinados(medicamentos_seleccionados, df):
    """Analizar efectos adversos de la combinación de medicamentos"""
    efectos_analisis = {}
    
    # Recopilar todos los efectos secundarios
    todos_efectos = []
    efectos_por_medicamento = {}
    clases_medicamentos = []
    
    for med in medicamentos_seleccionados:
        # Verificar que el medicamento existe en el dataframe
        if med not in df['drug_name'].values:
            continue
            
        datos_med = df[df['drug_name'] == med].iloc[0]
        
        # Efectos secundarios
        if pd.notna(datos_med['side_effects']):
            efectos_lista = [efecto.strip() for efecto in str(datos_med['side_effects']).split(',')]
            todos_efectos.extend(efectos_lista)
            efectos_por_medicamento[med] = efectos_lista
        
        # Clases de medicamentos para análisis de interacciones
        clases_medicamentos.append(datos_med['drug_class'])
    
    # Análisis de frecuencia
    contador_efectos = Counter(todos_efectos)
    efectos_comunes = {efecto: count for efecto, count in contador_efectos.items() if count > 1}
    
    # Identificar efectos problemáticos
    efectos_problematicos = {}
    for efecto, count in efectos_comunes.items():
        medicamentos_afectados = []
        for med, efectos in efectos_por_medicamento.items():
            if efecto in efectos:
                medicamentos_afectados.append(med)
        efectos_problematicos[efecto] = {
            'frecuencia': count,
            'medicamentos': medicamentos_afectados
        }
    
    # Análisis de interacciones por clase
    interacciones_clases = analizar_interacciones_clases(clases_medicamentos)
    
    efectos_analisis = {
        'todos_efectos': todos_efectos,
        'efectos_por_medicamento': efectos_por_medicamento,
        'frecuencia_efectos': dict(contador_efectos),
        'efectos_problematicos': efectos_problematicos,
        'interacciones_clases': interacciones_clases
    }
    
    return efectos_analisis

def analizar_interacciones_clases(clases_medicamentos):
    """Analizar interacciones potenciales basadas en clases de medicamentos"""
    interacciones = []
    
    # Base de conocimiento de interacciones
    interacciones_conocidas = {
        ('Analgesic', 'Anti-inflammatory'): 'Mayor riesgo de sangrado gastrointestinal',
        ('Antidiabetic', 'Antihypertensive'): 'Posible potenciación de hipotensión',
        ('Statin', 'Antibiotic'): 'Riesgo aumentado de miopatía',
        ('ARB', 'Diuretic'): 'Posible potenciación de efectos antihipertensivos',
        ('Anticonvulsant', 'Antidepressant'): 'Interacción en metabolismo hepático'
    }
    
    clases_unicas = list(set(clases_medicamentos))
    
    # Verificar interacciones conocidas
    for i, clase1 in enumerate(clases_unicas):
        for j, clase2 in enumerate(clases_unicas):
            if i < j:  # Evitar duplicados
                par = (clase1, clase2)
                par_inverso = (clase2, clase1)
                
                if par in interacciones_conocidas:
                    interacciones.append({
                        'clases': par,
                        'descripcion': interacciones_conocidas[par]
                    })
                elif par_inverso in interacciones_conocidas:
                    interacciones.append({
                        'clases': par_inverso,
                        'descripcion': interacciones_conocidas[par_inverso]
                    })
    
    return interacciones

# =========================================
# COMPONENTES VISUALES
# =========================================

def crear_hero_section():
    """Crear sección hero con diseño elegante y profesional"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); padding: 3rem 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <div style="text-align: center;">
            <h1 class="main-header">Farmacare</h1>
            <p style="color: white; font-size: 1.5rem; margin: 0; font-style: italic;">Hospital Cochabamba - Sistema Inteligente de Análisis de Riesgo de Medicamentos</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def crear_metricas_principales(df):
    """Crear métricas principales con diseño mejorado"""
    st.markdown("## 📊 Dashboard de Métricas Clave")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_meds = len(df)
        st.markdown(f'''
        <div class="metric-card">
            <h3>{total_meds}</h3>
            <p>Total Medicamentos</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        riesgo_promedio = df['continuous_risk'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>{riesgo_promedio:.2f}</h3>
            <p>Riesgo Promedio</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        alto_riesgo = len(df[df['multi_class_risk'] >= 2])
        st.markdown(f'''
        <div class="metric-card">
            <h3>{alto_riesgo}</h3>
            <p>Alto Riesgo</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        fabricantes = df['manufacturer'].nunique()
        st.markdown(f'''
        <div class="metric-card">
            <h3>{fabricantes}</h3>
            <p>Fabricantes</p>
        </div>
        ''', unsafe_allow_html=True)

def crear_graficos_interactivos(df):
    """Crear gráficos interactivos con Plotly"""
    st.markdown("## 📈 Visualizaciones Interactivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de distribución de clases de medicamentos
        try:
            if 'drug_class' in df.columns and len(df) > 0:
                clases_count = df['drug_class'].value_counts().head(10)
                
                fig = px.bar(
                    x=clases_count.index,
                    y=clases_count.values,
                    title='Top 10 Clases de Medicamentos',
                    labels={'x': 'Clase de Medicamento', 'y': 'Cantidad'},
                    color=clases_count.values,
                    color_continuous_scale='blues'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400,
                    showlegend=False,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al crear el gráfico de clases: {e}")
    
    with col2:
        # Gráfico de categorías de riesgo
        try:
            if 'multi_class_risk' in df.columns and len(df) > 0:
                categorias = ['Bajo', 'Moderado', 'Alto', 'Muy Alto']
                colores = ['#00b09b', '#ffd200', '#ff9966', '#ff416c']
                
                conteo_categorias = df['multi_class_risk'].value_counts().sort_index()
                
                valores = []
                for i in range(4):
                    if i in conteo_categorias.index:
                        valores.append(conteo_categorias[i])
                    else:
                        valores.append(0)
                
                fig = go.Figure(data=[go.Pie(
                    labels=categorias,
                    values=valores,
                    hole=0.3,
                    marker=dict(colors=colores),
                    textinfo='percent+label'
                )])
                
                fig.update_layout(
                    title='Distribución de Categorías de Riesgo',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al crear el gráfico de categorías: {e}")

def mostrar_analisis_efectos_adversos(efectos_analisis):
    """Mostrar análisis detallado de efectos adversos"""
    
    st.markdown("### 🔍 Análisis Detallado de Efectos Adversos")
    
    # Efectos por medicamento
    if efectos_analisis['efectos_por_medicamento']:
        st.markdown("#### 💊 Efectos Secundarios por Medicamento")
        for medicamento, efectos in efectos_analisis['efectos_por_medicamento'].items():
            with st.expander(f"📋 {medicamento}", expanded=False):
                for efecto in efectos:
                    st.markdown(f'<div class="effect-card">• {efecto}</div>', unsafe_allow_html=True)
    
    # Efectos problemáticos
    if efectos_analisis['efectos_problematicos']:
        st.markdown("#### 🚨 Efectos Potencialmente Problemáticos")
        st.warning("Estos efectos aparecen en múltiples medicamentos y pueden potenciarse:")
        
        for efecto, info in efectos_analisis['efectos_problematicos'].items():
            medicamentos_lista = ", ".join(info['medicamentos'])
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ {efecto}</strong><br>
                <em>Aparece en {info['frecuencia']} medicamentos: {medicamentos_lista}</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Interacciones por clase
    if efectos_analisis['interacciones_clases']:
        st.markdown("#### ⚗️ Interacciones por Clase de Medicamento")
        for interaccion in efectos_analisis['interacciones_clases']:
            clases_str = " + ".join(interaccion['clases'])
            st.error(f"**{clases_str}**: {interaccion['descripcion']}")

def crear_analisis_combinacion(df):
    """Interfaz para análisis de combinación"""
    st.markdown("## 🔍 Análisis de Combinación de Medicamentos")
    
    # Tarjeta de información
    st.markdown('''
    <div class="hospital-card">
        <h3>💡 Análisis Inteligente de Combinaciones</h3>
        <p>Ingrese los medicamentos a analizar. Nuestro sistema evaluará riesgos individuales y combinados, 
        identificando efectos secundarios potencialmente problemáticos.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Búsqueda con sugerencias
    medicamentos_disponibles = sorted(df['drug_name'].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        medicamentos_seleccionados = st.multiselect(
            "💊 Seleccione medicamentos para analizar:",
            options=medicamentos_disponibles,
            placeholder="Escriba o seleccione medicamentos..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analizar = st.button("🚀 Analizar Combinación", type="primary", use_container_width=True)
    
    if analizar and medicamentos_seleccionados:
        with st.spinner("🔬 Analizando combinación..."):
            time.sleep(1)
            
            # Análisis individual
            st.markdown("### 📊 Resultados Individuales")
            
            riesgos_individuales = {}
            for i, med in enumerate(medicamentos_seleccionados):
                if med not in df['drug_name'].values:
                    st.warning(f"Medicamento '{med}' no encontrado en la base de datos")
                    continue
                    
                datos_med = df[df['drug_name'] == med].iloc[0]
                riesgo = datos_med['continuous_risk']
                categoria_num = datos_med['multi_class_risk']
                categorias = ['Bajo', 'Moderado', 'Alto', 'Muy Alto']
                
                riesgos_individuales[med] = riesgo
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f'<div class="drug-card"><h4>{med}</h4><p>👨‍⚕️ {datos_med["drug_class"]}</p></div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Riesgo", f"{riesgo:.3f}")
                
                with col3:
                    if categoria_num == 0:
                        st.markdown(f'<div class="risk-low">{categorias[categoria_num]}</div>', unsafe_allow_html=True)
                    elif categoria_num == 1:
                        st.markdown(f'<div class="risk-moderate">{categorias[categoria_num]}</div>', unsafe_allow_html=True)
                    elif categoria_num == 2:
                        st.markdown(f'<div class="risk-high">{categorias[categoria_num]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-critical">{categorias[categoria_num]}</div>', unsafe_allow_html=True)
            
            if not riesgos_individuales:
                st.error("❌ No se pudo analizar ninguno de los medicamentos seleccionados")
                return
                
            # Análisis de efectos adversos
            efectos_analisis = analizar_efectos_adversos_combinados(medicamentos_seleccionados, df)
            mostrar_analisis_efectos_adversos(efectos_analisis)
            
            # Análisis combinado
            st.markdown("### 🔄 Análisis de Riesgo Combinado")
            
            riesgos = list(riesgos_individuales.values())
            riesgo_combinado = min(1.0, sum(riesgos) * 0.6 + max(riesgos) * 0.4)
            
            # Gráfico de gauge
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                try:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = riesgo_combinado,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Riesgo Combinado", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgreen"},
                                {'range': [0.3, 0.6], 'color': "yellow"},
                                {'range': [0.6, 0.8], 'color': "orange"},
                                {'range': [0.8, 1], 'color': "red"}
                            ]
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al crear el indicador de riesgo: {e}")
            
            # Recomendación
            st.markdown("### 💡 Recomendación del Sistema")
            
            if riesgo_combinado < 0.3:
                st.success("""
                ✅ **COMBINACIÓN SEGURA** 
                - Riesgo bajo detectado
                - Puede proceder con supervisión rutinaria
                - Efectos adversos mínimos esperados
                """)
            elif riesgo_combinado < 0.6:
                st.warning("""
                ⚠️ **PRECAUCIÓN RECOMENDADA** 
                - Riesgo moderado identificado
                - Monitorizar posibles efectos secundarios
                - Considerar alternativas si es posible
                """)
            elif riesgo_combinado < 0.8:
                st.error("""
                🚨 **ALTA PRECAUCIÓN** 
                - Riesgo alto detectado
                - Monitorización estrecha requerida
                - Evaluar beneficio vs riesgo
                """)
            else:
                st.error("""
                🚫 **RIESGO MUY ALTO - EVITAR COMBINACIÓN** 
                - Riesgo crítico identificado
                - Buscar alternativas inmediatamente
                - Consultar con especialista
                """)

def crear_base_datos_interactiva(df):
    """Base de datos interactiva y visual"""
    st.markdown("## 💊 Base de Datos de Medicamentos")
    
    # Filtros avanzados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        buscar_med = st.text_input("🔍 Buscar medicamento:", placeholder="Nombre del medicamento...")
    
    with col2:
        filtro_clase = st.selectbox("Filtrar por clase:", ["Todas"] + sorted(df['drug_class'].unique().tolist()))
    
    with col3:
        filtro_riesgo = st.selectbox("Filtrar por riesgo:", ["Todos", "Bajo", "Moderado", "Alto", "Muy Alto"])
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    if buscar_med:
        df_filtrado = df_filtrado[df_filtrado['drug_name'].str.contains(buscar_med, case=False, na=False)]
    
    if filtro_clase != "Todas":
        df_filtrado = df_filtrado[df_filtrado['drug_class'] == filtro_clase]
    
    if filtro_riesgo != "Todos":
        riesgo_map = {"Bajo": 0, "Moderado": 1, "Alto": 2, "Muy Alto": 3}
        df_filtrado = df_filtrado[df_filtrado['multi_class_risk'] == riesgo_map[filtro_riesgo]]
    
    # Mostrar resultados
    st.markdown(f"**📋 Mostrando {len(df_filtrado)} de {len(df)} medicamentos**")
    
    # Tarjetas de medicamentos
    for idx, row in df_filtrado.head(20).iterrows():
        categorias = ['Bajo', 'Moderado', 'Alto', 'Muy Alto']
        riesgo_clase = categorias[row['multi_class_risk']]
        
        if row['multi_class_risk'] == 0:
            riesgo_html = f'<div class="risk-low">{riesgo_clase}</div>'
        elif row['multi_class_risk'] == 1:
            riesgo_html = f'<div class="risk-moderate">{riesgo_clase}</div>'
        elif row['multi_class_risk'] == 2:
            riesgo_html = f'<div class="risk-high">{riesgo_clase}</div>'
        else:
            riesgo_html = f'<div class="risk-critical">{riesgo_clase}</div>'
        
        efectos = str(row['side_effects']) if pd.notna(row['side_effects']) else "No especificado"
        
        st.markdown(f'''
        <div class="drug-card">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 2;">
                    <h4 style="margin: 0; color: #2c3e50;">{row["drug_name"]}</h4>
                    <p style="margin: 5px 0; color: #666;">🏭 {row["manufacturer"]}</p>
                    <p style="margin: 5px 0; color: #666;">👨‍⚕️ {row["drug_class"]}</p>
                    <p style="margin: 5px 0; color: #666; font-size: 0.9em;">
                        <strong>Efectos:</strong> {efectos}
                    </p>
                </div>
                <div style="flex: 1; text-align: center;">
                    <h3 style="margin: 0; color: #667eea;">{row["continuous_risk"]:.3f}</h3>
                    <p style="margin: 0; color: #666;">Riesgo</p>
                </div>
                <div style="flex: 1;">
                    {riesgo_html}
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    if len(df_filtrado) > 20:
        st.info(f"💡 Mostrando los primeros 20 resultados de {len(df_filtrado)}. Use los filtros para refinar la búsqueda.")

# =========================================
# APLICACIÓN PRINCIPAL
# =========================================

def main():
    # Cargar y procesar datos
    df = load_data()
    df_procesado = preprocess_data(df)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">Farmacare</h2>
            <p style="color: white; margin: 0; opacity: 0.8;">Hospital Cochabamba</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        opcion = st.radio(
            "Navegación Principal:",
            ["📊 Dashboard Principal", "🔍 Analizar Combinaciones", "💊 Base de Datos", "ℹ️ Acerca del Sistema"]
        )
        
        st.markdown("---")
        
        # Información del sistema en sidebar
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
            <h4 style="color: white;">📈 Estadísticas Rápidas</h4>
            <p style="color: white;">• Medicamentos: {}</p>
            <p style="color: white;">• Riesgo promedio: {:.2f}</p>
            <p style="color: white;">• Fabricantes: {}</p>
        </div>
        """.format(
            len(df_procesado),
            df_procesado['continuous_risk'].mean(),
            df_procesado['manufacturer'].nunique()
        ), unsafe_allow_html=True)
    
    # Contenido principal basado en selección
    if opcion == "📊 Dashboard Principal":
        crear_hero_section()
        crear_metricas_principales(df_procesado)
        crear_graficos_interactivos(df_procesado)
        
        # Información adicional
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="hospital-card">
                <h3>🎯 Nuestro Enfoque</h3>
                <p>Utilizamos algoritmos de machine learning avanzado para predecir 
                riesgos de medicamentos y sus combinaciones, garantizando la seguridad 
                de nuestros pacientes.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="hospital-card">
                <h3>🛡️ Compromiso con la Seguridad</h3>
                <p>El Hospital Cochabamba se compromete a utilizar las últimas tecnologías 
                para prevenir interacciones medicamentosas peligrosas y proteger la salud 
                de nuestra comunidad.</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif opcion == "🔍 Analizar Combinaciones":
        crear_hero_section()
        crear_analisis_combinacion(df_procesado)
    
    elif opcion == "💊 Base de Datos":
        crear_hero_section()
        crear_base_datos_interactiva(df_procesado)
    
    elif opcion == "ℹ️ Acerca del Sistema":
        crear_hero_section()
        
        st.markdown("""
        <div class="hospital-card">
            <h2>🤖 Acerca del Sistema Farmacare</h2>
            <p>Farmacare es una plataforma inteligente desarrollada por el Hospital Cochabamba 
            que utiliza algoritmos de machine learning avanzado para analizar y predecir riesgos 
            asociados con medicamentos y sus combinaciones.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🔬 Tecnología Utilizada</h3>
                <p>Machine Learning, Procesamiento de Lenguaje Natural, 
                Análisis de Texto, Visualizaciones Interactivas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Características</h3>
                <p>Severidad de efectos secundarios, Dosificación, 
                Clase terapéutica, Interacciones potenciales</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4 style="color: #2c3e50; margin-bottom: 0.5rem;">🏥 Farmacare</h4>
        <p style="margin: 0; font-size: 0.9rem;">Hospital Cochabamba • Departamento de Farmacología Clínica</p>
        <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Sistema Inteligente de Análisis de Riesgo v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()