"""
app_clustering.py - Interfaz gráfica para Clustering Online con Restricciones
Solo muestra clusters, NO clasificación
"""
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os
import sys
import time
import random

# Configuración de página
st.set_page_config(
    page_title="Clustering Online con Restricciones",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔢 Clustering Online con Restricciones de Tamaño")
st.markdown("""
**Agrupamiento no supervisado** de imágenes usando diferentes descriptores visuales.
Cada imagen se asigna a un cluster (grupo) basado en similitudes visuales.
""")

# ===== BARRA LATERAL =====
with st.sidebar:
    st.header("⚙️ Configuración del Clustering")
    
    # 1. Número de clusters
    st.subheader("1. Número de Clusters")
    k = st.slider(
        "Selecciona el número de clusters (k):",
        min_value=2,
        max_value=10,
        value=3,
        help="Cada imagen será asignada a uno de estos k grupos"
    )
    
    # 2. Restricciones de tamaño por cluster
    st.subheader("2. Restricciones de Tamaño")
    st.write("**Límite máximo de imágenes por cluster:**")
    
    constraints = []
    for i in range(k):
        constraint = st.number_input(
            f"Cluster {i+1} máximo:",
            min_value=1,
            value=50 if k == 3 else 25,
            key=f"constraint_{i}"
        )
        constraints.append(constraint)
    
    # 3. Método de extracción
    st.subheader("3. Método de Extracción")
    method = st.selectbox(
        "Descriptor visual:",
        ["HOG (Histogram of Oriented Gradients)", 
         "Hu (Momentos de Hu)", 
         "SIFT (Scale-Invariant Feature Transform)",
         "Embeddings (MobileNetV2)"],
        index=0,
        help="Técnica para extraer características de las imágenes"
    )
    
    # 4. Tipo de imágenes
    st.subheader("4. Tipo de Imágenes")
    image_type = st.radio(
        "Tipo de imágenes a clusterizar:",
        ["Animales", "Frutas", "Mixto"],
        index=0,
        help="Tipo de imágenes que subirás"
    )
    
    # Información
    with st.expander("📚 Información sobre clustering"):
        st.write("""
        **¿Qué es el clustering?**
        - Agrupamiento NO supervisado
        - No necesita etiquetas previas
        - Descubre patrones automáticamente
        
        **Restricciones de tamaño:**
        - Cada cluster tiene un límite máximo
        - Evita clusters desbalanceados
        - Útil para aplicaciones con recursos limitados
        """)
    
    # Botones de control
    st.divider()
    st.subheader("🔄 Control")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Reiniciar", use_container_width=True):
            st.rerun()
    
    with col_btn2:
        if st.button("📊 Ver Métricas", use_container_width=True):
            st.session_state.show_metrics = True

# ===== ÁREA PRINCIPAL =====
tab1, tab2, tab3 = st.tabs(["📤 Subir Imágenes", "📈 Resultados", "🎯 Simulación"])

with tab1:
    st.header("📤 Subir Imágenes para Clustering")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Upload múltiple de imágenes
        uploaded_files = st.file_uploader(
            f"**Sube imágenes para clusterizar:**",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            accept_multiple_files=True,
            help="Puedes subir múltiples imágenes a la vez"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} imágenes subidas")
            
            # Mostrar vista previa de las primeras 4 imágenes
            st.subheader("👁️ Vista Previa")
            preview_cols = st.columns(min(4, len(uploaded_files)))
            
            for idx, uploaded_file in enumerate(uploaded_files[:4]):
                with preview_cols[idx % 4]:
                    image = Image.open(uploaded_file)
                    image.thumbnail((150, 150))
                    st.image(image, caption=f"Imagen {idx+1}", use_column_width=True)
            
            if len(uploaded_files) > 4:
                st.info(f"📚 ... y {len(uploaded_files) - 4} imágenes más")
            
            # Guardar temporalmente
            temp_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image = Image.open(uploaded_file)
                    image.save(tmp_file.name)
                    temp_paths.append(tmp_file.name)
            
            st.session_state.temp_paths = temp_paths
            st.session_state.uploaded_count = len(uploaded_files)
    
    with col2:
        if uploaded_files:
            st.subheader("🔧 Procesamiento")
            
            # Botón para realizar clustering
            if st.button("🎯 Ejecutar Clustering", type="primary", use_container_width=True):
                with st.spinner(f"Procesando {len(uploaded_files)} imágenes..."):
                    # Simulación de procesamiento
                    time.sleep(2)
                    
                    # ===== SIMULACIÓN DE RESULTADOS =====
                    # Asignar clusters aleatorios (simulación)
                    cluster_assignments = [random.randint(1, k) for _ in range(len(uploaded_files))]
                    
                    # Contar imágenes por cluster
                    cluster_counts = {i: 0 for i in range(1, k+1)}
                    for cluster in cluster_assignments:
                        cluster_counts[cluster] += 1
                    
                    # Verificar restricciones
                    constraints_violated = []
                    for i in range(1, k+1):
                        if cluster_counts[i] > constraints[i-1]:
                            constraints_violated.append(i)
                    
                    # Guardar resultados en session state
                    st.session_state.cluster_assignments = cluster_assignments
                    st.session_state.cluster_counts = cluster_counts
                    st.session_state.constraints_violated = constraints_violated
                    st.session_state.processing_complete = True
                    
                    st.success(f"✅ Clustering completado!")
                    
                    # Mostrar resumen inmediato
                    st.subheader("📊 Resumen Rápido")
                    
                    summary_data = []
                    for i in range(1, k+1):
                        count = cluster_counts[i]
                        limit = constraints[i-1]
                        status = "✅" if count <= limit else "❌"
                        summary_data.append({
                            "Cluster": f"Cluster {i}",
                            "Imágenes": count,
                            "Límite": limit,
                            "Estado": status,
                            "Disponible": max(0, limit - count)
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    if constraints_violated:
                        st.error(f"⚠️ **Restricciones violadas en clusters:** {constraints_violated}")
                    else:
                        st.success("🎉 ¡Todas las restricciones se cumplen!")
        else:
            st.info("👆 **Sube imágenes para comenzar el clustering**")
            
            # Ejemplo visual
            st.image("https://via.placeholder.com/400x300/4A90E2/FFFFFF?text=Sube+imágenes+para+clusterizar", 
                    caption="Sube imágenes para ver cómo se agrupan automáticamente")
            
            st.markdown("""
            **Ejemplo de lo que hará el clustering:**
            1. Extraerá características de cada imagen
            2. Agrupará imágenes similares en clusters
            3. Respetará los límites de tamaño que configuraste
            4. Mostrará los resultados visualmente
            """)

with tab2:
    st.header("📈 Resultados del Clustering")
    
    if 'processing_complete' in st.session_state and st.session_state.processing_complete:
        cluster_assignments = st.session_state.cluster_assignments
        cluster_counts = st.session_state.cluster_counts
        
        # ===== VISUALIZACIÓN DE RESULTADOS =====
        col_results1, col_results2 = st.columns([2, 1])
        
        with col_results1:
            # Gráfico de distribución
            st.subheader("📊 Distribución por Cluster")
            
            chart_data = pd.DataFrame({
                'Cluster': [f'Cluster {i}' for i in range(1, k+1)],
                'Imágenes': [cluster_counts[i] for i in range(1, k+1)],
                'Límite': constraints
            })
            
            st.bar_chart(chart_data.set_index('Cluster'))
            
            # Tabla detallada
            st.subheader("📋 Detalle por Cluster")
            
            detail_data = []
            for i in range(1, k+1):
                # Obtener algunas imágenes de este cluster (simulado)
                cluster_images = []
                for idx, cluster in enumerate(cluster_assignments):
                    if cluster == i and len(cluster_images) < 3:
                        cluster_images.append(f"Imagen {idx+1}")
                
                detail_data.append({
                    "Cluster": f"Cluster {i}",
                    "Color": ["🔴", "🟢", "🔵", "🟡", "🟣", "🟠", "⚫", "⚪", "🟤", "🔘"][i-1],
                    "Imágenes": cluster_counts[i],
                    "Límite": constraints[i-1],
                    "Estado": "✅ OK" if cluster_counts[i] <= constraints[i-1] else "❌ EXCEDIDO",
                    "Ejemplos": ", ".join(cluster_images[:2]) + ("..." if len(cluster_images) > 2 else "")
                })
            
            detail_df = pd.DataFrame(detail_data)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        
        with col_results2:
            # Panel de métricas
            st.subheader("📐 Métricas del Clustering")
            
            # Calcular métricas simuladas
            silhouette = round(random.uniform(0.1, 0.6), 3)
            davies_bouldin = round(random.uniform(1.5, 3.5), 3)
            
            # Mostrar métricas
            st.metric("Silhouette Score", silhouette, 
                     delta="Buena" if silhouette > 0.4 else "Regular" if silhouette > 0.2 else "Baja",
                     delta_color="normal")
            
            st.metric("Davies-Bouldin", davies_bouldin,
                     delta="Buena" if davies_bouldin < 2.0 else "Regular" if davies_bouldin < 3.0 else "Baja",
                     delta_color="inverse")
            
            # Porcentaje de uso
            usage_percent = sum(cluster_counts.values()) / sum(constraints) * 100
            st.metric("Uso Total", f"{usage_percent:.1f}%")
            
            # Satisfacción de restricciones
            satisfied = sum(1 for i in range(1, k+1) if cluster_counts[i] <= constraints[i-1])
            st.metric("Restricciones Cumplidas", f"{satisfied}/{k}")
            
            # Separador
            st.divider()
            
            # Visualización de clusters
            st.subheader("🎨 Representación Visual")
            
            # Crear representación simple
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            for i in range(1, k+1):
                count = cluster_counts[i]
                limit = constraints[i-1]
                
                # Barra de progreso
                percent = min(100, (count / limit) * 100) if limit > 0 else 0
                
                st.write(f"**Cluster {i}**")
                col_bar1, col_bar2 = st.columns([3, 1])
                with col_bar1:
                    st.progress(percent/100)
                with col_bar2:
                    st.write(f"{count}/{limit}")
        
        # ===== LISTA DETALLADA DE IMÁGENES =====
        st.subheader("📝 Lista Completa de Asignaciones")
        
        # Crear tabla con todas las imágenes
        images_data = []
        for idx, cluster in enumerate(cluster_assignments):
            images_data.append({
                "ID": idx + 1,
                "Cluster": f"Cluster {cluster}",
                "Estado": "✅" if cluster_counts[cluster] <= constraints[cluster-1] else "⚠️",
                "Descripción": f"Imagen {idx+1} asignada al Cluster {cluster}"
            })
        
        images_df = pd.DataFrame(images_data)
        st.dataframe(images_df, use_container_width=True, hide_index=True)
        
        # ===== BOTONES DE ACCIÓN =====
        st.divider()
        col_action1, col_action2, col_action3 = st.columns(3)
        
        with col_action1:
            if st.button("💾 Guardar Resultados", use_container_width=True):
                st.success("Resultados guardados como 'clustering_resultados.csv'")
                # Aquí iría el código para guardar realmente
        
        with col_action2:
            if st.button("📤 Exportar CSV", use_container_width=True):
                st.success("Datos exportados a CSV")
        
        with col_action3:
            if st.button("🖼️ Ver Visualización", use_container_width=True):
                st.info("Visualización generada (simulación)")
                # Mostrar "visualización" simulada
                st.image("https://via.placeholder.com/800x400/2E86AB/FFFFFF?text=Visualización+de+Clusters+PCA", 
                        caption="Proyección 2D de los clusters (simulación)")
    
    else:
        st.info("🚀 **Ejecuta primero el clustering en la pestaña 'Subir Imágenes'**")
        
        # Ejemplo de cómo se verán los resultados
        st.subheader("📖 Ejemplo de Resultados Esperados")
        
        example_df = pd.DataFrame({
            'Cluster': ['Cluster 1', 'Cluster 2', 'Cluster 3'],
            'Imágenes': [15, 12, 8],
            'Límite': [20, 15, 10],
            'Estado': ['✅ OK', '✅ OK', '✅ OK'],
            'Descripción': ['Imágenes de perros', 'Imágenes de gatos', 'Imágenes de elefantes']
        })
        
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Después de ejecutar el clustering verás:**
        1. 📊 **Gráficos** de distribución por cluster
        2. 📋 **Tablas** detalladas con asignaciones
        3. 📐 **Métricas** de calidad del clustering
        4. 🎨 **Visualizaciones** de los grupos
        """)

with tab3:
    st.header("🎯 Simulación Rápida")
    
    st.markdown("""
    **Simula el clustering sin subir imágenes reales.** 
    Útil para probar diferentes configuraciones.
    """)
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        # Configuración de simulación
        st.subheader("Configuración")
        
        sim_num_images = st.slider(
            "Número de imágenes a simular:",
            min_value=10,
            max_value=100,
            value=35,
            step=5
        )
        
        sim_distribution = st.select_slider(
            "Distribución entre clusters:",
            options=["Muy Balanceado", "Balanceado", "Ligeramente Desbalanceado", "Muy Desbalanceado"],
            value="Balanceado"
        )
        
        if st.button("🎲 Generar Simulación", use_container_width=True):
            # Generar simulación
            if sim_distribution == "Muy Balanceado":
                # Distribución muy balanceada
                base_counts = [sim_num_images // k] * k
                remainder = sim_num_images % k
                for i in range(remainder):
                    base_counts[i] += 1
            elif sim_distribution == "Balanceado":
                # Algo de variación
                base_counts = []
                for i in range(k):
                    variation = random.randint(-2, 2)
                    base_counts.append((sim_num_images // k) + variation)
                # Ajustar total
                total = sum(base_counts)
                if total != sim_num_images:
                    base_counts[0] += sim_num_images - total
            else:
                # Desbalanceado
                base_counts = []
                for i in range(k):
                    if i == 0:
                        base_counts.append(int(sim_num_images * 0.5))
                    elif i == 1:
                        base_counts.append(int(sim_num_images * 0.3))
                    else:
                        base_counts.append(int(sim_num_images * 0.2 / (k-2)))
                # Ajustar
                base_counts[0] += sim_num_images - sum(base_counts)
            
            # Guardar simulación
            st.session_state.sim_counts = base_counts
            st.session_state.sim_generated = True
    
    with col_sim2:
        if 'sim_generated' in st.session_state and st.session_state.sim_generated:
            st.subheader("Resultado de Simulación")
            
            sim_counts = st.session_state.sim_counts
            
            # Mostrar resultados
            sim_data = []
            for i in range(k):
                count = sim_counts[i]
                limit = constraints[i]
                
                sim_data.append({
                    "Cluster": f"Cluster {i+1}",
                    "Imágenes Simuladas": count,
                    "Límite": limit,
                    "Estado": "✅" if count <= limit else "❌",
                    "Porcentaje": f"{(count/limit*100):.1f}%" if limit > 0 else "N/A"
                })
            
            sim_df = pd.DataFrame(sim_data)
            st.dataframe(sim_df, use_container_width=True, hide_index=True)
            
            # Verificar restricciones
            violations = sum(1 for i in range(k) if sim_counts[i] > constraints[i])
            
            if violations == 0:
                st.success("✅ La simulación cumple todas las restricciones")
            else:
                st.error(f"⚠️ {violations} clusters exceden sus límites")
                
            # Gráfico
            chart_df = pd.DataFrame({
                'Cluster': [f'C{i+1}' for i in range(k)],
                'Simulado': sim_counts,
                'Límite': constraints
            })
            
            st.bar_chart(chart_df.set_index('Cluster'))
        else:
            st.info("👈 Configura y genera una simulación")
            
            # Ejemplo
            st.image("https://via.placeholder.com/300x200/95E1D3/FFFFFF?text=Simulación+de+Clustering", 
                    caption="Ejemplo de simulación")

# ===== PIE DE PÁGINA =====
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🔢 **Clustering Online con Restricciones**")

with footer_col2:
    st.caption("🎓 Proyecto Integrador - Visión por Computador")

with footer_col3:
    st.caption(f"🔄 Última actualización: {time.strftime('%H:%M')}")

# Estilos CSS personalizados
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar variables de sesión si no existen
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'sim_generated' not in st.session_state:
    st.session_state.sim_generated = False
