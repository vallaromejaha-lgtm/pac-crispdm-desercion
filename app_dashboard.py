import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PAC | Deserción estudiantil", page_icon="🎓", layout="wide")
st.title("🎓 Análisis de deserción estudiantil")
st.caption("Proyecto académico CRISP-DM · Datos educativos anonimizados · Clasificación binaria (Deserción vs Graduado)")
BASE = Path(__file__).resolve().parent
NECESARIOS = ["datos_dashboard.csv", "metricas_modelos.csv", "predicciones_prueba.csv", "importancia_variables.csv", "matrices_confusion.json"]
faltan = [x for x in NECESARIOS if not (BASE / x).exists()]
if faltan:
    st.error("Faltan archivos generados por el notebook: " + ", ".join(faltan))
    st.stop()

df = pd.read_csv(BASE / "datos_dashboard.csv")
metricas = pd.read_csv(BASE / "metricas_modelos.csv")
pred = pd.read_csv(BASE / "predicciones_prueba.csv")
importancias = pd.read_csv(BASE / "importancia_variables.csv")
matrices = json.loads((BASE / "matrices_confusion.json").read_text(encoding="utf-8"))

clases = ["Deserción", "Matriculado", "Graduado"]
clases_modelo = clases if len(next(iter(matrices.values()))) == 3 else ["Deserción", "Graduado"]
st.sidebar.header("Filtros de estudiantes")
selec_clases = st.sidebar.multiselect("Situación académica", clases, default=clases)
selec_carrera = st.sidebar.multiselect("Código de carrera", sorted(df["carrera"].unique().tolist()), default=[])
rango_edad = st.sidebar.slider("Edad al matricularse", int(df["edad_al_matricularse"].min()), int(df["edad_al_matricularse"].max()), (int(df["edad_al_matricularse"].min()),int(df["edad_al_matricularse"].max())))

f = df[df["situacion_es"].isin(selec_clases) & df["edad_al_matricularse"].between(*rango_edad)].copy()
if selec_carrera:
    f = f[f["carrera"].isin(selec_carrera)]

c1,c2,c3,c4 = st.columns(4)
cont = len(f)
num_des = int((f["situacion_es"] == "Deserción").sum())
c1.metric("Registros filtrados", f"{cont:,}")
c2.metric("Deserción", f"{num_des:,}")
c3.metric("% de deserción", f"{num_des/cont:.1%}" if cont else "—")
c4.metric("Carreras representadas", f["carrera"].nunique())

t1,t2,t3 = st.tabs(["Exploración de datos", "Comparación de modelos", "Predicciones y explicabilidad"])
with t1:
    if f.empty:
        st.warning("Los filtros no tienen resultados.")
    else:
        a,b=st.columns(2)
        cnt=f["situacion_es"].value_counts().reindex(clases, fill_value=0).rename_axis("Situación").reset_index(name="Cantidad")
        a.plotly_chart(px.bar(cnt,x="Situación",y="Cantidad",title="Distribución por situación académica",text="Cantidad"),use_container_width=True)
        b.plotly_chart(px.histogram(f,x="edad_al_matricularse",color="situacion_es",nbins=25, barmode="overlay",title="Edad de matrícula por situación"),use_container_width=True)
        clas=f.groupby(["carrera","situacion_es"]).size().reset_index(name="cantidad")
        clas["carrera"] = clas["carrera"].astype(str)
        st.plotly_chart(px.bar(clas,x="carrera",y="cantidad",color="situacion_es",title="Carrera (código original) y situación"),use_container_width=True)
        st.caption("Los códigos de carrera se mantienen como en el conjunto original. No implican orden ni magnitud.")
with t2:
    st.subheader("Evaluación final — conjunto de prueba")
    st.dataframe(metricas.round(4),hide_index=True,use_container_width=True)
    m=metricas.melt(id_vars="modelo",value_vars=["f1_macro","recall_desercion","accuracy"],var_name="Métrica",value_name="Valor")
    st.plotly_chart(px.bar(m,x="modelo",y="Valor",color="Métrica",barmode="group",range_y=[0,1],title="Comparación de métricas"),use_container_width=True)
    sel=st.selectbox("Matriz de confusión del modelo",list(matrices.keys()))
    st.plotly_chart(px.imshow(matrices[sel],x=clases_modelo,y=clases_modelo,text_auto=True,color_continuous_scale="Blues",labels={"x":"Predicción","y":"Clase real"},title=f"Matriz de confusión · {sel}"),use_container_width=True)
    st.caption("Los modelos se seleccionaron por F1 macro de validación. Los resultados de prueba no se utilizaron para ajustar hiperparámetros.")
with t3:
    st.subheader("Importancia de variables · Bosque aleatorio")
    im=importancias.head(12).sort_values("importancia_promedio")
    st.plotly_chart(px.bar(im,x="importancia_promedio",y="variable",orientation="h",title="Importancia por permutación en validación"),use_container_width=True)
    st.subheader("Predicciones en conjunto de prueba")
    st.dataframe(pred.head(150),hide_index=True,use_container_width=True)
    st.caption("Herramienta de análisis académico. No es un sistema validado para tomar decisiones individuales sobre estudiantes.")

st.divider()
st.markdown("**Fuente de referencia:** Realinho et al. (2021), UCI Machine Learning Repository, DOI: 10.24432/C5MC89. Verificar procedencia del CSV compartido por el equipo.")
