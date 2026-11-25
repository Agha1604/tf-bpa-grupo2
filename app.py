import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Grupo 2 - Intención de compra", layout="wide")


# Mismos pasos de limpieza que el notebook
@st.cache_data
def cargar_datos():
    df = pd.read_csv("online_shoppers_intention.csv")
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "productrelated": "product_related",
        "productrelated_duration": "product_related_duration",
        "bouncerates": "bounce_rates",
        "exitrates": "exit_rates",
        "pagevalues": "page_values",
        "specialday": "special_day",
        "operatingsystems": "operating_systems",
        "traffictype": "traffic_type",
        "visitortype": "visitor_type",
    })
    df = df.astype({"month": "string", "visitor_type": "string"})
    df = df.drop_duplicates()
    df = df.drop(columns=["operating_systems", "browser", "region",
                          "bounce_rates", "exit_rates", "traffic_type"])
    df["venta"] = df["revenue"].astype(str).str.lower().eq("true").astype("int8")
    return df


cols_num = [
    "administrative", "administrative_duration",
    "informational", "informational_duration",
    "product_related", "product_related_duration",
    "page_values", "special_day"
]
cols_nominales = ["month", "visitor_type"]
cols_bool = ["weekend"]
cols_modelo = cols_num + cols_nominales + cols_bool


def a_entero(x):
    return x.astype(int)


# Entrenamos el modelo al abrir la app (XGBoost con los parámetros del GridSearch del notebook)
@st.cache_resource
def entrenar_modelo():
    df_one = cargar_datos()
    X = df_one.drop(columns=["venta", "revenue"])
    y = df_one["venta"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # winsorización con el P98 de train
    tope_alto = round(np.percentile(X_train["product_related_duration"], 98), 2)
    X_train["product_related_duration"] = X_train["product_related_duration"].clip(upper=tope_alto)
    X_test["product_related_duration"] = X_test["product_related_duration"].clip(upper=tope_alto)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]), cols_num),
        ("cat_nom", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
        ]), cols_nominales),
        ("bool", Pipeline([
            ("to_int", FunctionTransformer(a_entero, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False))
        ]), cols_bool)
    ])

    modelo = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                subsample=0.8, random_state=42, eval_metric="logloss", n_jobs=-1))
    ])
    modelo.fit(X_train[cols_modelo], y_train)

    prob = modelo.predict_proba(X_test[cols_modelo])[:, 1]
    pred = (prob >= 0.5).astype(int)
    metricas = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precisión (compra)": precision_score(y_test, pred),
        "Recall (compra)": recall_score(y_test, pred),
        "F1 (compra)": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }
    return modelo, metricas, X_train, y_train


df_one = cargar_datos()
modelo, metricas_test, X_train, y_train = entrenar_modelo()

st.title("Predicción de intención de compra en e-commerce")
st.caption("Business Predictive Analytics - Grupo 2 - Sección 9341")

tab_datos, tab_modelo, tab_demo = st.tabs(["Datos", "Modelo", "Demo"])

with tab_datos:
    c1, c2, c3 = st.columns(3)
    c1.metric("Sesiones (sin duplicados)", f"{len(df_one):,}".replace(",", " "))
    c2.metric("Sesiones con compra", f"{df_one['venta'].sum():,}".replace(",", " "))
    c3.metric("Tasa de compra", f"{df_one['venta'].mean() * 100:.1f}%")

    st.write("Supuestos del proyecto: cada sesión es independiente (no hay ID de usuario) y cada mes "
             "es un ciclo de compra independiente. Los gráficos usan solo los datos de entrenamiento.")

    train = X_train.copy()
    train["venta"] = y_train

    st.subheader("Tasa de conversión por mes (train)")
    orden_meses = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    meses = (train.groupby("month")["venta"].mean() * 100).reindex(orden_meses).round(2)
    st.bar_chart(meses, y_label="Conversión (%)", x_label="Mes")

    st.subheader("Tasa de conversión por tiempo en páginas de producto (train)")
    rangos = pd.cut(train["product_related_duration"] / 60, bins=[-1, 2, 10, 30, 1000],
                    labels=["0 - 2 min", "2 - 10 min", "10 - 30 min", "+30 min"])
    tabla_rangos = train.groupby(rangos, observed=True)["venta"].agg(sesiones="size", compras="sum")
    tabla_rangos["conversión (%)"] = (tabla_rangos["compras"] / tabla_rangos["sesiones"] * 100).round(2)
    st.dataframe(tabla_rangos)

with tab_modelo:
    st.subheader("Comparación en validación cruzada (5 pliegues, train)")
    st.write("Resultados del notebook. El modelo se eligió con ROC-AUC antes de usar test.")
    tabla_cv = pd.DataFrame({
        "Modelo": ["XGBoost (optimizado)", "Random Forest", "Decision Tree",
                   "Regresión Logística", "Baseline (aleatorio)"],
        "ROC-AUC (CV)": [0.9302, 0.9212, 0.9114, 0.8919, 0.5011],
        "Desv. estándar": [0.0050, 0.0076, 0.0069, 0.0110, 0.0083],
    })
    st.dataframe(tabla_cv, hide_index=True)

    st.subheader("XGBoost optimizado en test (2 441 sesiones)")
    cols = st.columns(len(metricas_test))
    for col, (nombre, valor) in zip(cols, metricas_test.items()):
        col.metric(nombre, f"{valor:.4f}")
    st.write("Sin balanceo y con umbral de 0.5: el balanceo subía el recall, pero también las ventas "
             "perdidas (falsos positivos), que son el error más caro para la tienda.")
    st.write("Limitación: el modelo depende sobre todo de `page_values`. Sin esa variable el ROC-AUC "
             "baja de 0.93 a 0.75, así que sirve cuando la sesión ya avanzó por el catálogo.")

with tab_demo:
    st.subheader("Simular una sesión")
    st.write("Ingresa los datos acumulados de una sesión. `revenue` no se ingresa: es lo que el modelo predice.")

    ejemplos = {
        "Sesión 1: recurrente en noviembre, mucho catálogo": dict(
            administrative=3, administrative_duration=90.0, informational=0, informational_duration=0.0,
            product_related=45, product_related_duration=1800.0, page_values=40.0, special_day=0.0,
            month="Nov", visitor_type="Returning_Visitor", weekend=False),
        "Sesión 2: recurrente en diciembre, navegación media": dict(
            administrative=2, administrative_duration=60.0, informational=1, informational_duration=30.0,
            product_related=25, product_related_duration=900.0, page_values=8.0, special_day=0.0,
            month="Dec", visitor_type="Returning_Visitor", weekend=False),
        "Sesión 3: nueva en mayo, pocas páginas": dict(
            administrative=0, administrative_duration=0.0, informational=0, informational_duration=0.0,
            product_related=6, product_related_duration=150.0, page_values=0.0, special_day=0.0,
            month="May", visitor_type="New_Visitor", weekend=True),
    }
    eleccion = st.selectbox("Cargar un ejemplo", list(ejemplos.keys()))
    base = ejemplos[eleccion]

    c1, c2, c3 = st.columns(3)
    with c1:
        month = st.selectbox("Mes", orden_meses, index=orden_meses.index(base["month"]))
        visitor_type = st.selectbox("Tipo de visitante", ["Returning_Visitor", "New_Visitor", "Other"],
                                    index=["Returning_Visitor", "New_Visitor", "Other"].index(base["visitor_type"]))
        weekend = st.checkbox("Visita en fin de semana", value=base["weekend"])
        special_day = st.slider("Cercanía a fecha especial", 0.0, 1.0, base["special_day"], 0.2)
    with c2:
        product_related = st.number_input("Páginas de producto", 0, 700, base["product_related"])
        product_related_duration = st.number_input("Tiempo en páginas de producto (s)", 0.0, 64000.0,
                                                   base["product_related_duration"])
        page_values = st.number_input("page_values", 0.0, 400.0, base["page_values"])
    with c3:
        administrative = st.number_input("Páginas administrativas", 0, 30, base["administrative"])
        administrative_duration = st.number_input("Tiempo en administrativas (s)", 0.0, 4000.0,
                                                  base["administrative_duration"])
        informational = st.number_input("Páginas informativas", 0, 30, base["informational"])
        informational_duration = st.number_input("Tiempo en informativas (s)", 0.0, 3000.0,
                                                 base["informational_duration"])

    sesion = pd.DataFrame([{
        "administrative": administrative, "administrative_duration": administrative_duration,
        "informational": informational, "informational_duration": informational_duration,
        "product_related": product_related, "product_related_duration": product_related_duration,
        "page_values": page_values, "special_day": special_day,
        "month": month, "visitor_type": visitor_type, "weekend": weekend
    }])[cols_modelo]

    if st.button("Predecir", type="primary"):
        prob = float(modelo.predict_proba(sesion)[0, 1])
        r1, r2 = st.columns(2)
        r1.metric("Probabilidad de compra (revenue = True)", f"{prob * 100:.1f}%")
        r2.metric("Predicción", "Compra" if prob >= 0.5 else "No compra")
        st.progress(prob)
        if prob >= 0.5:
            st.success("Sin cupón: es probable que compre al precio normal.")
        else:
            st.warning("Candidata a cupón: es probable que se vaya sin comprar.")
        st.caption(f"En test el modelo tuvo ROC-AUC de {metricas_test['ROC-AUC']:.4f} y precisión de "
                   f"{metricas_test['Precisión (compra)'] * 100:.1f}% (de cada 10 sesiones que marca "
                   "como compra, unas 7 compran).")
