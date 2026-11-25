# Predicción de intención de compra en e-commerce

Trabajo final de Business Predictive Analytics (UPC), Grupo 2, sección 9341.

App de Streamlit con la demo del modelo: se ingresa una sesión simulada y el modelo (XGBoost) devuelve la probabilidad de que termine en compra (`revenue = True`).

- `app.py`: la app. Entrena el modelo al abrirse, con los mismos pasos y parámetros del notebook.
- `online_shoppers_intention.csv`: dataset de UCI (Sakar et al., 2019).
- `requirements.txt`: librerías.

Para correrla en local: `pip install -r requirements.txt` y luego `streamlit run app.py`.
