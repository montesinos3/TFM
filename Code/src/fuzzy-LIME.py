import pandas as pd
import numpy as np
import random
import lime
from lime.lime_tabular import LimeTabularExplainer
from lime import lime_tabular, submodular_pick

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
import skfuzzy as fuzz
from simpful import *

import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import IPython
from IPython.display import display, HTML
import sys

# Forzamos que IPython.core.display tenga el atributo display
IPython.core.display.display = IPython.display.display

import warnings
warnings.filterwarnings('ignore')

class ExplicacionDifusa:
    """Objeto devuelto por el explicador con la interfaz visual interactiva."""
    def __init__(self, paquete, valor_real=None):
        self.pred_ling = paquete['Prediccion_Linguistica']
        self.pred_num = paquete['Prediccion_Numerica']
        self.explicaciones = paquete['Explicaciones']
        self.valor_real = valor_real

    def show_in_notebook(self):
        html = f"""
        <div style="font-family: Arial, sans-serif; border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; max-width: 650px; background-color: #f9f9f9; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h3 style="color: #4CAF50; margin-top: 0;">🧠 Explicación Fuzzy-LIME</h3>
        """
        if self.valor_real is not None:
             html += f"<p><b>Predicción de la Caja Negra:</b> {self.valor_real:.2f}</p>"
             
        html += f"""
            <p style="font-size: 1.1em;"><b>Diagnóstico Oficial:</b> <span style="background-color: #4CAF50; color: white; padding: 4px 10px; border-radius: 5px; font-weight: bold;">{self.pred_ling}</span></p>
            <p style="color: #666; font-size: 0.9em;">(Valor numérico inferido del centroide: {self.pred_num:.2f})</p>
            <hr style="border: 1px solid #ddd;">
            <h4 style="color: #333;">Desglose de Reglas Activadas:</h4>
            <ul style="list-style-type: none; padding-left: 0;">
        """
        if not self.explicaciones:
            html += "<li style='color: red;'>Ninguna regla del modelo global cubre esta instancia concreta.</li>"
        else:
            for exp in self.explicaciones:
                fuerza = exp['Fuerza_Activacion'] * 100
                color = "#4CAF50" if fuerza > 50 else "#FFC107" if fuerza > 20 else "#F44336"
                html += f"""
                <li style="margin-bottom: 15px;">
                    <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; margin-bottom: 5px;">
                        <div style="background-color: {color}; width: {fuerza}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                    <b>[{fuerza:05.2f}%]</b> {exp['Texto_Regla']}
                </li>
                """
        html += "</ul></div>"
        display(HTML(html))

class FuzzyLimeExplainer:
    """
    Explicador Global basado en Lógica Difusa.
    Encapsula extracción, traducción, clustering e inferencia Mamdani.
    """
    def __init__(self, predict_fn, training_data, feature_names=None, mode='regression', 
                 max_depth=3, num_clusters=None, num_instancias_a_procesar=None, random_state=1234):
        """
        Parámetros:
        - predict_fn: Función de predicción del modelo caja negra (ej: rf.predict).
        - training_data: DataFrame o Numpy Array con los datos de entrenamiento.
        - feature_names: Lista de nombres de las variables (si training_data es DataFrame, se pueden inferir, si no, se generan nombres por defecto).
        - mode: 'regression' o 'classification' (LIME lo necesita para saber cómo generar el vecindario). En nuestro caso de momento solo hemos implementado para regresión, pero se podría adaptar a clasificación.
        - max_depth: Profundidad máxima para los árboles rígidos locales (fase de extracción).
        - num_clusters: Número de reglas maestras a generar en la fase de clustering.
        - num_instancias_a_procesar: Cuántas instancias del training_data usar para la extracción masiva (fase 1). Si None, se usan todas.
        """
        # GESTIÓN INTELIGENTE DE DATOS Y NOMBRES DE VARIABLES
        if isinstance(training_data, pd.DataFrame):
            self.training_data = training_data
            # Si no pasan nombres, cogemos las columnas del DataFrame
            self.feature_names = feature_names if feature_names is not None else list(training_data.columns)
        else:
            # Si pasan un Numpy Array, generamos nombres por defecto si no los dan
            self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(training_data.shape[1])]
            self.training_data = pd.DataFrame(training_data, columns=self.feature_names)

        self.predict_fn = predict_fn
        self.mode = mode
        
        self.max_depth = max_depth
        self.num_clusters = num_clusters
        self.num_instancias = num_instancias_a_procesar if num_instancias_a_procesar else len(training_data)
        self.random_state = random_state


        self.dict_cortes = {}
        self.lista_reglas_crudas = []
        self.vocabulario_entrada = {}
        self.vocabulario_salida = {}
        self.modelo_global_difuso = None
        
        self._temp_X = None
        self._temp_preds = None

    def _funcion_espia_interna(self, X):
        self._temp_X = X
        preds = self.predict_fn(X)
        self._temp_preds = preds
        return preds

    def _extraer_reglas_arbol(self, dt, feature_names):
        tree_ = dt.tree_
        def recorrer(nodo, camino_actual):
            if tree_.children_left[nodo] == tree_.children_right[nodo]:
                self.lista_reglas_crudas.append({
                    'condiciones': camino_actual,
                    'prediccion': tree_.value[nodo][0][0]
                })
                return
            feature_name = feature_names[tree_.feature[nodo]]
            threshold = tree_.threshold[nodo]
            
            if feature_name not in self.dict_cortes:
                self.dict_cortes[feature_name] = []
            self.dict_cortes[feature_name].append(threshold)
            
            recorrer(tree_.children_left[nodo], camino_actual + [(feature_name, '<=', threshold)])
            recorrer(tree_.children_right[nodo], camino_actual + [(feature_name, '>', threshold)])
        recorrer(0, [])

    def fit(self):
        """Ejecuta todo el pipeline (Fases 1 a 5)"""
        print(f"Iniciando entrenamiento de Fuzzy-LIME ({self.num_instancias} instancias)...")
        np.random.seed(self.random_state)
        
        # --- FASE 1: Extracción ---
        lime_base = LimeTabularExplainer(self.training_data.values, feature_names=self.feature_names, mode=self.mode, discretize_continuous=False, random_state=self.random_state)
        
        for i in range(self.num_instancias):
            instancia = self.training_data.iloc[i]
            _ = lime_base.explain_instance(instancia, self._funcion_espia_interna, num_features=len(self.feature_names))
            
            df_vecindario = pd.DataFrame(self._temp_X, columns=self.feature_names)
            dt = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            dt.fit(df_vecindario, self._temp_preds)
            self._extraer_reglas_arbol(dt, self.feature_names)

        # --- FASE 2: Vocabularios ---
        for feature, cortes in self.dict_cortes.items():
            min_val, max_val = self.training_data[feature].min(), self.training_data[feature].max()
            universo = np.linspace(min_val, max_val, 200)
            p20, p40, p60, p80 = np.percentile(cortes, [20, 40, 60, 80])
            self.vocabulario_entrada[feature] = {
                'universo': universo,
                'MUY BAJO': fuzz.trapmf(universo, [min_val, min_val, p20, p40]),
                'BAJO': fuzz.trimf(universo, [p20, p40, p60]),
                'MEDIO': fuzz.trimf(universo, [p40, p60, p80]),
                'ALTO': fuzz.trimf(universo, [p60, p80, max_val]),
                'MUY ALTO': fuzz.trapmf(universo, [p80, max_val, max_val, max_val])
            }
            
        todas_preds = [r['prediccion'] for r in self.lista_reglas_crudas]
        min_p, max_p = min(todas_preds), max(todas_preds)
        universo_p = np.linspace(min_p, max_p, 200)
        p20, p40, p60, p80 = np.percentile(todas_preds, [20, 40, 60, 80])
        self.vocabulario_salida = {
            'universo': universo_p,
            'MUY BAJA': fuzz.trapmf(universo_p, [min_p, min_p, p20, p40]),
            'BAJA': fuzz.trimf(universo_p, [p20, p40, p60]),
            'MEDIA': fuzz.trimf(universo_p, [p40, p60, p80]),
            'ALTA': fuzz.trimf(universo_p, [p60, p80, max_p]),
            'MUY ALTA': fuzz.trapmf(universo_p, [p80, max_p, max_p, max_p])
        }

        # --- FASE 3: Traducción con Consolidación de Intervalos ---
        lista_reglas_difusas_completas = []
        for regla in self.lista_reglas_crudas:
            intervalos = {}
            for (feature, op, umbral) in regla['condiciones']:
                if feature not in intervalos:
                    vocab = self.vocabulario_entrada[feature]
                    intervalos[feature] = {'min': vocab['universo'][0], 'max': vocab['universo'][-1]}
                if op == '<=':
                    intervalos[feature]['max'] = min(intervalos[feature]['max'], umbral)
                elif op == '>':
                    intervalos[feature]['min'] = max(intervalos[feature]['min'], umbral)
                    
            condiciones_difusas = []
            for feature, limites in intervalos.items():
                punto_medio = (limites['min'] + limites['max']) / 2.0
                vocab = self.vocabulario_entrada[feature]
                idx_umbral = (np.abs(vocab['universo'] - punto_medio)).argmin()
                pertenencia = {etiq: vocab[etiq][idx_umbral] for etiq in vocab.keys() if etiq != 'universo'}
                etiqueta_ganadora = max(pertenencia, key=pertenencia.get)
                condiciones_difusas.append(f"{feature} es {etiqueta_ganadora}")
                
            condicion_ordenada = " Y ".join(sorted(condiciones_difusas))
            pred = regla['prediccion']
            idx_pred = (np.abs(self.vocabulario_salida['universo'] - pred)).argmin()
            pertenencia_pred = {etiq: self.vocabulario_salida[etiq][idx_pred] for etiq in self.vocabulario_salida.keys() if etiq != 'universo'}
            etiqueta_pred_ganadora = max(pertenencia_pred, key=pertenencia_pred.get)
            
            lista_reglas_difusas_completas.append({
                'Condicion': condicion_ordenada,
                'Prediccion_Linguistica': etiqueta_pred_ganadora
            })

        # --- FASE 4: Resolución de Conflictos ---
        df_reglas = pd.DataFrame(lista_reglas_difusas_completas)
        df_frecuencias = df_reglas.groupby(['Condicion', 'Prediccion_Linguistica']).size().reset_index(name='Frecuencia')
        df_global_sin_conflictos = df_frecuencias.sort_values(by='Frecuencia', ascending=False).drop_duplicates(subset=['Condicion'], keep='first')

        # --- FASE 5: Fuzzy C-Means Clustering ---
        if self.num_clusters is not None and len(df_global_sin_conflictos) > self.num_clusters:
            etiquetas_ordenadas = ['MUY BAJA', 'BAJA', 'MEDIA', 'ALTA', 'MUY ALTA']
            etiquetas_var = ['MUY BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY ALTO']
            mapa_var_a_num = {etiqueta: i+1 for i, etiqueta in enumerate(etiquetas_var)}
            mapa_num_a_var = {i+1: etiqueta for i, etiqueta in enumerate(etiquetas_var)}
            mapa_pred_a_num = {etiqueta: i+1 for i, etiqueta in enumerate(etiquetas_ordenadas)}
            mapa_num_a_pred = {i+1: etiqueta for i, etiqueta in enumerate(etiquetas_ordenadas)}
            
            variables_activas = list(self.vocabulario_entrada.keys())
            matriz_vectores = []
            
            for _, row in df_global_sin_conflictos.iterrows():
                vector = {var: 3 for var in variables_activas}
                for parte in row['Condicion'].split(" Y "):
                    var_nombre, _, etiqueta = parte.partition(" es ")
                    if var_nombre in vector:
                        vector[var_nombre] = mapa_var_a_num.get(etiqueta, 3)
                vector['Prediccion'] = mapa_pred_a_num.get(row['Prediccion_Linguistica'], 3)
                for _ in range(row['Frecuencia']):
                    matriz_vectores.append(list(vector.values()))
                    
            datos_cluster = np.array(matriz_vectores).T 
            num_centros = self.num_clusters
            
            if num_centros > 1:
                centroides, _, _, _, _, _, _ = fuzz.cluster.cmeans(
                    datos_cluster, c=num_centros, m=2, error=0.005, maxiter=1000, init=None
                )
                reglas_finales_cluster = []
                for i, centroide in enumerate(centroides):
                    cond_texto = []
                    for j, var in enumerate(variables_activas):
                        valor_num = int(round(max(1, min(5, centroide[j]))))
                        cond_texto.append(f"{var} es {mapa_num_a_var[valor_num]}")
                    pred_num = int(round(max(1, min(5, centroide[-1]))))
                    
                    reglas_finales_cluster.append({
                        'Regla_Maestra': f"Centroide {i+1}",
                        'Condicion': " Y ".join(cond_texto),
                        'Prediccion_Linguistica': mapa_num_a_pred[pred_num]
                    })
                self.modelo_global_difuso = pd.DataFrame(reglas_finales_cluster).drop_duplicates(subset=['Condicion'])
            else:
                self.modelo_global_difuso = df_global_sin_conflictos
        else:
            self.modelo_global_difuso = df_global_sin_conflictos

        print(f"✅ Entrenamiento completado. Modelo Subrogado encapsulado en {len(self.modelo_global_difuso)} reglas maestras.")

    def explain_instance(self, data_row):
        """Motor Mamdani para la explicación local en tiempo real"""
        if self.modelo_global_difuso is None:
             raise ValueError("Debes llamar al método .fit() antes de predecir.")
             
        pertenencias_entrada = {}
        for feature in self.vocabulario_entrada.keys():
            val = data_row[feature]
            vocab = self.vocabulario_entrada[feature]
            universo = vocab['universo']
            val_clipped = np.clip(val, universo[0], universo[-1])
            idx = (np.abs(universo - val_clipped)).argmin()
            pertenencias_entrada[feature] = {etiq: vocab[etiq][idx] for etiq in vocab.keys() if etiq != 'universo'}
                    
        activaciones_salida = {etiq: 0.0 for etiq in self.vocabulario_salida.keys() if etiq != 'universo'}
        reglas_disparadas = []
        
        for _, row in self.modelo_global_difuso.iterrows():
            condicion = row['Condicion']
            pred_ling = row['Prediccion_Linguistica']
            
            grados_condicion = []
            for parte in condicion.split(" Y "):
                var, _, etiq = parte.partition(" es ")
                if var in pertenencias_entrada and etiq in pertenencias_entrada[var]:
                    grados_condicion.append(pertenencias_entrada[var][etiq])
                else:
                    grados_condicion.append(0.0)
                    
            grado_activacion = min(grados_condicion) if grados_condicion else 0.0
            
            if grado_activacion > 0:
                activaciones_salida[pred_ling] = max(activaciones_salida[pred_ling], grado_activacion)
                reglas_disparadas.append({
                    'Fuerza_Activacion': grado_activacion,
                    'Texto_Regla': f"SI {condicion} ENTONCES Prediccion = {pred_ling}"
                })
                
        reglas_disparadas = sorted(reglas_disparadas, key=lambda x: x['Fuerza_Activacion'], reverse=True)
            
        if sum(activaciones_salida.values()) == 0:
            pred_linguistica_oficial = "DESCONOCIDA (Falta cobertura)"
            pred_crisp_oculta = np.mean(self.vocabulario_salida['universo'])
        else:
            pred_linguistica_oficial = max(activaciones_salida, key=activaciones_salida.get)
            universo_out = self.vocabulario_salida['universo']
            agregacion_final = np.zeros_like(universo_out)
            for etiq, activacion in activaciones_salida.items():
                if activacion > 0:
                    agregacion_final = np.fmax(agregacion_final, np.fmin(activacion, self.vocabulario_salida[etiq]))
            pred_crisp_oculta = fuzz.defuzz(universo_out, agregacion_final, 'centroid')
            
        paquete_explicacion = {
            'Prediccion_Linguistica': pred_linguistica_oficial,
            'Prediccion_Numerica': pred_crisp_oculta,
            'Explicaciones': reglas_disparadas
        }
        
        valor_caja_negra = self.predict_fn(data_row.values.reshape(1, -1))[0]
        
        return ExplicacionDifusa(paquete_explicacion, valor_caja_negra)