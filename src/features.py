
# Importamos Path para gestionar rutas de archivos y carpetas de forma
# sencilla, evitando concatenar manualmente cadenas para construir rutas.
from pathlib import Path

# Importamos joblib para guardar y cargar objetos de Python.
# En este script lo utilizaremos para guardar el scaler y la lista de
# variables utilizadas por los modelos, para reutilizarlos posteriormente.
import joblib

# Importamos NumPy, una librería fundamental para trabajar con arrays,
# matrices, operaciones matemáticas y datos numéricos.
import numpy as np

# Importamos pandas para manipular los DataFrames que contienen
# los datos de los motores y sus sensores.
import pandas as pd

# Importamos yaml para leer el archivo de configuración config.yaml.
import yaml

# Importamos MinMaxScaler, que permite escalar variables numéricas
# para que se encuentren dentro de un rango determinado.
# Por defecto, este rango es [0, 1].
from sklearn.preprocessing import MinMaxScaler


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Lee el archivo YAML de configuración.

    Parámetros:
        config_path: Ruta del archivo de configuración.
                     Por defecto utiliza configs/config.yaml.

    Retorno:
        Diccionario de Python con la configuración del proyecto.
    """

    # Abrimos el archivo de configuración en modo lectura.
    # encoding="utf-8" permite leer correctamente caracteres especiales.
    with open(config_path, "r", encoding="utf-8") as f:

        # Convertimos el contenido YAML en un diccionario de Python
        # y lo devolvemos.
        return yaml.safe_load(f)


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas de sensores activos,
    excluyendo metadatos y la variable objetivo RUL.

    Parámetros:
        df: DataFrame que contiene los datos procesados.

    Retorno:
        Lista con los nombres de las columnas de sensores.
    """

    # Definimos un conjunto (set) con las columnas que NO son sensores.
    #
    # unit_id: identificador del motor.
    # cycle: ciclo temporal del motor.
    # op_setting_1, 2 y 3: parámetros operacionales.
    # RUL: variable objetivo que queremos predecir.
    #
    # Estas columnas no deben considerarse sensores para esta función.
    #
    # Utilizamos un set porque las comprobaciones de pertenencia
    # (col not in excluded) son eficientes.
    excluded = {
        "unit_id",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        "RUL"
    }

    # Recorremos todas las columnas del DataFrame.
    #
    # Conservamos únicamente aquellas columnas cuyo nombre no está
    # en el conjunto excluded.
    #
    # De esta manera obtenemos las columnas de sensores que permanecen
    # en el dataset después de la limpieza.
    return [col for col in df.columns if col not in excluded]


def add_temporal_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    rolling_windows: list[int],
    ewma_span: int,
) -> pd.DataFrame:
    """
    Calcula variables físicas de tendencia y ruido por motor (unit_id).

    REGLA INDUSTRIAL:
    Todas las ventanas miran EXCLUSIVAMENTE hacia atrás
    (min_periods=1) para garantizar cero data leakage del futuro.

    Parámetros:
        df: DataFrame con los datos originales y el RUL.
        sensor_cols: Lista de sensores sobre los que crear nuevas variables.
        rolling_windows: Tamaños de las ventanas móviles.
        ewma_span: Parámetro que controla el suavizado de la EWMA.

    Retorno:
        DataFrame original al que se añaden las nuevas variables temporales.
    """

    # Ordenamos el DataFrame primero por motor y después por ciclo.
    #
    # sort_values(["unit_id", "cycle"]) garantiza que las observaciones
    # temporales de cada motor estén ordenadas cronológicamente.
    #
    # Esto es fundamental para calcular correctamente medias móviles,
    # desviaciones móviles y medias exponenciales.
    #
    # copy() crea una copia para trabajar sobre ella sin modificar
    # directamente el DataFrame original recibido como argumento.
    df = df.sort_values(["unit_id", "cycle"]).copy()

    # Agrupamos los datos por identificador de motor.
    #
    # groupby("unit_id") hace que las operaciones posteriores se realicen
    # independientemente para cada motor.
    #
    # group_keys=False evita añadir claves de agrupación adicionales
    # al resultado de determinadas operaciones de GroupBy.
    #
    # Esto es importante porque cada motor tiene su propia evolución
    # temporal y no debemos mezclar los datos de motores diferentes.
    grouped = df.groupby("unit_id", group_keys=False)

    # Creamos un diccionario vacío donde almacenaremos todas las nuevas
    # variables temporales que vayamos calculando.
    #
    # Cada clave será el nombre de una nueva columna y cada valor
    # contendrá los datos calculados para dicha variable.
    #
    # Lo hacemos así para acumular las variables y crear posteriormente
    # un DataFrame con todas ellas de una sola vez.
    new_features = {}

    # Recorremos uno a uno los sensores seleccionados.
    #
    # col representa el nombre de un sensor en cada iteración.
    # Por ejemplo: sensor de temperatura, presión, velocidad, etc.
    #
    # Para cada sensor vamos a generar distintas variables temporales.
    for col in sensor_cols:

        # --------------------------------------------------------------
        # 1. SUAVIZADO EXPONENCIAL (EWMA)
        # --------------------------------------------------------------

        # Creamos una nueva variable cuyo nombre será el nombre original
        # del sensor seguido del sufijo "_ewma".
        #
        # Por ejemplo:
        #   sensor_original -> sensor_original_ewma
        #
        # grouped[col] selecciona el sensor dentro de cada grupo de motores.
        #
        # transform(...) aplica una operación a cada motor y devuelve
        # un resultado con la misma longitud y alineación que el DataFrame.
        #
        # lambda x: define una función anónima que recibe la serie temporal
        # de un motor concreto.
        #
        # x.ewm(...) calcula la media móvil exponencialmente ponderada.
        #
        # span=ewma_span controla la velocidad de suavizado:
        #   span pequeño -> más peso a los datos recientes, responde rápido.
        #   span grande  -> suavizado más fuerte, responde más lentamente.
        #
        # adjust=False utiliza el cálculo recursivo de la EWMA.
        #
        # .mean() calcula el valor suavizado de la señal.
        #
        # La EWMA ayuda a reducir ruido y resaltar la tendencia del sensor.
        new_features[f"{col}_ewma"] = grouped[col].transform(
            lambda x: x.ewm(span=ewma_span, adjust=False).mean()
        )

        # --------------------------------------------------------------
        # 2. VENTANAS MÓVILES HACIA ATRÁS
        # --------------------------------------------------------------

        # Recorremos los diferentes tamaños de ventana definidos
        # en el archivo de configuración.
        #
        # Por ejemplo, rolling_windows podría ser [5, 10, 20].
        #
        # Cada ventana representa el número de ciclos recientes
        # que utilizaremos para calcular estadísticas.
        for w in rolling_windows:

            # ----------------------------------------------------------
            # 2.1 MEDIA MÓVIL (ROLLING MEAN)
            # ----------------------------------------------------------

            # Creamos una variable que representa la media del sensor
            # durante los últimos w ciclos.
            #
            # Su nombre tendrá el formato:
            #   nombre_sensor_roll_mean_tamaño_ventana
            #
            # Por ejemplo:
            #   sensor_1_roll_mean_5
            #
            # grouped[col].transform(...) calcula esta operación
            # independientemente para cada motor.
            #
            # x.rolling(window=w) crea una ventana móvil de tamaño w.
            #
            # La ventana se desplaza a medida que avanzan los ciclos
            # y utiliza la observación actual y las anteriores.
            #
            # min_periods=1 indica que basta con tener una observación
            # para calcular la media.
            #
            # Esto evita que los primeros ciclos generen valores NaN
            # por no disponer todavía de una ventana completa.
            #
            # .mean() calcula la media de los valores dentro de la ventana.
            #
            # Esta variable puede ayudar al modelo a identificar tendencias
            # de degradación que no son evidentes en una lectura aislada.
            new_features[f"{col}_roll_mean_{w}"] = grouped[col].transform(
                lambda x: x.rolling(window=w, min_periods=1).mean()
            )

            # ----------------------------------------------------------
            # 2.2 DESVIACIÓN ESTÁNDAR MÓVIL (ROLLING STD)
            # ----------------------------------------------------------

            # Creamos una variable que representa la desviación estándar
            # del sensor durante los últimos w ciclos.
            #
            # Su nombre tendrá el formato:
            #   nombre_sensor_roll_std_tamaño_ventana
            #
            # Por ejemplo:
            #   sensor_1_roll_std_5
            #
            # La desviación estándar mide la dispersión de los valores.
            #
            # Una media móvil informa sobre el nivel o tendencia de la señal,
            # mientras que la desviación estándar informa sobre cuánto fluctúa.
            #
            # x.rolling(window=w, min_periods=1) crea una ventana móvil
            # de w ciclos, permitiendo calcular con menos observaciones
            # al comienzo de la serie.
            #
            # .std() calcula la desviación estándar de los valores
            # que se encuentran dentro de cada ventana.
            #
            # .fillna(0.0) sustituye los valores NaN por cero.
            #
            # Esto es necesario porque la desviación estándar requiere
            # normalmente al menos dos observaciones para calcularse.
            # En el primer ciclo, por tanto, puede resultar NaN.
            #
            # Lo sustituimos por 0.0 para evitar valores faltantes
            # en las variables que posteriormente utilizará el modelo.
            new_features[f"{col}_roll_std_{w}"] = grouped[col].transform(
                lambda x: x.rolling(window=w, min_periods=1).std().fillna(0.0)
            )

    # Convertimos el diccionario new_features en un DataFrame.
    #
    # Cada clave del diccionario pasa a ser el nombre de una columna.
    # Cada valor pasa a ser el conjunto de datos de esa columna.
    #
    # index=df.index garantiza que las nuevas variables estén asociadas
    # al índice del DataFrame df.
    #
    # Esto es importante para que posteriormente las nuevas columnas
    # se alineen correctamente con las filas originales.
    features_df = pd.DataFrame(new_features, index=df.index)

    # Concatenamos horizontalmente (axis=1) el DataFrame original df
    # con el DataFrame de nuevas variables features_df.
    #
    # El resultado conserva las columnas originales y añade todas
    # las variables temporales calculadas.
    #
    # Por ejemplo, un sensor originalmente puede tener una columna:
    #   sensor_1
    #
    # Y después de esta función podría tener:
    #   sensor_1
    #   sensor_1_ewma
    #   sensor_1_roll_mean_5
    #   sensor_1_roll_std_5
    #   sensor_1_roll_mean_10
    #   sensor_1_roll_std_10
    #
    # axis=1 significa concatenar columnas, no filas.
    return pd.concat([df, features_df], axis=1)


def split_train_val_by_unit(
    train_df: pd.DataFrame,
    train_ratio: float,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide train y validación por ID de motor (unit_id),
    nunca por filas sueltas.

    Parámetros:
        train_df: DataFrame de entrenamiento original.
        train_ratio: Proporción de motores que se asignará a train.
        seed: Semilla aleatoria para reproducibilidad.

    Retorno:
        Dos DataFrames:
            tr_split -> motores de entrenamiento.
            va_split -> motores de validación.
    """

    # Creamos un generador de números aleatorios de NumPy.
    #
    # seed establece la semilla que controla la generación aleatoria.
    #
    # Utilizar una semilla fija permite reproducir exactamente la misma
    # división de motores en diferentes ejecuciones del programa.
    #
    # Esto es importante para poder comparar modelos de manera justa.
    rng = np.random.default_rng(seed)

    # Extraemos los identificadores únicos de los motores del DataFrame.
    #
    # unique() devuelve cada unit_id una sola vez.
    #
    # Si un motor tiene 200 filas, aparecerá una sola vez en esta lista.
    #
    # La división se hará sobre motores completos, no sobre observaciones.
    units = train_df["unit_id"].unique()

    # Aleatorizamos el orden de los identificadores de los motores.
    #
    # rng.permutation(units) devuelve los mismos elementos de units,
    # pero en un orden aleatorio controlado por seed.
    #
    # Así evitamos que la división dependa del orden original
    # en el que aparecen los motores en el dataset.
    shuffled_units = rng.permutation(units)

    # Calculamos el índice donde se separarán los motores de train
    # y los motores de validación.
    #
    # len(shuffled_units) representa el número total de motores.
    #
    # Multiplicamos ese número por train_ratio.
    #
    # int(...) convierte el resultado a un entero, truncando los decimales.
    #
    # Por ejemplo:
    #   100 motores * 0.8 = 80
    #   split_idx = 80
    #
    # Los primeros 80 motores se utilizarán para train y los restantes
    # para validación.
    split_idx = int(len(shuffled_units) * train_ratio)

    # Creamos un set con los identificadores asignados a entrenamiento.
    #
    # shuffled_units[:split_idx] selecciona desde el principio
    # hasta el índice split_idx, sin incluirlo.
    #
    # set(...) convierte los identificadores en un conjunto.
    # Esto facilita posteriormente comprobar si un motor pertenece
    # al grupo de entrenamiento.
    train_units = set(shuffled_units[:split_idx])

    # Creamos un set con los identificadores destinados a validación.
    #
    # shuffled_units[split_idx:] selecciona todos los motores
    # desde el índice de separación hasta el final.
    #
    # Los motores de train y validación son distintos:
    # ningún motor estará simultáneamente en ambos conjuntos.
    val_units = set(shuffled_units[split_idx:])

    # Seleccionamos del DataFrame original todas las filas cuyo unit_id
    # pertenece al conjunto train_units.
    #
    # .isin(train_units) devuelve una máscara booleana:
    #   True  -> el motor pertenece al grupo de entrenamiento.
    #   False -> el motor no pertenece al grupo de entrenamiento.
    #
    # De esta manera seleccionamos todos los ciclos de los motores
    # asignados a train, no solo algunas filas.
    #
    # copy() crea una copia independiente para poder modificarla
    # posteriormente sin alterar el DataFrame original.
    tr_split = train_df[train_df["unit_id"].isin(train_units)].copy()

    # Seleccionamos todas las filas de los motores que pertenecen
    # al grupo de validación.
    #
    # De nuevo, .isin(val_units) selecciona los motores correspondientes
    # y copy() genera una copia independiente.
    va_split = train_df[train_df["unit_id"].isin(val_units)].copy()

    # Devolvemos los dos DataFrames resultantes de la división.
    return tr_split, va_split


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye ventanas deslizantes 3D:
        (N, seq_length, num_features)
    para PyTorch LSTM.

    Si un motor en test tiene menos ciclos que seq_length,
    rellena al inicio con su primer ciclo (edge padding)
    para no perder ningún activo en planta.

    Parámetros:
        df: DataFrame con las observaciones de los motores.
        feature_cols: Variables que utilizará la red neuronal.
        seq_length: Número de ciclos de cada secuencia.

    Retorno:
        X: Array de secuencias.
        y: Array de RUL asociado a cada secuencia.
        unit_list: Array con el identificador del motor de cada secuencia.
    """

    # Creamos tres listas vacías donde iremos almacenando
    # las secuencias de entrada, los valores objetivo y los IDs de motor.
    #
    # X_list contendrá las secuencias de sensores.
    # y_list contendrá el RUL correspondiente a cada secuencia.
    # unit_list contendrá el identificador del motor correspondiente.
    #
    # Utilizamos listas inicialmente porque es más sencillo añadir
    # elementos de diferentes iteraciones con append().
    X_list, y_list, unit_list = [], [], []

    # Agrupamos el DataFrame por unit_id.
    #
    # unit_id representa el motor.
    #
    # Cada iteración devolverá:
    #   unit_id -> identificador del motor.
    #   group   -> DataFrame con todas las observaciones de ese motor.
    #
    # De esta manera construimos secuencias independientes por motor,
    # sin mezclar observaciones de motores distintos.
    for unit_id, group in df.groupby("unit_id"):

        # Ordenamos las filas del motor por ciclo cronológico.
        #
        # Esto garantiza que las secuencias se construyan respetando
        # el orden temporal de las observaciones.
        group = group.sort_values("cycle")

        # Seleccionamos exclusivamente las columnas indicadas en feature_cols
        # y las convertimos en una matriz NumPy.
        #
        # .values devuelve los valores sin los nombres de las columnas
        # ni el índice de pandas.
        #
        # La matriz resultante tendrá dimensiones:
        #   (número de ciclos del motor, número de variables)
        #
        # Ejemplo:
        #   200 ciclos y 5 sensores -> matriz de tamaño (200, 5).
        data_matrix = group[feature_cols].values

        # Extraemos los valores de RUL del motor y los convertimos
        # en un array de NumPy.
        #
        # rul_vector tendrá un valor por cada ciclo del motor.
        #
        # Estos valores son las etiquetas que el modelo deberá aprender
        # a predecir a partir de las secuencias de sensores.
        rul_vector = group["RUL"].values

        # --------------------------------------------------------------
        # CASO 1: EL MOTOR TIENE MENOS CICLOS QUE seq_length
        # --------------------------------------------------------------

        # Comprobamos si el motor tiene menos observaciones que la longitud
        # requerida para construir una secuencia completa.
        #
        # Por ejemplo:
        #   seq_length = 30
        #   motor con 15 ciclos
        #
        # No podemos construir una secuencia de 30 ciclos utilizando
        # únicamente sus 15 observaciones sin aplicar padding.
        if len(group) < seq_length:

            # Calculamos cuántas observaciones faltan para alcanzar
            # la longitud deseada de la secuencia.
            #
            # Por ejemplo:
            #   seq_length = 30
            #   len(group) = 15
            #   pad_len = 15
            pad_len = seq_length - len(group)

            # Repetimos el primer ciclo del motor pad_len veces.
            #
            # data_matrix[0:1] selecciona la primera fila de la matriz,
            # conservando la dimensión de matriz (1, num_features).
            #
            # np.repeat(..., pad_len, axis=0) repite esa fila
            # verticalmente pad_len veces.
            #
            # Esto se conoce como edge padding:
            # rellenamos el inicio de la secuencia repitiendo
            # el primer valor disponible.
            #
            # Se utiliza para conseguir una secuencia de longitud fija
            # sin inventar valores nuevos ni descartar el motor.
            pad_data = np.repeat(data_matrix[0:1], pad_len, axis=0)

            # Concatenamos verticalmente las observaciones de padding
            # con las observaciones originales del motor.
            #
            # np.vstack une matrices por filas.
            #
            # Primero aparecen las filas repetidas del primer ciclo
            # y después los ciclos originales.
            #
            # Si teníamos 15 ciclos y añadimos 15:
            #   resultado -> 30 ciclos.
            #
            # La secuencia final tendrá exactamente seq_length filas.
            data_matrix = np.vstack([pad_data, data_matrix])

            # Añadimos la secuencia completa de este motor a X_list.
            #
            # En este caso, como el motor tenía menos ciclos que seq_length,
            # solo generamos una secuencia: la secuencia completa rellenada.
            X_list.append(data_matrix)

            # Añadimos el RUL correspondiente al último ciclo original
            # del motor.
            #
            # rul_vector[-1] selecciona el último elemento del array.
            #
            # Como el padding se añade al principio, el último ciclo
            # de la secuencia sigue siendo el último ciclo real del motor.
            #
            # La etiqueta representa el RUL del último ciclo de la secuencia.
            y_list.append(rul_vector[-1])

            # Guardamos el identificador del motor asociado a esta secuencia.
            #
            # Esto permite posteriormente saber de qué motor procede
            # cada predicción.
            unit_list.append(unit_id)

        # --------------------------------------------------------------
        # CASO 2: EL MOTOR TIENE seq_length CICLOS O MÁS
        # --------------------------------------------------------------

        # Si el motor tiene suficientes ciclos, podemos crear múltiples
        # secuencias mediante una ventana deslizante.
        #
        # Por ejemplo:
        #   longitud del motor = 100 ciclos
        #   seq_length = 30
        #
        # Podremos crear:
        #   Secuencia 1 -> ciclos 1-30
        #   Secuencia 2 -> ciclos 2-31
        #   Secuencia 3 -> ciclos 3-32
        #   ...
        #   Secuencia final -> ciclos 71-100
        #
        # len(group) - seq_length + 1 determina cuántas ventanas completas
        # caben si avanzamos de una en una.
        #
        # range(...) genera todos los índices iniciales posibles.
        else:
            for i in range(len(group) - seq_length + 1):

                # Extraemos una ventana de longitud seq_length
                # a partir del índice i.
                #
                # data_matrix[i : i + seq_length] selecciona:
                #   desde i
                #   hasta i + seq_length, sin incluir el extremo final.
                #
                # Por ejemplo, con seq_length=30:
                #   i=0 -> filas 0 a 29.
                #   i=1 -> filas 1 a 30.
                #   i=2 -> filas 2 a 31.
                #
                # Cada ventana se añade a X_list.
                #
                # Las ventanas se solapan porque el índice i avanza
                # de uno en uno.
                X_list.append(data_matrix[i : i + seq_length])

                # Asociamos a la secuencia el RUL del último ciclo
                # que contiene dicha secuencia.
                #
                # i + seq_length - 1 representa el índice del último
                # elemento de la ventana.
                #
                # Ejemplo:
                #   ventana ciclos 1-30 -> etiqueta RUL del ciclo 30.
                #   ventana ciclos 2-31 -> etiqueta RUL del ciclo 31.
                #
                # Así el modelo aprende a estimar el RUL correspondiente
                # al instante temporal final de cada ventana.
                y_list.append(rul_vector[i + seq_length - 1])

                # Guardamos el unit_id del motor al que pertenece
                # la secuencia que acabamos de generar.
                #
                # Aunque un motor genere muchas secuencias, cada una
                # tendrá asociado su identificador.
                unit_list.append(unit_id)

    # Convertimos X_list en un array NumPy de tipo float32.
    #
    # float32 utiliza 32 bits por valor numérico.
    #
    # Es el tipo de dato habitual para entrenar redes neuronales
    # en PyTorch porque:
    #   - Consume menos memoria que float64.
    #   - Es compatible con la mayoría de operaciones de deep learning.
    #
    # La forma esperada de X será:
    #   (número de secuencias, seq_length, número de features)
    #
    # Es decir, un tensor tridimensional.
    X = np.array(X_list, dtype=np.float32)

    # Convertimos la lista de RUL en un array NumPy float32.
    #
    # Cada elemento de y corresponde a una secuencia de X.
    #
    # La forma habitual será:
    #   (número de secuencias,)
    #
    # Cada secuencia tiene una única etiqueta objetivo: el RUL
    # del último ciclo de dicha secuencia.
    y = np.array(y_list, dtype=np.float32)

    # Convertimos la lista de identificadores de motor en un array NumPy.
    #
    # dtype=np.int32 utiliza enteros de 32 bits.
    #
    # unit_list permite identificar qué motor generó cada secuencia.
    #
    # La forma será:
    #   (número de secuencias,)
    unit_array = np.array(unit_list, dtype=np.int32)

    # Devolvemos los tres arrays:
    #
    # X -> datos de entrada para la LSTM.
    # y -> valores de RUL que el modelo deberá predecir.
    # unit_array -> identificador del motor de cada secuencia.
    return X, y, unit_array


def run_feature_pipeline(config_path: str = "configs/config.yaml") -> None:
    """
    Ejecuta el pipeline completo de Feature Engineering y preparación
    de datos para modelos tabulares y modelos secuenciales.

    Incluye:
        1. Carga de configuración.
        2. Lectura de datos procesados.
        3. Generación de variables temporales.
        4. División train/validation por motor.
        5. Escalado de variables.
        6. Guardado de datasets tabulares.
        7. Construcción de secuencias para LSTM.
        8. Guardado de arrays NumPy.
    """

    # --------------------------------------------------------------
    # 1. CARGA DE CONFIGURACIÓN
    # --------------------------------------------------------------

    # Cargamos la configuración del proyecto desde el archivo YAML.
    # cfg será un diccionario con todos los parámetros del pipeline.
    cfg = load_config(config_path)

    # Extraemos el identificador del dataset.
    #
    # Por ejemplo, FD001, FD002, FD003 o FD004.
    # Este identificador permite construir los nombres de los archivos.
    dataset_id = cfg["data"]["dataset_id"]

    # Obtenemos el directorio donde se guardaron los datos procesados
    # en el pipeline de ingesta anterior.
    #
    # Lo convertimos en Path para gestionar las rutas.
    processed_dir = Path(cfg["data"]["processed_dir"])

    # Leemos el porcentaje de motores que queremos asignar a entrenamiento.
    #
    # Por ejemplo:
    #   train_split_ratio = 0.8
    #
    # Esto significa que aproximadamente el 80% de los motores
    # se utilizarán para train y el 20% para validation.
    train_ratio = cfg["data"]["train_split_ratio"]

    # Leemos la semilla aleatoria definida en la configuración.
    #
    # Se utilizará para reproducir la misma división de motores
    # en train y validation en cada ejecución.
    seed = cfg["data"]["random_seed"]

    # Obtenemos los tamaños de ventana que utilizaremos para calcular
    # las medias móviles y desviaciones estándar móviles.
    #
    # Por ejemplo: [5, 10, 20].
    rolling_windows = cfg["features"]["rolling_windows"]

    # Obtenemos el parámetro span de la EWMA.
    #
    # Este parámetro controla el suavizado exponencial de las señales.
    ewma_span = cfg["features"]["ewma_span"]

    # Obtenemos la longitud de las secuencias que utilizaremos
    # como entrada de la red LSTM.
    #
    # Por ejemplo, seq_length = 30 significa que la LSTM recibirá
    # 30 ciclos consecutivos por secuencia.
    seq_length = cfg["features"]["sequence_length"]

    # --------------------------------------------------------------
    # 2. CARGA DE LOS DATASETS PROCESADOS
    # --------------------------------------------------------------

    # Leemos el archivo Parquet de entrenamiento generado
    # por el pipeline de ingesta.
    #
    # El archivo contiene:
    #   - unit_id
    #   - cycle
    #   - parámetros operacionales
    #   - sensores activos
    #   - RUL calculado
    #
    # dataset_id permite seleccionar el escenario correspondiente.
    train_df = pd.read_parquet(
        processed_dir / f"train_{dataset_id}_clean.parquet"
    )

    # Leemos el archivo Parquet de test.
    #
    # Tiene la misma estructura de variables que train,
    # aunque los motores de test no llegan necesariamente al fallo.
    test_df = pd.read_parquet(
        processed_dir / f"test_{dataset_id}_clean.parquet"
    )

    # --------------------------------------------------------------
    # 3. IDENTIFICACIÓN DE LOS SENSORES
    # --------------------------------------------------------------

    # Identificamos las columnas de sensores presentes en train.
    #
    # Se excluyen las columnas de identificación, ciclo,
    # parámetros operacionales y RUL.
    #
    # Utilizamos train_df para obtener la lista de variables.
    # Como train y test tienen la misma estructura, utilizaremos
    # esta lista también para test.
    sensor_cols = get_sensor_columns(train_df)

    # Mostramos un mensaje informativo indicando cuántos sensores
    # activos vamos a utilizar para crear variables temporales.
    #
    # len(sensor_cols) cuenta los sensores que permanecen tras la limpieza.
    print(
        f"Generando variables temporales "
        f"para {len(sensor_cols)} sensores activos..."
    )

    # --------------------------------------------------------------
    # 4. FEATURE ENGINEERING TEMPORAL
    # --------------------------------------------------------------

    # Generamos las variables temporales en train.
    #
    # Para cada sensor se calculan:
    #   - EWMA.
    #   - Rolling mean para cada ventana.
    #   - Rolling std para cada ventana.
    #
    # Todas estas variables se calculan independientemente por motor
    # y utilizando únicamente el ciclo actual y ciclos anteriores.
    train_feat = add_temporal_features(
        train_df,
        sensor_cols,
        rolling_windows,
        ewma_span
    )

    # Generamos las mismas variables temporales en test.
    #
    # Utilizamos exactamente los mismos sensores, ventanas móviles
    # y parámetro EWMA que en train.
    #
    # Esto garantiza que train y test tengan la misma estructura
    # de variables para posteriormente aplicar el mismo scaler
    # y alimentar los modelos.
    test_feat = add_temporal_features(
        test_df,
        sensor_cols,
        rolling_windows,
        ewma_span
    )

    # --------------------------------------------------------------
    # 5. DIVISIÓN TRAIN / VALIDATION POR MOTOR
    # --------------------------------------------------------------

    # Dividimos los motores del dataset de entrenamiento en dos grupos:
    #
    #   tr_df -> motores utilizados para entrenar los modelos.
    #   va_df -> motores utilizados para evaluar la generalización
    #            durante el desarrollo del modelo.
    #
    # train_ratio determina el porcentaje asignado a train.
    # seed garantiza que la división sea reproducible.
    #
    # La división se hace por unit_id y no por filas.
    # Así evitamos que un mismo motor aparezca simultáneamente
    # en entrenamiento y validación.
    #
    # Esto es especialmente importante en datos temporales:
    # si mezcláramos ciclos del mismo motor en ambos conjuntos,
    # el modelo podría aprender características específicas de ese motor
    # y obtener métricas artificialmente optimistas.
    tr_df, va_df = split_train_val_by_unit(
        train_feat,
        train_ratio=train_ratio,
        seed=seed
    )

    # --------------------------------------------------------------
    # 6. ESCALADO DE VARIABLES (SIN DATA LEAKAGE)
    # --------------------------------------------------------------

    # Definimos las columnas que NO deben considerarse features
    # de entrada al scaler.
    #
    # unit_id es un identificador, no una variable física.
    # cycle es el número de ciclo, que aquí se excluye del escalado.
    # RUL es la variable objetivo que queremos predecir.
    #
    # Excluir RUL es fundamental porque no debe formar parte
    # de las variables de entrada del modelo.
    non_feature_cols = {"unit_id", "cycle", "RUL"}

    # Construimos la lista de variables que utilizaremos como features.
    #
    # Recorremos las columnas de tr_df y conservamos todas aquellas
    # que no están incluidas en non_feature_cols.
    #
    # Esto incluye:
    #   - Parámetros operacionales.
    #   - Sensores originales.
    #   - Variables EWMA.
    #   - Rolling means.
    #   - Rolling standard deviations.
    #
    # La lista se obtiene de train para establecer la estructura
    # de features que se utilizará también en validation y test.
    feature_cols = [
        c for c in tr_df.columns
        if c not in non_feature_cols
    ]

    # Creamos el objeto MinMaxScaler.
    #
    # Por defecto, transforma los valores de cada feature
    # al rango [0, 1].
    #
    # El scaler aprende los valores mínimos y máximos de cada variable
    # cuando se ejecuta fit().
    #
    # Después utiliza esos valores aprendidos para transformar
    # los datasets mediante transform().
    scaler = MinMaxScaler()

    # Ajustamos el scaler a las features del conjunto de entrenamiento
    # y transformamos simultáneamente esas mismas features.
    #
    # fit_transform() realiza dos operaciones:
    #
    # 1. fit():
    #    Aprende el mínimo y el máximo de cada feature utilizando
    #    exclusivamente tr_df.
    #
    # 2. transform():
    #    Transforma los valores de tr_df al rango definido por el scaler.
    #
    # Esto es importante para evitar data leakage:
    # el scaler no debe aprender información de validation ni test.
    #
    # tr_df[feature_cols] selecciona las columnas que escalaremos.
    #
    # El resultado sustituye los valores originales por los valores
    # escalados dentro de tr_df.
    tr_df[feature_cols] = scaler.fit_transform(
        tr_df[feature_cols]
    )

    # Aplicamos el scaler previamente ajustado a validation.
    #
    # transform() utiliza los mínimos y máximos aprendidos de train.
    #
    # NO ejecutamos fit() ni fit_transform() en validation,
    # porque validation debe representar datos no utilizados
    # para aprender las transformaciones.
    #
    # Esto permite evaluar el modelo sobre datos que no han influido
    # en el proceso de ajuste del scaler.
    va_df[feature_cols] = scaler.transform(
        va_df[feature_cols]
    )

    # Aplicamos exactamente el mismo scaler a test.
    #
    # Utilizamos los parámetros aprendidos exclusivamente de train.
    #
    # De esta forma los valores de test se escalan bajo la misma
    # referencia que train y validation.
    #
    # Es posible que algunos valores de test queden por debajo de 0
    # o por encima de 1 si superan los mínimos o máximos observados
    # en train, ya que MinMaxScaler no limita automáticamente
    # los valores al intervalo [0, 1] salvo que se configure clip=True.
    test_feat[feature_cols] = scaler.transform(
        test_feat[feature_cols]
    )

    # --------------------------------------------------------------
    # 7. GUARDADO DE ARTEFACTOS Y DATASETS TABULARES
    # --------------------------------------------------------------

    # Definimos el directorio donde guardaremos los modelos
    # y los artefactos utilizados durante el entrenamiento.
    #
    # Path("models") representa una carpeta llamada models
    # en el directorio de trabajo del proyecto.
    artifacts_dir = Path("models")

    # Creamos el directorio models si todavía no existe.
    #
    # parents=True permite crear directorios intermedios.
    # exist_ok=True evita errores si la carpeta ya existe.
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Guardamos el objeto scaler en un archivo joblib.
    #
    # joblib.dump() serializa el objeto y lo guarda en disco.
    #
    # Guardar el scaler es importante porque, cuando entrenemos
    # el modelo y posteriormente hagamos predicciones con datos nuevos,
    # tendremos que aplicar exactamente la misma transformación.
    #
    # El nombre incluye dataset_id para distinguir los scalers
    # de diferentes escenarios.
    joblib.dump(
        scaler,
        artifacts_dir / f"scaler_{dataset_id}.joblib"
    )

    # Guardamos la lista de nombres de features utilizadas.
    #
    # Esta lista es necesaria para garantizar que los datos futuros
    # se presenten al modelo con las mismas columnas y en el mismo orden
    # que durante el entrenamiento.
    #
    # Si el modelo espera una estructura concreta de variables,
    # debemos conservarla para evitar errores o predicciones incorrectas.
    joblib.dump(
        feature_cols,
        artifacts_dir / f"feature_cols_{dataset_id}.joblib"
    )

    # Guardamos el dataset tabular de entrenamiento en formato Parquet.
    #
    # Este dataset contiene las variables originales y las nuevas
    # variables temporales, ya escaladas.
    #
    # Será utilizado posteriormente por modelos tabulares
    # como LightGBM.
    #
    # index=False evita guardar el índice de pandas como columna adicional.
    tr_df.to_parquet(
        processed_dir / f"train_tab_{dataset_id}.parquet",
        index=False
    )

    # Guardamos el dataset tabular de validación.
    #
    # Contiene los motores que no se utilizaron para ajustar el scaler
    # ni para entrenar el modelo.
    #
    # Se utilizará para evaluar el rendimiento del modelo durante
    # el desarrollo y comparar diferentes configuraciones.
    va_df.to_parquet(
        processed_dir / f"val_tab_{dataset_id}.parquet",
        index=False
    )

    # Guardamos el dataset tabular de test.
    #
    # Este dataset se utilizará posteriormente para realizar
    # las predicciones finales.
    #
    # Mantiene las mismas features y el mismo escalado
    # que train y validation.
    test_feat.to_parquet(
        processed_dir / f"test_tab_{dataset_id}.parquet",
        index=False
    )

    # --------------------------------------------------------------
    # 8. CONSTRUCCIÓN DE SECUENCIAS PARA PYTORCH LSTM
    # --------------------------------------------------------------

    # Construimos las secuencias del conjunto de entrenamiento.
    #
    # Utilizamos tr_df, que contiene:
    #   - Los motores asignados a entrenamiento.
    #   - Los sensores base escalados.
    #   - El RUL.
    #
    # sensor_cols contiene exclusivamente los sensores originales,
    # no las variables temporales adicionales.
    #
    # seq_length determina cuántos ciclos consecutivos tendrá
    # cada secuencia.
    #
    # create_sequences devuelve:
    #   X_train_seq -> secuencias de sensores.
    #   y_train_seq -> RUL asociado a cada secuencia.
    #   _           -> IDs de motores, que aquí no necesitamos guardar.
    X_train_seq, y_train_seq, _ = create_sequences(
        tr_df,
        sensor_cols,
        seq_length
    )

    # Construimos las secuencias del conjunto de validación.
    #
    # Se utiliza exactamente el mismo procedimiento que en train,
    # con los mismos sensores y la misma longitud de secuencia.
    #
    # Esto garantiza que la estructura de los datos de validación
    # sea compatible con la red LSTM.
    X_val_seq, y_val_seq, _ = create_sequences(
        va_df,
        sensor_cols,
        seq_length
    )

    # Construimos las secuencias de test.
    #
    # En este caso guardamos también los identificadores de motor
    # en u_test_seq, porque posteriormente pueden ser necesarios
    # para asociar las predicciones a cada motor.
    #
    # y_test_seq contiene los RUL de test disponibles en este dataset.
    # Se utilizarán para evaluar las predicciones, no como entrada
    # de la red neuronal.
    X_test_seq, y_test_seq, u_test_seq = create_sequences(
        test_feat,
        sensor_cols,
        seq_length
    )

    # --------------------------------------------------------------
    # 9. GUARDADO DE SECUENCIAS EN FORMATO NPZ
    # --------------------------------------------------------------

    # Guardamos todos los arrays generados en un único archivo comprimido
    # de NumPy con extensión .npz.
    #
    # np.savez_compressed() permite guardar múltiples arrays bajo
    # diferentes nombres dentro del mismo archivo.
    #
    # La compresión reduce el espacio ocupado en disco.
    #
    # Este archivo será posteriormente utilizado para cargar los datos
    # de forma sencilla durante el entrenamiento de la LSTM.
    np.savez_compressed(
        processed_dir / f"sequences_{dataset_id}.npz",

        # Secuencias de entrada del conjunto de entrenamiento.
        X_train=X_train_seq,

        # Valores objetivo de RUL correspondientes a train.
        y_train=y_train_seq,

        # Secuencias de entrada de validation.
        X_val=X_val_seq,

        # Valores objetivo de RUL correspondientes a validation.
        y_val=y_val_seq,

        # Secuencias de entrada del conjunto test.
        X_test=X_test_seq,

        # Valores objetivo de RUL de test, para evaluación.
        y_test=y_test_seq,

        # Identificador del motor asociado a cada secuencia de test.
        # Permite recuperar qué motor produjo cada predicción.
        unit_test=u_test_seq,
    )

    # --------------------------------------------------------------
    # 10. INFORMACIÓN FINAL DEL PIPELINE
    # --------------------------------------------------------------

    # Imprimimos un mensaje indicando que el Feature Pipeline
    # ha terminado correctamente.
    #
    # \n añade una línea en blanco antes del encabezado.
    print("\n[FEATURE PIPELINE COMPLETADO]")

    # Mostramos el número total de variables tabulares utilizadas.
    #
    # len(feature_cols) cuenta las features que hemos creado y escalado.
    #
    # Estas variables se utilizarán posteriormente en modelos
    # tabulares como LightGBM.
    print(
        f" -> Variables tabulares (LightGBM): "
        f"{len(feature_cols)} features"
    )

    # Mostramos cuántos motores pertenecen a train y validation.
    #
    # nunique() cuenta los identificadores unit_id únicos.
    #
    # Es importante mostrar el número de motores y no solo las filas,
    # porque la división se realizó por motores completos.
    print(
        f" -> Split por motor: "
        f"Train={tr_df['unit_id'].nunique()} motores | "
        f"Val={va_df['unit_id'].nunique()} motores"
    )

    # Mostramos las dimensiones de los arrays de secuencias
    # utilizados para la LSTM.
    #
    # .shape devuelve las dimensiones de cada array.
    #
    # Por ejemplo:
    #   X_train=(8000, 30, 14)
    #
    # Significa:
    #   8000 secuencias.
    #   30 ciclos por secuencia.
    #   14 variables por ciclo.
    #
    # Mostrar las dimensiones permite comprobar rápidamente
    # que los datos tienen la estructura esperada.
    print(
        f" -> Tensores PyTorch LSTM (seq={seq_length}): "
        f"X_train={X_train_seq.shape}, "
        f"X_val={X_val_seq.shape}, "
        f"X_test={X_test_seq.shape}"
    )


# ----------------------------------------------------------------------
# PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------

# Comprobamos si este archivo se está ejecutando directamente.
#
# Si es así, __name__ tendrá el valor "__main__".
#
# Si el archivo se importa desde otro script, esta condición será falsa
# y el pipeline no se ejecutará automáticamente.
if __name__ == "__main__":

    # Ejecutamos todo el Feature Pipeline utilizando la configuración
    # por defecto ubicada en configs/config.yaml.
    #
    # Se ejecutan todas las etapas:
    #   - Carga de configuración.
    #   - Lectura de datos procesados.
    #   - Creación de variables temporales.
    #   - División train/validation.
    #   - Escalado.
    #   - Guardado de datasets tabulares.
    #   - Creación de secuencias.
    #   - Guardado de arrays para LSTM.
    run_feature_pipeline()