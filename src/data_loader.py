
# Importamos Path para trabajar con rutas de archivos y directorios de una forma
# más cómoda y segura que concatenando cadenas de texto manualmente.
from pathlib import Path

# Importamos yaml para poder leer archivos de configuración en formato YAML.
# En este proyecto se utiliza para leer config.yaml.
import yaml

# Importamos pandas, que es la librería principal que utilizaremos para manejar,
# limpiar, transformar y analizar los datos tabulares.
import pandas as pd


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Lee el archivo YAML de configuración industrial.

    Parámetros:
        config_path: Ruta donde se encuentra el archivo de configuración.
                     Si no se especifica otra, utiliza configs/config.yaml.

    Retorno:
        Un diccionario de Python con la configuración cargada.
    """

    # Abrimos el archivo de configuración en modo lectura ("r").
    # encoding="utf-8" permite leer correctamente caracteres especiales.
    with open(config_path, "r", encoding="utf-8") as f:

        # yaml.safe_load convierte el contenido YAML en estructuras de Python.
        # Por ejemplo, los parámetros del YAML se convierten en diccionarios,
        # listas, números, cadenas de texto, booleanos, etc.
        return yaml.safe_load(f)


def build_column_names(sensor_map: dict) -> list[str]:
    """
    Construye la lista ordenada de las 26 columnas del dataset CMAPSS.

    Parámetros:
        sensor_map: Diccionario que relaciona el nombre original del sensor
                    (s1, s2, ..., s21) con su nombre físico.

    Retorno:
        Una lista con los nombres de las 26 columnas del dataset.
    """

    # Definimos las primeras cinco columnas del dataset.
    # Estas contienen la identificación del motor, el ciclo y los tres
    # parámetros operacionales.
    base_cols = [
        "unit_id",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3"
    ]

    # Recorremos los sensores desde s1 hasta s21.
    # El range(1, 22) genera los números del 1 al 21 (el 22 no se incluye).
    # Utilizamos sensor_map para sustituir los nombres genéricos s1, s2...
    # por los nombres físicos definidos en nuestro archivo de configuración.
    #
    # El orden es importante porque las columnas del archivo original
    # no tienen encabezados y debemos asignar correctamente cada nombre
    # a la posición correspondiente.
    sensor_cols = [sensor_map[f"s{i}"] for i in range(1, 22)]

    # Concatenamos las cinco columnas iniciales con las 21 columnas
    # correspondientes a los sensores.
    # El resultado contiene 26 nombres de columnas ordenados.
    return base_cols + sensor_cols


def compute_train_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """
    Calcula el RUL (Remaining Useful Life) para el conjunto de entrenamiento.

    Cada motor en train corre hasta el fallo (max_cycle).

    Parámetros:
        df: DataFrame del conjunto de entrenamiento.
        rul_cap: Valor máximo permitido para el RUL.

    Retorno:
        El DataFrame original con una nueva columna llamada RUL.
    """

    # Agrupamos los datos por motor (unit_id).
    # Para cada motor calculamos el ciclo máximo registrado, que representa
    # el ciclo en el que ese motor llega al final de su vida útil/fallo.
    #
    # transform("max") devuelve el máximo de cada grupo, pero manteniendo
    # el mismo número de filas y el mismo índice que el DataFrame original.
    # Así podemos asignar ese valor a cada fila de su motor correspondiente.
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")

    # Calculamos el RUL sin limitar:
    # RUL = ciclo máximo del motor - ciclo actual.
    #
    # Por ejemplo, si un motor llega al fallo en el ciclo 200:
    #   Ciclo actual 100 -> RUL = 100
    #   Ciclo actual 150 -> RUL = 50
    #   Ciclo actual 199 -> RUL = 1
    #   Ciclo actual 200 -> RUL = 0
    raw_rul = max_cycles - df["cycle"]

    # Aplicamos el límite superior (rul_cap) al RUL.
    # Los valores superiores al límite se sustituyen por rul_cap.
    #
    # Por ejemplo, si rul_cap = 125:
    #   RUL = 200 -> 125
    #   RUL = 150 -> 125
    #   RUL = 100 -> 100
    #
    # Esto genera un RUL piecewise linear (lineal por tramos):
    # primero permanece constante en el límite y después decrece linealmente.
    df["RUL"] = raw_rul.clip(upper=rul_cap)

    # Devolvemos el DataFrame, que ahora contiene la nueva columna RUL.
    return df


def compute_test_rul(
    df: pd.DataFrame,
    rul_df: pd.DataFrame,
    rul_cap: int
) -> pd.DataFrame:
    """
    Calcula el RUL para el conjunto de test.

    En test, la grabación se corta antes del fallo y los ciclos restantes
    al final de la serie vienen dados en el archivo RUL_FD00X.txt.

    Parámetros:
        df: DataFrame de los datos de test.
        rul_df: DataFrame que contiene el RUL restante de cada motor
                después del último ciclo registrado en test.
        rul_cap: Valor máximo permitido para el RUL.

    Retorno:
        El DataFrame de test con la nueva columna RUL.
    """

    # Creamos una copia del DataFrame que contiene los RUL finales.
    # copy() evita modificar directamente el DataFrame original rul_df.
    rul_df = rul_df.copy()

    # Creamos una columna unit_id con números del 1 al número de motores.
    #
    # range(1, len(rul_df) + 1) genera:
    #   1, 2, 3, ..., N
    #
    # Esto permite relacionar el RUL final de cada motor con su identificador.
    # Se asume que el orden de los motores en el archivo RUL corresponde
    # al orden de los motores del archivo de test.
    rul_df["unit_id"] = range(1, len(rul_df) + 1)

    # Para cada motor del conjunto test calculamos su último ciclo registrado.
    #
    # groupby("unit_id") agrupa los datos por motor.
    # ["cycle"] selecciona la columna de ciclos.
    # transform("max") devuelve el último ciclo registrado de cada motor
    # en todas las filas de ese motor, manteniendo el índice original.
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")

    # Unimos el DataFrame de test con rul_df mediante la columna unit_id.
    #
    # how="left" significa que conservamos todas las filas del DataFrame df
    # (test) y añadimos la información de rul_df cuando existe coincidencia.
    #
    # De esta forma, cada fila del motor recibe su RUL restante después
    # del último ciclo de test, almacenado en rul_at_end.
    df = df.merge(rul_df, on="unit_id", how="left")

    # Calculamos el RUL real de cada ciclo de test.
    #
    # Primera parte:
    # max_cycles - df["cycle"]
    # Calcula los ciclos que faltan desde el ciclo actual hasta el último
    # ciclo registrado en el archivo de test.
    #
    # Segunda parte:
    # df["rul_at_end"]
    # Añade los ciclos restantes que faltaban desde el final del test
    # hasta el fallo real del motor.
    #
    # Por tanto:
    # RUL = ciclos hasta el final del test + RUL restante tras el test.
    #
    # Ejemplo:
    # Último ciclo de test = 150
    # RUL restante después del test = 30
    # Ciclo actual = 100
    # RUL = (150 - 100) + 30 = 80
    raw_rul = (max_cycles - df["cycle"]) + df["rul_at_end"]

    # Aplicamos el límite superior al RUL de test.
    # Si el RUL calculado supera rul_cap, lo limitamos a rul_cap.
    # Esto mantiene la misma definición de RUL que en train.
    df["RUL"] = raw_rul.clip(upper=rul_cap)

    # Eliminamos la columna rul_at_end porque era una columna auxiliar
    # utilizada para calcular el RUL y ya no necesitamos conservarla.
    #
    # drop(columns=[...]) elimina la columna especificada.
    # Devolvemos el DataFrame sin esa columna auxiliar.
    return df.drop(columns=["rul_at_end"])


def audit_and_drop_dead_sensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor_cols: list[str],
    min_std: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Auditoría industrial de instrumentación.

    Calcula la desviación estándar SOLO en train (para evitar data leakage)
    y descarta sensores congelados o sin varianza.

    Parámetros:
        train_df: DataFrame de entrenamiento.
        test_df: DataFrame de test.
        sensor_cols: Lista de nombres de las columnas de sensores.
        min_std: Umbral mínimo de desviación estándar para considerar
                 que un sensor tiene suficiente variabilidad.

    Retorno:
        train_clean: Train sin los sensores descartados.
        test_clean: Test sin los sensores descartados.
        active_sensors: Lista de sensores que se conservan.
    """

    # Calculamos la desviación estándar de cada sensor utilizando únicamente
    # los datos de entrenamiento.
    #
    # train_df[sensor_cols] selecciona exclusivamente las columnas de sensores.
    # .std() calcula la desviación estándar de cada columna.
    #
    # La desviación estándar mide cuánto varían los valores respecto
    # a su media:
    #   std alta -> sensor presenta variabilidad.
    #   std baja -> sensor apenas cambia.
    #   std = 0  -> sensor tiene siempre el mismo valor.
    #
    # Utilizamos exclusivamente train para evitar data leakage:
    # no utilizamos información estadística del conjunto test para decidir
    # qué variables utilizar en el modelo.
    stds = train_df[sensor_cols].std()

    # Identificamos los sensores cuya desviación estándar es inferior
    # al umbral min_std definido en la configuración.
    #
    # stds < min_std devuelve una máscara booleana:
    #   True  -> sensor por debajo del umbral.
    #   False -> sensor por encima o igual al umbral.
    #
    # .index.tolist() extrae los nombres de las columnas que cumplen
    # la condición y los convierte en una lista de Python.
    dead_sensors = stds[stds < min_std].index.tolist()

    # Creamos una lista con los sensores que vamos a conservar.
    #
    # Recorremos todos los sensores originales y comprobamos que su nombre
    # no esté incluido en la lista de sensores descartados.
    active_sensors = [col for col in sensor_cols if col not in dead_sensors]

    # Imprimimos en consola un encabezado informativo de la auditoría
    # y el valor del umbral de desviación estándar utilizado.
    #
    # \n añade un salto de línea antes del mensaje para mejorar la lectura.
    # f-string permite insertar el valor de min_std directamente en el texto.
    print(f"\n[AUDITORÍA DE SENSORES] Umbral mínimo std: {min_std}")

    # Imprimimos el número de sensores descartados y sus nombres.
    # len(dead_sensors) calcula cuántos sensores se han eliminado.
    # {dead_sensors} muestra la lista completa de sus nombres.
    print(f" -> Sensores descartados ({len(dead_sensors)}): {dead_sensors}")

    # Imprimimos el número de sensores activos y los nombres de los sensores
    # que han superado la auditoría y se mantienen en el dataset.
    print(f" -> Sensores activos ({len(active_sensors)}): {active_sensors}\n")

    # Eliminamos las columnas correspondientes a los sensores descartados
    # del DataFrame de entrenamiento.
    #
    # drop(columns=dead_sensors) elimina las columnas indicadas.
    # Si dead_sensors está vacío, no se elimina ninguna columna.
    train_clean = train_df.drop(columns=dead_sensors)

    # Eliminamos exactamente las mismas columnas del conjunto test.
    #
    # Es fundamental aplicar en test la misma selección de variables
    # decidida a partir de train, para que ambos datasets tengan
    # las mismas características y evitar data leakage.
    test_clean = test_df.drop(columns=dead_sensors)

    # Devolvemos tres elementos:
    #   1. El DataFrame de train limpio.
    #   2. El DataFrame de test limpio.
    #   3. La lista de sensores que se han conservado.
    #
    # tuple[...] indica el tipo esperado de los tres elementos devueltos.
    return train_clean, test_clean, active_sensors


def run_ingestion(config_path: str = "configs/config.yaml") -> None:
    """
    Ejecuta el pipeline completo de ingesta, cálculo de RUL y limpieza.

    Esta función coordina todas las funciones anteriores:
    carga la configuración, lee los archivos, calcula el RUL,
    audita los sensores y guarda los datos procesados.
    """

    # Cargamos los parámetros del proyecto desde el archivo YAML.
    # cfg será un diccionario que contiene toda la configuración.
    cfg = load_config(config_path)

    # Accedemos al identificador del dataset definido en la configuración.
    #
    # Por ejemplo, si dataset_id = "FD001", utilizaremos los archivos:
    # train_FD001.txt, test_FD001.txt y RUL_FD001.txt.
    dataset_id = cfg["data"]["dataset_id"]

    # Convertimos la ruta del directorio donde están los datos brutos
    # en un objeto Path.
    #
    # raw_dir será el directorio desde el que leeremos los archivos originales.
    raw_dir = Path(cfg["data"]["raw_dir"])

    # Convertimos la ruta del directorio donde guardaremos los datos
    # procesados en un objeto Path.
    processed_dir = Path(cfg["data"]["processed_dir"])

    # Creamos el directorio de salida si todavía no existe.
    #
    # parents=True permite crear también directorios superiores
    # si fuera necesario.
    #
    # exist_ok=True evita que se produzca un error si el directorio
    # ya existe previamente.
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Leemos el valor máximo de RUL permitido desde la configuración.
    # Este valor se utilizará para truncar los RUL de train y test.
    rul_cap = cfg["physics_and_sensors"]["rul_cap"]

    # Leemos el umbral mínimo de desviación estándar que debe tener
    # un sensor para considerarlo suficientemente variable.
    min_std = cfg["physics_and_sensors"]["min_sensor_std"]

    # Extraemos del archivo de configuración el diccionario que relaciona
    # los nombres genéricos de los sensores con sus nombres físicos.
    #
    # Por ejemplo:
    #   s1 -> nombre físico del sensor 1
    #   s2 -> nombre físico del sensor 2
    #   ...
    #   s21 -> nombre físico del sensor 21
    sensor_map = cfg["physics_and_sensors"]["sensor_names"]

    # Construimos la lista completa de nombres de columnas del dataset.
    # Esta lista contiene las 5 columnas iniciales y los 21 sensores.
    # La utilizaremos para asignar nombres al leer los archivos originales.
    col_names = build_column_names(sensor_map)

    # Construimos una lista que contiene únicamente los nombres
    # de las 21 columnas de sensores.
    #
    # Recorremos s1 hasta s21 y obtenemos sus nombres físicos
    # utilizando sensor_map.
    #
    # Esta lista será necesaria para realizar la auditoría de sensores.
    sensor_cols = [sensor_map[f"s{i}"] for i in range(1, 22)]

    # ------------------------------------------------------------------
    # 1. LECTURA DE LOS ARCHIVOS BRUTOS
    # ------------------------------------------------------------------

    # Construimos la ruta completa al archivo de entrenamiento.
    #
    # raw_dir / nombre_archivo utiliza el operador / de Path para unir
    # directorios y nombres de archivo de forma compatible con el sistema.
    #
    # El nombre del archivo depende del dataset_id.
    train_path = raw_dir / f"train_{dataset_id}.txt"

    # Construimos la ruta completa al archivo de test.
    test_path = raw_dir / f"test_{dataset_id}.txt"

    # Construimos la ruta completa al archivo que contiene los RUL finales
    # de los motores de test.
    rul_path = raw_dir / f"RUL_{dataset_id}.txt"

    # Mostramos por consola qué escenario estamos cargando
    # y desde qué directorio se leen los datos.
    # Esto facilita comprobar que estamos usando el dataset correcto.
    print(f"Cargando escenario {dataset_id} desde {raw_dir}...")

    # Leemos el archivo de entrenamiento utilizando pandas.
    #
    # sep=r"\s+" indica que los valores están separados por uno o varios
    # espacios en blanco (espacios o tabulaciones).
    #
    # La r delante de la cadena indica que es una raw string,
    # evitando problemas con caracteres especiales como la barra invertida.
    #
    # header=None indica que el archivo original no tiene encabezados.
    #
    # names=col_names asigna los nombres de columnas que hemos construido
    # previamente, respetando el orden de los datos originales.
    train_df = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=col_names
    )

    # Leemos el archivo de test exactamente de la misma forma que train.
    #
    # Utilizamos los mismos nombres de columnas porque ambos archivos
    # tienen la misma estructura de variables.
    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=col_names
    )

    # Leemos el archivo de RUL final de test.
    #
    # Este archivo contiene un valor por motor, indicando cuántos ciclos
    # faltan desde el último ciclo registrado en test hasta el fallo.
    #
    # header=None indica que tampoco tiene encabezado.
    # names=["rul_at_end"] asigna un nombre a su única columna.
    rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["rul_at_end"]
    )

    # ------------------------------------------------------------------
    # 2. CÁLCULO FÍSICO DEL RUL
    # ------------------------------------------------------------------

    # Calculamos el RUL de todas las filas del conjunto de entrenamiento.
    #
    # En train conocemos el ciclo de fallo de cada motor porque la serie
    # llega hasta el fallo, por lo que podemos calcular directamente
    # max_cycle - ciclo_actual.
    #
    # rul_cap establece el límite máximo permitido del RUL.
    train_df = compute_train_rul(train_df, rul_cap=rul_cap)

    # Calculamos el RUL de todas las filas del conjunto test.
    #
    # Como test no llega hasta el fallo, utilizamos también el archivo
    # rul_df, que contiene los ciclos restantes después del último ciclo
    # observado para cada motor.
    #
    # La función compute_test_rul combina ambas cantidades y aplica
    # el mismo límite superior rul_cap.
    test_df = compute_test_rul(test_df, rul_df, rul_cap=rul_cap)

    # ------------------------------------------------------------------
    # 3. AUDITORÍA DE SENSORES MUERTOS (SIN DATA LEAKAGE)
    # ------------------------------------------------------------------

    # Ejecutamos la auditoría de sensores.
    #
    # train_df y test_df son los datasets que ya contienen el RUL.
    #
    # sensor_cols indica las columnas que debemos analizar.
    #
    # min_std define el umbral mínimo de variabilidad.
    #
    # La función devuelve:
    #   train_clean -> train sin los sensores descartados.
    #   test_clean  -> test sin los mismos sensores.
    #   _           -> lista de sensores activos, que en este punto
    #                  no necesitamos guardar en una variable.
    #
    # El guion bajo "_" se utiliza convencionalmente en Python cuando
    # recibimos un valor que no necesitamos utilizar posteriormente.
    train_clean, test_clean, _ = audit_and_drop_dead_sensors(
        train_df,
        test_df,
        sensor_cols,
        min_std=min_std
    )

    # ------------------------------------------------------------------
    # 4. GUARDADO EN FORMATO PARQUET
    # ------------------------------------------------------------------

    # Construimos la ruta del archivo de entrenamiento procesado.
    #
    # El sufijo "_clean" indica que los datos han sido procesados
    # y que se han eliminado los sensores considerados muertos.
    #
    # La extensión .parquet indica el formato de almacenamiento Parquet,
    # un formato columnar eficiente para guardar y leer datasets.
    train_out = processed_dir / f"train_{dataset_id}_clean.parquet"

    # Construimos la ruta del archivo de test procesado.
    # Utilizamos la misma nomenclatura que para train.
    test_out = processed_dir / f"test_{dataset_id}_clean.parquet"

    # Guardamos el DataFrame de entrenamiento en formato Parquet.
    #
    # index=False indica que no guardamos el índice de pandas como
    # una columna adicional del archivo.
    #
    # Parquet permite conservar los tipos de datos y suele ocupar
    # menos espacio y ser más eficiente que CSV para este tipo de datos.
    train_clean.to_parquet(train_out, index=False)

    # Guardamos el DataFrame de test en formato Parquet.
    # Utilizamos exactamente el mismo formato y configuración que train.
    test_clean.to_parquet(test_out, index=False)

    # Mostramos un mensaje indicando que el procesamiento y guardado
    # de los datasets ha finalizado correctamente.
    print(f"Datos procesados guardados correctamente:")

    # Mostramos la ruta del archivo train, el número de filas y
    # el número de motores únicos que contiene.
    #
    # len(train_clean) devuelve el número total de filas.
    #
    # {:,} formatea el número con separador de miles.
    # Por ejemplo: 20,531 en lugar de 20531.
    #
    # train_clean["unit_id"].nunique() cuenta cuántos identificadores
    # de motor distintos existen en el DataFrame.
    print(
        f" -> {train_out} | "
        f"Filas: {len(train_clean):,} | "
        f"Motores: {train_clean['unit_id'].nunique()}"
    )

    # Mostramos la misma información para el conjunto test.
    # Así podemos comprobar rápidamente el tamaño de ambos datasets
    # y el número de motores que contienen.
    print(
        f" -> {test_out} | "
        f"Filas: {len(test_clean):,} | "
        f"Motores: {test_clean['unit_id'].nunique()}"
    )


# ----------------------------------------------------------------------
# PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------

# Esta condición comprueba si este archivo Python se está ejecutando
# directamente o si se está importando desde otro archivo.
#
# __name__ es una variable especial de Python:
#   Si ejecutamos este archivo directamente, __name__ == "__main__".
#   Si lo importamos desde otro módulo, __name__ tendrá el nombre del módulo.
#
# Esto permite ejecutar automáticamente el pipeline cuando lanzamos
# este archivo desde la terminal, pero evita que se ejecute automáticamente
# si importamos sus funciones desde otro archivo Python.
if __name__ == "__main__":

    # Ejecutamos el pipeline completo de ingesta.
    #
    # Como no proporcionamos ningún argumento, utilizará la ruta por defecto:
    # configs/config.yaml
    #
    # A partir de aquí se ejecutan todas las etapas:
    # configuración -> lectura -> RUL -> auditoría -> guardado.
    run_ingestion()