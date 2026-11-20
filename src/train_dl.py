# Importamos Path para gestionar rutas de archivos y directorios
# de una forma sencilla y compatible con diferentes sistemas operativos.
from pathlib import Path

# Importamos time para medir cuánto tiempo tarda el entrenamiento.
# Utilizaremos time.perf_counter(), que es un reloj de alta precisión
# adecuado para medir duraciones de procesos.
import time

# Importamos NumPy para trabajar con arrays, operaciones matemáticas,
# generación de números aleatorios y cálculo de métricas.
import numpy as np

# Importamos PyTorch, la librería que utilizaremos para construir,
# entrenar y ejecutar nuestra red neuronal.
import torch

# Importamos nn (neural network), que contiene las herramientas
# para construir redes neuronales, capas, funciones de activación
# y funciones de pérdida.
import torch.nn as nn

# Importamos DataLoader y TensorDataset.
#
# TensorDataset permite agrupar tensores de entrada y sus etiquetas.
# DataLoader permite cargar esos datos en batches (lotes),
# mezclarlos y recorrerlos durante el entrenamiento.
from torch.utils.data import DataLoader, TensorDataset

# Importamos yaml para leer la configuración del proyecto
# almacenada en config.yaml.
import yaml

# Importamos mean_squared_error, una métrica de Scikit-learn
# que calcula el error cuadrático medio entre predicciones
# y valores reales.
from sklearn.metrics import mean_squared_error

# Importamos una función definida en nuestro propio proyecto.
#
# asymmetric_plant_cost calcula el coste de las predicciones
# teniendo en cuenta que los errores tardíos y tempranos
# tienen consecuencias diferentes en mantenimiento industrial.
#
# Esta función se encuentra en el módulo src.train_baseline.
from src.train_baseline import asymmetric_plant_cost


# ----------------------------------------------------------------------
# 1. CARGA DE CONFIGURACIÓN
# ----------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Lee el archivo YAML de configuración.

    Parámetros:
        config_path: Ruta al archivo de configuración.
                     Por defecto utiliza configs/config.yaml.

    Retorno:
        Diccionario de Python con los parámetros del proyecto.
    """

    # Abrimos el archivo YAML en modo lectura.
    # encoding="utf-8" permite interpretar correctamente caracteres especiales.
    with open(config_path, "r", encoding="utf-8") as f:

        # Convertimos el contenido del YAML a estructuras de Python,
        # normalmente un diccionario con los diferentes parámetros.
        #
        # safe_load evita utilizar mecanismos de carga de YAML
        # que puedan ejecutar código arbitrario.
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# 2. CONTROL DE ALEATORIEDAD (REPRODUCIBILIDAD)
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Fija las semillas aleatorias de las librerías utilizadas.

    Parámetros:
        seed: Número entero que se utilizará como semilla aleatoria.

    Objetivo:
        Facilitar que los experimentos sean reproducibles.
    """

    # Fijamos la semilla del generador aleatorio de NumPy.
    #
    # NumPy utiliza números pseudoaleatorios en distintas operaciones.
    # Fijar la semilla permite reproducir la misma secuencia aleatoria
    # cuando ejecutamos nuevamente el programa.
    np.random.seed(seed)

    # Fijamos la semilla del generador aleatorio de PyTorch en CPU.
    #
    # PyTorch utiliza aleatoriedad, entre otras cosas, para inicializar
    # los pesos de las redes neuronales.
    #
    # Esto permite que la inicialización del modelo sea reproducible
    # bajo las mismas condiciones.
    torch.manual_seed(seed)

    # Comprobamos si el ordenador dispone de una GPU compatible
    # con CUDA y accesible desde PyTorch.
    #
    # CUDA permite ejecutar operaciones de PyTorch en GPUs NVIDIA.
    if torch.cuda.is_available():

        # Fijamos también las semillas de todos los dispositivos GPU.
        #
        # Esto es importante si el entrenamiento utiliza una GPU,
        # ya que también existen operaciones aleatorias en CUDA.
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# 3. FUNCIÓN DE PÉRDIDA ASIMÉTRICA
# ----------------------------------------------------------------------

class AsymmetricMSELoss(nn.Module):
    """
    Función de pérdida industrial diferenciable para PyTorch.

    Penaliza más fuerte cuando la red predice tarde
    (y_pred > y_true), para proteger la integridad física
    del activo en planta.

    Hereda de nn.Module porque en PyTorch las funciones de pérdida
    personalizadas se construyen normalmente como módulos.
    """

    # ------------------------------------------------------------------
    # CONSTRUCTOR DE LA FUNCIÓN DE PÉRDIDA
    # ------------------------------------------------------------------

    def __init__(
        self,
        late_penalty: float = 2.0,
        early_penalty: float = 1.0
    ):

        # Ejecutamos el constructor de la clase padre nn.Module.
        #
        # Esto inicializa correctamente el objeto como un módulo PyTorch
        # y permite integrarlo con el resto de la arquitectura.
        super().__init__()

        # Guardamos el peso de penalización para las predicciones tardías.
        #
        # Una predicción tardía significa que el modelo estima que quedan
        # más ciclos de vida de los que realmente quedan.
        #
        # Es decir, el modelo está siendo demasiado optimista
        # sobre la vida útil restante del motor.
        #
        # Por ejemplo:
        #   RUL real = 20
        #   RUL predicho = 40
        #
        # El modelo cree que el motor puede funcionar 40 ciclos,
        # aunque realmente solo quedan 20.
        self.late_penalty = late_penalty

        # Guardamos el peso de penalización para las predicciones tempranas.
        #
        # Una predicción temprana significa que el modelo estima
        # menos vida útil de la que realmente queda.
        #
        # Por ejemplo:
        #   RUL real = 40
        #   RUL predicho = 20
        #
        # El modelo está siendo conservador y recomienda actuar
        # antes de lo estrictamente necesario.
        self.early_penalty = early_penalty

    # ------------------------------------------------------------------
    # CÁLCULO DE LA PÉRDIDA
    # ------------------------------------------------------------------

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula la pérdida entre las predicciones y los valores reales.

        Parámetros:
            y_pred: RUL predicho por la red neuronal.
            y_true: RUL real.

        Retorno:
            Un único valor de pérdida que representa el error penalizado.
        """

        # Calculamos la diferencia entre el valor predicho y el real.
        #
        # diff = y_pred - y_true
        #
        # Si diff > 0:
        #   El modelo predice más RUL del que realmente queda.
        #   Es una predicción tardía / demasiado optimista.
        #
        # Si diff < 0:
        #   El modelo predice menos RUL del que realmente queda.
        #   Es una predicción temprana / conservadora.
        #
        # Si diff = 0:
        #   La predicción coincide exactamente con el valor real.
        diff = y_pred - y_true

        # Asignamos un peso diferente dependiendo del tipo de error.
        #
        # torch.where(condición, valor_si_true, valor_si_false)
        # funciona de manera vectorizada sobre los tensores.
        #
        # Si diff > 0:
        #   Utilizamos late_penalty.
        #
        # Si diff <= 0:
        #   Utilizamos early_penalty.
        #
        # Esto permite penalizar más los errores que pueden provocar
        # que el sistema de mantenimiento actúe demasiado tarde.
        weights = torch.where(
            diff > 0,
            self.late_penalty,
            self.early_penalty
        )

        # Calculamos el error cuadrático de cada predicción.
        #
        # diff ** 2 eleva al cuadrado el error.
        #
        # Esto hace que los errores grandes tengan un impacto mayor
        # que los errores pequeños.
        #
        # Multiplicamos el error cuadrático por el peso correspondiente
        # a cada predicción.
        #
        # Finalmente, torch.mean() calcula la media de todos los errores
        # penalizados del batch.
        #
        # El resultado es un único número que representa la pérdida
        # de ese lote de datos.
        return torch.mean(weights * (diff ** 2))


# ----------------------------------------------------------------------
# 4. ARQUITECTURA DE LA RED NEURONAL LSTM
# ----------------------------------------------------------------------

class PredictiveMaintenanceLSTM(nn.Module):
    """
    Red neuronal LSTM de 2 capas con Dropout
    para estimación de RUL a partir de ventanas temporales de sensores.

    La arquitectura está formada por:

        Entrada de secuencia
              |
              v
            LSTM
              |
              v
      Último paso temporal
              |
              v
       Capa Fully Connected
              |
              v
             ReLU
              |
              v
       Capa Fully Connected
              |
              v
        Predicción de RUL
    """

    # ------------------------------------------------------------------
    # CONSTRUCTOR DE LA ARQUITECTURA
    # ------------------------------------------------------------------

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):

        # Inicializamos correctamente la clase padre nn.Module.
        #
        # Esto permite que PyTorch registre las capas, sus parámetros
        # y los pesos que posteriormente aprenderá el modelo.
        super().__init__()

        # --------------------------------------------------------------
        # CAPA LSTM
        # --------------------------------------------------------------

        # Creamos la capa recurrente LSTM.
        #
        # input_size:
        #   Número de variables que recibe la red en cada ciclo.
        #
        # Por ejemplo:
        #   Si utilizamos 14 sensores, input_size = 14.
        #
        # hidden_size:
        #   Número de unidades internas de la LSTM.
        #
        # Estas unidades almacenan y procesan información temporal
        # de la secuencia.
        #
        # hidden_size = 64 significa que la representación interna
        # de cada paso temporal tendrá 64 dimensiones.
        #
        # num_layers:
        #   Número de capas LSTM apiladas.
        #
        # num_layers = 2 significa que tenemos dos capas recurrentes,
        # donde la salida de la primera alimenta a la segunda.
        #
        # batch_first=True:
        #   Indica que los tensores de entrada tendrán el formato:
        #
        #   (batch_size, seq_length, input_size)
        #
        #   En lugar de:
        #
        #   (seq_length, batch_size, input_size)
        #
        # Esto facilita trabajar con la estructura de nuestros datos.
        #
        # dropout:
        #   Aplica dropout entre las capas LSTM durante el entrenamiento.
        #
        # dropout if num_layers > 1 else 0.0:
        #   PyTorch aplica dropout interno entre capas recurrentes,
        #   pero no existe dropout entre capas si solo hay una capa LSTM.
        #
        # Por eso:
        #   Si num_layers > 1 -> utilizamos el dropout definido.
        #   Si num_layers = 1 -> dropout = 0.0.
        #
        # Dropout consiste en desactivar aleatoriamente algunas conexiones
        # durante el entrenamiento, ayudando a reducir overfitting.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --------------------------------------------------------------
        # CAPAS DENSAS DE SALIDA (HEAD)
        # --------------------------------------------------------------

        # Creamos una secuencia de capas totalmente conectadas
        # que transforman la representación temporal de la LSTM
        # en una predicción numérica de RUL.
        #
        # nn.Sequential permite ejecutar varias capas consecutivamente
        # en el orden en que se definen.
        self.head = nn.Sequential(

            # Primera capa lineal.
            #
            # Recibe el vector de representación de la LSTM,
            # de tamaño hidden_size (64 por defecto).
            #
            # Lo transforma en un vector de 32 valores.
            #
            # Matemáticamente:
            #   salida = entrada * pesos + bias
            #
            # Los pesos y bias se aprenden durante el entrenamiento.
            nn.Linear(hidden_size, 32),

            # Aplicamos la función de activación ReLU.
            #
            # ReLU(x) = max(0, x)
            #
            # Los valores negativos se convierten en cero,
            # mientras que los valores positivos se mantienen.
            #
            # Introduce no linealidad en la red, permitiendo que
            # aprenda relaciones más complejas que una transformación
            # lineal simple.
            nn.ReLU(),

            # Segunda capa lineal.
            #
            # Recibe los 32 valores de la capa anterior
            # y genera una única salida.
            #
            # Como queremos predecir una variable continua (RUL),
            # esta capa genera un valor numérico.
            #
            # No aplicamos una función de activación final,
            # por lo que la salida puede ser cualquier valor real.
            # Posteriormente, durante la evaluación, se limita
            # al rango físico permitido.
            nn.Linear(32, 1),
        )

    # ------------------------------------------------------------------
    # FORWARD: PROPAGACIÓN HACIA DELANTE
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define cómo los datos atraviesan la red neuronal.

        Parámetro:
            x: Tensor de entrada con forma:
               (batch, seq_len, num_sensors)

        Retorno:
            Predicción de RUL para cada secuencia del batch.
        """

        # --------------------------------------------------------------
        # 1. PASO DE LA SECUENCIA POR LA LSTM
        # --------------------------------------------------------------

        # Enviamos el tensor de entrada x a la capa LSTM.
        #
        # x tiene la forma:
        #   (batch_size, seq_length, num_sensors)
        #
        # Por ejemplo:
        #   (64, 30, 14)
        #
        # Significa:
        #   64 secuencias en el batch.
        #   30 ciclos por secuencia.
        #   14 sensores por ciclo.
        #
        # La LSTM procesa los ciclos secuencialmente y genera
        # una representación interna de cada paso temporal.
        #
        # lstm_out contiene las salidas de la LSTM para todos
        # los pasos temporales de la secuencia.
        #
        # Su forma será:
        #   (batch_size, seq_length, hidden_size)
        #
        # Por ejemplo:
        #   (64, 30, 64)
        #
        # El segundo elemento "_" contiene los estados ocultos finales
        # y de celda de la LSTM (h_n y c_n).
        #
        # En este código no los utilizamos directamente,
        # por lo que los almacenamos en "_" para indicar
        # que no necesitamos ese resultado.
        lstm_out, _ = self.lstm(x)

        # --------------------------------------------------------------
        # 2. SELECCIÓN DEL ÚLTIMO PASO TEMPORAL
        # --------------------------------------------------------------

        # Seleccionamos la salida de la LSTM correspondiente
        # al último ciclo de cada secuencia.
        #
        # lstm_out tiene forma:
        #   (batch_size, seq_length, hidden_size)
        #
        # lstm_out[:, -1, :] significa:
        #
        #   [:]   -> todos los elementos del batch.
        #   -1    -> último paso temporal de la secuencia.
        #   [:]   -> todas las dimensiones ocultas.
        #
        # El resultado tiene forma:
        #   (batch_size, hidden_size)
        #
        # Esta representación contiene la información temporal
        # procesada por la LSTM hasta el final de la ventana.
        #
        # La utilizamos para predecir el RUL correspondiente
        # al último ciclo de cada secuencia.
        last_step = lstm_out[:, -1, :]

        # --------------------------------------------------------------
        # 3. PREDICCIÓN DEL RUL
        # --------------------------------------------------------------

        # Pasamos la representación del último paso temporal
        # a través de las capas densas definidas en self.head.
        #
        # La red transforma el vector de 64 dimensiones:
        #   -> 32 neuronas.
        #   -> ReLU.
        #   -> 1 salida.
        #
        # out tendrá inicialmente la forma:
        #   (batch_size, 1)
        #
        # Cada valor representa el RUL predicho para una secuencia.
        out = self.head(last_step)

        # Eliminamos la última dimensión de tamaño 1.
        #
        # squeeze(-1) elimina únicamente la última dimensión
        # cuando tiene tamaño 1.
        #
        # Por ejemplo:
        #   (64, 1) -> (64,)
        #
        # Esto facilita comparar las predicciones con y_true,
        # que normalmente también tendrá forma (batch_size,).
        #
        # Devolvemos las predicciones finales.
        return out.squeeze(-1)


# ----------------------------------------------------------------------
# 5. EVALUACIÓN DEL MODELO SOBRE TEST
# ----------------------------------------------------------------------

def evaluate_test_last_window(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    unit_test: np.ndarray,
    rul_cap: int,
    late_pen: float,
    early_pen: float,
    device: torch.device,
) -> dict:
    """
    Evalúa la última ventana temporal disponible de cada motor de test.

    Parámetros:
        model: Red neuronal entrenada.
        X_test: Secuencias de test.
        y_test: RUL real asociado a las secuencias.
        unit_test: Identificador del motor de cada secuencia.
        rul_cap: Máximo RUL permitido.
        late_pen: Penalización de predicción tardía.
        early_pen: Penalización de predicción temprana.
        device: CPU o GPU utilizada para ejecutar el modelo.

    Retorno:
        Diccionario con:
            rmse: Error cuadrático medio en raíz.
            asym_cost: Coste asimétrico.
            late_rate_pct: Porcentaje de predicciones tardías.
    """

    # --------------------------------------------------------------
    # 1. MODO EVALUACIÓN
    # --------------------------------------------------------------

    # Ponemos el modelo en modo evaluación.
    #
    # model.eval() modifica el comportamiento de determinadas capas.
    #
    # Por ejemplo:
    #   - Dropout deja de desactivar neuronas aleatoriamente.
    #   - BatchNorm utiliza sus estadísticas aprendidas.
    #
    # Es importante utilizar eval() cuando evaluamos el modelo
    # o realizamos predicciones, ya que no queremos que actúe
    # como durante el entrenamiento.
    model.eval()

    # --------------------------------------------------------------
    # 2. PREDICCIÓN SIN CALCULAR GRADIENTES
    # --------------------------------------------------------------

    # Desactivamos temporalmente el cálculo de gradientes.
    #
    # torch.no_grad() indica que no necesitamos almacenar información
    # para realizar posteriormente backpropagation.
    #
    # Esto:
    #   - Reduce consumo de memoria.
    #   - Reduce tiempo de ejecución.
    #   - Es apropiado porque aquí solo estamos realizando predicciones.
    with torch.no_grad():

        # Convertimos el array NumPy de test en un tensor de PyTorch.
        #
        # dtype=torch.float32 establece el tipo de dato numérico
        # que utiliza la red neuronal.
        #
        # device=device coloca el tensor en el dispositivo seleccionado:
        #   CPU o GPU.
        #
        # De esta manera los datos estarán en el mismo dispositivo
        # que el modelo.
        X_tensor = torch.tensor(
            X_test,
            dtype=torch.float32,
            device=device
        )

        # Ejecutamos el modelo sobre todas las secuencias de test.
        #
        # model(X_tensor) llama automáticamente a forward().
        #
        # El resultado contiene una predicción de RUL por secuencia.
        #
        # .cpu() traslada el resultado a CPU.
        #
        # .numpy() convierte el tensor PyTorch en un array NumPy.
        #
        # De esta manera podemos utilizar posteriormente
        # métricas de Scikit-learn y operaciones de NumPy.
        all_preds = model(X_tensor).cpu().numpy()

    # --------------------------------------------------------------
    # 3. LIMITACIÓN DEL RUL PREDICHO
    # --------------------------------------------------------------

    # Limitamos las predicciones al rango físicamente permitido:
    #   mínimo = 0 ciclos.
    #   máximo = rul_cap.
    #
    # np.clip(array, mínimo, máximo):
    #
    #   Valores inferiores a 0 -> se convierten en 0.
    #   Valores superiores a rul_cap -> se convierten en rul_cap.
    #   Valores intermedios -> se mantienen.
    #
    # Esto evita predicciones negativas de vida útil
    # o predicciones superiores al máximo establecido.
    all_preds = np.clip(all_preds, 0, rul_cap)

    # --------------------------------------------------------------
    # 4. SELECCIÓN DE LA ÚLTIMA VENTANA DE CADA MOTOR
    # --------------------------------------------------------------

    # Creamos dos listas vacías.
    #
    # last_preds almacenará la predicción correspondiente
    # a la última ventana disponible de cada motor.
    #
    # last_trues almacenará el RUL real correspondiente
    # a esa misma ventana.
    #
    # El objetivo es evaluar el momento de decisión final:
    # cuando el motor ha acumulado todos los ciclos disponibles
    # en test y debemos decidir cuánto tiempo de vida le queda.
    last_preds, last_trues = [], []

    # Obtenemos todos los identificadores de motor únicos.
    #
    # np.unique(unit_test) devuelve cada unit_id una única vez.
    #
    # Recorremos todos los motores del conjunto test.
    for uid in np.unique(unit_test):

        # Buscamos las posiciones del array unit_test
        # que corresponden al motor uid.
        #
        # np.where(unit_test == uid)[0] devuelve los índices
        # donde se cumple la condición.
        #
        # [-1] selecciona el último índice encontrado.
        #
        # Como las secuencias se generan por motor y en orden temporal,
        # ese índice corresponde a la última secuencia disponible
        # de dicho motor.
        #
        # Por ejemplo:
        #   Motor 1 -> índices 0, 1, 2, 3, 4
        #   idx = 4 -> última secuencia del motor 1.
        idx = np.where(unit_test == uid)[0][-1]

        # Añadimos a last_preds la predicción de la última secuencia
        # del motor actual.
        #
        # all_preds[idx] contiene el RUL predicho para esa ventana.
        last_preds.append(all_preds[idx])

        # Añadimos a last_trues el RUL real de esa misma secuencia.
        #
        # y_test[idx] contiene la etiqueta real asociada a la ventana.
        #
        # De esta manera comparamos predicción y valor real
        # en el mismo instante temporal.
        last_trues.append(y_test[idx])

    # --------------------------------------------------------------
    # 5. CONVERSIÓN DE LISTAS A ARRAYS
    # --------------------------------------------------------------

    # Convertimos la lista de predicciones finales en un array NumPy.
    #
    # Tendremos una predicción por motor.
    #
    # Si test contiene 100 motores:
    #   last_preds.shape = (100,)
    last_preds = np.array(last_preds)

    # Convertimos la lista de valores reales finales en un array NumPy.
    #
    # Tendremos un RUL real por motor, correspondiente
    # al mismo instante que la predicción seleccionada.
    #
    # Si test contiene 100 motores:
    #   last_trues.shape = (100,)
    last_trues = np.array(last_trues)

    # --------------------------------------------------------------
    # 6. CÁLCULO DE MÉTRICAS
    # --------------------------------------------------------------

    # Calculamos el RMSE (Root Mean Squared Error).
    #
    # mean_squared_error calcula primero el error cuadrático medio:
    #
    #   MSE = media((predicción - valor_real)^2)
    #
    # np.sqrt() calcula la raíz cuadrada:
    #
    #   RMSE = sqrt(MSE)
    #
    # El RMSE expresa el error en las mismas unidades
    # que la variable objetivo.
    #
    # En este caso, ciclos de vida útil.
    #
    # float(...) convierte el resultado a un número float estándar
    # de Python, facilitando su almacenamiento y presentación.
    rmse = float(
        np.sqrt(
            mean_squared_error(last_trues, last_preds)
        )
    )

    # Calculamos el coste asimétrico industrial.
    #
    # Utilizamos la función asymmetric_plant_cost importada
    # desde nuestro módulo de entrenamiento baseline.
    #
    # last_trues contiene los valores reales.
    # last_preds contiene las predicciones.
    #
    # late_penalty establece cuánto penalizamos
    # los errores de predicción tardía.
    #
    # early_penalty establece cuánto penalizamos
    # los errores de predicción temprana.
    #
    # Esta métrica permite evaluar el modelo considerando
    # que ambos tipos de errores no tienen el mismo impacto industrial.
    asym_cost = asymmetric_plant_cost(
        last_trues,
        last_preds,
        late_penalty=late_pen,
        early_penalty=early_pen
    )

    # Calculamos el porcentaje de predicciones tardías.
    #
    # last_preds > last_trues devuelve una máscara booleana:
    #
    #   True  -> predicción tardía.
    #   False -> predicción temprana o exacta.
    #
    # np.mean() calcula la media de los valores booleanos:
    #
    #   True = 1
    #   False = 0
    #
    # Por tanto, obtenemos la proporción de predicciones tardías.
    #
    # Multiplicamos por 100 para expresarla en porcentaje.
    #
    # float(...) convierte el resultado a un float de Python.
    late_rate = float(
        np.mean(last_preds > last_trues) * 100.0
    )

    # Devolvemos un diccionario con las tres métricas calculadas.
    #
    # rmse:
    #   Error general de predicción en ciclos.
    #
    # asym_cost:
    #   Coste que considera la diferente gravedad de los errores.
    #
    # late_rate_pct:
    #   Porcentaje de motores cuya predicción fue tardía.
    #
    # Este diccionario será utilizado posteriormente
    # para imprimir los resultados del entrenamiento.
    return {
        "rmse": rmse,
        "asym_cost": asym_cost,
        "late_rate_pct": late_rate
    }


# ----------------------------------------------------------------------
# 6. PIPELINE COMPLETO DE ENTRENAMIENTO
# ----------------------------------------------------------------------

def run_dl_training(
    config_path: str = "configs/config.yaml"
) -> None:
    """
    Ejecuta el entrenamiento completo de la red neuronal LSTM.

    Etapas:
        1. Carga de configuración.
        2. Configuración de aleatoriedad.
        3. Selección de dispositivo CPU/GPU.
        4. Carga de secuencias.
        5. Creación de DataLoaders.
        6. Creación del modelo.
        7. Definición de loss y optimizer.
        8. Entrenamiento por épocas.
        9. Evaluación de validation.
        10. Guardado del mejor modelo.
        11. Evaluación final en test.
    """

    # --------------------------------------------------------------
    # 1. CARGA DE CONFIGURACIÓN
    # --------------------------------------------------------------

    # Cargamos la configuración del proyecto desde el archivo YAML.
    cfg = load_config(config_path)

    # Extraemos el identificador del dataset.
    #
    # Por ejemplo, FD001, FD002, FD003 o FD004.
    # Se utiliza para localizar los archivos correspondientes
    # y nombrar el modelo entrenado.
    dataset_id = cfg["data"]["dataset_id"]

    # Extraemos la semilla aleatoria definida en la configuración.
    #
    # La utilizaremos para controlar la aleatoriedad
    # de NumPy y PyTorch.
    seed = cfg["data"]["random_seed"]

    # Obtenemos la ruta del directorio donde se encuentran
    # los datasets procesados y las secuencias generadas
    # en el Feature Pipeline.
    processed_dir = Path(cfg["data"]["processed_dir"])

    # Obtenemos el límite máximo de RUL establecido
    # en la configuración física del proyecto.
    #
    # Se utilizará para limitar las predicciones finales.
    rul_cap = cfg["physics_and_sensors"]["rul_cap"]

    # Extraemos el peso de penalización para las predicciones tardías.
    #
    # Una predicción tardía sobreestima el tiempo restante de vida
    # y puede retrasar una intervención de mantenimiento.
    late_pen = cfg["cost_asymmetry"]["late_prediction_penalty"]

    # Extraemos el peso de penalización para las predicciones tempranas.
    #
    # Una predicción temprana puede provocar mantenimiento
    # antes de que sea realmente necesario.
    early_pen = cfg["cost_asymmetry"]["early_prediction_penalty"]

    # --------------------------------------------------------------
    # 2. CONFIGURACIÓN DE ALEATORIEDAD Y DISPOSITIVO
    # --------------------------------------------------------------

    # Fijamos las semillas aleatorias.
    #
    # Esto ayuda a reproducir los resultados de los experimentos.
    set_seed(seed)

    # Seleccionamos el dispositivo donde ejecutaremos el modelo.
    #
    # torch.cuda.is_available() comprueba si hay una GPU CUDA disponible.
    #
    # Si está disponible:
    #   device = "cuda"
    #
    # Si no está disponible:
    #   device = "cpu"
    #
    # El modelo y los tensores deberán estar en el mismo dispositivo
    # para poder realizar operaciones entre ellos.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------------------
    # 3. CARGA DE LAS SECUENCIAS
    # --------------------------------------------------------------

    # Cargamos el archivo NPZ generado en el Feature Pipeline.
    #
    # Este archivo contiene las secuencias y etiquetas de:
    #   - Train.
    #   - Validation.
    #   - Test.
    #
    # np.load() permite acceder a los arrays almacenados
    # dentro del archivo comprimido.
    data = np.load(
        processed_dir / f"sequences_{dataset_id}.npz"
    )

    # Extraemos las secuencias de entrenamiento.
    #
    # X_train contiene las ventanas de sensores.
    #
    # Su forma habitual es:
    #   (n_secuencias_train, seq_length, n_sensores)
    X_train = data["X_train"]

    # Extraemos las etiquetas de entrenamiento.
    #
    # y_train contiene el RUL real asociado
    # a cada secuencia de entrenamiento.
    y_train = data["y_train"]

    # Extraemos las secuencias de validación.
    #
    # Se utilizarán para evaluar el modelo durante el entrenamiento,
    # pero no para actualizar sus pesos.
    X_val = data["X_val"]

    # Extraemos los RUL reales de validación.
    #
    # Se utilizan para calcular la función de pérdida
    # de validación y seleccionar el mejor modelo.
    y_val = data["y_val"]

    # Extraemos las secuencias de test.
    #
    # Estas secuencias se utilizarán para la evaluación final
    # del modelo una vez terminado el entrenamiento.
    X_test = data["X_test"]

    # Extraemos los RUL reales de test.
    #
    # Se utilizarán para calcular las métricas finales
    # y evaluar el rendimiento del modelo.
    y_test = data["y_test"]

    # Extraemos el identificador del motor correspondiente
    # a cada secuencia de test.
    #
    # Este array permite posteriormente seleccionar
    # la última secuencia de cada motor.
    unit_test = data["unit_test"]

    # --------------------------------------------------------------
    # 4. CREACIÓN DE DATALOADERS
    # --------------------------------------------------------------

    # Creamos el DataLoader de entrenamiento.
    #
    # TensorDataset agrupa los tensores de entrada X_train
    # y sus etiquetas y_train.
    #
    # torch.tensor(X_train) convierte el array NumPy
    # en un tensor PyTorch.
    #
    # torch.tensor(y_train) hace lo mismo con las etiquetas.
    #
    # batch_size=64 indica que el modelo procesará
    # 64 secuencias por lote.
    #
    # shuffle=True mezcla aleatoriamente las secuencias
    # al comenzar cada época.
    #
    # Esto ayuda a evitar que el modelo dependa del orden
    # en el que se presentan los ejemplos de entrenamiento.
    #
    # El DataLoader genera automáticamente los batches
    # durante el entrenamiento.
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train)
        ),
        batch_size=64,
        shuffle=True,
    )

    # Creamos el DataLoader de validación.
    #
    # TensorDataset agrupa las secuencias de validación
    # con sus correspondientes etiquetas.
    #
    # batch_size=128 significa que procesaremos
    # 128 secuencias por lote de validación.
    #
    # shuffle=False indica que no mezclamos las secuencias.
    #
    # No es necesario mezclar validation porque no se utiliza
    # para actualizar los pesos del modelo.
    #
    # El DataLoader permite evaluar los datos en batches,
    # evitando tener que procesarlos todos a la vez.
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_val)
        ),
        batch_size=128,
        shuffle=False,
    )

    # --------------------------------------------------------------
    # 5. CREACIÓN DEL MODELO
    # --------------------------------------------------------------

    # Obtenemos el número de sensores/features que contiene
    # cada ciclo de las secuencias de entrenamiento.
    #
    # X_train.shape devuelve las dimensiones del array.
    #
    # Si:
    #   X_train.shape = (8000, 30, 14)
    #
    # Entonces:
    #   X_train.shape[0] = 8000 secuencias.
    #   X_train.shape[1] = 30 ciclos por secuencia.
    #   X_train.shape[2] = 14 sensores.
    #
    # Por tanto, num_sensors = 14.
    #
    # Este valor será input_size de la LSTM.
    num_sensors = X_train.shape[2]

    # Creamos la red neuronal PredictiveMaintenanceLSTM.
    #
    # input_size=num_sensors:
    #   La red recibe el número de sensores que tenemos.
    #
    # hidden_size=64:
    #   Cada capa LSTM tiene 64 unidades ocultas.
    #
    # num_layers=2:
    #   Utilizamos dos capas LSTM apiladas.
    #
    # dropout=0.2:
    #   Aplicamos dropout del 20% entre capas LSTM durante train.
    #
    # .to(device) traslada todos los parámetros del modelo
    # al dispositivo seleccionado, CPU o GPU.
    model = PredictiveMaintenanceLSTM(
        input_size=num_sensors,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    # --------------------------------------------------------------
    # 6. DEFINICIÓN DE LA FUNCIÓN DE PÉRDIDA
    # --------------------------------------------------------------

    # Creamos la función de pérdida asimétrica.
    #
    # late_penalty:
    #   Penalización para predicciones tardías.
    #
    # early_penalty:
    #   Penalización para predicciones tempranas.
    #
    # La red intentará minimizar esta función durante el entrenamiento.
    criterion = AsymmetricMSELoss(
        late_penalty=late_pen,
        early_penalty=early_pen
    )

    # --------------------------------------------------------------
    # 7. DEFINICIÓN DEL OPTIMIZADOR
    # --------------------------------------------------------------

    # Creamos el optimizador Adam.
    #
    # model.parameters() proporciona todos los pesos y bias
    # que el modelo debe aprender.
    #
    # lr=1e-3 equivale a 0.001.
    #
    # Learning rate controla el tamaño de los pasos que damos
    # al actualizar los parámetros de la red.
    #
    # Un learning rate demasiado grande puede impedir que el modelo
    # converja correctamente.
    #
    # Un learning rate demasiado pequeño puede hacer que el entrenamiento
    # sea excesivamente lento.
    #
    # weight_decay=1e-4 equivale a 0.0001.
    #
    # Es un término de regularización que penaliza pesos demasiado grandes
    # y puede ayudar a reducir overfitting.
    #
    # Adam adapta la magnitud de las actualizaciones de los parámetros
    # utilizando información de los gradientes anteriores.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # --------------------------------------------------------------
    # 8. CONFIGURACIÓN DEL ENTRENAMIENTO
    # --------------------------------------------------------------

    # Definimos el número máximo de épocas de entrenamiento.
    #
    # Una época significa que el modelo ha recorrido una vez
    # todo el conjunto de entrenamiento.
    #
    # epochs=35 significa que realizaremos hasta 35 recorridos completos
    # sobre los datos de entrenamiento.
    epochs = 35

    # Inicializamos best_val_loss con infinito.
    #
    # float("inf") representa un valor infinitamente grande.
    #
    # De esta manera, la pérdida de validación de la primera época
    # será necesariamente inferior y podremos guardar
    # el primer modelo como mejor modelo inicial.
    #
    # Posteriormente solo guardaremos modelos que mejoren
    # la pérdida de validación anterior.
    best_val_loss = float("inf")

    # Definimos la ruta donde guardaremos los mejores pesos del modelo.
    #
    # El nombre incluye dataset_id para distinguir los modelos
    # entrenados con diferentes escenarios.
    #
    # .pt es una extensión habitual para guardar modelos/estados
    # de PyTorch.
    best_model_path = Path(
        f"models/lstm_{dataset_id}.pt"
    )

    # Mostramos un mensaje indicando:
    #   - Que estamos entrenando una LSTM.
    #   - El dispositivo utilizado.
    #   - El número de épocas previsto.
    #
    # device mostrará "cpu" o "cuda".
    print(
        f"Entrenando PyTorch LSTM en dispositivo: "
        f"{device} ({epochs} épocas)..."
    )

    # Registramos el instante inicial del entrenamiento.
    #
    # perf_counter() devuelve un contador de alta precisión.
    # Lo utilizaremos posteriormente para medir el tiempo total.
    t0 = time.perf_counter()

    # --------------------------------------------------------------
    # 9. BUCLE PRINCIPAL DE ENTRENAMIENTO
    # --------------------------------------------------------------

    # Recorremos las épocas desde 1 hasta epochs, incluyendo 35.
    #
    # range(1, epochs + 1) genera:
    #   1, 2, 3, ..., 35
    #
    # En cada época:
    #   1. Entrenamos con todos los batches de train.
    #   2. Evaluamos con todos los batches de validation.
    #   3. Comprobamos si el modelo ha mejorado.
    #   4. Guardamos el modelo si es el mejor hasta el momento.
    for epoch in range(1, epochs + 1):

        # ----------------------------------------------------------
        # 9.1 MODO ENTRENAMIENTO
        # ----------------------------------------------------------

        # Activamos el modo entrenamiento del modelo.
        #
        # model.train() permite que determinadas capas actúen
        # en modo entrenamiento.
        #
        # Por ejemplo:
        #   Dropout vuelve a desactivar aleatoriamente neuronas.
        #
        # Esto permite utilizar regularización durante el entrenamiento.
        model.train()

        # ----------------------------------------------------------
        # 9.2 RECORRIDO DE LOS BATCHES DE TRAIN
        # ----------------------------------------------------------

        # Recorremos todos los batches generados por train_loader.
        #
        # bx contiene las secuencias de sensores de un batch.
        #
        # by contiene los RUL reales asociados a esas secuencias.
        #
        # Como batch_size=64, normalmente cada batch tendrá
        # 64 secuencias, salvo el último batch que puede ser más pequeño.
        for bx, by in train_loader:

            # Trasladamos los datos del batch al dispositivo seleccionado.
            #
            # bx.to(device) mueve las secuencias.
            #
            # by.to(device) mueve las etiquetas.
            #
            # Si el modelo está en GPU, los datos también deben estar
            # en GPU para poder realizar operaciones entre ellos.
            bx, by = bx.to(device), by.to(device)

            # ------------------------------------------------------
            # 9.2.1 LIMPIEZA DE GRADIENTES
            # ------------------------------------------------------

            # Ponemos a cero los gradientes almacenados
            # en los parámetros del modelo.
            #
            # PyTorch acumula gradientes por defecto.
            #
            # Si no los limpiáramos antes de cada batch,
            # se acumularían los gradientes de diferentes batches,
            # lo cual no es lo que queremos en este entrenamiento.
            #
            # Por tanto, limpiamos los gradientes antes
            # de calcular los del batch actual.
            optimizer.zero_grad()

            # ------------------------------------------------------
            # 9.2.2 PROPAGACIÓN HACIA DELANTE
            # ------------------------------------------------------

            # Enviamos las secuencias bx al modelo.
            #
            # Esto ejecuta el método forward() de la red.
            #
            # La LSTM procesa las secuencias y la capa head
            # genera las predicciones de RUL.
            #
            # preds contiene las predicciones del batch.
            preds = model(bx)

            # ------------------------------------------------------
            # 9.2.3 CÁLCULO DE LA PÉRDIDA
            # ------------------------------------------------------

            # Calculamos la pérdida comparando:
            #
            #   preds -> RUL predicho por el modelo.
            #   by    -> RUL real.
            #
            # criterion aplica nuestra función de pérdida asimétrica.
            #
            # Si el modelo predice demasiado RUL:
            #   aplica late_penalty.
            #
            # Si predice menos RUL del real:
            #   aplica early_penalty.
            #
            # El resultado loss es un tensor con la pérdida
            # del batch actual.
            loss = criterion(preds, by)

            # ------------------------------------------------------
            # 9.2.4 BACKPROPAGATION
            # ------------------------------------------------------

            # Calculamos los gradientes de la función de pérdida
            # respecto a los parámetros de la red neuronal.
            #
            # loss.backward() realiza backpropagation.
            #
            # PyTorch calcula cómo afecta cada peso y bias
            # al error final de las predicciones.
            #
            # Estos gradientes indican en qué dirección
            # debemos modificar los parámetros para reducir la pérdida.
            loss.backward()

            # ------------------------------------------------------
            # 9.2.5 ACTUALIZACIÓN DE LOS PESOS
            # ------------------------------------------------------

            # Actualizamos los parámetros del modelo utilizando
            # los gradientes calculados y el optimizador Adam.
            #
            # optimizer.step() modifica los pesos y bias
            # de la red neuronal.
            #
            # Este es el paso en el que el modelo realmente aprende.
            #
            # El ciclo completo por batch es:
            #
            #   Predicción
            #       ↓
            #   Cálculo del error
            #       ↓
            #   Backpropagation
            #       ↓
            #   Actualización de pesos
            optimizer.step()

        # ----------------------------------------------------------
        # 9.3 EVALUACIÓN EN VALIDATION
        # ----------------------------------------------------------

        # Una vez que hemos terminado de entrenar con todos los batches
        # de esta época, cambiamos el modelo a modo evaluación.
        #
        # En este modo, Dropout deja de desactivar neuronas aleatoriamente.
        #
        # Esto permite evaluar el comportamiento del modelo
        # de forma consistente.
        model.eval()

        # Creamos una lista vacía donde almacenaremos
        # las pérdidas de los diferentes batches de validation.
        #
        # Posteriormente calcularemos la media de esas pérdidas.
        val_losses = []

        # ----------------------------------------------------------
        # 9.3.1 EVALUACIÓN SIN GRADIENTES
        # ----------------------------------------------------------

        # Desactivamos el cálculo de gradientes durante validation.
        #
        # No estamos entrenando con estos datos.
        #
        # Solo queremos comprobar cómo se comporta el modelo
        # sobre motores que no se utilizaron para actualizar sus pesos.
        #
        # Esto reduce consumo de memoria y acelera la evaluación.
        with torch.no_grad():

            # Recorremos todos los batches de validation.
            #
            # bx contiene las secuencias.
            # by contiene sus RUL reales.
            for bx, by in val_loader:

                # Movemos las secuencias y etiquetas
                # al dispositivo donde está el modelo.
                bx, by = bx.to(device), by.to(device)

                # Calculamos la pérdida de validación del batch.
                #
                # model(bx) genera las predicciones.
                #
                # criterion compara esas predicciones con by.
                #
                # .item() convierte el tensor de pérdida
                # en un número Python.
                #
                # Añadimos ese valor a la lista val_losses.
                val_losses.append(
                    criterion(model(bx), by).item()
                )

        # ----------------------------------------------------------
        # 9.4 PÉRDIDA MEDIA DE VALIDACIÓN
        # ----------------------------------------------------------

        # Calculamos la media de todas las pérdidas de validation.
        #
        # np.mean(val_losses) calcula la media aritmética.
        #
        # float(...) convierte el resultado en un float estándar.
        #
        # Este valor representa el rendimiento del modelo
        # en validation durante la época actual.
        mean_val_loss = float(np.mean(val_losses))

        # ----------------------------------------------------------
        # 9.5 SELECCIÓN DEL MEJOR MODELO
        # ----------------------------------------------------------

        # Comprobamos si la pérdida de validation actual
        # es inferior a la mejor pérdida obtenida hasta ahora.
        #
        # Cuanto menor sea la pérdida, mejor será el modelo
        # según nuestra función de pérdida asimétrica.
        if mean_val_loss < best_val_loss:

            # Actualizamos el registro de la mejor pérdida de validation.
            #
            # A partir de ahora, este será el valor de referencia
            # con el que compararemos las siguientes épocas.
            best_val_loss = mean_val_loss

            # Guardamos los pesos del modelo actual.
            #
            # model.state_dict() contiene los parámetros aprendidos:
            # pesos y bias de las diferentes capas.
            #
            # torch.save(...) guarda esos parámetros en disco.
            #
            # No guardamos necesariamente toda la estructura del modelo,
            # sino su state_dict.
            #
            # Posteriormente podemos reconstruir la arquitectura
            # y cargar estos pesos para recuperar el modelo entrenado.
            #
            # Guardamos únicamente cuando validation mejora.
            # De esta manera, aunque las últimas épocas empeoren,
            # conservaremos los pesos del mejor modelo encontrado.
            torch.save(
                model.state_dict(),
                best_model_path
            )

        # ----------------------------------------------------------
        # 9.6 INFORMACIÓN POR ÉPOCA
        # ----------------------------------------------------------

        # Mostramos información de entrenamiento cada cinco épocas
        # y también durante la primera época.
        #
        # epoch % 5 == 0 comprueba si el número de época
        # es múltiplo de 5.
        #
        # Por ejemplo:
        #   5, 10, 15, 20, 25, 30, 35.
        #
        # epoch == 1 permite mostrar también la primera época.
        #
        # Así evitamos imprimir información en cada época,
        # reduciendo la cantidad de mensajes en consola.
        if epoch % 5 == 0 or epoch == 1:

            # Imprimimos:
            #
            #   Época actual / épocas totales.
            #
            # {epoch:02d} muestra el número con dos dígitos.
            # Por ejemplo:
            #   1 -> 01
            #   5 -> 05
            #
            # {mean_val_loss:.2f} muestra la pérdida con dos decimales.
            #
            # Esto permite seguir la evolución del modelo
            # a lo largo del entrenamiento.
            print(
                f" -> Época {epoch:02d}/{epochs} | "
                f"Val Asymmetric Loss: {mean_val_loss:.2f}"
            )

    # --------------------------------------------------------------
    # 10. MEDICIÓN DEL TIEMPO TOTAL DE ENTRENAMIENTO
    # --------------------------------------------------------------

    # Calculamos cuánto tiempo ha transcurrido desde t0.
    #
    # time.perf_counter() devuelve el contador actual.
    #
    # Restando el tiempo inicial obtenemos la duración total
    # del bucle de entrenamiento.
    #
    # El tiempo incluye las épocas de entrenamiento y validation,
    # pero no incluye la evaluación final de test que realizaremos después.
    train_time = time.perf_counter() - t0

    # --------------------------------------------------------------
    # 11. CARGA DEL MEJOR MODELO
    # --------------------------------------------------------------

    # Cargamos los mejores pesos guardados durante el entrenamiento.
    #
    # model.load_state_dict(...) reemplaza los parámetros actuales
    # del modelo por los parámetros almacenados en el archivo.
    #
    # torch.load(...) lee el archivo de pesos.
    #
    # map_location=device indica que los pesos se cargarán
    # en el dispositivo seleccionado, aunque se hayan guardado
    # en otro dispositivo.
    #
    # weights_only=True indica que queremos cargar únicamente
    # los pesos del modelo, utilizando el modo de carga restringido
    # de PyTorch para este tipo de archivo.
    #
    # Es importante hacer esto porque la última época no necesariamente
    # es la mejor: validation podría haber empeorado.
    #
    # Por tanto, recuperamos el modelo que consiguió
    # la mejor pérdida de validation.
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True
        )
    )

    # --------------------------------------------------------------
    # 12. EVALUACIÓN FINAL SOBRE TEST
    # --------------------------------------------------------------

    # Evaluamos el mejor modelo sobre el conjunto de test.
    #
    # Esta evaluación se realiza después del entrenamiento
    # y de la selección del mejor modelo mediante validation.
    #
    # model:
    #   Red neuronal con los mejores pesos guardados.
    #
    # X_test:
    #   Secuencias de test.
    #
    # y_test:
    #   RUL reales de test.
    #
    # unit_test:
    #   Identificador de motor de cada secuencia.
    #
    # rul_cap:
    #   Límite máximo de RUL permitido.
    #
    # late_pen y early_pen:
    #   Penalizaciones asimétricas.
    #
    # device:
    #   Dispositivo de ejecución.
    #
    # La función devuelve un diccionario con las métricas finales.
    test_metrics = evaluate_test_last_window(
        model,
        X_test,
        y_test,
        unit_test,
        rul_cap,
        late_pen,
        early_pen,
        device
    )

    # --------------------------------------------------------------
    # 13. PRESENTACIÓN DE RESULTADOS
    # --------------------------------------------------------------

    # Imprimimos un encabezado con el nombre del modelo
    # y el tiempo de entrenamiento.
    #
    # train_time:.2f muestra el tiempo con dos decimales.
    #
    # El tiempo se expresa en segundos.
    print(
        f"\n[RESULTADOS PYTORCH LSTM ASIMÉTRICO] "
        f"(Tiempo entreno: {train_time:.2f}s)"
    )

    # Imprimimos las métricas finales del conjunto test.
    #
    # test_metrics["rmse"]:
    #   RMSE en ciclos.
    #
    # test_metrics["asym_cost"]:
    #   Coste asimétrico de las predicciones.
    #
    # test_metrics["late_rate_pct"]:
    #   Porcentaje de predicciones tardías.
    #
    # .2f muestra los números con dos decimales.
    # .1f muestra el porcentaje con un decimal.
    #
    # El resultado permite evaluar el modelo desde tres perspectivas:
    #   - Error general.
    #   - Coste considerando el riesgo de los errores.
    #   - Frecuencia de predicciones tardías.
    print(
        f" -> Test NASA (100 motores) | "
        f"RMSE: {test_metrics['rmse']:.2f} ciclos | "
        f"Coste Asimétrico: {test_metrics['asym_cost']:.2f} | "
        f"% Avisos Tardíos: {test_metrics['late_rate_pct']:.1f}%"
    )


# ----------------------------------------------------------------------
# PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------

# Comprobamos si este archivo se está ejecutando directamente.
#
# Si ejecutamos este archivo desde la terminal:
#   __name__ == "__main__"
#
# Si lo importamos desde otro módulo:
#   __name__ tendrá el nombre del módulo.
#
# Esta condición permite que el entrenamiento se ejecute
# únicamente cuando lanzamos este archivo directamente.
if __name__ == "__main__":

    # Ejecutamos el pipeline completo de entrenamiento de la LSTM.
    #
    # Como no proporcionamos ningún argumento,
    # se utilizará configs/config.yaml.
    #
    # El programa:
    #   - Carga los datos.
    #   - Crea DataLoaders.
    #   - Construye la LSTM.
    #   - Entrena durante 35 épocas.
    #   - Evalúa en validation.
    #   - Guarda el mejor modelo.
    #   - Evalúa finalmente en test.
    run_dl_training()