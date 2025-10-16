# **FlexSAIze MLflow Lab – End-to-End MLOps Pipeline**

### 🧠 Overview
Este proyecto fue desarrollado como un laboratorio personal para construir un **pipeline de Machine Learning reproducible** utilizando herramientas reales de MLOps:  
- **DVC** para versionar datasets y controlar el flujo de datos.  
- **MLflow** para registrar y comparar experimentos.  
- **Raspberry Pi 5** configurada como *data lake* y *servidor local de MLflow*.  
- Código modular en **Python**, siguiendo buenas prácticas de ingeniería y diseño orientado a clases.

---

## ⚙️ Project Architecture
```
PRUEBAS_MLFLOW_FLEXSAIZE/
│
├── data/                         # Datasets (raw y procesados)
│
├── scripts/                      # Scripts ejecutables para DVC o CLI
│   ├── run_preprocess.py         # Ejecuta preprocesamiento de datos
│   ├── run_plit.py               # Particiona los datos por grupos
│   ├── run_train.py              # Entrena modelos y genera métricas
│
├── src/flexsaize/                # Código fuente principal
│   ├── data/
│   │   └── preprocessor.py       # Clase DataPreprocessor: carga y limpieza
│   ├── split/
│   │   └── data_splitter.py      # Clase DataSplitter: partición train/val/test
│   ├── train/
│   │   └── model_trainer.py      # Clase ModelTrainer: entrenamiento y métricas
│   └── models/
│       └── model_config.py       # Configuración de modelos y parámetros
│
├── dvc.yaml / dvc.lock           # Pipeline reproducible DVC
├── params.yaml                   # Parámetros globales (paths, modelo, seed, etc.)
├── requirements.txt              # Dependencias del proyecto
├── pyproject.toml                # Configuración del entorno
└── .env                          # Variables locales (no versionadas)
```

---

## 🧩 Pipeline Overview

El flujo de datos y entrenamiento sigue estas etapas principales:

1. **Preprocesamiento**  
   - Limpieza de datos, eliminación de valores nulos y cálculo de nuevas features (aspect ratio, relative area, etc.).
   - Guardado del dataset limpio (`data_clean.csv`) y versión extendida con metadatos (`data_clean_withmeta.csv`).

2. **Partición de Datos**  
   - División `train/val/test` respetando la columna `file` para evitar fuga de banners entre conjuntos.
   - Soporte para `GroupShuffleSplit` y validación de consistencia.

3. **Entrenamiento y Evaluación**  
   - Entrenamiento con modelos tipo árbol: `RandomForestRegressor`, `XGBoostRegressor`, etc.
   - Cálculo de métricas: RMSE, MAE y R² por variable objetivo (`x, y, width, height`).
   - Registro automático de resultados y parámetros en **MLflow**.

4. **Tracking y Versionado**  
   - DVC controla la versión de cada dataset y modelo.  
   - MLflow guarda todos los experimentos, métricas y artefactos.  
   - Raspberry Pi 5 actúa como servidor de MLflow y data lake.

---

## 🚀 Getting Started

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/DavidCrzBlt/flex-saize-core.git
cd FlexSAIze_MLflow_Lab
```

### 2️⃣ Crear entorno virtual
```bash
python -m venv myvirtualenv
source myvirtualenv/bin/activate   # En Linux/Mac
myvirtualenv\Scripts\activate      # En Windows
```

### 3️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4️⃣ Ejecutar pipeline básico
```bash
# Preprocesamiento
python scripts/run_preprocess.py --input data/data.csv --output data/data_clean.csv

# Entrenamiento
python scripts/run_train.py --input data/data_clean.csv
```

---

## 📊 Experiment Tracking
Los experimentos se registran automáticamente en MLflow.  
Ejemplo de dashboard:

![alt text](<Imagen de WhatsApp 2025-10-16 a las 16.23.39_f6952e73.jpg>)

---

## 🧱 Tools & Technologies
| Categoría | Herramienta | Propósito |
|------------|--------------|------------|
| Versionado de datos | **DVC** | Control de datasets y outputs |
| Tracking de experimentos | **MLflow** | Registro de runs y métricas |
| Infraestructura | **Raspberry Pi 5** | Data lake + MLflow Server |
| Lenguaje | **Python 3.12** | Implementación principal |
| Librerías clave | `pandas`, `scikit-learn`, `xgboost`, `yaml` | Preprocesamiento y modelado |

---

## 📘 Aprendizajes y Buenas Prácticas
- Validar antes de suponer (`if df.empty:` / manejo de excepciones).  
- Diseñar clases con responsabilidad única (SRP).  
- Encapsular la lógica del pipeline para facilitar reproducibilidad.  
- Documentar cada fase con comentarios y *docstrings*.  
- Versionar datasets y experimentos para asegurar trazabilidad.

---

## 🧩 Próximos pasos
- Integrar CI/CD con GitHub Actions para ejecución automática del pipeline.  
- Desplegar modelos vía FastAPI para predicciones en tiempo real.  
- Automatizar actualización de datasets en la Raspberry Pi.  

---

## 👨‍💻 Autor
**David Cruz Beltrán**  
Estudiante de Maestría en Inteligencia Artificial (MNA) | Ingeniero Mecatrónico  
📍 Guadalajara, México  

📫 [LinkedIn](https://www.linkedin.com/in/david-cruz-beltran/)  
📂 [Repositorio del Proyecto](https://github.com/DavidCrzBlt/flex-saize-core)
