# Quick Start Guide / Guía de Inicio Rápido
## Synthetic Payment Data Generator

### English

This guide will help you get started quickly with the Synthetic Payment Data Generator.

#### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB of free disk space
- (Optional) CUDA-capable GPU for faster training

#### Quick Installation

1. **Clone and navigate to the repository:**
```bash
git clone https://github.com/MaxPelu/Synthetic-Payments-Data-Generator.git
cd Synthetic-Payments-Data-Generator
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

#### Quick Start - Complete Pipeline

Run the entire pipeline with default settings (this will take some time):

```bash
python src/train_pipeline.py
```

This will:
- Generate 10,000 synthetic payment transactions
- Perform exploratory data analysis with visualizations
- Train both GAN and VAE models (100 epochs each)
- Generate synthetic samples from both models
- Compare the results

#### Quick Start - Step by Step

If you prefer to run each step individually:

**Step 1: Generate initial data**
```bash
python src/data_generator.py
```

**Step 2: Run exploratory data analysis**
```bash
python src/eda_analysis.py
```

**Step 3: Train GAN model**
```bash
python src/models/gan_model.py
```

**Step 4: Train VAE model**
```bash
python src/models/vae_model.py
```

#### Using Trained Models

After training, use the example script to generate more synthetic data:

```bash
python src/example_usage.py
```

#### Custom Configuration

Customize the pipeline with command-line arguments:

```bash
# Generate 5000 transactions with 10% anomaly rate
python src/train_pipeline.py --n-transactions 5000 --anomaly-rate 0.1

# Train for only 50 epochs
python src/train_pipeline.py --epochs 50

# Skip data generation and EDA (use existing data)
python src/train_pipeline.py --skip-data-gen --skip-eda
```

#### Output Files

After running the pipeline, you'll find:

- `data/raw/synthetic_payments.csv` - Initial synthetic data
- `data/processed/gan_generated_payments.csv` - GAN-generated data
- `data/processed/vae_generated_payments.csv` - VAE-generated data
- `models/gan_model.pth` - Trained GAN model
- `models/vae_model.pth` - Trained VAE model
- `visualizations/*.png` - EDA visualizations

#### Common Issues

**Issue: Out of memory during training**
- Solution: Reduce batch size: `--batch-size 32`

**Issue: Training is too slow**
- Solution: Reduce number of epochs: `--epochs 50`
- Or install CUDA version of PyTorch for GPU acceleration

**Issue: Import errors**
- Solution: Make sure you're in the project root directory and virtual environment is activated

---

### Español

Esta guía te ayudará a comenzar rápidamente con el Generador de Datos de Pago Sintéticos.

#### Requisitos Previos

- Python 3.8 o superior
- Gestor de paquetes pip
- Al menos 2GB de espacio libre en disco
- (Opcional) GPU compatible con CUDA para entrenamiento más rápido

#### Instalación Rápida

1. **Clonar y navegar al repositorio:**
```bash
git clone https://github.com/MaxPelu/Synthetic-Payments-Data-Generator.git
cd Synthetic-Payments-Data-Generator
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

#### Inicio Rápido - Pipeline Completo

Ejecuta todo el pipeline con configuración por defecto (esto tomará tiempo):

```bash
python src/train_pipeline.py
```

Esto hará:
- Generar 10,000 transacciones de pago sintéticas
- Realizar análisis exploratorio de datos con visualizaciones
- Entrenar modelos GAN y VAE (100 épocas cada uno)
- Generar muestras sintéticas de ambos modelos
- Comparar los resultados

#### Inicio Rápido - Paso a Paso

Si prefieres ejecutar cada paso individualmente:

**Paso 1: Generar datos iniciales**
```bash
python src/data_generator.py
```

**Paso 2: Ejecutar análisis exploratorio de datos**
```bash
python src/eda_analysis.py
```

**Paso 3: Entrenar modelo GAN**
```bash
python src/models/gan_model.py
```

**Paso 4: Entrenar modelo VAE**
```bash
python src/models/vae_model.py
```

#### Usar Modelos Entrenados

Después del entrenamiento, usa el script de ejemplo para generar más datos sintéticos:

```bash
python src/example_usage.py
```

#### Configuración Personalizada

Personaliza el pipeline con argumentos de línea de comandos:

```bash
# Generar 5000 transacciones con 10% de tasa de anomalía
python src/train_pipeline.py --n-transactions 5000 --anomaly-rate 0.1

# Entrenar por solo 50 épocas
python src/train_pipeline.py --epochs 50

# Omitir generación de datos y EDA (usar datos existentes)
python src/train_pipeline.py --skip-data-gen --skip-eda
```

#### Archivos de Salida

Después de ejecutar el pipeline, encontrarás:

- `data/raw/synthetic_payments.csv` - Datos sintéticos iniciales
- `data/processed/gan_generated_payments.csv` - Datos generados por GAN
- `data/processed/vae_generated_payments.csv` - Datos generados por VAE
- `models/gan_model.pth` - Modelo GAN entrenado
- `models/vae_model.pth` - Modelo VAE entrenado
- `visualizations/*.png` - Visualizaciones de EDA

#### Problemas Comunes

**Problema: Falta de memoria durante entrenamiento**
- Solución: Reducir tamaño de lote: `--batch-size 32`

**Problema: Entrenamiento muy lento**
- Solución: Reducir número de épocas: `--epochs 50`
- O instalar versión CUDA de PyTorch para aceleración por GPU

**Problema: Errores de importación**
- Solución: Asegúrate de estar en el directorio raíz del proyecto y que el entorno virtual esté activado

---

## Next Steps / Próximos Pasos

### English

After successfully running the pipeline:

1. **Explore the visualizations** in the `visualizations/` directory
2. **Examine generated data** to understand the patterns
3. **Experiment with parameters** to see how they affect output
4. **Use generated data** for your fraud detection or analysis projects
5. **Fine-tune models** by adjusting architecture or hyperparameters

### Español

Después de ejecutar exitosamente el pipeline:

1. **Explora las visualizaciones** en el directorio `visualizations/`
2. **Examina los datos generados** para entender los patrones
3. **Experimenta con parámetros** para ver cómo afectan la salida
4. **Usa los datos generados** para tus proyectos de detección de fraude o análisis
5. **Ajusta los modelos** modificando la arquitectura o hiperparámetros
