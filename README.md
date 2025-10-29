# Synthetic Payments Data Generator 
## Generador de Datos Sintéticos de Pagos

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive AI-powered solution for generating realistic synthetic B2B payment transaction data using Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).

Una solución integral impulsada por IA para generar datos sintéticos realistas de transacciones de pagos B2B utilizando Redes Generativas Antagónicas (GANs) y Autocodificadores Variacionales (VAEs).

---

## 📋 Table of Contents / Tabla de Contenidos

- [Features / Características](#features--características)
- [Project Structure / Estructura del Proyecto](#project-structure--estructura-del-proyecto)
- [Installation / Instalación](#installation--instalación)
- [Usage / Uso](#usage--uso)
- [Models / Modelos](#models--modelos)
- [Exploratory Data Analysis / Análisis Exploratorio de Datos](#exploratory-data-analysis--análisis-exploratorio-de-datos)
- [Results / Resultados](#results--resultados)
- [Contributing / Contribuir](#contributing--contribuir)
- [License / Licencia](#license--licencia)

---

## ✨ Features / Características

### English
- **Realistic B2B Payment Data Generation**: Creates synthetic payment transactions between companies with realistic patterns
- **Multiple Generative Models**: Implements both GANs and VAEs for comparison
- **Anomaly Simulation**: Includes configurable anomaly patterns (high/low amounts, duplicate transactions, etc.)
- **Comprehensive EDA**: Full exploratory data analysis with visualizations
- **Industry-Specific Patterns**: Simulates transactions across multiple industries
- **Temporal Patterns**: Includes realistic date patterns and payment delays
- **Bilingual Documentation**: Complete documentation in English and Spanish

### Español
- **Generación Realista de Datos de Pagos B2B**: Crea transacciones de pago sintéticas entre empresas con patrones realistas
- **Múltiples Modelos Generativos**: Implementa tanto GANs como VAEs para comparación
- **Simulación de Anomalías**: Incluye patrones de anomalías configurables (montos altos/bajos, transacciones duplicadas, etc.)
- **EDA Completo**: Análisis exploratorio completo de datos con visualizaciones
- **Patrones Específicos por Industria**: Simula transacciones entre múltiples industrias
- **Patrones Temporales**: Incluye patrones de fechas realistas y retrasos de pago
- **Documentación Bilingüe**: Documentación completa en inglés y español

---

## 📁 Project Structure / Estructura del Proyecto

```
Synthetic-Payments-Data-Generator/
│
├── data/
│   ├── raw/                      # Raw synthetic data / Datos sintéticos crudos
│   └── processed/                # Processed and generated data / Datos procesados y generados
│
├── src/
│   ├── data_generator.py         # Initial data generation / Generación inicial de datos
│   ├── eda_analysis.py           # Exploratory data analysis / Análisis exploratorio
│   ├── models/
│   │   ├── gan_model.py          # GAN implementation / Implementación de GAN
│   │   └── vae_model.py          # VAE implementation / Implementación de VAE
│   └── utils/                    # Utility functions / Funciones auxiliares
│
├── visualizations/               # EDA visualizations / Visualizaciones de EDA
├── models/                       # Trained model checkpoints / Puntos de control de modelos
├── notebooks/                    # Jupyter notebooks / Notebooks de Jupyter
│
├── requirements.txt              # Python dependencies / Dependencias de Python
├── .gitignore
└── README.md                     # This file / Este archivo
```

---

## 🚀 Installation / Instalación

### English

1. **Clone the repository:**
```bash
git clone https://github.com/MaxPelu/Synthetic-Payments-Data-Generator.git
cd Synthetic-Payments-Data-Generator
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

**Option A: Using pip (recommended for development)**
```bash
pip install -r requirements.txt
```

**Option B: Using setup.py (recommended for production)**
```bash
pip install -e .
```

This will install the package and create command-line tools:
- `synpay-generate` - Generate initial data
- `synpay-eda` - Run exploratory data analysis
- `synpay-train` - Run complete training pipeline
- `synpay-example` - Run usage examples

### Español

1. **Clonar el repositorio:**
```bash
git clone https://github.com/MaxPelu/Synthetic-Payments-Data-Generator.git
cd Synthetic-Payments-Data-Generator
```

2. **Crear un entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**

**Opción A: Usando pip (recomendado para desarrollo)**
```bash
pip install -r requirements.txt
```

**Opción B: Usando setup.py (recomendado para producción)**
```bash
pip install -e .
```

Esto instalará el paquete y creará herramientas de línea de comandos:
- `synpay-generate` - Generar datos iniciales
- `synpay-eda` - Ejecutar análisis exploratorio
- `synpay-train` - Ejecutar pipeline completo de entrenamiento
- `synpay-example` - Ejecutar ejemplos de uso

---

## 💻 Usage / Uso

### English

#### 1. Generate Initial Synthetic Data
```bash
python src/data_generator.py
```
This creates a baseline dataset of synthetic B2B payment transactions with:
- Realistic company profiles
- Transaction amounts based on company size
- Multiple payment methods and currencies
- Configurable anomaly patterns

#### 2. Run Exploratory Data Analysis
```bash
python src/eda_analysis.py
```
Generates comprehensive visualizations including:
- Amount distributions
- Temporal patterns
- Categorical distributions
- Correlation heatmaps
- Anomaly analysis

#### 3. Train GAN Model
```bash
python src/models/gan_model.py
```
Trains a Generative Adversarial Network to learn payment data patterns and generate new synthetic samples.

#### 4. Train VAE Model
```bash
python src/models/vae_model.py
```
Trains a Variational Autoencoder to generate synthetic payment data with a probabilistic approach.

### Español

#### 1. Generar Datos Sintéticos Iniciales
```bash
python src/data_generator.py
```
Crea un conjunto de datos base de transacciones de pago B2B sintéticas con:
- Perfiles de empresas realistas
- Montos de transacciones basados en tamaño de empresa
- Múltiples métodos de pago y monedas
- Patrones de anomalías configurables

#### 2. Ejecutar Análisis Exploratorio de Datos
```bash
python src/eda_analysis.py
```
Genera visualizaciones completas incluyendo:
- Distribuciones de montos
- Patrones temporales
- Distribuciones categóricas
- Mapas de calor de correlación
- Análisis de anomalías

#### 3. Entrenar Modelo GAN
```bash
python src/models/gan_model.py
```
Entrena una Red Generativa Antagónica para aprender patrones de datos de pago y generar nuevas muestras sintéticas.

#### 4. Entrenar Modelo VAE
```bash
python src/models/vae_model.py
```
Entrena un Autocodificador Variacional para generar datos de pago sintéticos con un enfoque probabilístico.

---

## 🤖 Models / Modelos

### GAN (Generative Adversarial Network)

**English:**
The GAN model consists of two neural networks:
- **Generator**: Creates synthetic payment data from random noise
- **Discriminator**: Distinguishes between real and synthetic data

The models compete in a minimax game, improving data generation quality over time.

**Español:**
El modelo GAN consiste en dos redes neuronales:
- **Generador**: Crea datos de pago sintéticos a partir de ruido aleatorio
- **Discriminador**: Distingue entre datos reales y sintéticos

Los modelos compiten en un juego minimax, mejorando la calidad de generación de datos con el tiempo.

### VAE (Variational Autoencoder)

**English:**
The VAE model uses:
- **Encoder**: Compresses payment data into a latent representation
- **Decoder**: Reconstructs payment data from latent space

VAE learns a probabilistic distribution of the data, allowing controlled generation.

**Español:**
El modelo VAE utiliza:
- **Codificador**: Comprime datos de pago en una representación latente
- **Decodificador**: Reconstruye datos de pago desde el espacio latente

VAE aprende una distribución probabilística de los datos, permitiendo generación controlada.

---

## 📊 Exploratory Data Analysis / Análisis Exploratorio de Datos

### Visualizations / Visualizaciones

The EDA module generates multiple visualization types:

1. **Amount Distribution / Distribución de Montos**
   - Histograms with log-scale transformations
   - Box plots by payment status
   - Violin plots for anomaly detection

2. **Temporal Patterns / Patrones Temporales**
   - Daily transaction volumes
   - Monthly aggregations
   - Day-of-week patterns
   - Anomaly trends over time

3. **Categorical Analysis / Análisis Categórico**
   - Payment method distributions
   - Currency breakdowns
   - Industry-specific patterns
   - Company size comparisons

4. **Correlation Analysis / Análisis de Correlación**
   - Feature correlation heatmaps
   - Anomaly correlation patterns

5. **Anomaly Analysis / Análisis de Anomalías**
   - Anomaly rates by industry
   - Anomaly rates by company size
   - Amount comparisons (normal vs anomaly)
   - Payment method anomaly patterns

---

## 📈 Results / Resultados

### English

The system generates:
- **10,000+ synthetic transactions** with realistic patterns
- **~5% anomaly rate** for fraud detection training
- **Multiple industries** represented (Technology, Healthcare, Finance, etc.)
- **Four company sizes** (Small, Medium, Large, Enterprise)
- **Multiple currencies** (USD, EUR, GBP)
- **Various payment methods** (Wire Transfer, ACH, Check, Credit Card)

Generated data can be used for:
- Training fraud detection models
- Testing payment processing systems
- Financial forecasting and analysis
- Risk assessment models
- Privacy-preserving data sharing

### Español

El sistema genera:
- **10,000+ transacciones sintéticas** con patrones realistas
- **~5% de tasa de anomalías** para entrenamiento de detección de fraude
- **Múltiples industrias** representadas (Tecnología, Salud, Finanzas, etc.)
- **Cuatro tamaños de empresas** (Pequeña, Mediana, Grande, Empresa)
- **Múltiples monedas** (USD, EUR, GBP)
- **Varios métodos de pago** (Transferencia, ACH, Cheque, Tarjeta de Crédito)

Los datos generados pueden usarse para:
- Entrenar modelos de detección de fraude
- Probar sistemas de procesamiento de pagos
- Pronóstico y análisis financiero
- Modelos de evaluación de riesgo
- Compartir datos preservando la privacidad

---

## 🤝 Contributing / Contribuir

### English
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Español
¡Las contribuciones son bienvenidas! Por favor, no dudes en enviar un Pull Request. Para cambios importantes, por favor abre un issue primero para discutir qué te gustaría cambiar.

---

## 📄 License / Licencia

This project is licensed under the MIT License - see the LICENSE file for details.

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo LICENSE para más detalles.

---

## 👥 Authors / Autores

MaxPelu - [GitHub](https://github.com/MaxPelu)

---

## 🙏 Acknowledgments / Agradecimientos

### English
- Built with PyTorch and TensorFlow
- Data visualization with Matplotlib and Seaborn
- Data processing with Pandas and NumPy
- Inspired by modern GAN and VAE architectures

### Español
- Construido con PyTorch y TensorFlow
- Visualización de datos con Matplotlib y Seaborn
- Procesamiento de datos con Pandas y NumPy
- Inspirado en arquitecturas modernas de GAN y VAE

---

## 📞 Contact / Contacto

For questions or suggestions, please open an issue on GitHub.

Para preguntas o sugerencias, por favor abre un issue en GitHub.
