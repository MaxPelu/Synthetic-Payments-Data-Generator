# Technical Architecture / Arquitectura Técnica
## Synthetic Payment Data Generator

### English

## System Architecture

The system consists of four main components:

### 1. Data Generation Layer (`data_generator.py`)

**Purpose**: Generate realistic synthetic B2B payment transaction data

**Key Features**:
- Company profile generation with Faker library
- Transaction simulation with temporal patterns
- Anomaly injection (configurable rate)
- Industry-specific patterns
- Company size-based amount generation

**Output**: CSV file with 19 columns including:
- Transaction metadata (ID, invoice, dates)
- Payer/Payee information (company, industry, size)
- Financial data (amount, currency, payment method)
- Status and anomaly flags

### 2. Exploratory Data Analysis (`eda_analysis.py`)

**Purpose**: Comprehensive analysis and visualization of payment data

**Visualizations**:
1. **Amount Distribution**: Histograms, log-scale, box plots, violin plots
2. **Temporal Patterns**: Daily/monthly volumes, day-of-week analysis
3. **Categorical Distributions**: Payment methods, currencies, industries
4. **Correlation Analysis**: Feature correlation heatmaps
5. **Anomaly Analysis**: Anomaly rates by various dimensions

**Technologies**: Pandas, Matplotlib, Seaborn

### 3. Generative Models (`models/`)

#### GAN Model (`gan_model.py`)

**Architecture**:
- **Generator**:
  - Input: Random latent vector (100 dimensions)
  - Hidden layers: [256, 512, 256] neurons
  - Output: Synthetic transaction features
  - Activation: ReLU + Tanh output
  - Batch normalization and dropout

- **Discriminator**:
  - Input: Transaction features
  - Hidden layers: [256, 128] neurons
  - Output: Real/Fake probability
  - Activation: LeakyReLU + Sigmoid output
  - Dropout for regularization

**Training**:
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999))
- Adversarial training loop

#### VAE Model (`vae_model.py`)

**Architecture**:
- **Encoder**:
  - Input: Transaction features
  - Hidden layers: [256, 128] neurons
  - Output: Mean and log-variance (64 dimensions)
  - Batch normalization and dropout

- **Decoder**:
  - Input: Latent vector (64 dimensions)
  - Hidden layers: [128, 256] neurons
  - Output: Reconstructed transaction features
  - Batch normalization and dropout

**Training**:
- Loss: Reconstruction (MSE) + KL Divergence
- Optimizer: Adam (lr=0.001)
- Beta-VAE formulation (β=1.0)

### 4. Training Pipeline (`train_pipeline.py`)

**Workflow**:
1. Data generation (configurable parameters)
2. Exploratory data analysis
3. GAN model training
4. VAE model training
5. Model comparison and evaluation

**Features**:
- Command-line interface with argparse
- Step skipping for partial execution
- Progress reporting
- Bilingual output

## Data Flow

```
User Input → Data Generator → Raw Data (CSV)
                ↓
            EDA Analysis → Visualizations (PNG)
                ↓
        Model Training (GAN/VAE) → Trained Models (.pth)
                ↓
        Sample Generation → Synthetic Data (CSV)
```

## Feature Engineering

**Numerical Features**:
- Amount (continuous, log-normal distribution)

**Categorical Features**:
- Industry (8 categories)
- Company size (4 categories)
- Currency (3 categories)
- Payment method (4 categories)
- Status (3 categories)

**Preprocessing**:
- Label encoding for categorical features
- Standard scaling (zero mean, unit variance)
- Feature normalization to [-1, 1] range

## Model Selection Rationale

### Why GAN?
- Excellent for generating realistic, diverse samples
- Adversarial training improves quality over time
- Can capture complex multimodal distributions
- Good for generating novel transactions

### Why VAE?
- Provides structured latent space
- Better for controlled generation
- Probabilistic framework
- Smoother interpolation between samples
- More stable training than GAN

## Performance Considerations

**Memory**:
- Data generation: ~100MB for 10K transactions
- Model training: ~2GB RAM (CPU) / ~4GB VRAM (GPU)
- Batch size adjustable based on available memory

**Training Time** (approximate):
- Data generation: <1 minute
- EDA: ~30 seconds
- GAN training (100 epochs): 10-20 minutes (CPU), 2-5 minutes (GPU)
- VAE training (100 epochs): 5-10 minutes (CPU), 1-3 minutes (GPU)

**Scalability**:
- Can generate millions of transactions
- Batch generation supported
- Models can be retrained on larger datasets

## Security & Privacy

**Data Privacy**:
- All generated data is synthetic
- No real transaction data used
- Safe for sharing and testing

**Security Scans**:
- CodeQL analysis: 0 vulnerabilities
- No external API dependencies for generation
- Deterministic with seed for reproducibility

---

### Español

## Arquitectura del Sistema

El sistema consiste en cuatro componentes principales:

### 1. Capa de Generación de Datos (`data_generator.py`)

**Propósito**: Generar datos sintéticos realistas de transacciones de pago B2B

**Características Clave**:
- Generación de perfiles de empresas con biblioteca Faker
- Simulación de transacciones con patrones temporales
- Inyección de anomalías (tasa configurable)
- Patrones específicos por industria
- Generación de montos basada en tamaño de empresa

**Salida**: Archivo CSV con 19 columnas incluyendo:
- Metadatos de transacción (ID, factura, fechas)
- Información Pagador/Receptor (empresa, industria, tamaño)
- Datos financieros (monto, moneda, método de pago)
- Estado y banderas de anomalía

### 2. Análisis Exploratorio de Datos (`eda_analysis.py`)

**Propósito**: Análisis y visualización completos de datos de pago

**Visualizaciones**:
1. **Distribución de Montos**: Histogramas, escala log, diagramas de caja
2. **Patrones Temporales**: Volúmenes diarios/mensuales, análisis por día
3. **Distribuciones Categóricas**: Métodos de pago, monedas, industrias
4. **Análisis de Correlación**: Mapas de calor de correlación
5. **Análisis de Anomalías**: Tasas de anomalía por varias dimensiones

**Tecnologías**: Pandas, Matplotlib, Seaborn

### 3. Modelos Generativos (`models/`)

#### Modelo GAN (`gan_model.py`)

**Arquitectura**:
- **Generador**:
  - Entrada: Vector latente aleatorio (100 dimensiones)
  - Capas ocultas: [256, 512, 256] neuronas
  - Salida: Características de transacción sintéticas
  - Activación: ReLU + salida Tanh
  - Normalización por lotes y dropout

- **Discriminador**:
  - Entrada: Características de transacción
  - Capas ocultas: [256, 128] neuronas
  - Salida: Probabilidad Real/Falso
  - Activación: LeakyReLU + salida Sigmoid
  - Dropout para regularización

**Entrenamiento**:
- Pérdida: Entropía Cruzada Binaria
- Optimizador: Adam (lr=0.0002, betas=(0.5, 0.999))
- Bucle de entrenamiento adversarial

#### Modelo VAE (`vae_model.py`)

**Arquitectura**:
- **Codificador**:
  - Entrada: Características de transacción
  - Capas ocultas: [256, 128] neuronas
  - Salida: Media y log-varianza (64 dimensiones)
  - Normalización por lotes y dropout

- **Decodificador**:
  - Entrada: Vector latente (64 dimensiones)
  - Capas ocultas: [128, 256] neuronas
  - Salida: Características reconstruidas
  - Normalización por lotes y dropout

**Entrenamiento**:
- Pérdida: Reconstrucción (MSE) + Divergencia KL
- Optimizador: Adam (lr=0.001)
- Formulación Beta-VAE (β=1.0)

### 4. Pipeline de Entrenamiento (`train_pipeline.py`)

**Flujo de Trabajo**:
1. Generación de datos (parámetros configurables)
2. Análisis exploratorio de datos
3. Entrenamiento modelo GAN
4. Entrenamiento modelo VAE
5. Comparación y evaluación de modelos

**Características**:
- Interfaz de línea de comandos con argparse
- Omisión de pasos para ejecución parcial
- Reporte de progreso
- Salida bilingüe

## Flujo de Datos

```
Entrada Usuario → Generador Datos → Datos Crudos (CSV)
                      ↓
                Análisis EDA → Visualizaciones (PNG)
                      ↓
          Entrenamiento Modelos → Modelos Entrenados (.pth)
                      ↓
          Generación Muestras → Datos Sintéticos (CSV)
```

## Consideraciones de Rendimiento

**Memoria**:
- Generación de datos: ~100MB para 10K transacciones
- Entrenamiento: ~2GB RAM (CPU) / ~4GB VRAM (GPU)
- Tamaño de lote ajustable según memoria disponible

**Tiempo de Entrenamiento** (aproximado):
- Generación de datos: <1 minuto
- EDA: ~30 segundos
- Entrenamiento GAN (100 épocas): 10-20 minutos (CPU), 2-5 minutos (GPU)
- Entrenamiento VAE (100 épocas): 5-10 minutos (CPU), 1-3 minutos (GPU)

**Escalabilidad**:
- Puede generar millones de transacciones
- Generación por lotes soportada
- Modelos pueden reentrenarse en datasets más grandes

## Seguridad y Privacidad

**Privacidad de Datos**:
- Todos los datos generados son sintéticos
- No se usan datos de transacciones reales
- Seguro para compartir y probar

**Escaneos de Seguridad**:
- Análisis CodeQL: 0 vulnerabilidades
- Sin dependencias de API externas para generación
- Determinístico con semilla para reproducibilidad
