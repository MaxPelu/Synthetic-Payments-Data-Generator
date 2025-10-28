"""
Complete Training Pipeline for Synthetic Payment Data Generation
Pipeline Completo de Entrenamiento para Generación de Datos de Pago Sintéticos

This script orchestrates the entire workflow from data generation to model training.
Este script orquesta todo el flujo de trabajo desde generación de datos hasta entrenamiento de modelos.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add src to path
# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_generator import PaymentDataGenerator
from eda_analysis import PaymentEDA
from models.gan_model import PaymentGAN
from models.vae_model import PaymentVAE


def ensure_directories():
    """
    Ensure all necessary directories exist.
    Asegurar que todos los directorios necesarios existan.
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'visualizations'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready / Directorio listo: {directory}")


def step_generate_initial_data(n_transactions=10000, n_companies=150, anomaly_rate=0.05):
    """
    Step 1: Generate initial synthetic payment data.
    Paso 1: Generar datos de pago sintéticos iniciales.
    """
    print("\n" + "="*80)
    print("STEP 1: GENERATING INITIAL SYNTHETIC DATA")
    print("PASO 1: GENERANDO DATOS SINTÉTICOS INICIALES")
    print("="*80 + "\n")
    
    generator = PaymentDataGenerator(n_companies=n_companies, seed=42)
    df = generator.generate_transactions(
        n_transactions=n_transactions,
        start_date='2023-01-01',
        end_date='2024-12-31',
        anomaly_rate=anomaly_rate
    )
    
    output_path = 'data/raw/synthetic_payments.csv'
    generator.save_data(df, output_path)
    
    print(f"\n✓ Initial data generated successfully!")
    print(f"✓ ¡Datos iniciales generados exitosamente!")
    print(f"  File / Archivo: {output_path}")
    print(f"  Records / Registros: {len(df)}")
    print(f"  Anomalies / Anomalías: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
    
    return df


def step_run_eda(df):
    """
    Step 2: Run exploratory data analysis.
    Paso 2: Ejecutar análisis exploratorio de datos.
    """
    print("\n" + "="*80)
    print("STEP 2: RUNNING EXPLORATORY DATA ANALYSIS")
    print("PASO 2: EJECUTANDO ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*80 + "\n")
    
    eda = PaymentEDA(df, output_dir='visualizations')
    eda.run_full_eda()
    
    print("\n✓ EDA completed successfully!")
    print("✓ ¡EDA completado exitosamente!")


def step_train_gan(df, epochs=100, batch_size=64):
    """
    Step 3: Train GAN model.
    Paso 3: Entrenar modelo GAN.
    """
    print("\n" + "="*80)
    print("STEP 3: TRAINING GAN MODEL")
    print("PASO 3: ENTRENANDO MODELO GAN")
    print("="*80 + "\n")
    
    gan = PaymentGAN(latent_dim=100, learning_rate=0.0002)
    gan.train(df, epochs=epochs, batch_size=batch_size, save_path='models/gan_model.pth')
    
    # Generate samples
    # Generar muestras
    print("\nGenerating synthetic samples with GAN...")
    print("Generando muestras sintéticas con GAN...")
    synthetic_data = gan.generate_samples(n_samples=1000)
    synthetic_data.to_csv('data/processed/gan_generated_payments.csv', index=False)
    
    print("\n✓ GAN training completed successfully!")
    print("✓ ¡Entrenamiento de GAN completado exitosamente!")
    print(f"  Model saved / Modelo guardado: models/gan_model.pth")
    print(f"  Synthetic data / Datos sintéticos: data/processed/gan_generated_payments.csv")
    
    return gan, synthetic_data


def step_train_vae(df, epochs=100, batch_size=64):
    """
    Step 4: Train VAE model.
    Paso 4: Entrenar modelo VAE.
    """
    print("\n" + "="*80)
    print("STEP 4: TRAINING VAE MODEL")
    print("PASO 4: ENTRENANDO MODELO VAE")
    print("="*80 + "\n")
    
    vae = PaymentVAE(latent_dim=64, hidden_dims=[256, 128], learning_rate=0.001)
    vae.train(df, epochs=epochs, batch_size=batch_size, beta=1.0, save_path='models/vae_model.pth')
    
    # Generate samples
    # Generar muestras
    print("\nGenerating synthetic samples with VAE...")
    print("Generando muestras sintéticas con VAE...")
    synthetic_data = vae.generate_samples(n_samples=1000)
    synthetic_data.to_csv('data/processed/vae_generated_payments.csv', index=False)
    
    print("\n✓ VAE training completed successfully!")
    print("✓ ¡Entrenamiento de VAE completado exitosamente!")
    print(f"  Model saved / Modelo guardado: models/vae_model.pth")
    print(f"  Synthetic data / Datos sintéticos: data/processed/vae_generated_payments.csv")
    
    return vae, synthetic_data


def step_compare_models():
    """
    Step 5: Compare original, GAN, and VAE generated data.
    Paso 5: Comparar datos originales, GAN y VAE generados.
    """
    print("\n" + "="*80)
    print("STEP 5: COMPARING MODELS")
    print("PASO 5: COMPARANDO MODELOS")
    print("="*80 + "\n")
    
    # Load all datasets
    # Cargar todos los conjuntos de datos
    original = pd.read_csv('data/raw/synthetic_payments.csv')
    gan_data = pd.read_csv('data/processed/gan_generated_payments.csv')
    vae_data = pd.read_csv('data/processed/vae_generated_payments.csv')
    
    print("=== Dataset Comparison / Comparación de Conjuntos de Datos ===\n")
    
    print("Original Data / Datos Originales:")
    print(f"  Records / Registros: {len(original)}")
    print(f"  Amount range / Rango de montos: ${original['amount'].min():.2f} - ${original['amount'].max():.2f}")
    print(f"  Mean amount / Monto promedio: ${original['amount'].mean():.2f}")
    print(f"  Std amount / Desviación estándar de monto: ${original['amount'].std():.2f}")
    
    print("\nGAN Generated Data / Datos Generados por GAN:")
    print(f"  Records / Registros: {len(gan_data)}")
    print(f"  Amount range / Rango de montos: ${gan_data['amount'].min():.2f} - ${gan_data['amount'].max():.2f}")
    print(f"  Mean amount / Monto promedio: ${gan_data['amount'].mean():.2f}")
    print(f"  Std amount / Desviación estándar de monto: ${gan_data['amount'].std():.2f}")
    
    print("\nVAE Generated Data / Datos Generados por VAE:")
    print(f"  Records / Registros: {len(vae_data)}")
    print(f"  Amount range / Rango de montos: ${vae_data['amount'].min():.2f} - ${vae_data['amount'].max():.2f}")
    print(f"  Mean amount / Monto promedio: ${vae_data['amount'].mean():.2f}")
    print(f"  Std amount / Desviación estándar de monto: ${vae_data['amount'].std():.2f}")
    
    print("\n✓ Model comparison completed!")
    print("✓ ¡Comparación de modelos completada!")


def main():
    """
    Main training pipeline.
    Pipeline principal de entrenamiento.
    """
    parser = argparse.ArgumentParser(
        description='Train synthetic payment data generation models / '
                   'Entrenar modelos de generación de datos de pago sintéticos'
    )
    parser.add_argument('--n-transactions', type=int, default=10000,
                       help='Number of initial transactions to generate / '
                            'Número de transacciones iniciales a generar')
    parser.add_argument('--n-companies', type=int, default=150,
                       help='Number of companies to simulate / '
                            'Número de empresas a simular')
    parser.add_argument('--anomaly-rate', type=float, default=0.05,
                       help='Proportion of anomalous transactions / '
                            'Proporción de transacciones anómalas')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs / '
                            'Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training / '
                            'Tamaño de lote para entrenamiento')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip data generation step / '
                            'Omitir paso de generación de datos')
    parser.add_argument('--skip-eda', action='store_true',
                       help='Skip EDA step / '
                            'Omitir paso de EDA')
    parser.add_argument('--skip-gan', action='store_true',
                       help='Skip GAN training / '
                            'Omitir entrenamiento de GAN')
    parser.add_argument('--skip-vae', action='store_true',
                       help='Skip VAE training / '
                            'Omitir entrenamiento de VAE')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SYNTHETIC PAYMENT DATA GENERATOR - COMPLETE PIPELINE")
    print("GENERADOR DE DATOS DE PAGO SINTÉTICOS - PIPELINE COMPLETO")
    print("="*80)
    
    # Ensure directories exist
    # Asegurar que los directorios existan
    ensure_directories()
    
    # Step 1: Generate initial data
    # Paso 1: Generar datos iniciales
    if not args.skip_data_gen:
        df = step_generate_initial_data(
            n_transactions=args.n_transactions,
            n_companies=args.n_companies,
            anomaly_rate=args.anomaly_rate
        )
    else:
        print("\n⊘ Skipping data generation (using existing data)")
        print("⊘ Omitiendo generación de datos (usando datos existentes)")
        df = pd.read_csv('data/raw/synthetic_payments.csv')
    
    # Step 2: Run EDA
    # Paso 2: Ejecutar EDA
    if not args.skip_eda:
        step_run_eda(df)
    else:
        print("\n⊘ Skipping EDA")
        print("⊘ Omitiendo EDA")
    
    # Step 3: Train GAN
    # Paso 3: Entrenar GAN
    if not args.skip_gan:
        gan, gan_data = step_train_gan(df, epochs=args.epochs, batch_size=args.batch_size)
    else:
        print("\n⊘ Skipping GAN training")
        print("⊘ Omitiendo entrenamiento de GAN")
    
    # Step 4: Train VAE
    # Paso 4: Entrenar VAE
    if not args.skip_vae:
        vae, vae_data = step_train_vae(df, epochs=args.epochs, batch_size=args.batch_size)
    else:
        print("\n⊘ Skipping VAE training")
        print("⊘ Omitiendo entrenamiento de VAE")
    
    # Step 5: Compare models
    # Paso 5: Comparar modelos
    if not args.skip_gan and not args.skip_vae:
        step_compare_models()
    
    print("\n" + "="*80)
    print("✓✓✓ PIPELINE COMPLETED SUCCESSFULLY! ✓✓✓")
    print("✓✓✓ ¡PIPELINE COMPLETADO EXITOSAMENTE! ✓✓✓")
    print("="*80 + "\n")
    
    print("Generated files / Archivos generados:")
    print("  - data/raw/synthetic_payments.csv")
    print("  - data/processed/gan_generated_payments.csv")
    print("  - data/processed/vae_generated_payments.csv")
    print("  - models/gan_model.pth")
    print("  - models/vae_model.pth")
    print("  - visualizations/*.png")
    print("\nYou can now use these models to generate more synthetic payment data!")
    print("¡Ahora puedes usar estos modelos para generar más datos de pago sintéticos!")


if __name__ == '__main__':
    main()
