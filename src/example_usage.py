"""
Example Usage Script for Synthetic Payment Data Generation
Script de Ejemplo de Uso para Generación de Datos de Pago Sintéticos

This script demonstrates how to use the trained models to generate synthetic data.
Este script demuestra cómo usar los modelos entrenados para generar datos sintéticos.
"""

import os
import sys
import pandas as pd
import torch

# Add parent directory to path for proper imports
# Agregar directorio padre al path para importaciones correctas
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.gan_model import PaymentGAN
from src.models.vae_model import PaymentVAE


def example_load_and_generate_gan():
    """
    Example: Load trained GAN and generate synthetic data.
    Ejemplo: Cargar GAN entrenado y generar datos sintéticos.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: GENERATING DATA WITH TRAINED GAN")
    print("EJEMPLO 1: GENERANDO DATOS CON GAN ENTRENADO")
    print("="*80 + "\n")
    
    # Check if model exists
    # Verificar si existe el modelo
    model_path = 'models/gan_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Error: Modelo no encontrado en {model_path}")
        print("Please run training first: python src/train_pipeline.py")
        print("Por favor ejecuta el entrenamiento primero: python src/train_pipeline.py")
        return
    
    # Initialize and load GAN
    # Inicializar y cargar GAN
    print("Loading GAN model...")
    print("Cargando modelo GAN...")
    gan = PaymentGAN(latent_dim=100)
    gan.load_model(model_path)
    
    # Generate synthetic samples
    # Generar muestras sintéticas
    print("\nGenerating 5000 synthetic payment transactions...")
    print("Generando 5000 transacciones de pago sintéticas...")
    
    synthetic_data = gan.generate_samples(n_samples=5000)
    
    # Display sample
    # Mostrar muestra
    print("\n=== Sample Generated Data / Muestra de Datos Generados ===")
    print(synthetic_data.head(10))
    
    # Display statistics
    # Mostrar estadísticas
    print("\n=== Statistics / Estadísticas ===")
    print(f"Total records / Total de registros: {len(synthetic_data)}")
    print(f"Amount range / Rango de montos: ${synthetic_data['amount'].min():.2f} - ${synthetic_data['amount'].max():.2f}")
    print(f"Mean amount / Monto promedio: ${synthetic_data['amount'].mean():.2f}")
    print(f"Median amount / Monto mediano: ${synthetic_data['amount'].median():.2f}")
    
    print("\n=== Payment Method Distribution / Distribución de Métodos de Pago ===")
    print(synthetic_data['payment_method'].value_counts())
    
    print("\n=== Currency Distribution / Distribución de Monedas ===")
    print(synthetic_data['currency'].value_counts())
    
    # Save to file
    # Guardar a archivo
    output_path = 'data/processed/example_gan_output.csv'
    synthetic_data.to_csv(output_path, index=False)
    print(f"\n✓ Generated data saved to / Datos generados guardados en: {output_path}")
    
    return synthetic_data


def example_load_and_generate_vae():
    """
    Example: Load trained VAE and generate synthetic data.
    Ejemplo: Cargar VAE entrenado y generar datos sintéticos.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: GENERATING DATA WITH TRAINED VAE")
    print("EJEMPLO 2: GENERANDO DATOS CON VAE ENTRENADO")
    print("="*80 + "\n")
    
    # Check if model exists
    # Verificar si existe el modelo
    model_path = 'models/vae_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Error: Modelo no encontrado en {model_path}")
        print("Please run training first: python src/train_pipeline.py")
        print("Por favor ejecuta el entrenamiento primero: python src/train_pipeline.py")
        return
    
    # Initialize and load VAE
    # Inicializar y cargar VAE
    print("Loading VAE model...")
    print("Cargando modelo VAE...")
    vae = PaymentVAE(latent_dim=64, hidden_dims=[256, 128])
    vae.load_model(model_path)
    
    # Generate synthetic samples
    # Generar muestras sintéticas
    print("\nGenerating 5000 synthetic payment transactions...")
    print("Generando 5000 transacciones de pago sintéticas...")
    
    synthetic_data = vae.generate_samples(n_samples=5000)
    
    # Display sample
    # Mostrar muestra
    print("\n=== Sample Generated Data / Muestra de Datos Generados ===")
    print(synthetic_data.head(10))
    
    # Display statistics
    # Mostrar estadísticas
    print("\n=== Statistics / Estadísticas ===")
    print(f"Total records / Total de registros: {len(synthetic_data)}")
    print(f"Amount range / Rango de montos: ${synthetic_data['amount'].min():.2f} - ${synthetic_data['amount'].max():.2f}")
    print(f"Mean amount / Monto promedio: ${synthetic_data['amount'].mean():.2f}")
    print(f"Median amount / Monto mediano: ${synthetic_data['amount'].median():.2f}")
    
    print("\n=== Industry Distribution / Distribución de Industrias ===")
    print("Payer industries / Industrias pagadoras:")
    print(synthetic_data['payer_industry'].value_counts())
    
    print("\n=== Company Size Distribution / Distribución de Tamaño de Empresas ===")
    print("Payer sizes / Tamaños de pagadores:")
    print(synthetic_data['payer_size'].value_counts())
    
    # Save to file
    # Guardar a archivo
    output_path = 'data/processed/example_vae_output.csv'
    synthetic_data.to_csv(output_path, index=False)
    print(f"\n✓ Generated data saved to / Datos generados guardados en: {output_path}")
    
    return synthetic_data


def example_batch_generation():
    """
    Example: Generate large batches of synthetic data.
    Ejemplo: Generar grandes lotes de datos sintéticos.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: BATCH GENERATION OF SYNTHETIC DATA")
    print("EJEMPLO 3: GENERACIÓN POR LOTES DE DATOS SINTÉTICOS")
    print("="*80 + "\n")
    
    # Check if models exist
    # Verificar si existen los modelos
    gan_path = 'models/gan_model.pth'
    vae_path = 'models/vae_model.pth'
    
    if not os.path.exists(gan_path) or not os.path.exists(vae_path):
        print("Error: Models not found. Please train models first.")
        print("Error: Modelos no encontrados. Por favor entrena los modelos primero.")
        return
    
    # Load models
    # Cargar modelos
    print("Loading models...")
    print("Cargando modelos...")
    gan = PaymentGAN(latent_dim=100)
    gan.load_model(gan_path)
    
    vae = PaymentVAE(latent_dim=64, hidden_dims=[256, 128])
    vae.load_model(vae_path)
    
    # Generate multiple batches
    # Generar múltiples lotes
    n_batches = 5
    batch_size = 2000
    
    print(f"\nGenerating {n_batches} batches of {batch_size} records each...")
    print(f"Generando {n_batches} lotes de {batch_size} registros cada uno...")
    
    all_gan_data = []
    all_vae_data = []
    
    for i in range(n_batches):
        print(f"\nBatch {i+1}/{n_batches}...")
        
        # GAN generation
        gan_batch = gan.generate_samples(n_samples=batch_size)
        all_gan_data.append(gan_batch)
        
        # VAE generation
        vae_batch = vae.generate_samples(n_samples=batch_size)
        all_vae_data.append(vae_batch)
    
    # Combine all batches
    # Combinar todos los lotes
    final_gan_data = pd.concat(all_gan_data, ignore_index=True)
    final_vae_data = pd.concat(all_vae_data, ignore_index=True)
    
    print(f"\n✓ Total GAN records generated / Total de registros GAN generados: {len(final_gan_data)}")
    print(f"✓ Total VAE records generated / Total de registros VAE generados: {len(final_vae_data)}")
    
    # Save large datasets
    # Guardar conjuntos de datos grandes
    final_gan_data.to_csv('data/processed/large_batch_gan.csv', index=False)
    final_vae_data.to_csv('data/processed/large_batch_vae.csv', index=False)
    
    print("\n✓ Large batch data saved successfully!")
    print("✓ ¡Datos de lotes grandes guardados exitosamente!")


def main():
    """
    Run all example scenarios.
    Ejecutar todos los escenarios de ejemplo.
    """
    print("\n" + "="*80)
    print("SYNTHETIC PAYMENT DATA GENERATOR - USAGE EXAMPLES")
    print("GENERADOR DE DATOS DE PAGO SINTÉTICOS - EJEMPLOS DE USO")
    print("="*80)
    
    # Example 1: GAN generation
    # Ejemplo 1: Generación con GAN
    try:
        example_load_and_generate_gan()
    except Exception as e:
        print(f"\nError in Example 1 / Error en Ejemplo 1: {e}")
    
    # Example 2: VAE generation
    # Ejemplo 2: Generación con VAE
    try:
        example_load_and_generate_vae()
    except Exception as e:
        print(f"\nError in Example 2 / Error en Ejemplo 2: {e}")
    
    # Example 3: Batch generation
    # Ejemplo 3: Generación por lotes
    try:
        example_batch_generation()
    except Exception as e:
        print(f"\nError in Example 3 / Error en Ejemplo 3: {e}")
    
    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETED!")
    print("✓ ¡TODOS LOS EJEMPLOS COMPLETADOS!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
