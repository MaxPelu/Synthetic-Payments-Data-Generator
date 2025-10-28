"""
Quick test script to verify models can be instantiated and run.
Script de prueba rápida para verificar que los modelos pueden ser instanciados y ejecutados.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.gan_model import PaymentGAN
from models.vae_model import PaymentVAE

def test_gan():
    """Test GAN model instantiation and quick training."""
    print("\n=== Testing GAN Model ===")
    print("=== Probando Modelo GAN ===")
    
    # Load data
    df = pd.read_csv('data/raw/synthetic_payments.csv')
    print(f"Loaded {len(df)} transactions")
    
    # Use a small subset for quick testing
    df_small = df.head(500)
    
    # Initialize GAN
    gan = PaymentGAN(latent_dim=50, hidden_dims_g=[128, 256, 128], hidden_dims_d=[128, 64])
    print("✓ GAN initialized successfully")
    
    # Quick training (just 2 epochs for testing)
    print("Training for 2 epochs (quick test)...")
    gan.train(df_small, epochs=2, batch_size=32, save_path='models/test_gan_model.pth')
    print("✓ GAN training completed")
    
    # Generate samples
    print("Generating samples...")
    samples = gan.generate_samples(n_samples=10)
    print("✓ Generated samples:")
    print(samples.head())
    
    return True

def test_vae():
    """Test VAE model instantiation and quick training."""
    print("\n=== Testing VAE Model ===")
    print("=== Probando Modelo VAE ===")
    
    # Load data
    df = pd.read_csv('data/raw/synthetic_payments.csv')
    print(f"Loaded {len(df)} transactions")
    
    # Use a small subset for quick testing
    df_small = df.head(500)
    
    # Initialize VAE
    vae = PaymentVAE(latent_dim=32, hidden_dims=[128, 64])
    print("✓ VAE initialized successfully")
    
    # Quick training (just 2 epochs for testing)
    print("Training for 2 epochs (quick test)...")
    vae.train(df_small, epochs=2, batch_size=32, save_path='models/test_vae_model.pth')
    print("✓ VAE training completed")
    
    # Generate samples
    print("Generating samples...")
    samples = vae.generate_samples(n_samples=10)
    print("✓ Generated samples:")
    print(samples.head())
    
    return True

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING MODEL TESTS")
    print("EJECUTANDO PRUEBAS DE MODELOS")
    print("="*80)
    
    try:
        gan_success = test_gan()
        print("\n✓ GAN test passed!")
    except Exception as e:
        print(f"\n✗ GAN test failed: {e}")
        gan_success = False
    
    try:
        vae_success = test_vae()
        print("\n✓ VAE test passed!")
    except Exception as e:
        print(f"\n✗ VAE test failed: {e}")
        vae_success = False
    
    print("\n" + "="*80)
    if gan_success and vae_success:
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("✓✓✓ ¡TODAS LAS PRUEBAS PASARON! ✓✓✓")
    else:
        print("✗ SOME TESTS FAILED")
        print("✗ ALGUNAS PRUEBAS FALLARON")
    print("="*80)

if __name__ == '__main__':
    main()
