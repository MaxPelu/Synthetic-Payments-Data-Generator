"""
GAN (Generative Adversarial Network) for Synthetic Payment Data Generation
GAN (Red Generativa Antagónica) para Generación de Datos de Pago Sintéticos

This module implements a GAN model to generate synthetic payment transaction data.
Este módulo implementa un modelo GAN para generar datos sintéticos de transacciones de pago.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


class PaymentDataset(Dataset):
    """
    PyTorch Dataset for payment transaction data.
    Dataset de PyTorch para datos de transacciones de pago.
    """
    
    def __init__(self, data):
        """
        Initialize the dataset.
        Inicializar el dataset.
        
        Args:
            data (np.ndarray): Normalized transaction data
                              Datos de transacción normalizados
        """
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Generator(nn.Module):
    """
    Generator network for GAN.
    Red generadora para GAN.
    """
    
    def __init__(self, latent_dim, output_dim, hidden_dims=[256, 512, 256]):
        """
        Initialize the generator.
        Inicializar el generador.
        
        Args:
            latent_dim (int): Dimension of latent space
                             Dimensión del espacio latente
            output_dim (int): Dimension of output data
                             Dimensión de los datos de salida
            hidden_dims (list): Dimensions of hidden layers
                               Dimensiones de las capas ocultas
        """
        super(Generator, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Output normalization
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Forward pass through generator.
        Paso hacia adelante por el generador.
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network for GAN.
    Red discriminadora para GAN.
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        """
        Initialize the discriminator.
        Inicializar el discriminador.
        
        Args:
            input_dim (int): Dimension of input data
                            Dimensión de los datos de entrada
            hidden_dims (list): Dimensions of hidden layers
                               Dimensiones de las capas ocultas
        """
        super(Discriminator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through discriminator.
        Paso hacia adelante por el discriminador.
        """
        return self.model(x)


class PaymentGAN:
    """
    GAN model for generating synthetic payment data.
    Modelo GAN para generar datos de pago sintéticos.
    """
    
    def __init__(self, latent_dim=100, hidden_dims_g=[256, 512, 256], 
                 hidden_dims_d=[256, 128], learning_rate=0.0002, device=None):
        """
        Initialize the GAN model.
        Inicializar el modelo GAN.
        
        Args:
            latent_dim (int): Dimension of latent space
                             Dimensión del espacio latente
            hidden_dims_g (list): Hidden dimensions for generator
                                 Dimensiones ocultas para generador
            hidden_dims_d (list): Hidden dimensions for discriminator
                                 Dimensiones ocultas para discriminador
            learning_rate (float): Learning rate for optimizers
                                  Tasa de aprendizaje para optimizadores
            device (str): Device to run model on ('cuda' or 'cpu')
                         Dispositivo para ejecutar modelo
        """
        self.latent_dim = latent_dim
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.output_dim = None
        
        self.generator = None
        self.discriminator = None
        self.hidden_dims_g = hidden_dims_g
        self.hidden_dims_d = hidden_dims_d
        self.learning_rate = learning_rate
        
        print(f"Using device / Usando dispositivo: {self.device}")
    
    def preprocess_data(self, df):
        """
        Preprocess payment data for training.
        Preprocesar datos de pago para entrenamiento.
        
        Args:
            df (pd.DataFrame): Raw payment data / Datos de pago crudos
        
        Returns:
            np.ndarray: Preprocessed and normalized data
                       Datos preprocesados y normalizados
        """
        df = df.copy()
        
        # Select numerical and categorical features
        # Seleccionar características numéricas y categóricas
        numerical_features = ['amount']
        categorical_features = ['payer_industry', 'payer_size', 'payee_industry', 
                               'payee_size', 'currency', 'payment_method', 'status']
        
        processed_data = []
        
        # Process numerical features
        # Procesar características numéricas
        for feature in numerical_features:
            if feature in df.columns:
                processed_data.append(df[feature].values.reshape(-1, 1))
        
        # Process categorical features with label encoding
        # Procesar características categóricas con codificación de etiquetas
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    encoded = self.label_encoders[feature].fit_transform(df[feature])
                else:
                    encoded = self.label_encoders[feature].transform(df[feature])
                processed_data.append(encoded.reshape(-1, 1))
        
        # Concatenate all features
        # Concatenar todas las características
        processed_data = np.concatenate(processed_data, axis=1)
        
        # Normalize data
        # Normalizar datos
        normalized_data = self.scaler.fit_transform(processed_data)
        
        self.output_dim = normalized_data.shape[1]
        self.feature_names = numerical_features + categorical_features
        
        return normalized_data
    
    def build_models(self):
        """
        Build generator and discriminator models.
        Construir modelos generador y discriminador.
        """
        self.generator = Generator(
            self.latent_dim, 
            self.output_dim, 
            self.hidden_dims_g
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.output_dim, 
            self.hidden_dims_d
        ).to(self.device)
        
        print("Models built successfully / Modelos construidos exitosamente")
        print(f"Generator parameters / Parámetros del generador: "
              f"{sum(p.numel() for p in self.generator.parameters())}")
        print(f"Discriminator parameters / Parámetros del discriminador: "
              f"{sum(p.numel() for p in self.discriminator.parameters())}")
    
    def train(self, df, epochs=100, batch_size=64, save_path='models/gan_model.pth'):
        """
        Train the GAN model.
        Entrenar el modelo GAN.
        
        Args:
            df (pd.DataFrame): Training data / Datos de entrenamiento
            epochs (int): Number of training epochs / Número de épocas
            batch_size (int): Batch size / Tamaño de lote
            save_path (str): Path to save model / Ruta para guardar modelo
        """
        # Preprocess data
        # Preprocesar datos
        data = self.preprocess_data(df)
        
        # Build models if not already built
        # Construir modelos si no están construidos
        if self.generator is None or self.discriminator is None:
            self.build_models()
        
        # Create dataset and dataloader
        # Crear dataset y dataloader
        dataset = PaymentDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizers
        # Función de pérdida y optimizadores
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), 
                                lr=self.learning_rate, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), 
                                lr=self.learning_rate, betas=(0.5, 0.999))
        
        # Training loop
        # Bucle de entrenamiento
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Iniciando entrenamiento por {epochs} épocas...")
        
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            
            for real_data in dataloader:
                batch_size_current = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # Train Discriminator
                # Entrenar Discriminador
                optimizer_d.zero_grad()
                
                # Real data
                # Datos reales
                real_labels = torch.ones(batch_size_current, 1).to(self.device)
                real_output = self.discriminator(real_data)
                d_loss_real = criterion(real_output, real_labels)
                
                # Fake data
                # Datos falsos
                z = torch.randn(batch_size_current, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                fake_labels = torch.zeros(batch_size_current, 1).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                # Pérdida total del discriminador
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()
                
                # Train Generator
                # Entrenar Generador
                optimizer_g.zero_grad()
                
                z = torch.randn(batch_size_current, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                optimizer_g.step()
                
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
            
            # Print progress
            # Imprimir progreso
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"D Loss: {np.mean(d_losses):.4f}, "
                      f"G Loss: {np.mean(g_losses):.4f}")
        
        # Save model
        # Guardar modelo
        self.save_model(save_path)
        print(f"\nTraining completed! Model saved to / "
              f"¡Entrenamiento completado! Modelo guardado en: {save_path}")
    
    def generate_samples(self, n_samples=1000):
        """
        Generate synthetic samples using trained generator.
        Generar muestras sintéticas usando el generador entrenado.
        
        Args:
            n_samples (int): Number of samples to generate
                            Número de muestras a generar
        
        Returns:
            pd.DataFrame: Generated synthetic data
                         Datos sintéticos generados
        """
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            generated_data = self.generator(z).cpu().numpy()
        
        # Denormalize data
        # Desnormalizar datos
        generated_data = self.scaler.inverse_transform(generated_data)
        
        # Create DataFrame
        # Crear DataFrame
        df_generated = pd.DataFrame(generated_data, columns=self.feature_names)
        
        # Decode categorical features
        # Decodificar características categóricas
        for feature, encoder in self.label_encoders.items():
            if feature in df_generated.columns:
                # Round and clip to valid range
                # Redondear y limitar a rango válido
                encoded_values = np.round(df_generated[feature].values).astype(int)
                encoded_values = np.clip(encoded_values, 0, len(encoder.classes_) - 1)
                df_generated[feature] = encoder.inverse_transform(encoded_values)
        
        # Ensure amount is positive
        # Asegurar que el monto sea positivo
        df_generated['amount'] = np.abs(df_generated['amount'])
        
        return df_generated
    
    def save_model(self, path='models/gan_model.pth'):
        """
        Save the trained model and preprocessors.
        Guardar el modelo entrenado y preprocesadores.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'hidden_dims_g': self.hidden_dims_g,
            'hidden_dims_d': self.hidden_dims_d,
            'feature_names': self.feature_names
        }, path)
        
        # Save preprocessors
        # Guardar preprocesadores
        with open(path.replace('.pth', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(path.replace('.pth', '_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
    
    def load_model(self, path='models/gan_model.pth'):
        """
        Load a trained model and preprocessors.
        Cargar un modelo entrenado y preprocesadores.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.latent_dim = checkpoint['latent_dim']
        self.output_dim = checkpoint['output_dim']
        self.hidden_dims_g = checkpoint['hidden_dims_g']
        self.hidden_dims_d = checkpoint['hidden_dims_d']
        self.feature_names = checkpoint['feature_names']
        
        self.build_models()
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load preprocessors
        # Cargar preprocesadores
        with open(path.replace('.pth', '_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(path.replace('.pth', '_encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        print(f"Model loaded from / Modelo cargado desde: {path}")


def main():
    """
    Main function to train GAN model.
    Función principal para entrenar modelo GAN.
    """
    print("Training GAN for synthetic payment data generation...")
    print("Entrenando GAN para generación de datos de pago sintéticos...")
    
    # Load training data
    # Cargar datos de entrenamiento
    df = pd.read_csv('data/raw/synthetic_payments.csv')
    
    # Initialize and train GAN
    # Inicializar y entrenar GAN
    gan = PaymentGAN(latent_dim=100, learning_rate=0.0002)
    gan.train(df, epochs=100, batch_size=64)
    
    # Generate samples
    # Generar muestras
    print("\nGenerating synthetic samples...")
    print("Generando muestras sintéticas...")
    synthetic_data = gan.generate_samples(n_samples=1000)
    
    # Save synthetic data
    # Guardar datos sintéticos
    synthetic_data.to_csv('data/processed/gan_generated_payments.csv', index=False)
    print("Synthetic data saved to / Datos sintéticos guardados en: "
          "data/processed/gan_generated_payments.csv")


if __name__ == '__main__':
    main()
