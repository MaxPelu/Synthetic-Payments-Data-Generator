"""
VAE (Variational Autoencoder) for Synthetic Payment Data Generation
VAE (Autocodificador Variacional) para Generación de Datos de Pago Sintéticos

This module implements a VAE model to generate synthetic payment transaction data.
Este módulo implementa un modelo VAE para generar datos sintéticos de transacciones de pago.
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
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class VAE(nn.Module):
    """
    Variational Autoencoder for payment data generation.
    Autocodificador Variacional para generación de datos de pago.
    """
    
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[256, 128]):
        """
        Initialize the VAE.
        Inicializar el VAE.
        
        Args:
            input_dim (int): Dimension of input data
                            Dimensión de los datos de entrada
            latent_dim (int): Dimension of latent space
                             Dimensión del espacio latente
            hidden_dims (list): Dimensions of hidden layers
                               Dimensiones de las capas ocultas
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder layers
        # Capas del codificador
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        # Parámetros del espacio latente
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layers
        # Capas del decodificador
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input to latent space parameters.
        Codificar entrada a parámetros del espacio latente.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent distribution.
        Truco de reparametrización para muestreo de distribución latente.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent representation to output space.
        Decodificar representación latente al espacio de salida.
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        Paso hacia adelante por el VAE.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


class PaymentVAE:
    """
    VAE model wrapper for generating synthetic payment data.
    Envoltorio del modelo VAE para generar datos de pago sintéticos.
    """
    
    def __init__(self, latent_dim=64, hidden_dims=[256, 128], 
                 learning_rate=0.001, device=None):
        """
        Initialize the VAE model.
        Inicializar el modelo VAE.
        
        Args:
            latent_dim (int): Dimension of latent space
                             Dimensión del espacio latente
            hidden_dims (list): Hidden dimensions for encoder/decoder
                               Dimensiones ocultas para codificador/decodificador
            learning_rate (float): Learning rate for optimizer
                                  Tasa de aprendizaje para optimizador
            device (str): Device to run model on ('cuda' or 'cpu')
                         Dispositivo para ejecutar modelo
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.input_dim = None
        self.model = None
        
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
        
        self.input_dim = normalized_data.shape[1]
        self.feature_names = numerical_features + categorical_features
        
        return normalized_data
    
    def vae_loss(self, reconstructed, original, mu, logvar, beta=1.0):
        """
        VAE loss function combining reconstruction and KL divergence.
        Función de pérdida VAE combinando reconstrucción y divergencia KL.
        
        Args:
            reconstructed: Reconstructed data / Datos reconstruidos
            original: Original data / Datos originales
            mu: Mean of latent distribution / Media de distribución latente
            logvar: Log variance of latent distribution / Log varianza de distribución latente
            beta: Weight for KL divergence term / Peso para término de divergencia KL
        """
        # Reconstruction loss (MSE)
        # Pérdida de reconstrucción (MSE)
        recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
        
        # KL divergence loss
        # Pérdida de divergencia KL
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss
    
    def train(self, df, epochs=100, batch_size=64, beta=1.0, save_path='models/vae_model.pth'):
        """
        Train the VAE model.
        Entrenar el modelo VAE.
        
        Args:
            df (pd.DataFrame): Training data / Datos de entrenamiento
            epochs (int): Number of training epochs / Número de épocas
            batch_size (int): Batch size / Tamaño de lote
            beta: Weight for KL divergence / Peso para divergencia KL
            save_path (str): Path to save model / Ruta para guardar modelo
        """
        # Preprocess data
        # Preprocesar datos
        data = self.preprocess_data(df)
        
        # Create model
        # Crear modelo
        self.model = VAE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        print("Model built successfully / Modelo construido exitosamente")
        print(f"Model parameters / Parámetros del modelo: "
              f"{sum(p.numel() for p in self.model.parameters())}")
        
        # Create dataset and dataloader
        # Crear dataset y dataloader
        dataset = PaymentDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        # Optimizador
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        # Bucle de entrenamiento
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Iniciando entrenamiento por {epochs} épocas...")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                # Paso hacia adelante
                reconstructed, mu, logvar = self.model(batch_data)
                
                # Calculate loss components
                # Calcular componentes de pérdida
                recon_loss = nn.functional.mse_loss(
                    reconstructed, batch_data, reduction='sum'
                )
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss
                
                # Backward pass
                # Paso hacia atrás
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            # Print progress
            # Imprimir progreso
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataset)
                avg_recon = total_recon_loss / len(dataset)
                avg_kl = total_kl_loss / len(dataset)
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Total Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon:.4f}, "
                      f"KL Loss: {avg_kl:.4f}")
        
        # Save model
        # Guardar modelo
        self.save_model(save_path)
        print(f"\nTraining completed! Model saved to / "
              f"¡Entrenamiento completado! Modelo guardado en: {save_path}")
    
    def generate_samples(self, n_samples=1000):
        """
        Generate synthetic samples using trained VAE.
        Generar muestras sintéticas usando el VAE entrenado.
        
        Args:
            n_samples (int): Number of samples to generate
                            Número de muestras a generar
        
        Returns:
            pd.DataFrame: Generated synthetic data
                         Datos sintéticos generados
        """
        self.model.eval()
        
        with torch.no_grad():
            # Sample from standard normal distribution
            # Muestrear de distribución normal estándar
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            generated_data = self.model.decode(z).cpu().numpy()
        
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
    
    def save_model(self, path='models/vae_model.pth'):
        """
        Save the trained model and preprocessors.
        Guardar el modelo entrenado y preprocesadores.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'feature_names': self.feature_names
        }, path)
        
        # Save preprocessors
        # Guardar preprocesadores
        with open(path.replace('.pth', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(path.replace('.pth', '_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
    
    def load_model(self, path='models/vae_model.pth'):
        """
        Load a trained model and preprocessors.
        Cargar un modelo entrenado y preprocesadores.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.latent_dim = checkpoint['latent_dim']
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.feature_names = checkpoint['feature_names']
        
        self.model = VAE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load preprocessors
        # Cargar preprocesadores
        with open(path.replace('.pth', '_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(path.replace('.pth', '_encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        print(f"Model loaded from / Modelo cargado desde: {path}")


def main():
    """
    Main function to train VAE model.
    Función principal para entrenar modelo VAE.
    """
    print("Training VAE for synthetic payment data generation...")
    print("Entrenando VAE para generación de datos de pago sintéticos...")
    
    # Load training data
    # Cargar datos de entrenamiento
    df = pd.read_csv('data/raw/synthetic_payments.csv')
    
    # Initialize and train VAE
    # Inicializar y entrenar VAE
    vae = PaymentVAE(latent_dim=64, hidden_dims=[256, 128], learning_rate=0.001)
    vae.train(df, epochs=100, batch_size=64, beta=1.0)
    
    # Generate samples
    # Generar muestras
    print("\nGenerating synthetic samples...")
    print("Generando muestras sintéticas...")
    synthetic_data = vae.generate_samples(n_samples=1000)
    
    # Save synthetic data
    # Guardar datos sintéticos
    synthetic_data.to_csv('data/processed/vae_generated_payments.csv', index=False)
    print("Synthetic data saved to / Datos sintéticos guardados en: "
          "data/processed/vae_generated_payments.csv")


if __name__ == '__main__':
    main()
