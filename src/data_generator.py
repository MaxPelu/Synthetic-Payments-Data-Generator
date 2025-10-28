"""
Initial Data Generator for B2B Payment Transactions
Generador de Datos Inicial para Transacciones de Pagos B2B

This module generates realistic synthetic payment transaction data between companies.
Este módulo genera datos sintéticos realistas de transacciones de pagos entre empresas.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import json

# Initialize Faker for generating realistic company names
# Inicializar Faker para generar nombres de empresas realistas
fake = Faker()


class PaymentDataGenerator:
    """
    Generator for synthetic B2B payment transaction data.
    Generador de datos sintéticos de transacciones de pagos B2B.
    """
    
    def __init__(self, n_companies=100, seed=42):
        """
        Initialize the payment data generator.
        Inicializar el generador de datos de pagos.
        
        Args:
            n_companies (int): Number of companies to simulate
                              Número de empresas a simular
            seed (int): Random seed for reproducibility
                       Semilla aleatoria para reproducibilidad
        """
        np.random.seed(seed)
        random.seed(seed)
        Faker.seed(seed)
        
        self.n_companies = n_companies
        self.companies = self._generate_companies()
        
    def _generate_companies(self):
        """
        Generate a list of synthetic companies.
        Generar una lista de empresas sintéticas.
        
        Returns:
            list: List of company dictionaries
                  Lista de diccionarios de empresas
        """
        companies = []
        for i in range(self.n_companies):
            company = {
                'company_id': f'COMP_{i:04d}',
                'company_name': fake.company(),
                'industry': random.choice([
                    'Technology', 'Healthcare', 'Finance', 'Retail',
                    'Manufacturing', 'Energy', 'Transportation', 'Consulting'
                ]),
                'size': random.choice(['Small', 'Medium', 'Large', 'Enterprise'])
            }
            companies.append(company)
        return companies
    
    def generate_transactions(self, n_transactions=10000, 
                            start_date='2023-01-01', 
                            end_date='2024-12-31',
                            anomaly_rate=0.05):
        """
        Generate synthetic payment transactions.
        Generar transacciones de pago sintéticas.
        
        Args:
            n_transactions (int): Number of transactions to generate
                                 Número de transacciones a generar
            start_date (str): Start date for transactions (YYYY-MM-DD)
                             Fecha de inicio para transacciones
            end_date (str): End date for transactions (YYYY-MM-DD)
                           Fecha de fin para transacciones
            anomaly_rate (float): Proportion of anomalous transactions (0-1)
                                 Proporción de transacciones anómalas
        
        Returns:
            pd.DataFrame: DataFrame containing synthetic transactions
                         DataFrame con transacciones sintéticas
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        transactions = []
        
        for i in range(n_transactions):
            # Select payer and payee companies
            # Seleccionar empresas pagadora y receptora
            payer = random.choice(self.companies)
            payee = random.choice([c for c in self.companies if c['company_id'] != payer['company_id']])
            
            # Generate transaction date
            # Generar fecha de transacción
            days_between = (end - start).days
            random_days = random.randint(0, days_between)
            transaction_date = start + timedelta(days=random_days)
            
            # Determine if this is an anomaly
            # Determinar si es una anomalía
            is_anomaly = random.random() < anomaly_rate
            
            # Generate amount based on company size and anomaly status
            # Generar monto basado en tamaño de empresa y estado de anomalía
            base_amount = self._generate_amount(payer['size'], payee['size'])
            
            if is_anomaly:
                # Anomaly patterns: unusually high/low amounts, round numbers, etc.
                # Patrones de anomalía: montos inusualmente altos/bajos, números redondos, etc.
                anomaly_type = random.choice(['high_amount', 'low_amount', 'round_number', 'duplicate'])
                amount = self._apply_anomaly(base_amount, anomaly_type)
            else:
                amount = base_amount
            
            # Generate invoice details
            # Generar detalles de factura
            invoice_id = f'INV-{transaction_date.year}-{i:06d}'
            payment_method = random.choice(['Wire Transfer', 'ACH', 'Check', 'Credit Card'])
            currency = random.choice(['USD', 'EUR', 'GBP'], p=[0.7, 0.2, 0.1])
            
            # Payment status with realistic delays
            # Estado de pago con retrasos realistas
            due_date = transaction_date + timedelta(days=random.choice([15, 30, 45, 60]))
            payment_date = self._generate_payment_date(transaction_date, due_date, is_anomaly)
            
            status = 'Paid' if payment_date <= datetime.now() else 'Pending'
            if status == 'Paid':
                days_to_pay = (payment_date - transaction_date).days
                status = 'Late' if payment_date > due_date else 'On Time'
            
            transaction = {
                'transaction_id': f'TXN_{i:08d}',
                'invoice_id': invoice_id,
                'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                'due_date': due_date.strftime('%Y-%m-%d'),
                'payment_date': payment_date.strftime('%Y-%m-%d') if status != 'Pending' else None,
                'payer_id': payer['company_id'],
                'payer_name': payer['company_name'],
                'payer_industry': payer['industry'],
                'payer_size': payer['size'],
                'payee_id': payee['company_id'],
                'payee_name': payee['company_name'],
                'payee_industry': payee['industry'],
                'payee_size': payee['size'],
                'amount': round(amount, 2),
                'currency': currency,
                'payment_method': payment_method,
                'status': status,
                'is_anomaly': is_anomaly,
                'description': self._generate_description(payer['industry'], payee['industry'])
            }
            
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def _generate_amount(self, payer_size, payee_size):
        """
        Generate transaction amount based on company sizes.
        Generar monto de transacción basado en tamaños de empresas.
        """
        size_multipliers = {
            'Small': (1000, 50000),
            'Medium': (10000, 200000),
            'Large': (50000, 1000000),
            'Enterprise': (100000, 5000000)
        }
        
        min_amount, max_amount = size_multipliers.get(payer_size, (1000, 100000))
        
        # Log-normal distribution for realistic amount distribution
        # Distribución log-normal para distribución realista de montos
        mu = np.log((min_amount + max_amount) / 2)
        sigma = 0.8
        amount = np.random.lognormal(mu, sigma)
        
        # Clip to reasonable range
        # Limitar a rango razonable
        amount = np.clip(amount, min_amount, max_amount)
        
        return amount
    
    def _apply_anomaly(self, base_amount, anomaly_type):
        """
        Apply anomaly pattern to transaction amount.
        Aplicar patrón de anomalía al monto de transacción.
        """
        if anomaly_type == 'high_amount':
            return base_amount * random.uniform(5, 20)
        elif anomaly_type == 'low_amount':
            return base_amount * random.uniform(0.01, 0.1)
        elif anomaly_type == 'round_number':
            return round(base_amount, -4)  # Round to nearest 10,000
        elif anomaly_type == 'duplicate':
            return base_amount
        return base_amount
    
    def _generate_payment_date(self, transaction_date, due_date, is_anomaly):
        """
        Generate payment date with realistic delays.
        Generar fecha de pago con retrasos realistas.
        """
        if is_anomaly:
            # Anomalies might be very late or very early payments
            # Anomalías pueden ser pagos muy tardíos o muy tempranos
            if random.random() < 0.5:
                days_delay = random.randint(60, 180)  # Very late
            else:
                days_delay = random.randint(-30, 0)  # Very early
        else:
            # Normal payment patterns
            # Patrones normales de pago
            days_delay = int(np.random.normal(15, 10))
        
        payment_date = transaction_date + timedelta(days=days_delay)
        return payment_date
    
    def _generate_description(self, payer_industry, payee_industry):
        """
        Generate transaction description based on industries.
        Generar descripción de transacción basada en industrias.
        """
        descriptions = [
            f'Professional services - {payee_industry}',
            f'Product purchase - {payee_industry} supplies',
            f'Consulting services - {payer_industry} sector',
            f'Software licensing and support',
            f'Equipment and materials',
            f'Maintenance and support services',
            f'Marketing and advertising services',
            f'Research and development services'
        ]
        return random.choice(descriptions)
    
    def save_data(self, df, output_path='data/raw/synthetic_payments.csv'):
        """
        Save generated data to CSV file.
        Guardar datos generados en archivo CSV.
        
        Args:
            df (pd.DataFrame): DataFrame to save / DataFrame a guardar
            output_path (str): Output file path / Ruta del archivo de salida
        """
        df.to_csv(output_path, index=False)
        print(f"Data saved to / Datos guardados en: {output_path}")
        print(f"Total transactions / Total de transacciones: {len(df)}")
        print(f"Anomalies / Anomalías: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")


def main():
    """
    Main function to generate synthetic payment data.
    Función principal para generar datos de pago sintéticos.
    """
    print("Generating synthetic B2B payment transaction data...")
    print("Generando datos sintéticos de transacciones de pago B2B...")
    
    # Initialize generator
    # Inicializar generador
    generator = PaymentDataGenerator(n_companies=150, seed=42)
    
    # Generate transactions
    # Generar transacciones
    df = generator.generate_transactions(
        n_transactions=10000,
        start_date='2023-01-01',
        end_date='2024-12-31',
        anomaly_rate=0.05
    )
    
    # Save data
    # Guardar datos
    generator.save_data(df)
    
    # Display summary statistics
    # Mostrar estadísticas resumen
    print("\n=== Summary Statistics / Estadísticas Resumen ===")
    print(df.describe())
    print("\n=== Data Sample / Muestra de Datos ===")
    print(df.head())


if __name__ == '__main__':
    main()
