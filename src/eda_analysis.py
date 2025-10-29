"""
Exploratory Data Analysis (EDA) for Payment Transaction Data
Análisis Exploratorio de Datos (EDA) para Datos de Transacciones de Pago

This module performs comprehensive EDA and creates visualizations.
Este módulo realiza EDA completo y crea visualizaciones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style for better-looking plots
# Establecer estilo para gráficos más atractivos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PaymentEDA:
    """
    Exploratory Data Analysis for payment transaction data.
    Análisis Exploratorio de Datos para datos de transacciones de pago.
    """
    
    def __init__(self, df, output_dir='visualizations'):
        """
        Initialize EDA with payment data.
        Inicializar EDA con datos de pago.
        
        Args:
            df (pd.DataFrame): Payment transaction data
                              Datos de transacciones de pago
            output_dir (str): Directory to save visualizations
                             Directorio para guardar visualizaciones
        """
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert date columns to datetime
        # Convertir columnas de fecha a datetime
        date_columns = ['transaction_date', 'due_date', 'payment_date']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
    
    def generate_summary_statistics(self):
        """
        Generate and display summary statistics.
        Generar y mostrar estadísticas resumen.
        """
        print("=" * 80)
        print("SUMMARY STATISTICS / ESTADÍSTICAS RESUMEN")
        print("=" * 80)
        
        print("\n### Dataset Overview / Visión General del Dataset ###")
        print(f"Total transactions / Total de transacciones: {len(self.df)}")
        print(f"Date range / Rango de fechas: {self.df['transaction_date'].min()} to {self.df['transaction_date'].max()}")
        print(f"Number of unique payers / Número de pagadores únicos: {self.df['payer_id'].nunique()}")
        print(f"Number of unique payees / Número de receptores únicos: {self.df['payee_id'].nunique()}")
        
        print("\n### Amount Statistics / Estadísticas de Montos ###")
        print(self.df['amount'].describe())
        
        print("\n### Payment Status Distribution / Distribución de Estado de Pago ###")
        print(self.df['status'].value_counts())
        print(f"\nPercentages / Porcentajes:")
        print(self.df['status'].value_counts(normalize=True) * 100)
        
        print("\n### Anomaly Distribution / Distribución de Anomalías ###")
        print(f"Normal transactions / Transacciones normales: {(~self.df['is_anomaly']).sum()}")
        print(f"Anomalous transactions / Transacciones anómalas: {self.df['is_anomaly'].sum()}")
        print(f"Anomaly rate / Tasa de anomalía: {self.df['is_anomaly'].mean()*100:.2f}%")
        
        print("\n### Currency Distribution / Distribución de Monedas ###")
        print(self.df['currency'].value_counts())
        
        print("\n### Payment Method Distribution / Distribución de Métodos de Pago ###")
        print(self.df['payment_method'].value_counts())
        
        print("\n### Industry Distribution / Distribución de Industrias ###")
        print("\nPayer industries / Industrias de pagadores:")
        print(self.df['payer_industry'].value_counts())
        print("\nPayee industries / Industrias de receptores:")
        print(self.df['payee_industry'].value_counts())
        
        print("\n### Company Size Distribution / Distribución de Tamaño de Empresas ###")
        print("\nPayer sizes / Tamaños de pagadores:")
        print(self.df['payer_size'].value_counts())
        print("\nPayee sizes / Tamaños de receptores:")
        print(self.df['payee_size'].value_counts())
    
    def plot_amount_distribution(self):
        """
        Plot distribution of transaction amounts.
        Graficar distribución de montos de transacciones.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram of amounts
        # Histograma de montos
        axes[0, 0].hist(self.df['amount'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Amount / Monto ($)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency / Frecuencia', fontsize=12)
        axes[0, 0].set_title('Distribution of Transaction Amounts\nDistribución de Montos de Transacciones', 
                             fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log-scale histogram
        # Histograma en escala logarítmica
        axes[0, 1].hist(np.log10(self.df['amount'] + 1), bins=50, 
                       edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Log10(Amount) / Log10(Monto)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency / Frecuencia', fontsize=12)
        axes[0, 1].set_title('Log-Scale Distribution of Amounts\nDistribución en Escala Log de Montos', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot by status
        # Diagrama de caja por estado
        self.df.boxplot(column='amount', by='status', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Payment Status / Estado de Pago', fontsize=12)
        axes[1, 0].set_ylabel('Amount / Monto ($)', fontsize=12)
        axes[1, 0].set_title('Amount Distribution by Payment Status\nDistribución de Montos por Estado de Pago', 
                            fontsize=14, fontweight='bold')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45)
        
        # Violin plot for anomalies
        # Gráfico de violín para anomalías
        sns.violinplot(data=self.df, x='is_anomaly', y='amount', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Is Anomaly / Es Anomalía', fontsize=12)
        axes[1, 1].set_ylabel('Amount / Monto ($)', fontsize=12)
        axes[1, 1].set_title('Amount Distribution: Normal vs Anomalous\nDistribución de Montos: Normal vs Anómalo', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels(['Normal', 'Anomaly / Anomalía'])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/amount_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved / Guardado: {self.output_dir}/amount_distribution.png")
        plt.close()
    
    def plot_temporal_patterns(self):
        """
        Plot temporal patterns in transactions.
        Graficar patrones temporales en transacciones.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Transactions over time
        # Transacciones a lo largo del tiempo
        daily_counts = self.df.groupby(self.df['transaction_date'].dt.date).size()
        axes[0, 0].plot(daily_counts.index, daily_counts.values, linewidth=1.5)
        axes[0, 0].set_xlabel('Date / Fecha', fontsize=12)
        axes[0, 0].set_ylabel('Number of Transactions / Número de Transacciones', fontsize=12)
        axes[0, 0].set_title('Daily Transaction Volume\nVolumen Diario de Transacciones', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        plt.sca(axes[0, 0])
        plt.xticks(rotation=45)
        
        # Monthly transaction amounts
        # Montos de transacciones mensuales
        self.df['year_month'] = self.df['transaction_date'].dt.to_period('M')
        monthly_amounts = self.df.groupby('year_month')['amount'].sum() / 1_000_000
        axes[0, 1].bar(range(len(monthly_amounts)), monthly_amounts.values, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Month / Mes', fontsize=12)
        axes[0, 1].set_ylabel('Total Amount / Monto Total (Millions $)', fontsize=12)
        axes[0, 1].set_title('Monthly Transaction Volume\nVolumen Mensual de Transacciones', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(range(0, len(monthly_amounts), 3))
        axes[0, 1].set_xticklabels([str(m) for m in monthly_amounts.index[::3]], rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Day of week patterns
        # Patrones por día de la semana
        self.df['day_of_week'] = self.df['transaction_date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.df['day_of_week'].value_counts().reindex(day_order)
        axes[1, 0].bar(range(len(day_counts)), day_counts.values, alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Day of Week / Día de la Semana', fontsize=12)
        axes[1, 0].set_ylabel('Number of Transactions / Número de Transacciones', fontsize=12)
        axes[1, 0].set_title('Transactions by Day of Week\nTransacciones por Día de la Semana', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(day_counts)))
        axes[1, 0].set_xticklabels(['Mon/Lun', 'Tue/Mar', 'Wed/Mié', 'Thu/Jue', 
                                    'Fri/Vie', 'Sat/Sáb', 'Sun/Dom'], rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Anomalies over time
        # Anomalías a lo largo del tiempo
        monthly_anomalies = self.df.groupby('year_month')['is_anomaly'].sum()
        axes[1, 1].plot(range(len(monthly_anomalies)), monthly_anomalies.values, 
                       marker='o', linewidth=2, markersize=6, color='red')
        axes[1, 1].set_xlabel('Month / Mes', fontsize=12)
        axes[1, 1].set_ylabel('Number of Anomalies / Número de Anomalías', fontsize=12)
        axes[1, 1].set_title('Anomalies Over Time\nAnomalías a lo Largo del Tiempo', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(0, len(monthly_anomalies), 3))
        axes[1, 1].set_xticklabels([str(m) for m in monthly_anomalies.index[::3]], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/temporal_patterns.png', dpi=300, bbox_inches='tight')
        print(f"Saved / Guardado: {self.output_dir}/temporal_patterns.png")
        plt.close()
    
    def plot_categorical_distributions(self):
        """
        Plot distributions of categorical variables.
        Graficar distribuciones de variables categóricas.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Payment method distribution
        # Distribución de métodos de pago
        payment_method_counts = self.df['payment_method'].value_counts()
        axes[0, 0].pie(payment_method_counts.values, labels=payment_method_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Payment Method Distribution\nDistribución de Métodos de Pago', 
                            fontsize=14, fontweight='bold')
        
        # Currency distribution
        # Distribución de monedas
        currency_counts = self.df['currency'].value_counts()
        axes[0, 1].bar(range(len(currency_counts)), currency_counts.values, 
                      alpha=0.7, color='teal')
        axes[0, 1].set_xlabel('Currency / Moneda', fontsize=12)
        axes[0, 1].set_ylabel('Count / Cantidad', fontsize=12)
        axes[0, 1].set_title('Currency Distribution\nDistribución de Monedas', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(range(len(currency_counts)))
        axes[0, 1].set_xticklabels(currency_counts.index)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Industry distribution (payers)
        # Distribución de industrias (pagadores)
        industry_counts = self.df['payer_industry'].value_counts()
        axes[1, 0].barh(range(len(industry_counts)), industry_counts.values, alpha=0.7)
        axes[1, 0].set_ylabel('Industry / Industria', fontsize=12)
        axes[1, 0].set_xlabel('Count / Cantidad', fontsize=12)
        axes[1, 0].set_title('Payer Industry Distribution\nDistribución de Industria de Pagadores', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_yticks(range(len(industry_counts)))
        axes[1, 0].set_yticklabels(industry_counts.index)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Company size distribution
        # Distribución de tamaño de empresas
        size_order = ['Small', 'Medium', 'Large', 'Enterprise']
        payer_size_counts = self.df['payer_size'].value_counts().reindex(size_order)
        payee_size_counts = self.df['payee_size'].value_counts().reindex(size_order)
        
        x = np.arange(len(size_order))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, payer_size_counts.values, width, 
                      label='Payer / Pagador', alpha=0.8)
        axes[1, 1].bar(x + width/2, payee_size_counts.values, width, 
                      label='Payee / Receptor', alpha=0.8)
        axes[1, 1].set_xlabel('Company Size / Tamaño de Empresa', fontsize=12)
        axes[1, 1].set_ylabel('Count / Cantidad', fontsize=12)
        axes[1, 1].set_title('Company Size Distribution\nDistribución de Tamaño de Empresas', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Small/Pequeña', 'Medium/Mediana', 
                                    'Large/Grande', 'Enterprise/Empresa'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/categorical_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Saved / Guardado: {self.output_dir}/categorical_distributions.png")
        plt.close()
    
    def plot_correlation_heatmap(self):
        """
        Plot correlation heatmap of numerical features.
        Graficar mapa de calor de correlación de características numéricas.
        """
        # Prepare numerical data
        # Preparar datos numéricos
        numerical_df = self.df[['amount']].copy()
        
        # Add payment delay
        # Agregar retraso de pago
        paid_df = self.df[self.df['payment_date'].notna()].copy()
        if len(paid_df) > 0:
            paid_df['payment_delay_days'] = (
                paid_df['payment_date'] - paid_df['transaction_date']
            ).dt.days
            paid_df['days_to_due'] = (
                paid_df['due_date'] - paid_df['transaction_date']
            ).dt.days
            
            numerical_df = paid_df[['amount', 'payment_delay_days', 'days_to_due']].copy()
        
        # Add encoded categorical variables
        # Agregar variables categóricas codificadas
        numerical_df['is_anomaly'] = self.df['is_anomaly'].astype(int)
        numerical_df['is_late'] = (self.df['status'] == 'Late').astype(int)
        
        # Calculate correlation matrix
        # Calcular matriz de correlación
        corr_matrix = numerical_df.corr()
        
        # Plot heatmap
        # Graficar mapa de calor
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap of Numerical Features\n'
                 'Mapa de Calor de Correlación de Características Numéricas', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved / Guardado: {self.output_dir}/correlation_heatmap.png")
        plt.close()
    
    def plot_anomaly_analysis(self):
        """
        Plot detailed anomaly analysis.
        Graficar análisis detallado de anomalías.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Anomaly rate by industry
        # Tasa de anomalía por industria
        anomaly_by_industry = self.df.groupby('payer_industry')['is_anomaly'].mean() * 100
        anomaly_by_industry = anomaly_by_industry.sort_values(ascending=False)
        axes[0, 0].barh(range(len(anomaly_by_industry)), anomaly_by_industry.values, alpha=0.7)
        axes[0, 0].set_xlabel('Anomaly Rate / Tasa de Anomalía (%)', fontsize=12)
        axes[0, 0].set_ylabel('Industry / Industria', fontsize=12)
        axes[0, 0].set_title('Anomaly Rate by Industry\nTasa de Anomalía por Industria', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_yticks(range(len(anomaly_by_industry)))
        axes[0, 0].set_yticklabels(anomaly_by_industry.index)
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Anomaly rate by company size
        # Tasa de anomalía por tamaño de empresa
        size_order = ['Small', 'Medium', 'Large', 'Enterprise']
        anomaly_by_size = self.df.groupby('payer_size')['is_anomaly'].mean() * 100
        anomaly_by_size = anomaly_by_size.reindex(size_order)
        axes[0, 1].bar(range(len(anomaly_by_size)), anomaly_by_size.values, alpha=0.7, color='coral')
        axes[0, 1].set_xlabel('Company Size / Tamaño de Empresa', fontsize=12)
        axes[0, 1].set_ylabel('Anomaly Rate / Tasa de Anomalía (%)', fontsize=12)
        axes[0, 1].set_title('Anomaly Rate by Company Size\nTasa de Anomalía por Tamaño de Empresa', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(range(len(anomaly_by_size)))
        axes[0, 1].set_xticklabels(['Small/Peq', 'Medium/Med', 'Large/Gde', 'Enterprise/Emp'])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Amount comparison: normal vs anomaly
        # Comparación de montos: normal vs anomalía
        normal_amounts = self.df[~self.df['is_anomaly']]['amount']
        anomaly_amounts = self.df[self.df['is_anomaly']]['amount']
        
        axes[1, 0].hist([normal_amounts, anomaly_amounts], bins=30, 
                       label=['Normal', 'Anomaly / Anomalía'], alpha=0.7)
        axes[1, 0].set_xlabel('Amount / Monto ($)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency / Frecuencia', fontsize=12)
        axes[1, 0].set_title('Amount Distribution: Normal vs Anomaly\n'
                            'Distribución de Montos: Normal vs Anomalía', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Anomaly rate by payment method
        # Tasa de anomalía por método de pago
        anomaly_by_method = self.df.groupby('payment_method')['is_anomaly'].mean() * 100
        axes[1, 1].bar(range(len(anomaly_by_method)), anomaly_by_method.values, 
                      alpha=0.7, color='lightcoral')
        axes[1, 1].set_xlabel('Payment Method / Método de Pago', fontsize=12)
        axes[1, 1].set_ylabel('Anomaly Rate / Tasa de Anomalía (%)', fontsize=12)
        axes[1, 1].set_title('Anomaly Rate by Payment Method\nTasa de Anomalía por Método de Pago', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(len(anomaly_by_method)))
        axes[1, 1].set_xticklabels(anomaly_by_method.index, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/anomaly_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved / Guardado: {self.output_dir}/anomaly_analysis.png")
        plt.close()
    
    def run_full_eda(self):
        """
        Run complete EDA pipeline.
        Ejecutar pipeline completo de EDA.
        """
        print("\n" + "="*80)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("INICIANDO ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*80 + "\n")
        
        self.generate_summary_statistics()
        
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS / GENERANDO VISUALIZACIONES")
        print("="*80 + "\n")
        
        self.plot_amount_distribution()
        self.plot_temporal_patterns()
        self.plot_categorical_distributions()
        self.plot_correlation_heatmap()
        self.plot_anomaly_analysis()
        
        print("\n" + "="*80)
        print("EDA COMPLETED! / ¡EDA COMPLETADO!")
        print(f"All visualizations saved to / Todas las visualizaciones guardadas en: {self.output_dir}")
        print("="*80 + "\n")


def main():
    """
    Main function to run EDA.
    Función principal para ejecutar EDA.
    """
    # Load data
    # Cargar datos
    df = pd.read_csv('data/raw/synthetic_payments.csv')
    
    # Run EDA
    # Ejecutar EDA
    eda = PaymentEDA(df)
    eda.run_full_eda()


if __name__ == '__main__':
    main()
