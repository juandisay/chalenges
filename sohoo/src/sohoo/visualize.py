import pandas as pd
import json
from datetime import datetime, timedelta
from shared.settings import DATA_WEATHER

def analyze_weather_data(cities):
    # Read JSON data
    data = json.loads(DATA_WEATHER)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['hourly']['time']),
        'temperature': data['hourly']['temperature_2m'],
        'humidity': data['hourly']['relative_humidity_2m'],
        'wind_speed': data['hourly']['wind_speed_10m']
    })
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Get last 7 days of data
    last_7_days = df.last('7D')
    
    # Daily statistics
    daily_stats = last_7_days.resample('D').agg({
        'temperature': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max'],
        'wind_speed': ['mean', 'max']
    }).round(2)
    
    # Rename columns for better readability
    daily_stats.columns = [
        'Suhu Rata-rata', 'Suhu Min', 'Suhu Max', 'Suhu Std',
        'Kelembaban Rata-rata', 'Kelembaban Min', 'Kelembaban Max',
        'Angin Rata-rata', 'Angin Max'
    ]
    
    # Format index to date string
    daily_stats.index = daily_stats.index.strftime('%Y-%m-%d')
    
    # Overall statistics
    overall_stats = pd.DataFrame({
        'Metrik': [
            'Suhu Rata-rata (°C)',
            'Suhu Minimum (°C)',
            'Suhu Maximum (°C)',
            'Standar Deviasi Suhu',
            'Kelembaban Rata-rata (%)',
            'Kecepatan Angin Rata-rata (km/h)'
        ],
        'Nilai': [
            last_7_days['temperature'].mean().round(2),
            last_7_days['temperature'].min().round(2),
            last_7_days['temperature'].max().round(2),
            last_7_days['temperature'].std().round(2),
            last_7_days['humidity'].mean().round(2),
            last_7_days['wind_speed'].mean().round(2)
        ]
    })
    
    # Print results
    print(f"\n=== Analisis Cuaca 7 Hari Terakhir di {cities}===\n")
    
    print("Statistik Harian:")
    print("-" * 80)
    print(daily_stats)
    print("\n")
    
    print("Statistik Keseluruhan:")
    print("-" * 40)
    print(overall_stats.to_string(index=False))
    
    # Additional analysis
    hottest_day = daily_stats['Suhu Max'].idxmax()
    coldest_day = daily_stats['Suhu Min'].idxmin()
    windiest_day = daily_stats['Angin Max'].idxmax()
    
    print("\nTemuan Menarik:")
    print("-" * 40)
    print(f"Hari Terpanas: {hottest_day} ({daily_stats.loc[hottest_day, 'Suhu Max']}°C)")
    print(f"Hari Terdingin: {coldest_day} ({daily_stats.loc[coldest_day, 'Suhu Min']}°C)")
    print(f"Hari Terangin: {windiest_day} ({daily_stats.loc[windiest_day, 'Angin Max']} km/h)")
    
    # Return DataFrames for further analysis if needed
    return daily_stats, overall_stats
