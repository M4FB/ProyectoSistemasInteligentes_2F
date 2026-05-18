import pandas as pd
import numpy as np

raw = pd.read_csv("dataset/house_price_kc_raw.csv")

# Limpiar outliers
raw = raw[(raw['bedrooms'] >= 1) & (raw['bedrooms'] <= 7)]
raw = raw[raw['price'] > 0]
raw = raw.dropna(subset=['price', 'sqft_living', 'bedrooms', 'bathrooms', 'floors', 'yr_built', 'condition', 'view'])

# Location: agrupar ciudades en 4 niveles por precio mediano
mediana_por_ciudad = raw.groupby('city')['price'].median()
cuartiles = mediana_por_ciudad.quantile([0.25, 0.5, 0.75])
def nivel_ciudad(ciudad):
    med = mediana_por_ciudad.get(ciudad, mediana_por_ciudad.median())
    if med < cuartiles[0.25]:
        return 0   # barato
    elif med < cuartiles[0.50]:
        return 1
    elif med < cuartiles[0.75]:
        return 2
    else:
        return 3   # caro

mapeoCondition = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

dataset = pd.DataFrame({
    'Id':        range(1, len(raw) + 1),
    'Area':      raw['sqft_living'].astype(int).values,
    'Bedrooms':  raw['bedrooms'].astype(int).values,
    'Bathrooms': raw['bathrooms'].round().astype(int).values,
    'Floors':    raw['floors'].round().astype(int).values,
    'YearBuilt': raw['yr_built'].astype(int).values,
    'Location':  raw['city'].map(nivel_ciudad).values,
    'Condition': raw['condition'].map(mapeoCondition).values,
    'Garage':    (raw['sqft_basement'] > 0).astype(int).values,
    'View':      raw['view'].astype(int).values,
    'Price':     raw['price'].astype(int).values,
})

dataset.to_csv("dataset/house_price_all_numeric.csv", index=False)
print(f"Dataset guardado: {len(dataset)} filas")
print()
print("Correlaciones con Price:")
print(dataset.corr()['Price'].sort_values(ascending=False).to_string())
