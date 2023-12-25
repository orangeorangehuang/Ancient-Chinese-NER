import pandas as pd
variant_csv = '/Users/orangehuang/Desktop/adl-final/data/variant.csv'

def clean(text):
  df = pd.read_csv(variant_csv)
  standard = df['standard']
  variant = df['variant']

  for i in range(len(variant)):
    text = text.replace(variant[i], standard[i])

  return text