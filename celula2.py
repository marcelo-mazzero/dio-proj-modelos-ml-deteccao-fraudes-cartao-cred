# Célula 2: Carga e Exploração de Dados
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print("Dimensões do dataset:", df.shape)
print("\nProporção de Fraudes:")
print(df['Class'].value_counts(normalize=True) * 100)
