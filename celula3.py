# Célula 3: Preparação dos Dados
# Criando a variável logarítmica do valor para reduzir a assimetria
df['Log_Amount'] = np.log1p(df['Amount'])

# Removendo colunas originais que não usaremos diretamente no formato bruto
X = df.drop(['Class', 'Amount', 'Time'], axis=1)
y = df['Class']

# Separação de treino e teste mantendo a proporção de fraudes (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Padronização (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
