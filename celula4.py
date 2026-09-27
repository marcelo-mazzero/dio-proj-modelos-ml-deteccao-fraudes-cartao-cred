# Célula 4: Balanceamento (Undersampling)
# Como a base tem 284 mil linhas, o undersampling torna o treinamento rápido e focado nas anomalias
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)

print(f"Distribuição após Undersampling:\n{y_train_rus.value_counts()}")
