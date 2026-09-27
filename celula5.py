# Célula 5: Treinamento dos Modelos
# Baseline: Regressão Logística
lr = LogisticRegression(random_state=42)
lr.fit(X_train_rus, y_train_rus)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Modelo 2: Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_rus, y_train_rus)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Modelo 3: XGBoost (Usando peso de classe na base desbalanceada original, como alternativa)
# scale_pos_weight = total_negativos / total_positivos
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train) # Treinado na base completa com pesos
y_pred_xgb = xgb.predict(X_test_scaled)
y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
