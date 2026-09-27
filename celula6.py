# Célula 6: Avaliação e Comparação (Foco no Recall)
print("--- Regressão Logística (Undersampling) ---")
print(classification_report(y_test, y_pred_lr))

print("\n--- Random Forest (Undersampling) ---")
print(classification_report(y_test, y_pred_rf))

print("\n--- XGBoost (Class Weight) ---")
print(classification_report(y_test, y_pred_xgb))
