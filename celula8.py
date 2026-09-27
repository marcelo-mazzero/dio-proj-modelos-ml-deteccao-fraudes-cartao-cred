# Célula 8: Ajuste de Limiar (Threshold) no XGBoost
threshold = 0.8  # Limiar mais rigoroso para reduzir falsos positivos, assumindo uma perda aceitável de Recall
y_pred_custom_threshold = (y_prob_xgb >= threshold).astype(int)

print(f"--- XGBoost com Limiar de {threshold} ---")
print(classification_report(y_test, y_pred_custom_threshold))
