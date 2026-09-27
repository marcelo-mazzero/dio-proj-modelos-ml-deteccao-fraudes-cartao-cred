# Célula 7: Curva de Precisão e Recall
plt.figure(figsize=(10, 6))

for name, prob in [('Regressão Logística', y_prob_lr), ('Random Forest', y_prob_rf), ('XGBoost', y_prob_xgb)]:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=name)

plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva Precisão-Recall')
plt.legend()
plt.show()
