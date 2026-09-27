# Célula 9: Explicabilidade com SHAP
# Usamos o TreeExplainer no modelo Random Forest para entender as decisões
explainer = shap.TreeExplainer(rf)

# Calculando SHAP values para uma amostra do teste (para rodar mais rápido)
X_test_sample = X_test_scaled[:500] 
shap_values = explainer.shap_values(X_test_sample)

# Tratamento para extrair a classe 1 (Fraude) independentemente da versão do SHAP
if isinstance(shap_values, list):
    # Versões antigas: retorna uma lista onde o índice 1 é a classe positiva
    shap_vals_fraude = shap_values[1]
elif len(shap_values.shape) == 3:
    # Versões novas: retorna array 3D (amostras, features, classes)
    shap_vals_fraude = shap_values[:, :, 1]
else:
    # Retorno 2D padrão em classificadores já filtrados
    shap_vals_fraude = shap_values

# Feature importance baseada em SHAP (Classe 1 = Fraude)
plt.title("Impacto das Variáveis na Detecção de Fraude (SHAP)")
shap.summary_plot(shap_vals_fraude, X_test_sample, feature_names=X.columns)
