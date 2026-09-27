# Detecção de Fraudes em Cartões de Crédito

> **Projeto prático desenvolvido por Marcelo Mazzero para o [Bootcamp da DIO & Bradesco sobre GenAI, Dados e Cibersegurança](https://www.dio.me/bootcamp/bradesco-dados-ciberseguranca-genai)**

O objetivo deste projeto é desenvolver e avaliar modelos de Machine Learning para a detecção de fraudes em transações de cartão de crédito utilizando dados reais e anonimizados.

---


## 1. O Problema do Desbalanceamento
Neste dataset, 99,8% das transações são legítimas e apenas 0,17% são fraudes. Isso inviabiliza o uso da **Acurácia** como métrica de avaliação. Um modelo que simplesmente aprove todas as compras terá 99,8% de acurácia, mas falhará completamente no seu propósito (deixará passar 100% das fraudes). 

Por isso, o projeto muda o foco para o **Recall** (capacidade de encontrar as fraudes reais), a **Precisão** (confiança de que o alerta gerado é realmente fraude) e o **F1-Score** (média harmônica entre ambos). O custo de um falso negativo (fraude aprovada) é financeiramente pior do que um falso positivo (cartão bloqueado temporariamente).

---


## 2. Estrutura do Código no Notebook
O código foi construído de forma linear para facilitar o acompanhamento do fluxo de dados em Machine Learning, dividido nas seguintes células lógicas:

*   **[Célula 1 (Importações)](celula1.py):** Carregamento das bibliotecas (Pandas, Scikit-Learn, XGBoost, SHAP).
*   **[Célula 2 (Carga e Exploração)](celula2.py):** Leitura do CSV diretamente da nuvem e análise da proporção de fraudes.
*   **[Célula 3 (Preparação dos Dados)](celula3.py):** Transformação logarítmica (`Amount`), descarte de colunas inúteis, separação (`train_test_split` com `stratify`) e padronização (`StandardScaler`).
*   **[Célula 4 (Balanceamento)](celula4.py):** Aplicação de `RandomUnderSampler` para igualar as classes no treino.
*   **[Célula 5 (Treinamento)](celula5.py):** Construção da Regressão Logística (Baseline), Random Forest e XGBoost.
*   **[Células 6 (Avaliação Visual e Numérica)](celula6.py):** Impressão do `classification_report`.
*   **[Célula 7 (Avaliação Visual e Numérica)](celula7.py):** Plotagem da Curva de Precisão-Recall.
*   **[Célula 8 (Ajuste de Limiar)](celula8.py):** Alteração do *threshold* no XGBoost para encontrar o melhor ponto entre Precisão e Recall.
*   **[Célula 9 (Explicabilidade)](celula9.py):** Uso da biblioteca SHAP para gerar o *Summary Plot* e entender as decisões matemáticas do Random Forest.

---


## 3. Comparação entre Modelos
Foram testadas três abordagens de modelagem, combinadas com diferentes tratamentos para o desbalanceamento:
1. **Regressão Logística (Baseline) com Undersampling:** Modelo rápido, mas apresentou alto índice de falsos positivos (precisão baixa para a classe minoritária).
2. **Random Forest com Undersampling:** Excelente capacidade de adaptação às não-linearidades, entregando um Recall altíssimo, às custas de uma precisão moderada na classe 1.
3. **XGBoost com Ponderação de Classes (scale_pos_weight):** O melhor equilíbrio de F1-Score geral, aprendendo com a base completa sem descartar as transações normais, e priorizando matematicamente os erros cometidos nas fraudes.

---


## 4. Limiar de Decisão e Explicabilidade (SHAP)
Foi aplicado um ajuste manual do **limiar de decisão (threshold)** de 0.5 para 0.8 no modelo XGBoost. Isso foi feito para controlar o número de falsos positivos, garantindo que a central de atendimento não fique sobrecarregada, sem sacrificar severamente o Recall.

Para abrir a "caixa preta" do modelo, aplicamos a biblioteca **SHAP**. O gráfico de `summary_plot` gerado no notebook demonstra que variáveis específicas decorrentes do PCA original (como V14, V4 e V12) são as mais determinantes para puxar a decisão do modelo em direção ao alerta de fraude.

---


## 5. Diferenciais da Abordagem
Em relação a um pipeline padrão, foram implementadas as seguintes adaptações:
- Utilização de matrizes de peso (`scale_pos_weight`) em boosting, em vez de depender exclusivamente de oversampling cego (SMOTE), economizando recursos computacionais e evitando overfitting sintético.
- Foco em curvas de Precisão-Recall em vez de apenas curvas ROC, uma vez que o ROC pode ser excessivamente otimista em bases extremamente desbalanceadas.

---


## 6. Como Reproduzir este Projeto no Google Colab
Para testar e rodar o código você mesmo sem precisar instalar nada na sua máquina, siga este passo a passo:

1.  Acesse o [Google Colab](https://colab.research.google.com/).
2.  Clique em **"File"** (Arquivo) > **"New notebook"** (Novo notebook).
3.  Você precisará instalar as bibliotecas de balanceamento e explicabilidade. Crie uma primeira célula, cole o comando abaixo e execute (botão de "Play" ao lado esquerdo da célula):
    ```bash
    !pip install shap imbalanced-learn xgboost
    ```
4.  Após a instalação, abra os arquivos `.py` com o bloco de código das células fornecidos neste repositório.
5.  Copie o código de cada Célula (da 1 a 9) e cole em células individuais e sequenciais no seu notebook do Colab.
6.  No menu superior do Colab, clique em **"Runtime"** (Ambiente de execução) e depois em **"Run all"** (Executar tudo). 
7.  Aguarde a execução. Como o dataset não precisa ser baixado (ele é lido pela URL oficial do TensorFlow), o modelo será treinado na nuvem e todas as métricas e gráficos (incluindo o SHAP) aparecerão diretamente na tela.

---


## 7. Aviso

Este repositório é apenas para fins de estudo pessoal.

---


Desenvolvido por **Marcelo Mazzero** em setembro/2026.
