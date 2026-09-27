# Célula 1: Importação de Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import shap

# Configuração visual
sns.set_theme(style="whitegrid")
