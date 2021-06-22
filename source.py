# Importar bibliotecas
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from pprint import pprint

# Importar dados do conjunto de treinamento
features = pd.read_csv("conjunto_de_treinamento.csv")

# Remover colunas
features = features.drop(
    [
         "id_solicitante",
         "grau_instrucao",
         "estado_onde_nasceu",
         "estado_onde_reside",
         "codigo_area_telefone_residencial",
         "estado_onde_trabalha",
         "codigo_area_telefone_trabalho",
         "local_onde_reside",
         "local_onde_trabalha"
    ],
    axis=1
)

# Correção da coluna "sexo"
features.loc[features["sexo"] == " ", "sexo"] = "N"

# Divisão dos dados categóricos
features = pd.get_dummies(features,columns=
    [
         "produto_solicitado",
         "sexo",
         "estado_civil",
         "nacionalidade",
         "tipo_residencia",
         "dia_vencimento",
         "profissao",
         "forma_envio_solicitacao",
         "profissao_companheiro",
         "grau_instrucao_companheiro",
         "ocupacao"
    ]
)

# Padronização dos dados binários
binarizer = LabelBinarizer()

binaries = [
    "possui_telefone_trabalho",
    "possui_telefone_residencial",
    "possui_cartao_visa",
    "possui_cartao_mastercard",
    "possui_cartao_diners",
    "possui_cartao_amex",
    "possui_outros_cartoes",
    "possui_carro",
    "vinculo_formal_com_empresa",
    "tipo_endereco",
    "possui_telefone_celular"
]

for v in binaries:
    features[v] = binarizer.fit_transform(features[v])

features = features.interpolate(method="nearest")

# Separação do conjunto em label e features
label = np.array(features["inadimplente"])
features = features.drop("inadimplente", axis=1)

feature_list = list(features.columns)

# Determinar os atributos com maior correlação com a inadimplência
reg = LassoCV(random_state=12345, cv=10, max_iter=2000, eps=0.000005)
reg.fit(features, label)

print("Alpha LassoCV: %f" % reg.alpha_)
print("Score LassoCV: %f" % reg.score(features, label))
coefs = pd.Series(reg.coef_, index = feature_list)
print("Lasso escolheu " + str(sum(coefs != 0)) + " variaveis e excluiu " + str(sum(coefs == 0)) + " variaveis.")

figure(figsize=(8,16), dpi=80)
sorted_coefs = coefs.sort_values()
sorted_coefs.plot(kind="barh")

# Excluir as colunas com baixa correlação
features = features.drop(coefs.loc[coefs == 0].index, axis=1)

# Standard Scaler
scaler = StandardScaler().fit(features)
features = scaler.transform(features)

features = np.array(features)

# Separar conjunto em treinamento e teste
train_features, test_features, train_label, test_label = train_test_split(features, label, test_size = 0.3, random_state = 12345)

# Modelo base
rf = RandomForestClassifier(n_estimators = 200, random_state = 12345)

# Treinar o modelo
rf.fit(train_features, train_label)

# Previsões
predictions = rf.predict(test_features)

# Erros
errors = abs(np.around(predictions, decimals=0) - test_label)

# Acurácia
print("Acuracia:", 1 - np.mean(errors))

# Busca automatizada
n_estimators = [int(x) for x in np.linspace(start=160, stop=180, num=5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(100, 120, num = 3)]
max_depth.append(None)
min_samples_split = [40, 50]
min_samples_leaf = [9]
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=12345, n_jobs = -1)
rf_random.fit(features, label)

# Melhores parâmetros
print("Melhores parametros:")
print(rf_random.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = 100 * abs(np.around(predictions, decimals=0) - test_labels)
    accuracy = 100 - np.mean(errors)
    print("Erro = {:0.4f}.".format(np.mean(errors)))
    print("Acuracia = {:0.2f}%.".format(accuracy))
    
    return accuracy

base_model = RandomForestClassifier(n_estimators = 165, random_state = 12345)
base_model.fit(train_features, train_label)
base_accuracy = evaluate(base_model, test_features, test_label)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_label)

# Grid Search
param_grid = {
    'n_estimators': [165],
    'min_samples_split': [40],
    'min_samples_leaf': [9],
    'max_features': ['sqrt'],
    'max_depth': [320],
    'bootstrap': [True]
}

rf = RandomForestClassifier(random_state=12345)

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 18, n_jobs = -1, verbose = 2)

grid_search.fit(train_features, train_label)
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_label)

# Conjunto de teste
# Importar dados
test_features = pd.read_csv("conjunto_de_teste.csv")

# Corrigir colunas
test_features.loc[test_features["sexo"] == " ", "sexo"] = "N"
test_features = test_features.interpolate(method="nearest")

# Colunas categóricas
test_features = pd.get_dummies(test_features,columns=
    [
         "produto_solicitado",
         "sexo",
         "estado_civil",
         "nacionalidade",
         "tipo_residencia",
         "dia_vencimento",
         "profissao",
         "forma_envio_solicitacao",
         "profissao_companheiro",
         "grau_instrucao_companheiro",
         "ocupacao"
    ]
)

# Colunas binárias
binarizer = LabelBinarizer()

binaries = [
    "possui_telefone_trabalho",
    "possui_telefone_residencial",
    "possui_cartao_visa",
    "possui_cartao_mastercard",
    "possui_cartao_diners",
    "possui_cartao_amex",
    "possui_outros_cartoes",
    "possui_carro",
    "vinculo_formal_com_empresa",
    "tipo_endereco",
    "possui_telefone_celular"
]

for v in binaries:
    test_features[v] = binarizer.fit_transform(test_features[v])
    
# Excluir colunas
test_features = test_features.drop(test_features.columns.difference(feature_list),axis=1)

# Standard scaler
test_features = scaler.transform(test_features)

test_features = np.array(test_features)

result = best_grid.predict(test_features)
result = np.around(result)
result = result.astype(int)

results = pd.DataFrame()
results['id_solicitante'] = np.linspace(start=20001, stop=25000, num=5000, dtype=int)
results['inadimplente'] = result

results.to_csv("results.csv", index=False, header=True)