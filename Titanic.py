import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Leitura dos arquivos

data = pd.read_csv('Coloque seu caminho aqui')
test = pd.read_csv('Coloque seu caminho aqui')

# Tratamento dos dados de treinamento

data['Sex'] = data['Sex'].map({ 'male': 0, 'female': 1 }) # Mapeamento para os prefixos dos nomes

title_map = { 'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3 }

data['Social_Class'] = data['Name'].str.extract(r', (\w+)\.')[0].map(title_map).fillna(-1)

data.drop(['PassengerId', 'Ticket', 'Cabin', 'Embarked', 'Name'], axis=1, inplace=True)

data.fillna({'Age': data['Age'].mean()}, inplace=True)

y = data['Survived']

data.drop(['Survived'], axis=1, inplace=True)

x = data

# Normalização das colunas "Age" e "Fare"

scaler_age = StandardScaler()
scaler_fare = StandardScaler()

x['Age'] = scaler_age.fit_transform(x[['Age']])
x['Fare'] = scaler_fare.fit_transform(x[['Fare']])

# Dividindo os dados para treinamento e teste

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Inicializando o modelo e o treinando

model = LogisticRegression(C=0.5, solver='liblinear')

model.fit(x_train, y_train)

# Tratando os dados de teste

test['Sex'] = test['Sex'].map({ 'male': 0, 'female': 1 })

test['Social_Class'] = test['Name'].str.extract(r', (\w+)\.')[0].map(title_map).fillna(-1)

test.drop(['PassengerId', 'Ticket', 'Cabin', 'Embarked', 'Name'], axis=1, inplace=True)

test.fillna({'Age': test['Age'].mean()}, inplace=True)

# Normalização das colunas "Age" e "Fare"

test['Age'] = scaler_age.transform(test[['Age']])
test['Fare'] = scaler_fare.transform(test[['Fare']])

# Predição

prediction = model.predict(x_test)

print(accuracy_score(y_test, prediction))