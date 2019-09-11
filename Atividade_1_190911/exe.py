'''
Unifacisa Campina Grande
Pós-graduação em Datascience
Disciplina: Introdução à Ciência e Análise de Dados
Professor: Jones Granatyr
Aluno: Ronycley Gonçalves Agra

Atividade final - Data: 11/09/2019

Base dados escolhida: Website Phishing
Site da base de dados: https://archive.ics.uci.edu/ml/datasets/Website+Phishing
Descrição breve da base: Dados coletados do www.phishtank.com, onde é possível verificar a ligitimidade de sites

'''
import pandas as pd
import numpy as np

'''
Leitura da base de dados
Obs.: A base original está em arff e foi convertida para CSV
'''
base = pd.read_csv('phishing_data.csv')
     
'''
Seleção dos previsores
'''          
previsores = base.iloc[:, 0:9].values #coluna 1 a 3 | : signifa todas as linhas

'''
Seleção da coluna de lassificação
'''   
classe = base.iloc[:, 9].values

'''
Preparação dos dados, selação e aplicação da estratégia
'''  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 0:9]) #fit: encaixa/prepara para aplicar a estratégia nos dados
previsores[:, 0:9] = imputer.transform(previsores[:, 0:9]) #transform: aplica a estratégia na base de dados

'''
Separação da base original em bases de treinamento e teste
'''  
from sklearn.model_selection import train_test_split #separar a base para treinamento e teste | random_state=0: pegar sempre os mesmos registros
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

'''
Aplicação do algoritmo de classificação, geração da tabela de probabilidad e realização da previsão
''' 
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB() #algoritmo de classificação
classificador.fit(previsores_treinamento, classe_treinamento) #gera a tabela de probabilidade (1014 registros de treinamento)
previsoes = classificador.predict(previsores_teste) #realiza a previsão, a múltiplicação das probabilidades (339 registro de teste)

'''
Comparação do valores corretos com a previsão e geração da matriz de confusão
'''
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes) #faz a comparação dos valores corretos com os valores da previsão,
#a diagonal principal dessa matriz é a quantidade acertos
#essa é bom pra isolar as classes e verificar o nível de acertos de cada uma

#classe_teste são as repostas reais
