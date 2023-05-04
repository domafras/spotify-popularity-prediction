
# Spotify Popularity Prediction
O projeto de previsão de popularidade no Spotify tem como objetivo prever a popularidade de uma música em particular no Spotify com base em suas características. Para isso, serão usados dados disponíveis no Kaggle, e técnicas de análise de dados e Machine Learning serão aplicadas utilizando a linguagem Python e algumas bibliotecas, como Pandas, Matplotlib, Seaborn e Scikit-learn.

O resultado esperado é um modelo de classificação capaz de prever com precisão a popularidade de novas músicas com base em suas características.

### Dataset

Base de dados disponível no Kaggle.

Link:  [https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

### Data Dictionary

-   **track_id**: O ID do Spotify para a faixa
-   **artists**: Os nomes dos artistas que tocaram a música. Se houver mais de um artista, eles são separados por um ;
-   **album_name**: O nome do álbum em que a faixa aparece
-   **track_name**: Nome da faixa
-   **popularity**: A popularidade de uma faixa é um valor entre 0 e 100, sendo 100 o mais popular. A popularidade é calculada por algoritmo e baseia-se principalmente no número total de reproduções que a faixa teve e na recenticidade dessas reproduções. Em geral, as músicas que estão sendo muito reproduzidas agora terão uma popularidade maior do que as músicas que foram muito reproduzidas no passado. Faixas duplicadas (por exemplo, a mesma faixa de um single e de um álbum) são avaliadas independentemente. A popularidade do artista e do álbum é derivada matematicamente da popularidade da faixa.
-   **duration_ms**: O comprimento da faixa em milissegundos
-   **explicit**: Se a faixa possui ou não letras explícitas (verdadeiro = sim, tem; falso = não tem OU desconhecido)
-   **danceability**: Danceability descreve o quão adequada uma faixa é para dançar com base em uma combinação de elementos musicais, incluindo tempo, estabilidade rítmica, força do beat e regularidade geral. Um valor de 0,0 é menos dançável e 1,0 é mais dançável
-   **energy**: Energy é uma medida de 0,0 a 1,0 e representa uma medida perceptual de intensidade e atividade. Tipicamente, faixas energéticas têm uma sensação rápida, alta e barulhenta. Por exemplo, death metal tem alta energia, enquanto um prelúdio de Bach tem baixa pontuação na escala
-   **key**: A tecla em que a faixa está. Inteiros mapeiam para notas usando a notação padrão da classe de pitch. Exemplo: 0 = C, 1 = C♯/D♭, 2 = D, e assim por diante. Se nenhuma chave foi detectada, o valor é -1
-   **loudness**: O volume geral de uma faixa em decibéis (dB)
-   **mode**: Mode indica a modalidade (maior ou menor) de uma faixa, o tipo de escala a partir da qual seu conteúdo melódico é derivado. Maior é representado por 1 e menor é 0
-   **speechiness**: Speechiness detecta a presença de palavras faladas em uma faixa. Quanto mais exclusivamente parecida com fala a gravação (por exemplo, talk show, livro em áudio, poesia), mais próximo de 1,0 o valor atribuído ao atributo. Valores acima de 0,66 descrevem faixas que provavelmente são compostas inteiramente por palavras faladas. Os valores entre 0,33 e 0,66 descrevem faixas que podem conter música e fala, seja em seções ou camadas, incluindo casos como a música rap. Valores abaixo de 0,33 provavelmente representam músicas e outras faixas não semelhantes à fala
-   **acousticness**: Uma medida de confiança de 0,0 a 1,0 se a faixa é acústica. 1.0 representa alta confiança de que a faixa é acústica
-   **instrumentalness**: Prevê se uma faixa não contém vocais. Sons "Ooh" e "aah" são tratados como instrumentais nesse contexto. Faixas de rap ou spoken word são claramente "vocais". Quanto mais próximo o valor de instrumentalidade é de 1,0, maior a probabilidade da faixa não conter conteúdo vocal
-   **liveness**: Detecta a presença de uma plateia na gravação. Valores de liveness mais altos representam uma probabilidade aumentada de que a faixa foi tocada ao vivo. Um valor acima de 0,8 fornece forte probabilidade de que a faixa seja ao vivo
-   **valence**: Uma medida de 0,0 a 1,0 que descreve a positividade musical transmitida
-   **tempo**: O tempo geral estimado de uma faixa em batidas por minuto (BPM). Em terminologia musical, tempo é a velocidade ou ritmo de uma determinada peça e deriva diretamente da duração média das batidas.
-   **time_signature**: Uma assinatura de tempo estimada. A assinatura de tempo (ou compasso) é uma convenção notacional que especifica quantas batidas há em cada compasso (ou medida). A assinatura de tempo varia de 3 a 7 indicando assinaturas de tempo de 3/4 a 7/4.
-   **track_genre**: O gênero ao qual a faixa pertence.


## Análise Exploratória de Dados

- Análise exploratória utilizando Pandas, Matplotlib e Seaborn para investigar dados, manipular e gerar visualizações em busca de insights valiosos sobre o tema de estudo (características que tornam uma música popular no Spotify).

Ver mais sobre:
- [Traduzir o DataFrame](https://www.linkedin.com/pulse/traduzindo-dados-pandasdataframe-com-google-translate-romerito-morais/?originalSubdomain=pt&utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%201/7:%20Coleta%20de%20dados%20e%20An%C3%A1lise%20Explorat%C3%B3ria&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%201/7)
- [Pandas Profiling](https://www.hashtagtreinamentos.com/pandas-profiling-no-python-ciencia-dados?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%201/7:%20Coleta%20de%20dados%20e%20An%C3%A1lise%20Explorat%C3%B3ria&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%201/7) 
- [Data Storytelling](https://paulovasconcellos.com.br/o-que-%C3%A9-data-storytelling-ac5a924dcdaf)
- [Teste de hipótese](https://ovictorviana.medium.com/teste-de-hip%C3%B3tese-com-python-ba5d751f156c)

## Pré-processamento de dados

- Modelo de classificação para prever se uma música será popular ou não
- Limpar, organizar e transformar os dados brutos em dados que possam ser usados para treinar os seus modelos
- Técnicas:  remoção de dados duplicados, preenchimento de dados ausentes, normalização dos dados, engenharia de recursos e outros
- Converter a coluna de popularidade (TARGET) em uma classe binária (1 para popular, 0 para não popular)

Ver mais sobre:
- [Data Preprocessing](https://medium.com/data-hackers/pr%C3%A9-processamento-de-dados-com-python-53b95bcf5ff4)
- [Modelo de Classificação](https://medium.com/leti-pires/predi%C3%A7%C3%A3o-da-necessidade-de-leitos-de-uti-no-hospital-s%C3%ADrio-liban%C3%AAs-811c88062f15)
- [Livro sobre Machine Learning](https://www.amazon.com.br/Machine-Learning-Refer%C3%AAncia-Trabalhando-Estruturados/dp/857522817X?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%202/7:%20%F0%9F%91%A9%F0%9F%8F%BD%E2%80%8D%F0%9F%92%BB%20Entendendo%20conceitos%20iniciais%20de%20Machine%20Learning%20e%20Pr%C3%A9-Processamento%20de%20Dados&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%202/7)

## Divisão dos dados e validação cruzada
- Divisão dos seus dados em treino, validação (ajustar hiperparâmetros) e teste
- Avaliar o desempenho do seu modelo de forma justa (generalizar) e evitar overfitting (superajustar) aos dados de treinamento
- Abordagens para **divisão dos dados**:
	- **divisão aleatória** (separação aleatória em 3 conjuntos, geralmente 70-80% para treino, 10-20% para teste, 10-20% para validação)
	- **validação cruzada** (avaliação da capacidade de generalização em diferentes conjuntos de dados, evitando overfitting - que é quando um modelo se ajusta demais aos dados de treinamento, mas não generaliza bem aos novos dados.
		- *StratifiedKFold:* especialmente útil para conjunto de dados desbalanceados
		- Existem outras abordagens, que devem ser experimentadas.
- Após a divisão dos dados, necessário fazer a **divisão dos conjuntos (X e Y)**:
	-  Nesse caso, as variáveis explicativas (X) são gênero musical, duração das músicas, instrumentação, etc. E a variável de saída (Y) representa o alvo que indicará a popularidade da música, o que temos o objetivo de prever.

Ver mais sobre:
- [Validação Cruzada: Uma abordagem Intuitiva](https://medium.com/tentando-ser-um-unic%C3%B3rnio/valida%C3%A7%C3%A3o-cruzada-uma-abordagem-intuitiva-697bb001a0ec)
- [Machine Learning: validação de modelos](https://www.alura.com.br/conteudo/machine-learning-validando-modelos?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%203/7:%20Divis%C3%A3o%20dos%20dados%20e%20valida%C3%A7%C3%A3o%20cruzada&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%203/7&vgo_ee=flM1SjllnhErYZpbPjmw1oLGdQWsJ8foDlRwj1ksNwN5aA==:xMmavEY2MXBE3iUQiSQVM%2bjRoW4xGzbf)
- [Overfitting e Underfitting](https://didatica.tech/underfitting-e-overfitting/?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%203/7:%20Divis%C3%A3o%20dos%20dados%20e%20valida%C3%A7%C3%A3o%20cruzada&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%203/7)

## Baseline e treinamento do primeiro modelo

- Criação do modelo inicial de Machine Learning (baseline). Baseline é o resultado de um modelo básico, que é a linha base para soluções mais complexas e resultados posteriores melhores.
- Um modelo base é importante para identificar problema de viés ou variância. Se esse modelo inicial tiver precisão baixa, pode indicar um problema complexo, que necessita uma abordagem mais sofisticada para resolvê-lo.
- **Modelos de classificação**:
	- **Regressão Logística** é frequentemente utilizado como baseline em problemas de classificação (por ser simples e de interpretação trivial)
		- Estima probabilidades usando função logística (apesar de se chamar "regressão", é utilizada para problemas de classificação)
	- Alternativas à explorar: **Naive Bayes, Random Forest, Decision Tree, XGBoost**, entre outros modelos.
- Para treinar: **.fit()** e para previsão: **.predict()**. 
	- "As previsões são feitas nos dados de validação/teste, que são diferentes dos dados de treinamento. Isso nos permite avaliar a capacidade do modelo de generalizar para novos dados que nunca foram vistos antes."

Ver mais sobre:
- [Scikit-Learn, biblioteca de Machine Learning](https://scikit-learn.org/stable/?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%204/7:%20%F0%9F%91%A9%F0%9F%8F%BD%E2%80%8D%F0%9F%92%BB%20Definindo%20a%20baseline%20e%20treinando%20o%20primeiro%20modelo&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%204/7)
- [Coeficiente **.coef_**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%204/7:%20%F0%9F%91%A9%F0%9F%8F%BD%E2%80%8D%F0%9F%92%BB%20Definindo%20a%20baseline%20e%20treinando%20o%20primeiro%20modelo&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%204/7)
- [Yellowbrick, biblioteca para FeatureImportance](https://www.scikit-yb.org/en/latest/api/model_selection/importances.html?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%204/7:%20%F0%9F%91%A9%F0%9F%8F%BD%E2%80%8D%F0%9F%92%BB%20Definindo%20a%20baseline%20e%20treinando%20o%20primeiro%20modelo&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%204/7)

## Validação de modelos de Machine Learning

- Utilização de métricas adequadas para validação de modelos de Machine Learning. O valor delas reflete a qualidade de um modelo, portanto, é crucial fazer as escolhas certas a fim de avaliar se o modelo atende aos requisitos.
- A comparação entre a **classe de predição do modelo** e a **classe real** é a maneira de se  avaliar um modelo.
- **Matriz de confusão** é uma forma de visualizar a distância do modelo para uma classificação perfeita. Segue exemplo:
![Performance de Machine Learning - Matriz de Confusão - Diego Nogare](https://diegonogare.net/wp-content/uploads/2020/04/matrizConfusao-600x381.png)

No contexto do projeto:
-   **TP:** verdadeiro positivo (a música é popular e modelo acertou)
-   **TN:** verdadeiro negativo (a música não é popular e o modelo acertou)
-   **FN**: falso negativo (o modelo diz que é popular, mas o valor real é não popular)
-   **FP:** falso positivo (o modelo diz que não é popular, mas o valor real é popular)

Outras métricas:
- **Acurácia**: 
	- (TP + TN) / (TP + FP + FN + TN)
	- dentre todas as classificações, mostra quantas o modelo classificou corretamente
- **Precisão**: 
	- TP / (TP + FP)
	- dentre todas as classificações de classe Popular (TP + FP) que o modelo fez, quantas estão corretas (TP)
- **Recall**:
	- TP / (TP + FN)
	- dentre todas as situações (real = SIM; TP + FN) de classe Popular como valor esperado, quais estão corretas (TP)
- **F1-score**:
	- 2 * ((precisão * recall) / (precisão + recall))
	- média harmônica entre precisão e recall

A escolha da métrica dependerá do objetivo do projeto e das características do conjunto de dados. Se o objetivo for minimizar os falsos positivos, a precisão será a métrica mais importante. Se o objetivo for minimizar os falsos negativos, o recall será a métrica mais importante.

Ver mais sobre:
- [Métricas de avaliação - Vitor Rodrigues](https://vitorborbarodrigues.medium.com/m%C3%A9tricas-de-avalia%C3%A7%C3%A3o-acur%C3%A1cia-precis%C3%A3o-recall-quais-as-diferen%C3%A7as-c8f05e0a513c)
- [Métricas de avaliação - Mario Filho](https://mariofilho.com/precisao-recall-e-f1-score-em-machine-learning/#:~:text=A%20precis%C3%A3o%20mede%20a%20quantidade%20de%20vezes%20que,que%20combina%20precis%C3%A3o%20e%20recall%20de%20maneira%20equilibrada.)

## Reamostragem de dados
- A reamostragem é uma técnica utilizada em Machine Learning para lidar com desequilíbrios nos dados (quando há grande diferença na quantidade de observações entre as classes positiva e negativa). 
- No caso do projeto, tem relação com o corte de popularidade (is_popular) em que a condição definida foi  >= 70, portanto, as músicas consideradas populares tem proporção bem menor que músicas não populares.
- Quando há esse desequilíbrio, o modelo tem tendência à "alarmes falsos", pois responderá bem para classes majoritárias (não popular), mas desempenho inferior às classes minoritárias (popular).
- Abordagens:
	- **Oversampling**: geração de novas instâncias pertencentes à classe minoritária para equilibrar a distribuição. 
	- **Undersampling**: redução da classe majoritária para um tamanho equivalente ao da classe minoritária. Isso pode ser feito aleatoriamente ou selecionando cuidadosamente quais instâncias devem ser mantidas.

![](https://ci4.googleusercontent.com/proxy/zDtzpO_LB4tgTkoZipNldrZq-_O7Gd8qV4K1-ItslicKSYzfwa6iBFDCjNDQC4ena-kDwggx8xpIaZfJyAqCNsmajITF6gtYyqStV495pKFeYPsC28cFnl-Z-k6eNiw_XTbjA0fuVytJF33N-RRMlejSgMkxa1sRYIWOvMZmpzrgbCuHalgcMbwJVABOpjULzNR4sL9y4niLrqvW4SXtMXIcrdXBin12M9ru64Hgl4zko42d9pfRmZEvrKybsrLe=s0-d-e1-ft#https://content.app-us1.com/cdn-cgi/image/dpr=2,fit=scale-down,format=auto,onerror=redirect,width=650/MpJmZ/2023/04/06/3a3ff6b8-a694-430c-87ba-00dabf019e7d.png?r=1370016386)

Para realizar o balanceamento das classes de popularidade de músicas, interessante utilizar mais de uma técnica, para comparar resultados de performance (métricas) e escolher a de melhor desempenho. 

Exemplos: *Distribuição Random UnderSampling, Distribuição Random OverSampling, Smote (Over-Sampling), Híbrido (OverSampling e UnderSampling)*

Após esse ajuste na amostra de dados, definir e testar **hiperparâmetros** para alcançar melhor performance em um modelo. Os hiperparâmetros são parâmetros do modelo que não são aprendidos durante o treinamento, mas que precisam ser definidos pelo usuário.

Exemplos:
- Número de árvores em modelo de Random Forest
- Taxa de aprendizado (learning rate) em algoritmos de gradiente descendente;
- Número de camadas e unidades em cada camada em redes neurais;
- Tipo de kernel usado em máquinas de suporte vetorial (SVMs);
- Profundidade máxima da árvore em modelos de Árvore de Decisão;
- Tamanho do lote (batch size) em modelos de Redes Neurais Convolucionais (CNNs);
- Regularização L1 ou L2 em modelos de regressão linear ou logística.

Ver mais sobre:
- [RandomizedSearchCV e a GridSearchCV](https://didatica.tech/como-encontrar-a-melhor-performance-machine-learning/?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%206/7:%20%F0%9F%91%A9%F0%9F%8F%BD%E2%80%8D%F0%9F%92%BB%20Reamostragem%20de%20dados&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%206/7)
- [AUC - ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

## Aplicando o modelo nos dados de teste e salvando os resultados

1.  Carregar o modelo previamente treinado com os dados de treinamento.
2.  Aplicar o modelo aos **dados de teste** e obter as predições resultantes.
3.  Calcular as métricas de desempenho para avaliar a qualidade das predições.
4.  Salvar as métricas de desempenho em um arquivo, juntamente com as predições obtidas.
5.  Serializar o modelo final, gerando um arquivo que pode ser carregado posteriormente para realizar novas predições ou compartilhar com outras pessoas.

Ver mais sobre:
- documentar transformações feitas aos dados originais
- [Como escrever um  README](https://www.alura.com.br/artigos/escrever-bom-readme?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Ci%C3%AAncia%20de%20Dados%207/7:%20Documente%20e%20crie%20seu%20portf%C3%B3lio&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Java%29%20Dia%207/7:%20Ordenando%20elementos&vgo_ee=qppZNTLafkIH8CTXY6bXLxwUnRnlmwiuCIJkd9A7F3A=&utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%207/7:%20Aplicando%20o%20modelo%20nos%20dados%20de%20teste%20e%20salvando%20os%20resultados&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%207/7&vgo_ee=74PBhmx9NIZms4kw%2bytGuNSxTKAr5osf8UuxemT1maV57w==:EfMicZLbR2mpPcpBj86xkq%2bHlwcrx1T7)
- Bibliotecas para salvar modelo ([Joblib](https://joblib.readthedocs.io/en/latest/?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Ci%C3%AAncia%20de%20Dados%204/7:%20%F0%9F%91%A9%F0%9F%8F%BD%E2%80%8D%F0%9F%92%BB%20Sistemas%20de%20recomenda%C3%A7%C3%A3o&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Java%29%20Dia%204/7:%20Gerando%20o%20HTML&utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%207/7:%20Aplicando%20o%20modelo%20nos%20dados%20de%20teste%20e%20salvando%20os%20resultados&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%207/7) ou [Pickle](https://docs.python.org/3/library/pickle.html?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Ci%C3%AAncia%20de%20Dados%204/7:%20%F0%9F%91%A9%F0%9F%8F%BD%E2%80%8D%F0%9F%92%BB%20Sistemas%20de%20recomenda%C3%A7%C3%A3o&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Java%29%20Dia%204/7:%20Gerando%20o%20HTML&utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%207/7:%20Aplicando%20o%20modelo%20nos%20dados%20de%20teste%20e%20salvando%20os%20resultados&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%207/7))
- Deploy ([introdução](https://blog.somostera.com/data-science/deploy-o-que-e?utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%207/7:%20Aplicando%20o%20modelo%20nos%20dados%20de%20teste%20e%20salvando%20os%20resultados&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%207/7) e [prática](https://medium.com/data-hackers/como-eu-fiz-o-deploy-do-meu-primeiro-modelo-de-machine-learning-9b416d9abc51))
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml?hl=pt-br&utm_source=ActiveCampaign&utm_medium=email&utm_content=#7DaysOfCode%20-%20Machine%20Learning%207/7:%20Aplicando%20o%20modelo%20nos%20dados%20de%20teste%20e%20salvando%20os%20resultados&utm_campaign=%5BAlura%20#7Days%20Of%20Code%5D%28Js%20e%20DOM%20-%203%C2%AA%20Ed%20%29%207/7#ml_phase_ii_feature_engineering) 
---
### Contato e licença

Sem restrições ao uso, modificações e distribuição do código fonte.
Projeto tem como referência o desafio #7DaysOfCode da Alura.

Feito com  ❤️  por  [Leonardo Mafra](https://www.linkedin.com/in/leomafra/)

