 <div align="center">
<img src="https://coursereport-production.imgix.net/uploads/school/logo/84/original/logo-ironhack-blue.png?w=200&h=200&dpr=1&q=75">
<img src="http://www.saocarlos.usp.br/wp-content/uploads/2020/11/logo_50anos_500.250.png">
</div>

<div align="left">

[![](https://readme-typing-svg.herokuapp.com/)](https://git.io/typing-svg)
</div>
<!--GITHUB_ACTIVITY:{"rows": 5}-->

---
<div align="center">

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=ggnicolau&show_icons=true&theme=darcula)
</div>
<!--GITHUB_ACTIVITY:{"rows": 5}-->

---

<div align="left">
<div class=''text-justify''>

TCC-vocabulary-mismatch-bertopic
==============================

We used BERTopic + BM25 to conduct experiments for vocabulary mismatch, clustering data, for improving information retrieval when we assume a user want similar documents. We chose entropy as our metric.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------
#### Libraries

* numpy
* pandas
* matplotlib
* sklearn
* nltk
* transformers
* sentence-transformers
* umap-learn
* hdbscan

#### Technologies

* Python version  3.9
* Git

#### Tools

* VS Studio
* Jupyter IPython

#### Services

* Github


# **DESENVOLVIMENTO E EXPERIMENTO: BM25 + BERTOPIC**

# _1. Introdução e problema de pesquisa_

Partimos do problema de incompatibilidade de vocabulário (&quot;_vocabulary mismatch_&quot;) em que os usuários usam termos de consulta diferentes daqueles usados ​​em documentos relevantes. Esse é um dos desafios centrais sobre recuperação de informação (_Information Retrieval_) (Nogueira _et_ al, 2019, p.1).

Nosso objetivo é melhorar a recuperação de informação (RI) de um sistema de busca para uma situação específica. Partimos da hipótese de que, se um usuário está interessado em um documento específico de um grupo, também interessa a ele que retorne outros documentos do mesmo grupo (Moura, 2009, p.17 _apud_ Chakrabarti, 2003; Kobayashi e Aono, 2004).

Nossa proposta foi utilizar uma técnica de expansão dos documentos com termos que são representativos do conteúdo dos documentos, compondo um novo campo no sistema de busca com os termos extraídos. Criamos esse campo através da modelagem de tópicos dos nossos documentos melhorado por modelos de embeddings pré-treinados (BERT).

Para isso, criamos um ambiente experimental que utiliza o Elasticsearch para recuperação de informação. O Elasticsearch tem como base o algoritmo BM25 (um algoritmo clássico de RI baseado em TF-IDF) (Beiske, 2013). Nesse sentido, obtemos um _framework_ que é composto de (BERT + modelo de tópicos) + BM25.

# _2. Metodologia_

A partir desse ambiente experimental, podemos comparar os resultados da RI entre:

1. Nosso baseline, composto dos documentos originais;
2. Os documentos enriquecidos com termos extraídos de um modelo de tópicos, compondo um campo no sistema de busca com os novos termos;

Optamos por uma variação do modelo de tópicos que incorpora também modelos de embeddings pré-treinados baseados no BERT, de forma a obter tópicos com melhor conteúdo semântico. Para isso, é utilizada a biblioteca de Python chamada BERTopic (Grootendorst, 2022) e o modelo pré-treinado &quot;_distiluse-base-multilingual-cased-v1_&quot;.

Será utilizada a métrica de entropia para comparar os resultados de (a) e (b).

## _2.1. Função de Ranking - BM25_

Precisamos encontrar a relevância dos termos de queries e documentos para constituir um sistema de busca (RI). &quot;Contagem de palavras&quot; não é uma métrica suficiente para encontrar a relevância dos termos. Geralmente, os termos que mais aparecem são irrelevantes, como por exemplo conjunções (a, o, da, do etc) (Jimenez _et_ al, 2018, p.2888). Uma maneira de contornar isso e encontrar os termos mais relevantes é através de algoritmos como TF-IDF (_term frequency–inverse document frequency_). Assim, conseguimos filtrar os termos que aparecem muito em todos os documentos, porque entendemos que eles são irrelevantes; como consequência, damos maior peso para os termos que aparecem bastante em alguns documentos e podemos ver a distribuição deles ao longo do corpus de documentos.

TF-IDF

Enquanto TF-IDF como um modelo de vetorial favorece a frequência de termos e penaliza a frequência de documentos, desconsidera também o tamanho dos documentos e a saturação da frequência dos termos (Seitz, 2022). Outro modelo, chamado BM25, é uma proposta para resolver esta limitação usando um modelo probabilístico. Assim, enquanto a frequência de termo é controlada por uma função de saturação para impedir o seu crescimento linear, o parâmetro usado para calcular o tamanho médio dos documentos penaliza os documentos longos com alta frequência de termos buscados (Jimenez _et_ al, 2018, p.2888).

BM25 :

BM25 é o algoritmo mais utilizado em sistemas de busca pelo seu _tradeoff_ de acurácia e performance computacional, incorporado em sistemas baseados no Lucene (como o Elasticsearch e Solr).

## _2.2. Geração de Tópicos_

Nosso baseline (a) utiliza os documentos originais sem enriquecimento semântico. Para enriquecê-los, geramos tópicos (conjunto de palavras hierarquizadas) que irão compor um novo campo no sistema de busca com os termos extraídos dos tópicos.

Para isso, utilizamos o _framework_ da biblioteca BERTopic (Grootendorst, 2022) para criar nosso modelo de tópicos. Ele passa por algumas etapas para produzir nossos tópicos.

![](RackMultipart20220710-1-a7jhna_html_5e3a9f8bc22104ef.png)

Em um primeiro momento, aproveitamos um modelo de _embeddings_ pré-treinado para projetar os nossos documentos nesse espaço vetorial e extrair os seus vetores utilizando a biblioteca _sentence-bert_. O modelo pré-treinado é utilizado para melhorar a qualidade de nossos vetores do que seria o caso se usássemos apenas o nosso conjunto de documentos para constituir nosso espaço vetorial (Nogueira _et al_, 2019).

Em um segundo momento, aplica-se o algoritmo UMAP para redução de dimensionalidade do nosso espaço vetorial. Em seguida, aplica-se o algoritmo HDBSCAN para clusterização dos nossos documentos (Grootendorst, 2022).

Por fim, aplicamos a fórmula CTF-ICF (uma variação da fórmula TF-IDF) para os clusters desenvolvidos na fase anterior. Assim, cada cluster é convertido em um único documento, em vez de um conjunto de documentos. De cada cluster, extraímos a frequência da palavra x no cluster c, onde c se refere ao cluster que criamos anteriormente. Isso resulta em nossa representação TF baseada em cluster. Como no TF-IDF clássico, multiplicamos TF por IDF para obter a pontuação de importância por palavra em cada classe (Grootendorst, 2022).

![](RackMultipart20220710-1-a7jhna_html_c1819398d168a498.png)

O TF-IDF é usado para comparar a importância das palavras entre todos os documentos do nosso corpus. Em vez disso, tratamos apenas dos documentos em cada cluster e depois aplicamos o TF-IDF respectivamente para cada cluster como se fosse um documento (Grootendorst, 2022). O resultado seria a medida de importância para palavras dentro de um cluster. Quanto mais palavras importantes estiverem dentro de um cluster, mais ele será representativo daquele tópico. Em outras palavras, se extrairmos as palavras mais importantes por cluster, obteremos descrições de tópicos. Este modelo é chamado de TF-IDF baseado em cluster (CTF-ICF) (Grootendorst, 2022).

Uma vez que temos o nosso modelo de tópicos, podemos inferir qual é o tópico mais importante para cada documento. Assim, expandimos nossos documentos, nos aproveitamos dos termos mais importantes do seu tópico relativo com o documento, para compor um novo campo no sistema de busca que possa melhorar nossa recuperação de informação.

# _3. Critérios de Avaliação_

## _3.1. Datasets_

Em 2013, o Instituto de Ciências Matemáticas e da Computação da Universidade de São Paulo (ICMC-USP) tornou disponíveis conjuntos de documentos para avaliação de experimentos computacionais (Rossi, Marcacini, Rezende, 2013). Escolhemos o dataset de computação extraído _Open Directory Project_ (_Dmoz-Computers-500 Collection_) (Netscape _apud_ Rossi, Marcacini, Rezende, 2013). Sua distribuição possui as seguintes características:

![](RackMultipart20220710-1-a7jhna_html_4ad7f21d10259a42.png)

_Fonte: (Rossi, Marcacini, Rezende, 2013)_

Cada documento possui sua classificação. Foram escolhidos 500 documentos estratificados em 19 categorias. São essas classificações que serão usadas para computar a qualidade do nosso information retrieval.

![](RackMultipart20220710-1-a7jhna_html_be83aaabfe28666e.png)

_Fonte: (Rossi, Marcacini, Rezende, 2013)_

## _3.2. Métricas_

Em aplicações de recuperação de informação é necessário ter uma medida de confiança para testar os resultados alcançados. A entropia (incerteza) fornece uma medida valiosa na verificação de sistemas de informação, que é calculado através de conjunto dos K ​​documentos mais bem classificados para uma consulta (Grivola _et_ al., 2005). É esperado que bons resultados de recuperação proporcionem um conjunto de documentos mais homogêneos. Portanto, a entropia do conjunto de documentos deve ser menor quando o desempenho obtido para uma determinada consulta for bom.

Se a entropia do conjunto for alta, a estrutura linguística dos documentos é altamente variável. Os documentos com uma grande variabilidade linguística geram maior risco de que alguns documentos recuperados sejam irrelevantes. A probabilidade de recuperar documentos irrelevantes aumenta com K (Grivola _et_ al., 2005). Como um único documento longo e não relevante pode causar um aumento significativo na entropia, é aconselhável manter K pequeno (Grivola _et_ al., 2005). Considerando estes fatores, usamos a fórmula seguinte de entropia:

Em nossa fórmula do cálculo da entropia _H_, _nf(c)_ representa a frequência normalizada de cada classe recuperada e _K_ é a quantidade de classes encontrada na busca.

## _3.4. Experimentos Preliminares_

Conduzimos nosso experimento em um Jupyter Notebook usando a linguagem Python. Começamos configurando um servidor de Elasticsearch, criando um ambiente de BM25 para realizar nossos testes empíricos. Alimentamos o sistema com os nossos dados textuais originários do dataset _Dmoz-Computers-500_.

Primeiro, fizemos nosso experimento com nosso modelo baseline, ou seja, nossos documentos originais sem enriquecimento semântico. Testamos apenas uma query (&#39;_neural networks for linux systems_&#39;) no nosso sistema de busca e calculamos a entropia a partir da distribuição das classes no resultado da recuperação de informação. Em seguida, encontramos bi-gramas mais importantes do nosso conjunto de documentos para usarmos cada um deles como uma nova query, reservando os resultados obtidos e o cálculo de entropia para cada uma delas.

Na segunda parte do nosso estudo, transformamos nossos dados em modelo de tópicos. Começamos eliminando _stopwords_, depois vetorizamos os nossos documentos. Em seguida, reduzimos a dimensionalidade do nosso espaço vetorial com o algoritmo UMAP e clusterizamos os vetores dos nossos documentos com o algoritmo HDBSCAN. Por fim, criamos um modelo de tópico CTF-ICF. Uma vez que tivemos nosso modelo de tópicos, enriquecemos os documentos com as palavras mais relevantes do tópico mais relevante de cada documento, criando um novo campo no sistema de busca com os termos dos tópicos. Repetimos o processo aplicado para o baseline, mas agora com o novo campo, ou seja, criamos bi-gramas para usar como queries e reservamos os resultados obtidos com nosso modelo enriquecido para calcular e entropia em cada uma das queries.

Assim que tivemos os scores do nosso baseline e nossa hipótese pudemos compará-los.

![](RackMultipart20220710-1-a7jhna_html_6f61874640054722.png)

Como podemos observar em nosso gráfico de _boxplot_, a média da entropia dos documentos enriquecidos foi mais baixa do que nosso baseline (0,68 e 0,74 respectivamente), lembrando que quanto menor a entropia melhor o nosso resultado. Além disso, o terceiro quartil obteve score mais baixo nos documentos enriquecidos. Podemos concluir então que o modelo de hipótese, (BERT + topic model) + BM25, melhorou os resultados da busca. No futuro, conduziremos novos experimentos buscando outros parâmetros que poderão gerar resultados ainda melhores.

# **REFERÊNCIAS BIBLIOGRÁFICAS**

Beiske, K. (2013) &quot;Similarity in elasticsearch,&quot; _Elastic Blog_. Elastic, 26 November. Available at: [https://www.elastic.co/pt/blog/found-similarity-in-elasticsearch](https://www.elastic.co/pt/blog/found-similarity-in-elasticsearch) (Accessed: July 3, 2022).

Grivolla, J., Jourlin, P. and De Mori, R. (no date) _Automatic classification of queries by expected retrieval performance_, _Grivolla.net_. Available at: [http://www.grivolla.net/articles/sigir2005-qp.pdf](http://www.grivolla.net/articles/sigir2005-qp.pdf) (Accessed: July 3, 2022).

Grootendorst, M. P. (no date) _The algorithm_, _Github.io_. Available at: [https://maartengr.github.io/BERTopic/algorithm/algorithm.html](https://maartengr.github.io/BERTopic/algorithm/algorithm.html) (Accessed: July 3, 2022).

Jimenez, S. _et_ al. &#39;BM25-CTF: Improving TF and IDF Factors in BM25 by Using Collection Term Frequencies&#39;. 1 Jan. 2018 : 2887 – 2899. Available at:[https://www.researchgate.net/profile/Sergio-Jimenez-7/publication/325231406\_BM25-CTF\_Improving\_TF\_and\_IDF\_factors\_in\_BM25\_by\_using\_collection\_term\_frequencies/links/5b0d8349aca2725783f140e5/BM25-CTF-Improving-TF-and-IDF-factors-in-BM25-by-using-collection-term-frequencies.pdf](https://www.researchgate.net/profile/Sergio-Jimenez-7/publication/325231406_BM25-CTF_Improving_TF_and_IDF_factors_in_BM25_by_using_collection_term_frequencies/links/5b0d8349aca2725783f140e5/BM25-CTF-Improving-TF-and-IDF-factors-in-BM25-by-using-collection-term-frequencies.pdf) (Accessed: July 3, 2022).

Kamal, A. (2021) _Building your favourite TV series search engine - Information Retrieval Using BM25 Ranking_, _Medium_. Available at: [https://abishek21.medium.com/building-your-favourite-tv-series-search-engine-information-retrieval-using-bm25-ranking-8e8c54bcdb38](https://abishek21.medium.com/building-your-favourite-tv-series-search-engine-information-retrieval-using-bm25-ranking-8e8c54bcdb38) (Accessed: July 3, 2022).

Manning, Christopher D., et al. (2008) _Introduction to information retrieval_. Cambridge University Press. Available at: [https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf) (Accessed: July 3, 2022).

Moura, M. F. (2009). &#39;Contribuições para a construção de taxonomias de tópicos em domínios restritos utilizando aprendizado estatístico&#39;. Tese de Doutorado. Instituto de Ciências Matemáticas e da Computação da Universidade de São Paulo (ICMC-USP), São Carlos. Available at:[https://teses.usp.br/teses/disponiveis/55/55134/tde-05042010-162834/publico/MFM\_Tese\_5318963.pdf](https://teses.usp.br/teses/disponiveis/55/55134/tde-05042010-162834/publico/MFM_Tese_5318963.pdf)(Accessed: July 3, 2022).

Nogueira, R. _et al._ (2019) &quot;Document expansion by query prediction,&quot; _arXiv [cs.IR]_. Available at: [http://arxiv.org/abs/1904.08375](http://arxiv.org/abs/1904.08375) (Accessed: July 3, 2022).

_Open directory project.org: ODP web directory built with the DMOZ RDF database_ (no date) _Odp.org_. Available at: [http://www.odp.org/homepage.php](http://www.odp.org/homepage.php) (Accessed: July 3, 2022).

Seitz, R. (no date) _Understanding TF-IDF and BM-25_, _KMW Technology_. Available at: [https://kmwllc.com/index.php/2020/03/20/understanding-tf-idf-and-bm-25/](https://kmwllc.com/index.php/2020/03/20/understanding-tf-idf-and-bm-25/) (Accessed: July 3, 2022).

Yates, A., Nogueira, R., &amp; Lin, J. (2021). Pretrained transformers for text ranking: BERT and beyond. _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval_. Available at: [https://arxiv.org/pdf/2010.06467v1.pdf](https://arxiv.org/pdf/2010.06467v1.pdf) (Accessed: July 3, 2022).
