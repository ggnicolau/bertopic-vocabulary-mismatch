 <div align="center">
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

BERTopic + BM25 for Vocabulary Mismatch
==============================

We used BERTopic + BM25 to conduct experiments for vocabulary mismatch, clustering data, for improving information retrieval when we assume a user want similar documents. We chose entropy as our metric.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets.
    │   └── raw            <- The original, immutable data dump.
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
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------
#### Libraries

* numpy
* pandas
* matplotlib
* sklearn
* nltk
* torch
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


# **DEVELOPMENT AND EXPERIMENT: BM25 + BERTOPIC**

# _1. Introduction and research problem_

We started from the vocabulary mismatch problem (&quot;_vocabulary mismatch_&quot;) in which users use different query terms from those used in relevant documents. This is one of the central challenges in information retrieval (_Information Retrieval_) (Nogueira _et_ al, 2019, p.1).

Our goal is to improve the information retrieval (IR) of a search engine for a specific situation. We start from the hypothesis that, if a user is interested in a specific document from a group, he is also interested in returning other documents from the same group (Moura, 2009, p.17 _apud_ Chakrabarti, 2003; Kobayashi and Aono, 2004).

Our proposal was to use a document expansion technique with terms that are representative of the content of the documents, composing a new field in the search system with the extracted terms. We created this field by modeling topics from our documents enhanced by pre-trained embeddings (BERT) models.

For this, we created an experimental environment that uses Elasticsearch for information retrieval. Elasticsearch is based on the BM25 algorithm (a classic TF-IDF-based IR algorithm) (Beiske, 2013). In this sense, we get a _framework_ that is composed of (BERT + topic template) + BM25.

# _2. Methodology_

From this experimental environment, we can compare the IR results between:

1. Our baseline, composed of the original documents;
2. Documents enriched with terms extracted from a topic model, composing a field in the search system with the new terms;

We opted for a variation of the topic model that also incorporates pre-trained embedding models based on BERT, in order to obtain topics with better semantic content. For this, the Python library called BERTopic (Grootendorst, 2022) and the pre-trained model &quot;_distiluse-base-multilingual-cased-v1_&quot; are used.

The entropy metric will be used to compare the results of (a) and (b).

## _2.1. Ranking Function - BM25_

We need to find the relevance of query terms and documents to constitute a search engine (RI). &quot;Word count&quot; is not a sufficient metric to find the relevance of terms. Generally, the terms that appear the most are irrelevant, such as conjunctions (a, o, da, do etc) (Jimenez _et_ al, 2018, p.2888). One way around this and finding the most relevant terms is through algorithms like TF-IDF (_term frequency–inverse document frequency_). Thus, we were able to filter the terms that appear a lot in all documents, because we understand that they are irrelevant; as a consequence, we give greater weight to the terms that appear a lot in some documents and we can see their distribution throughout the corpus of documents.

<div align="center">
<img src=https://github.com/ggnicolau/bertopic-vocabulary-mismatch/tree/main/reports/figures/tfidf.png>
</div>

While TF-IDF as a vector model favors the frequency of terms and penalizes the frequency of documents, it also disregards the size of the documents and the saturation of the frequency of the terms (Seitz, 2022). Another model, called BM25, is a proposal to solve this limitation using a probabilistic model. Thus, while the term frequency is controlled by a saturation function to prevent its linear growth, the parameter used to calculate the average size of documents penalizes long documents with a high frequency of search terms (Jimenez _et_ al, 2018, p. 2888).

<div align="center">
<img src=https://github.com/ggnicolau/bertopic-vocabulary-mismatch/tree/main/reports/figures/bm25.png>
</div>

BM25 is the most used algorithm in search engines due to its _tradeoff_ of accuracy and computational performance, incorporated in systems based on Lucene (such as Elasticsearch and Solr).

## _2.2. Topic Generation_

Our baseline (a) uses the original documents without semantic enrichment. To enrich them, we generate topics (set of hierarchical words) that will compose a new field in the search system with the terms extracted from the topics.

For this, we use the _BERTopic_ library framework (Grootendorst, 2022) to create our topic model. It goes through a few steps to produce our topics.

<div align="center">
<img src=https://maartengr.github.io/BERTopic/img/algorithm.png>
</div>

_Source: (__Grootendorst, 2022)_

At first, we took advantage of a _embeddings_ to project our documents in this vector space and extract their vectors using the _sentence-bert_. The pre-trained model is used to improve the quality of our vectors than would be the case if we only used our document set to constitute our vector space (Nogueira _et al_, 2019).

In a second moment, the UMAP algorithm is applied to reduce the dimensionality of our vector space. Then, the HDBSCAN algorithm is applied to cluster our documents (Grootendorst, 2022).

Finally, we apply the CTF-ICF formula (a variation of the TF-IDF formula) to the clusters developed in the previous phase. Thus, each cluster is converted into a single document rather than a set of documents. From each cluster, we extract the frequency of the word x in cluster c, where c refers to the cluster we created earlier. This results in our cluster-based TF representation. As in the classic TF-IDF, we multiply TF by IDF to obtain the per-word importance score in each class (Grootendorst, 2022).

<div align="center">
<img src=https://maartengr.github.io/BERTopic/img/ctfidf.png>

_Source: (__Grootendorst, 2022)_

The TF-IDF is used to compare the importance of words across all documents in our corpus. Instead, we only deal with the documents in each cluster and then apply the TF-IDF respectively to each cluster as if it were a document (Grootendorst, 2022). The result would be the measure of importance for words within a cluster. The more important words that are within a cluster, the more representative it is for that topic. In other words, if we extract the most important words by cluster, we get topic descriptions. This model is called cluster-based TF-IDF (CTF-ICF) (Grootendorst, 2022).

Once we have our topic model, we can infer which topic is most important for each document. Thus, we expand our documents, taking advantage of the most important terms of their topic relative to the document, to compose a new field in the search system that can improve our information retrieval.

# _3. Assessment Criteria_

## _3.1. Datasets_

In 2013, the Institute of Mathematics and Computer Sciences of the University of São Paulo (ICMC-USP) made available sets of documents for the evaluation of computational experiments (Rossi, Marcacini, Rezende, 2013). We chose the computing dataset extracted from the _Open Directory Project_ (_Dmoz-Computers-500 Collection_) (Netscape _apud_ Rossi, Marcacini, Rezende, 2013). Its distribution has the following characteristics:

<div align="center">
<img src=https://github.com/ggnicolau/bertopic-vocabulary-mismatch/tree/main/reports/figures/table1.png>
</div>

_Source: (Rossi, Marcacini, Rezende, 2013)_

Each document has its classification. 500 documents were chosen, stratified into 19 categories. It is these ratings that will be used to compute the quality of our information retrieval.

<div align="center">
<img src=https://github.com/ggnicolau/bertopic-vocabulary-mismatch/tree/main/reports/figures/table2.png>
</div>

_Source: (Rossi, Marcacini, Rezende, 2013)_

## _3.2. Metrics_

In information retrieval applications it is necessary to have a reliable measure to test the results achieved. Entropy (uncertainty) provides a valuable measure in the verification of information systems, which is calculated through the set of the K highest ranked documents for a query (Grivola _et_ al., 2005). Good retrieval results are expected to provide a more homogeneous set of documents. Therefore, the entropy of the document set should be lower when the performance obtained for a given query is good.

If the entropy of the set is high, the linguistic structure of the documents is highly variable. Documents with high linguistic variability pose a greater risk that some retrieved documents will be irrelevant. The probability of retrieving irrelevant documents increases with K (Grivola _et_ al., 2005). As a single long, non-relevant document can cause a significant increase in entropy, it is advisable to keep K small (Grivola _et_ al., 2005). Considering these factors, we use the following entropy formula:

In our entropy calculation formula _H_, _nf(c)_ represents the normalized frequency of each retrieved class and _K_ is the number of classes found in the search.

# _4. Preliminary Experiments_

We conducted our experiment in a Jupyter Notebook using the Python language. We started by configuring an Elasticsearch server, creating a BM25 environment to carry out our empirical tests. We feed the system our textual data from the _Dmoz-Computers-500_.

First, we did our experiment with our baseline model, that is, our original documents without semantic enrichment. We tested just one query (&#39;_neural networks for linux systems_&#39;) in our search engine and calculated entropy from the distribution of classes in the information retrieval result. Next, we find the most important bigrams from our set of documents to use each of them as a new query, reserving the results obtained and the entropy calculation for each one.

In the second part of our study, we transformed our data into a topic model. We start by eliminating _stopwords_, then vectorize our documents. We then reduce the dimensionality of our vector space with the UMAP algorithm and cluster the vectors of our documents with the HDBSCAN algorithm. Finally, we created a CTF-ICF topic template. Once we had our topic model, we enriched the documents with the most relevant words from the most relevant topic in each document, creating a new field in the search system with the topic terms. We repeat the process applied to the baseline, but now with the new field, that is, we create bigrams to use as queries and reserve the results obtained with our enriched model to calculate the entropy in each of the queries.

Once we had our baseline and our hypothesis we were able to compare them.

<div align="center">
<img src=https://github.com/ggnicolau/bertopic-vocabulary-mismatch/tree/main/reports/figures/ir_comparision_github.png>
</div>

As we can see in our _boxplot_, the mean entropy of the enriched documents was lower than our baseline (0.68 and 0.74 respectively), remembering that the lower the entropy, the better our result. In addition, the third quartile had a lower score in enriched documents. We can therefore conclude that the hypothesis model, (BERT + topic model) + BM25, improved the search results. In the future, we will conduct new experiments looking for other parameters that may generate even better results.

# **BIBLIOGRAPHIC REFERENCES**

Beiske, K. (2013) &quot;Similarity in elasticsearch,&quot; _Elastic Blog_. Elastic, 26 November. Available at: [https://www.elastic.co/en/blog/found-similarity-in-elasticsearch](https://www.elastic.co/pt/blog/found-similarity-in-elasticsearch) (Accessed: July 3, 2022).

Grivolla, J., Jourlin, P. and De Mori, R. (no date) _Automatic classification of queries by expected retrieval performance_, _Grivolla.net_. Available at: [http://www.grivolla.net/articles/sigir2005-qp.pdf](http://www.grivolla.net/articles/sigir2005-qp.pdf) (Accessed: July 3, 2022).

Grootendorst, MP (no date) _The algorithm_, _Github.io_. Available at: [https://maartengr.github.io/BERTopic/algorithm/algorithm.html](https://maartengr.github.io/BERTopic/algorithm/algorithm.html) (Accessed: July 3, 2022).

Jimenez, S. _et_ al. &#39;BM25-CTF: Improving TF and IDF Factors in BM25 by Using Collection Term Frequencies&#39;. 1 Jan. 2018 : 2887 – 2899. Available at:[https://www.researchgate.net/profile/Sergio-Jimenez-7/publication/325231406\_BM25-CTF\_Improving\_TF\_and\_IDF\_factors\_in\_BM25\_by\_using\_collection\_term\_frequencies/links/5b0d8349aca2725783f140e5/BM25-CTF-Improving-TF-and-in-using-factor-by-IDF-25 collection-term-frequencies.pdf](https://www.researchgate.net/profile/Sergio-Jimenez-7/publication/325231406_BM25-CTF_Improving_TF_and_IDF_factors_in_BM25_by_using_collection_term_frequencies/links/5b0d8349aca2725783f140e5/BM25-CTF-Improving-TF-and-IDF-factors-in-BM25-by-using-collection-term-frequencies.pdf) (Accessed: July 3, 2022).

Kamal, A. (2021) _Building your favorite TV series search engine - Information Retrieval Using BM25 Ranking_, _Medium_. Available at: [https://abishek21.medium.com/building-your-favourite-tv-series-search-engine-information-retrieval-using-bm25-ranking-8e8c54bcdb38](https://abishek21.medium.com/building-your-favourite-tv-series-search-engine-information-retrieval-using-bm25-ranking-8e8c54bcdb38) (Accessed: July 3, 2022).

Manning, Christopher D., et al. (2008) _Introduction to information retrieval_. Cambridge University Press. Available at: [https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf) (Accessed: July 3, 2022).

Moura, MF (2009). &#39;Contributions to the construction of topic taxonomies in restricted domains using statistical learning&#39;. Doctoral thesis. Institute of Mathematics and Computer Sciences of the University of São Paulo (ICMC-USP), São Carlos. Available at:[https://teses.usp.br/teses/disponiveis/55/55134/tde-05042010-162834/publico/MFM\_Tese\_5318963.pdf](https://teses.usp.br/teses/disponiveis/55/55134/tde-05042010-162834/publico/MFM_Tese_5318963.pdf)(Accessed: July 3, 2022).

Nogueira, R. _et al._ (2019) &quot;Document expansion by query prediction,&quot; _arXiv [cs.IR]_. Available at: [http://arxiv.org/abs/1904.08375](http://arxiv.org/abs/1904.08375) (Accessed: July 3, 2022).

_Open directory project.org: ODP web directory built with the DMOZ RDF database_ (no date) _Odp.org_. Available at: [http://www.odp.org/homepage.php](http://www.odp.org/homepage.php) (Accessed: July 3, 2022).

Seitz, R. (no date) _Understanding TF-IDF and BM-25_, _KMW Technology_. Available at: [https://kmwllc.com/index.php/2020/03/20/understanding-tf-idf-and-bm-25/](https://kmwllc.com/index.php/2020/03/20/understanding-tf-idf-and-bm-25/) (Accessed: July 3, 2022).

Yates, A., Nogueira, R., &amp; Lin, J. (2021). Pretrained transformers for text ranking: BERT and beyond. _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval_. Available at: [https://arxiv.org/pdf/2010.06467v1.pdf](https://arxiv.org/pdf/2010.06467v1.pdf) (Accessed: July 3, 2022).

## Version

0.0.5.0

## Author

* **Guilherme Giuliano Nicolau**: @ggnicolau (<https://github.com/ggnicolau>)
* **Ricardo Marcondes Marcacini**: @rmarcacini (<https://github.com/rmarcacini>)

</div>

<!--GITHUB_ACTIVITY:{"rows": 5}-->

---

<div align="center">

<br/><br/>
![Quote](https://github-readme-quotes.herokuapp.com/quote?theme=dark&animation=grow_out_in)

[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=ggnicolau&layout=compact)](https://github.com/anuraghazra/github-readme-stats)

![https://medium.com/@ggnicolau](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)

</div>
