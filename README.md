# LDA Topic Modeling and Bayesian Optimization

This repository contains the complete pipeline for tuning a Latent Dirichlet Allocation (LDA) topic model using Bayesian optimization, applied to the 20 Newsgroups dataset. The primary workflow and all associated code are located in the file `lda_20newsgroups_v3.ipynb`.

## Overview

Hyperparameter tuning for LDA is computationally expensive, with a standard fit on the full 18,000-document 20 Newsgroups corpus taking up to 413 seconds per evaluation. The provided notebook, `lda_20newsgroups_v3.ipynb`, implements a fast-track search strategy that reduces evaluation time to approximately 15 seconds. After 30 Bayesian evaluations, the optimal parameters are used to train a final model on the entire dataset, maximizing the $C_v$ coherence score.

## Speed and Optimization Strategy

The pipeline achieves rapid evaluation times by addressing specific bottlenecks:

* **Data Subsampling:** Instead of evaluating on 18,000 documents, the search phase uses a stratified 4,000-document subsample, maintaining an equal number of documents per category.


* **Reduced Iterations:** The LDA `max_iter` parameter is dropped to 7 during the search phase, while the final model utilizes 40 iterations for thorough training.


* **Targeted Metrics:** The $C_v$ coherence score is computed strictly on the subsample during the optimization phase.


* **Parallel Execution:** The pipeline keeps `n_jobs=-1` to maximize hardware utilization, balancing the reduced data load.



## Bayesian Search Space

The hyperparameter search is driven by `scikit-optimize` (`gp_minimize`) and targets the following parameters:

* **Number of Topics (`n_components`):** Integer range from 10 to 35 (uniform prior).


* **Document-Topic Prior (`doc_topic_prior` / Alpha):** Real range from 0.01 to 1.0 (log-uniform prior).


* **Topic-Word Prior (`topic_word_prior` / Beta):** Real range from 0.001 to 0.5 (log-uniform prior).


* **Learning Decay (`learning_decay`):** Real range from 0.50 to 0.90 (uniform prior).


* **Search Allocation:** The process performs 8 initial random points followed by 22 Gaussian Process evaluations using Expected Improvement, totaling 30 calls.



## Pipeline Architecture

The workflow in `lda_20newsgroups_bayesian_v2.ipynb` follows a linear sequence:

1. **Data Ingestion:** Downloads the 20 Newsgroups dataset using `kagglehub`.


2. **Preprocessing:** Cleans text by stripping email headers, quoted lines, words under three characters, and applies a custom list of stop words alongside standard English stop words.


3. **Vectorization:** Processes text using `CountVectorizer` capped at 5,000 maximum features.


4. **Baseline Modeling:** Trains a default 20-topic LDA model to establish baseline Log-Likelihood, Perplexity, and Coherence metrics.


5. **Optimization:** Executes the Bayesian search loop over the subsample, logging performance to `bayesian_search_results.csv`.


6. **Final Training:** Retrains the LDA model on the full corpus using the best discovered configuration.



## Visualization and Outputs

The script automatically generates and saves several analytical plots:

* **Search Diagnostics (`fig1_bayes_diagnostics.png`):** A 2x2 grid showing the convergence curve and scatter plots of parameter values versus coherence.


* **Metric Comparison (`fig2_comparison.png`):** Bar charts comparing baseline and optimized metrics.


* **Parallel Coordinates (`fig3_parallel_coords.png`):** A parallel coordinates plot highlighting the best hyperparameter configuration across all evaluations.


* **Document Overview (`fig4_doc_overview.png`):** A distribution bar chart of documents across topics alongside a final metrics scorecard.


* **Word Weights (`fig5_heatmap.png`):** A heatmap displaying the normalized weights of the top 10 words for each topic.


* **Word Clouds (`fig6_wordclouds.png`):** Individual word clouds generated for every topic.


* **Category Alignment (`fig7_confidence_category.png`):** A histogram of document-topic confidence and a row-normalized heatmap mapping dominant topics to original newsgroup categories.


* **Interactive Dashboard (`lda_interactive.html`):** A standalone interactive visualization generated via `pyLDAvis`.



## Dependencies

Ensure the following libraries are installed to run the notebook:

* `scikit-learn`

* `gensim`

* `scikit-optimize`

* `pandas` and `numpy`

* `matplotlib` and `seaborn`

* `wordcloud`

* `pyLDAvis`

* `kagglehub`
