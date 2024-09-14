# jdrama-kb-ratings-prediction
Improving Audience Ratings Prediction of Japanese TV Dramas using Knowledge-based Embeddings (ICMLA2024)

`datasets` contains the datasets used in this paper. `merged2.csv` provides the data post-collection from ARTV. `extracted_actor_positions_artv.csv` provides the same after subject to preprocessing.

`collection` contains the scripts used for data collection from ARTV and extraction of plot synopsis information from Japanese Wikipedia.

`analysis` contains the scripts used for implementing the supervised learning testbed, as presented in the paper.

To achieve speed-ups, the Python library scikit-learn-intelex (2024.4.0) and the GNU shell tool parallel are used.

O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
