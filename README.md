# Assessing the Effects of Method Hyperparameters on Knowledge Graph Embedding Quality

This repository contains the data and code used the analysis. Study was led by Oliver Lloyd as part of a PhD thesis supervised by Tom Gaunt, Yi Liu, and Patrick Rubin-Delanchy. 

UMLS-43 is a variant of the UMLS knowledge graph that is robust to data leakage through inverse relations. It has been derived by removing three edge types that should be considered problematic by [Dettmers' definition](https://arxiv.org/abs/1707.01476): 'degree_of', 'precedes', and 'derivative_of'. It is presented here as a .tsv edgelist, such that each line represents one edge in the (head, relation, tail) format.

Contact: oliver.lloyd@bristol.ac.uk.
