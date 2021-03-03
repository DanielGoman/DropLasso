# DropLasso

A work by Maximilian Matyash and Daniel Goman

A project in the course of Topics in Regression

This project features a definition of a regression problem, its features and an algorithm which finds the best linear predictors to the problem which were defined and suggested in a paper published by B. Khalfaoui and J.P. Vert.

Link to the paper: https://arxiv.org/pdf/1802.09381.pdf

The observation of the two properties of the DropLasso problem mentioned in the paper were interesting, but we found the proof not to be sufficient, so in our work we go deeper into the two properties and give thorough proofs to them.

Implementation wise, our work features the following:

    1. We implemented the algorithm which finds the BLM to the DropLasso problem

    2. We simulate data of various noise levels

    3. We run the DropLasso model on the data, as well as a Dropout model, which is DropLasso without Lasso regularization, and Elastic Net.
