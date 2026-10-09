# How the method works

KESurv combines three sources of information: a forest neighborhood, the survival trajectories predicted by base learners, and the observed right-censored outcomes of selected training subjects.

## 1. Forest proximity

The fitted `RandomSurvivalForest` maps each subject to a leaf in every tree. For a query subject, KESurv considers training subjects in the same leaves. Within each leaf, bootstrap multiplicity contributes to the tree's neighbor weights. The per-tree weights are averaged into a forest proximity distribution.

The implementation uses `RandomSurvivalForest.apply()` and `estimators_samples_`; the latter supplies the in-bag bootstrap draws. KESurv caches the training leaf and bootstrap lookup at fit time so prediction does not rebuild them for every query.

## 2. Agreement from survival curves

Every base estimator predicts a survival curve for each candidate and the query. KESurv samples these curves on a shared grid from zero to the similarity horizon, then computes the normalized absolute area between each pair. A base estimator votes for a candidate when that area is at most `epsilon`.

$$
d_m(i, x) \approx \frac{1}{\tau}\sum_{j=0}^{G-2}
\left|S_m(t_j\mid x_i)-S_m(t_j\mid x)\right|(t_{j+1}-t_j)
$$

Here $m$ is a base learner, $i$ a training candidate, $x$ a query, $\tau$ the grid horizon, and $G$ the number of grid points. `min_agreement` determines how many learner votes a candidate must receive.

You can choose either a uniform grid with `similarity_time` and `n_timepoints`, or an explicit array with `similarity_times`. Smaller grids generally reduce comparison work, while changing the approximation and potentially the selected neighbors.

## 3. Weighted Kaplan–Meier

KESurv keeps qualifying candidates, renormalizes their forest weights, and calculates a weighted Kaplan–Meier curve using their observed events and censoring times. It caches the sorted time groups at fit time and reuses them for queries. If no candidate with positive forest weight passes the vote filter, the forest weights themselves provide the fallback.

The final curve is a right-continuous step function. Events recorded at time zero are included, consistent with $S(t)=P(T>t)$. The paper's written product lists strictly positive event times, so this is a documented implementation choice.

## What the method assumes

KESurv handles **right-censored** outcomes. The base estimators must return valid survival curves: finite probabilities in $[0,1]$ that do not increase over time. Model quality, censoring assumptions, preprocessing, and evaluation design still matter. The package implements the estimator itself; it does not include the paper's benchmark experiments or claim to reproduce its reported tables.

Read [Predictions and inspection](predictions.md) to examine actual neighbor votes and weights.
