# Related work (first pass, 2026-10-16)

Notes are from abstracts and search snippets only; read before citing.

## Closest in spirit: many markers -> simple classifier -> generalisation
- Sitt et al. 2014, *Brain*: a large-scale screen of EEG markers of consciousness (spectral, complexity, information sharing); three partly independent marker families combine. https://pmc.ncbi.nlm.nih.gov/articles/PMC4610185
- Engemann et al. 2018, *Brain*: the same multi-marker set with tree ensembles, tested across sites and protocols. Theta/alpha dynamics were most influential. https://pubmed.ncbi.nlm.nih.gov/30285102/ (pdf: https://www.unicog.org/publications/awy251.pdf)
- hctsa (Fulcher & Jones): >7 700 time-series features with automatic selection; an EEG tutorial uses the Bonn data. https://github.com/benfulcher/hctsaTutorial_BonnEEG
- catch22: a reduced 22-feature set. Validation on neuroimaging: https://incf.org/sig/reduced-time-series-feature-library-efficiently-characterize-neural-dynamics

## Same tasks (MI / subtraction / word generation)
- Friedrich, Neuper, Scherer 2013, *PLoS ONE*, "Whatever works": 7 mental tasks; the best combination is user-specific. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3781165/
- Graz 7-task dataset (BNCI 004-2015): https://lampx.tugraz.at/~bci/database/004-2015/description.pdf
- MEG, MI vs mental arithmetic vs silent word generation (beta decrements, SVM): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10171552/

## Predictors / markers of BCI performance
- Blankertz et al. 2010, resting SMR predictor (r = 0.53, N = 80): https://pubmed.ncbi.nlm.nih.gov/20303409
- Children's BCI markers, SHAP on EEG features (alpha/theta connectivity and power): https://prism.ucalgary.ca/items/33b4434e-c6c1-4755-9c1d-0ce43cbd44cb/full
- Neuronal avalanches as a BCI-training biomarker (2025 preprint): https://www.biorxiv.org/content/10.1101/2025.05.31.657206.full.pdf
- Relevance of neurophysiological patterns for predicting MI-BCI performance: https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/hal-02971802v1

## Feature families on MI
- Entropy in MI-BCI, review of 63 papers (2025): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11853529/
- Higher-order interactions in MI EEG via information geometry (2025): https://arxiv.org/pdf/2510.14188
- Lotte et al. 2018, classifier review, 10-year update: https://www.doi.org/10.1088/1741-2552/AAB2F2

## Cross-subject generalisation surveys
- Deep-learning cross-subject EEG decoding survey (2026): https://arxiv.org/pdf/2604.27033
- Cross-subject/session emotion recognition review: https://arxiv.org/pdf/2212.08744

## Apparent gap
No paper found that ranks a large hand-crafted feature space (incl. HOI) by its
cross-subject stability under LOSO for multi-class mental-task BCI.
