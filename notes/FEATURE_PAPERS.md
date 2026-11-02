# Papers behind the feature bank (2026-11-01)

DOIs are filled in from memory and not yet verified; check before citing.

## catch22 family (7 216 columns)
- Fulcher, Little & Jones 2013, *J R Soc Interface*: highly comparative time-series analysis (hctsa). https://doi.org/10.1098/rsif.2013.0048
- Fulcher & Jones 2017, *Cell Systems*: hctsa, >7 700 features. https://doi.org/10.1016/j.cels.2017.10.001
- hctsa EEG tutorial (Bonn data): https://github.com/benfulcher/hctsaTutorial_BonnEEG
- Lubba et al. 2019, *Data Min Knowl Disc*: catch22. https://doi.org/10.1007/s10618-019-00647-x
- catch22 validation on neuroimaging (INCF): https://incf.org/sig/reduced-time-series-feature-library-efficiently-characterize-neural-dynamics

## Multi-marker approach
- Sitt et al. 2014, *Brain*: https://pmc.ncbi.nlm.nih.gov/articles/PMC4610185
- Engemann et al. 2018, *Brain*: https://pubmed.ncbi.nlm.nih.gov/30285102/

## Other feature families (~50 000 columns in total)

| family | paper | link |
|---|---|---|
| regions / bands, classic_ref | Friedrich, Scherer & Neuper 2012, *Int J Psychophysiol* | https://doi.org/10.1016/j.ijpsycho.2012.01.014 |
| spectral (30–45 Hz 1/f slope) | Gao, Peterson & Voytek 2017, *NeuroImage* | https://doi.org/10.1016/j.neuroimage.2017.06.078 |
| bursts | Shin et al. 2017, *eLife* | https://doi.org/10.7554/eLife.29086 |
| bursts | Little et al. 2019, *PLoS Biol* | https://doi.org/10.1371/journal.pbio.3000479 |
| complexity_ms (multiscale entropy) | Costa, Goldberger & Peng 2002, *PRL* | https://doi.org/10.1103/PhysRevLett.89.068102 |
| conn (wPLI) | Vinck et al. 2011, *NeuroImage* | https://doi.org/10.1016/j.neuroimage.2011.01.055 |
| conn (AEC-orth) | Hipp et al. 2012, *Nat Neurosci* | https://doi.org/10.1038/nn.3101 |
| conn (weighted clustering) | Onnela et al. 2005, *Phys Rev E* | https://doi.org/10.1103/PhysRevE.71.065103 |
| wsmi | King et al. 2013, *Curr Biol* | https://doi.org/10.1016/j.cub.2013.07.075 |
| wsmi | Sitt et al. 2014, *Brain* | https://pmc.ncbi.nlm.nih.gov/articles/PMC4610185 |
| hoi (O-information) | Rosas et al. 2019, *Phys Rev E* | https://doi.org/10.1103/PhysRevE.100.032305 |
| cfc (PAC) | Canolty et al. 2006, *Science* | https://doi.org/10.1126/science.1128115 |

temporal, temporal_band, wavelet, cov and cov_ch cite no paper in their code.
Background papers not tied to a feature family are in [RELATED.md](RELATED.md).
