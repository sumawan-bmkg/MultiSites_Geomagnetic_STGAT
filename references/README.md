# References Directory

This directory contains research papers, articles, and other reference materials related to the ANTIGRAVITY project.

## Key Papers

### Graph Neural Networks

1. **Kipf & Welling (2017)**  
   *Semi-Supervised Classification with Graph Convolutional Networks*  
   ICLR 2017  
   [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

2. **Veličković et al. (2018)**  
   *Graph Attention Networks*  
   ICLR 2018  
   [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

3. **Wu et al. (2020)**  
   *A Comprehensive Survey on Graph Neural Networks*  
   IEEE Transactions on Neural Networks and Learning Systems  
   [arXiv:1901.00596](https://arxiv.org/abs/1901.00596)

4. **Battaglia et al. (2018)**  
   *Relational inductive biases, deep learning, and graph networks*  
   [arXiv:1806.01261](https://arxiv.org/abs/1806.01261)

### Physics-Informed Neural Networks

5. **Raissi et al. (2019)**  
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*  
   Journal of Computational Physics  
   [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

6. **Karniadakis et al. (2021)**  
   *Physics-informed machine learning*  
   Nature Reviews Physics  
   DOI: 10.1038/s42254-021-00314-5

### Spatio-Temporal Learning

7. **Yu et al. (2018)**  
   *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*  
   IJCAI 2018  
   [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)

8. **Li et al. (2018)**  
   *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*  
   ICLR 2018  
   [arXiv:1707.01926](https://arxiv.org/abs/1707.01926)

### Earthquake Prediction & Seismology

9. **DeVries et al. (2018)**  
   *Deep learning of aftershock patterns following large earthquakes*  
   Nature  
   DOI: 10.1038/s41586-018-0438-y

10. **Rouet-Leduc et al. (2017)**  
    *Machine Learning Predicts Laboratory Earthquakes*  
    Geophysical Research Letters  
    DOI: 10.1002/2017GL074677

11. **Bergen et al. (2019)**  
    *Machine learning for data-driven discovery in solid Earth geoscience*  
    Science  
    DOI: 10.1126/science.aau0323

### Seismic Precursors

12. **Uyeda et al. (2009)**  
    *Electric and magnetic phenomena observed before the volcano-seismic activity in 2000 in the Izu Island Region, Japan*  
    PNAS  
    DOI: 10.1073/pnas.0900817106

13. **Hayakawa & Hobara (2010)**  
    *Current status of seismo-electromagnetics for short-term earthquake prediction*  
    Geomatics, Natural Hazards and Risk  
    DOI: 10.1080/19475705.2010.486933

### Space Weather Effects

14. **Love & Thomas (2013)**  
    *Insignificant solar-terrestrial triggering of earthquakes*  
    Geophysical Research Letters  
    DOI: 10.1002/grl.50211

15. **Marchitelli et al. (2020)**  
    *On the correlation between solar activity and large earthquakes worldwide*  
    Scientific Reports  
    DOI: 10.1038/s41598-020-67860-3

## Related Topics

### Multi-Task Learning
- Caruana (1997) - Multitask Learning
- Ruder (2017) - An Overview of Multi-Task Learning in Deep Neural Networks

### Attention Mechanisms
- Vaswani et al. (2017) - Attention Is All You Need
- Bahdanau et al. (2015) - Neural Machine Translation by Jointly Learning to Align and Translate

### Circular Statistics
- Mardia & Jupp (2000) - Directional Statistics
- Fisher (1993) - Statistical Analysis of Circular Data

### Geospatial Analysis
- Haversine formula for great-circle distance
- Vincenty formula for geodesic distance
- Tectonic plate boundaries and seismicity

## Online Resources

### Datasets
- USGS Earthquake Catalog: https://earthquake.usgs.gov/
- BMKG (Indonesia): https://www.bmkg.go.id/
- ISC Bulletin: http://www.isc.ac.uk/

### Space Weather
- NOAA Space Weather Prediction Center: https://www.swpc.noaa.gov/
- Kp Index: https://www.gfz-potsdam.de/en/kp-index/
- Dst Index: http://wdc.kugi.kyoto-u.ac.jp/

### Tools & Libraries
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- ObsPy (Seismology): https://docs.obspy.org/
- GeoPandas: https://geopandas.org/

## File Organization

```
references/
├── papers/
│   ├── gnn/                    # Graph Neural Networks
│   ├── pinn/                   # Physics-Informed Neural Networks
│   ├── earthquake/             # Earthquake prediction
│   └── space_weather/          # Space weather effects
│
├── books/
│   ├── deep_learning/
│   └── seismology/
│
├── datasets/
│   └── descriptions/           # Dataset documentation
│
└── notes/
    └── literature_review.md    # Summary of key findings
```

## Citation Format

When citing papers in documentation or code:

```
Author et al. (Year) - Title
Journal/Conference
DOI or arXiv link
```

Example:
```python
# Based on Graph Attention Networks (Veličković et al., 2018)
# https://arxiv.org/abs/1710.10903
class GATLayer(nn.Module):
    ...
```

## Notes

- PDF files are in `.gitignore` (copyright reasons)
- Keep only links and citations in this README
- For personal use, download papers to local `references/papers/` directory
- Always cite sources in code and documentation

## Contributing

When adding new references:
1. Add citation to this README
2. Categorize appropriately
3. Include DOI or arXiv link
4. Add brief description if relevant
5. Update literature review notes if applicable
