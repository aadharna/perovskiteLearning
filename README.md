This work has investigate the application of metric learning algorithms onto the Perovskite SD2E dataset. 

Experiments can be run by calling the `Perovskite_DistanceLearning.py` file. This has three arguments: 
  1. Which algorithm to use
  2. Whether or not to use a crystal score of 4 or (3&4) in the binary classification problem
  3. How many neighbors to use in the kNN classification after learning a metric defined by (1). 

It was shown in a recent paper that one of the better classification algorithms for determining if a perovskite crystal would form for a given reaction parameter set could be determined by using a k-nearest-neighbor algorithm where the value of k was equal to 1. In fact that classifier had a roughly 90% accuracy rate when controlling for imbalanced sample sizes of different amines. Given infinite data however, a 1-NN classifier is effectively a memorization of the training data. 

However, distance based methods such as kNN are often reliant on good underlying structure inherent in the data. This structure can be provided by reality often times as reality follows certain phyical rules, and can be exploited if the right data is captured. However, there is no reason to believe that the defaul distance metric used in distance based classifiers is appropriate for most problems. 

Given the sheer size of the perovskite data, an interesting question is `can a distance metric be learned from the data itself instead of being imposed by assumptions the scientist makes`? Furthermore, can a the kNN classifier be improved if a learned distance metric is used. 

This work investigated this hypothesis. 

In addition to invesigating the application of distance-learning algorithms, I uncovered and fixed bugs in an open-source python distance-metric-learning package: https://github.com/jlsuarezdiaz/pyDML. The squashed bugs include fixing numerical precision errors on several algorithms thereby aiding convergence of these algorithms, and updating code to use up-to-date effecient data packages. 

tl;dr -- it turns out that learning a distance function doesn't make a huge difference over the default euclidean distance, and while there are cases where classifiers can be improved, the standard euclidean distance is competitive with well-performing learned distance metrics. 

The first table here shows the best algorithm when using only the most visible crystals (crystalScore == 4) as the "good" result in a binary classification problem.
![](perovResults1.png)

The second table here shows the best algorithms when using crystals that have been scored as 3 or 4 as the "good" result in a binary classification problem.
![](perovResults2.png)


These tables above (in pictures) can also be seen in the `visualizeResults.ipynb` file. 
