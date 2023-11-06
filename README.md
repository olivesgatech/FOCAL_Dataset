# FOCAL(Ford - OLIVES Collaboration on Active Learning): A Cost-Aware Video Dataset for Active Learning

***

This work was done in the [Omni Lab for Intelligent Visual Engineering and Science (OLIVES) @ Georgia Tech](https://ghassanalregib.info/) in collaboration with the Ford Motor Company. 
It has recently been accepted for publication in the IEEE International Conference on Big Data (Acceptance Rate 17.4%)!!
Feel free to check our lab's [Website](https://ghassanalregib.info/publications) 
and [GitHub](https://github.com/olivesgatech) for other interesting work!!!

***
## Citation

K. Kokilepersaud*, Y. Logan*, R. Benkert, C. Zhou, M. Prabhushankar, G. AlRegib, E. Corona, K. Singh, A. Parchami, "FOCAL: A Cost-Aware, Video Dataset for Active Learning," in IEEE Conference on Big Data 2023, Sorento, Italy, Dec. 15-18, 2023.
```
@article{kokilepersaud2023focal,
  title={FOCAL: A Cost-Aware, Video Dataset for Active Learning},
  author={Kokilepersaud, Kiran and Logan, Yash-Yee and Benkert, Ryan and Zhou, Chen and Prabhushankar, Mohit and AlRegib, Ghassan and Corona, Enrique and Singh, Kunjan and Parchami, Armin},
  journal={IEEE International Conference on Big Data},
  year={2023},
  publisher={IEEE}
}
```
## Abstract
In this paper, we introduce the FOCAL (Ford-OLIVES Collaboration on Active Learning) dataset which enables the study of the impact of annotation-cost within a video active learning setting. Annotation-cost refers to the time it takes an annotator to label and quality-assure a given video sequence. A practical motivation for active learning research is to minimize annotation-cost by selectively labeling informative samples that will maximize performance within a given budget constraint. However, previous work in video active learning lacks real-time annotation labels for accurately assessing cost minimization and instead operates under the assumption that annotation-cost scales linearly with the amount of data to annotate. This assumption does not take into account a variety of real-world confounding factors that contribute to  a nonlinear cost such as the effect of an assistive labeling tool and the variety of interactions within a scene such as occluded objects, weather, and motion of objects. FOCAL addresses this discrepancy by providing real annotation-cost labels for 126 video sequences across 69 unique city
scenes with a variety of weather,
lighting, and seasonal conditions. These videos have a wide range of interactions that are at the intersection of infrastructure-assisted autonomy and autonomous vehicle communities. We show through a statistical analysis of the FOCAL dataset that cost is more correlated with a variety of factors beyond just the length of a video sequence. We also introduce a set of conformal active learning algorithms that take advantage of the sequential structure of video data in order to achieve a better trade-off between annotation-cost and performance while also reducing floating point operations (FLOPS) overhead by at least 77.67%. We show how these approaches better reflect how annotations on videos are done in practice through a sequence selection framework. We further demonstrate the advantage of these approaches by introducing two performance-cost metrics and show that the best conformal active learning method is cheaper than the best traditional active learning method by 113 hours.  


## Visual Abstract

## Data

The data for this work can be found at this  location, 
with the associated paper located .

## Code Usage

1. Clone the repository with:
   ```
  git clone https://github.com/olivesgatech/FOCAL_Dataset.git
   ```
2. Set the python path in the starting directory of the repo with:
   ```
export PYTHONPATH=$PYTHONPATH:$PWD
```
3. Configure the .toml file according to the desired settings of the project. An example is provided in the provided repository. Important parameters of interest are shown below:
   ```
hello
   ```



### Acknowledgements

This work was done in collaboration with the [Retina Consultants of Texas](https://www.retinaconsultantstexas.com/).

This codebase utilized was partly constructed with code from the [Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast) Github.
