## Welcome to GitHub Page for the project I've created as my Bachelor's Thesis and later presented at [BIOIMAGING '22 conference](https://http://www.bioimaging.biostec.org/).

The primary objective of this page is to present summary of the paper as well as to hold all the necessary links at one place (code, testing dataset, ...).

### Method Summary
Our method aims to bring automation in the process of orthodontics treatment. Some facts that stem from this:
  - We are dealing with difficult cases of orthodontics patients, i.e. the input meshes are obtained by scanning the dental arches of unhealthy patients with various kinds of teeth misplacements, shiftings, etc. 
  - It should not rely on *strong* PCs with modern GPUs - it should run on *normal* consumer computer (imagine for example the computer of your clinitian) in seconds.
 
#### Outline
![](outline.png)
 To meet the aforementioned needs, the learning task is to regress heatmaps of Gaussians in 2D. Afterward, using multiple viewpoints, the result postion is calculated from valid viewpoints only -- incorrect ones are eliminated using RANSAC and least-squares fit.
 
### Overall Results
On a testing dataset of complicated orthodontic cases, we report the landmarking accuracy of **0.75 +- 0.96 mm**. As for the missing teeth, our method detects correctly the presence of teeth in **97.68%** cases. These results are achieved using Attention U-Net, 100 viewpoints and RANSAC post-processing.

### Dataset Availability
Unfortunately, we are unable to share the dataset used for training, validation, and testing, as it contains private medical data for which we did not obtain the necessary approval for public release. Consequently, the trained model weights cannot be made publicly available either.

### Contact
Do not hesitate to ask in case you have any questions -- my LinkedIn profile: [click](https://www.linkedin.com/in/tibor-kub%C3%ADk-7a4364181/) or write me an email: [click](mailto:xkubik34@stud.fit.vutbr.cz).

### Special Thanks
Special thanks goes to Michal Španěl, the best supervisor, and to [TESCAN 3DIM, s.r.o.](https://www.linkedin.com/company/tescan-3dim/) for providing the dataset and funding.
