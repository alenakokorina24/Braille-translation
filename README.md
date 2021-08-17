## Optical Braille Recognition with Faster R-CNN

This repository contains a potential solution for optical Braille recognition problem — the problem that still remains not fully resolved due to lack of training data. The presented solution uses deep learning methods, specifically, neural networks.

> Braille is a raised-point tactile font designed for writing and reading by blind and visually impaired people.

**Data generation**

First of all, due to lack of (especially labeled) data, this system generates synthetic "pages" with texts in Braille. The script is presented in [braille-dataset-generation.ipynb](https://github.com/alenakokorina24/Braille-translation/blob/main/braille-dataset-generation.ipynb).

Real images were used as a base for creating artifical data. I used datasets (only those in Russian, initially not labeled) that were presented in the [World AI & Data Challenge](https://git.asi.ru/tasks/world-ai-and-data-challenge/braille-text-optical-recognition) competition for the same problem. 

- 350 samples were cut from multiple real photos of sheets with texts in Braille. These samples represent Russian alphabet, punctuation marks and special symbols (capital letter sign, digital sign). Each character is represented in approximately 8 samples. There are samples of noise too — I added samples of digits (0-9) into this selection to make network more noise-proof. Also eaсh sample had some sort of preprocessing — background removal and conversion into grayscale.
- Then, using these (already labeled) character samples, I created the sheets dataset. Each page was created according to some rules that are mentioned in the script itself. The algorithm uses certain indents between letters, words, lines. The number of lines in page or words in line or letters in word are generated randomly but with certain probabilities that were picked empirically to make synthetic pages look more realistic. Also each character has small random offset and rotation. At the end the gaussian noise filter is applied to generated page.  
 
![Screenshot from 2021-08-05 16-53-55](https://user-images.githubusercontent.com/65346868/128330921-6ddbb226-d9cd-4e3b-b66a-e0e316beff6d.png)
 > A crop from generated picture.

So there were generated 4000 labeled synthetic pages. All files are stored in Kaggle's cloud.
- [Samples](https://www.kaggle.com/alenakokorina/sheets-w-noise)
- [Labels](https://www.kaggle.com/alenakokorina/sheets-labels-w-noise)

**Neural network**

The whole training process is contained in [this notebook](https://github.com/alenakokorina24/Braille-translation/blob/main/faster-rcnn-train.ipynb).
Best model weights are stored [here](https://www.kaggle.com/alenakokorina/braille-weights?select=state_dict_model.pt).

![Screenshot from 2021-08-05 20-02-11](https://user-images.githubusercontent.com/65346868/128354298-665687f0-619d-4120-bb9a-b80c1a711f58.png)
> Object detection and classification results (on real sample).

## How to use

In order to use the system, you need to download the [weights](https://www.kaggle.com/alenakokorina/braille-weights?select=state_dict_model.pt), the [dictionary](https://www.kaggle.com/alenakokorina/braille-dict) and the script itself.




