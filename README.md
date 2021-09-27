Project Title: Generative adversarial networks to project the effects of
colorectal cancer treatments on human single-cells

### Part 1 - How to use the code

Please use Python 3.7. I used Pycharm on top of Anaconda but any IDE or your Python interpreter should do the trick.

Please find a list of the modules to be used/downloaded with their versions in the file "modules" in the google drive or zip.

Jupyter Notebooks does not work at all with scanpy when Google Colab works partially with scanpy. I recommend using
an IDE or your Python interpreter to run all the Python scripts where scanpy is used.

I recommend commenting out all the code for the kernel two-sample tests as it takes hours to run.

### Part 2 - How to download the data

Step(s) to download the datasets:
1) Please go on: http://tisch.comp-genomics.org/gallery/?cancer=CRC&celltype=&species=
2) Click on the dataset name
3) Click on "Download"
4) Download "Single-cell expression matrix" and "Meta information" for each dataset

The two datasets are:
- "CRC_GSE108989"
- "CRC_GSE112865_mouse_aPD1"
