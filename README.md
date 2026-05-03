# siRN2A

## Installation
We recommend installing siRN2A within a virtual environment. For Conda, use the following command to create and activate a dedicated environment:

```bash
conda create -n siRN2A python=3.10.6
conda activate siRN2A
```

Next, install the required dependencies with:
```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


## Tutorial
### Training
To train the siRN2A model, please use the following code. For demonstration purposes, we have provided sample data located at train_data_path and test_data_path.

```bash
python main.py --train_csv ../data/train_data.csv --test_csv ../data/test_data.csv
```

### Inference
For inference, simply input the file containing the relevant features to run the predictions. For your reference, we have provided a .ckpt checkpoint file along with an example sample for prediction.
```bash
python main.py --inference_csv ../data/inference_data.csv --ckpt_path ./checkpoint/checkpoint.pth
```
DOI: 10.5281/zenodo.19877592

# siRNAOD3 Database
As our work is currently under submission, we have provided sample data for demonstration purposes, which is located in the data/ directory. The complete siRNAOD3 database will be fully open-sourced and available for public download and citation upon the acceptance of this paper. For further inquiries and specific needs, please contact: huzixin@fudan.edu.cn, smwang23@m.fudan.edu.cn.

The siRNAOD3 database is dedicated to building a comprehensive and open-source multimodal database for siRNAs. It aims to accelerate the screening and design of siRNA sequences, thereby empowering the rapid development of the siRNA therapeutics field. 

Furthermore, we actively welcome collaborative opportunities with researchers and institutions worldwide.