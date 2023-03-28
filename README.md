# spinel_to_perovskite_TL

## Introduction

This project aims to investigate the feasibility of using deep neural networks for transfer learning in crystal formation energy prediction. Specifically, we explore the possibility of transferring the knowledge learned from training a model on spinel formation energy to predict perovskite formation energy.

## Repository Structure

The repository contains the following six folders:

- **1-source domain**: contains the code for training the source domain model using spinel formation energy data.
- **2-target domain**: contains the code for training the target domain model using perovskite formation energy data.
- **3-transfer learning**: contains the code for transferring the knowledge learned from the source domain model to the target domain.
- **4-source domain predict target domain**: contains the code for directly predicting perovskite formation energy using the source domain model.
- **CE-features-generation**: contains the code for generating chemical environment (CE) features that contain information about the composition and crystal structure of materials.
- **raw_data**: contains all the raw data required for training and evaluating the DNN models.

## Prerequisites
- Python 3.x
- For Python package dependencies, please refer to the requirements.txt file

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/liyihang1024/spinel_to_perovskite_TL.git
```

2. Navigate to the project directory:

```bash
cd spinel_to_perovskite_TL
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the project:

## License

This project is licensed under the MIT License. See the LICENSE file for details.
