# textVQG
This is repository for textVQG. Download the datasets from ICDAR 2019 St-VQA challenge "https://rrc.cvc.uab.es/?ch=11". The method also works on textVQA dataset which can be downloaded from "https://textvqa.org/".

1. Install requirements present in requirements.txt
2. To create vocabulary run python /utils/vocab.py
3. To store the textVQG dataset in .hdf5 file run python /utils/store_dataset.py
4. To train the model run python train_textvqg.py
5. To evaluate the model run python evaluate_textvqg.py

The following are some results of the proposed method:
![res1](https://user-images.githubusercontent.com/44959352/132222886-1fd59167-772d-457c-b0b9-165cc81ea25d.png)
![res2](https://user-images.githubusercontent.com/44959352/132222906-1ee94d1f-ce27-483a-8368-ad0b6b1b38a1.png)
