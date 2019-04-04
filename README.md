# ***This repo is WIP***

## Download
```
git clone https://github.com/yuki-mt/bert-ja-ERNIE.git
cd bert-ja-ERNIE
git submodule update --init --recursive
```

## Pretraining
1. follow bert-japanese [pretraining-from-scratch](https://github.com/yoheikikuta/bert-japanese#pretraining-from-scratch) section
  - if your machine have enough CPU cores, try `./script/create_pretraining_data.sh` instead of [this script](https://github.com/yoheikikuta/bert-japanese#creating-data-for-pretraining)
- install gcloud command and setup by running `gcloud init`
- run the following commands

```
$ YOUR_GCP_PROJECT="..."
$ ./script/cp_pretraining_data.sh
$ gsutil -m cp -r /work/data/wiki_record/ gs://${YOUR_GCP_PROJECT}/data
```

- run the script on Google Colab notebook.  [pretraining.ipynb](...)
