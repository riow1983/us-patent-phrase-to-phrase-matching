# us-patent-phrase-to-phrase-matching
![header](https://github.com/riow1983/us-patent-phrase-to-phrase-matching/blob/main/png/header.png)<br>
https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching<br>
どんなコンペ?:<br>
開催期間:<br>
![timeline](https://github.com/riow1983/us-patent-phrase-to-phrase-matching/blob/main/png/timeline.png)<br>
[結果](#2022-06-20)<br>  
<br>
<br>
<br>
***
## 実験管理テーブル
https://wandb.ai/riow1983/us-patent-phrase-to-phrase-matching?workspace=user-riow1983
|commitSHA|comment|Local CV|Public LB|
|----|----|----|----|
|872b1baad828122be1863753abcd5c010cf346fa|nb004/exp001;<br>one sentence to residual net w/ normal CV|0.84|0.8185|
|0501852ad4baf36948d390c67419adeccdf1cdbb|nb005/exp001;<br>two sentences w/ normal CV|0.85|0.8342|
|4a0c38371653c1c5b36bdc14898875511eae23b6|nb005/exp002;<br>two sentences w/ Closing gap CV|0.80|0.8314|
|338ac18388f5d9e22a8a836afb1b5fd2d985e3ca|nb005/exp003;<br>AWP added|0.8225|0.8383|
|b386a092e11d0bc91154b5958e76e4a56bbb6b56|nb005/exp004;<br>attention replaced w/ TFH|0.8125|0.8342|
|956e0083714cc52e0b343238ebeda9dbb65c679f|nb005/exp005;<br>TFH as weight|0.61925|-|
|0cbcef604fb14e4b8f8efc2658cbd47da4c40ea7|nb005/exp003.001;<br>aug data added|0.79|0.8257|
|c6d00daebf4babcc8a2029100178bc3911c4cecb|nb005/exp003.002;<br>aug removed and 10 folds added|0.823|0.8390|
|f6be59c4d8c0bcbcf901e057c2461196f3063458|nb005/exp003.002.001;<br>aug removed and 10 folds added w/ CPC bug fixed|0.821|0.8391|
<br>

## Late Submissions
|commitSHA|comment|Local CV|Private LB|Public LB|
|----|----|----|----|----|
<br>


## My Assets
[notebook命名規則]  
- kagglenb001{e,t,i}-hoge.ipynb: Kaggle platform上で新規作成されたKaggle notebook (kernel).
- nb001{e,t,i}-hoge.ipynb: localで新規作成されたnotebook. 
- {e:EDA, t:train, i:inference}
- kaggle platform上で新規作成され, localで編集を加えるnotebookはファイル名kagglenbをnbに変更し, 番号は変更しない.

#### Code
作成したnotebook等の説明  
|name|url|status|comment|
|----|----|----|----|
<br>





***
## 参考資料
#### Snipets
```python
# PyTorch device
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
<br>

```python
# Kaggle or Colab
import sys
if 'kaggle_web_client' in sys.modules:
    # Do something
elif 'google.colab' in sys.modules:
    # Do something
```
<br>

```python
# output dir for Kaggle or other
from pathlib import Path
KAGGLE_ENV = True if 'KAGGLE_URL_BASE' in set(os.environ.keys()) else False
INPUT_DIR = Path('../input/')

if KAGGLE_ENV:
    OUTPUT_DIR = Path('')
else:
    !mkdir nb001
    OUTPUT_DIR = INPUT_DIR / 'nb001'

# hoge_path = OUTPUT_DIR / 'hoge.csv'
```
<br>

```python
# Push to LINE

import requests

def send_line_notification(message):
    import json
    f = open("../../line.json", "r")
    json_data = json.load(f)
    line_token = json_data["kagglePush"]
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

if CFG.wandb:
    send_line_notification(f"Training of {CFG.wandbgroup} has been done. See {run.url}")
else:
    send_line_notification(f"Training of {CFG.wandbgroup} has been done.")
```
<br>

```python
# Seed everything
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)
```
<br>

```python
>>> import sys
>>> print(sys.argv)
['demo.py', 'one', 'two', 'three']

# reference: https://docs.python.org/ja/3.8/tutorial/stdlib.html
```
<br>

```bash
# JupyterのIOPub data rate exceeded エラー回避方法
!jupyter notebook --generate-config -y
!echo 'c.NotebookApp.iopub_data_rate_limit = 10000000' >> /root/.jupyter/jupyter_notebook_config.py
```
<br>

```python
# argparse example (reference: https://www.kaggle.com/code/currypurin/nbme-mlm/notebook)

%%writefile mlm.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="", required=False)
    parser.add_argument("--model_path", type=str, default="../input/deberta-v3-large/deberta-v3-large/", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--exp_num', type=str, required=True)
    parser.add_argument("--param_freeze", action='store_true', required=False)
    parser.add_argument("--num_train_epochs", type=int, default=5, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

# !python mlm.py --debug --exp_num 0
```
<br>

```bash
# 新しいブランチexp003を作成する
!git branch exp003

# ブランチを切り替える
!git checkout exp003

# ブランチを切り替えたのち, 特定のfileだけ過去バージョンに戻す
#!git log ./notebooks/hoge.ipynb
!git checkout {COMMIT SHA} ./notebooks/hoge.ipynb
```
<br>


#### Papers
|name|url|status|comment|
|----|----|----|----|
<br>


#### Blogs (Medium / Qiita / Others)
|name|url|status|comment|
|----|----|----|----|
<br>


#### Documentation (incl. Tutorial)
|name|url|status|comment|
|----|----|----|----|
<br>

#### BBC (StackOverflow / StackExchange / Quora / Reddit / Others)
|name|url|status|comment|
|----|----|----|----|
<br>

#### GitHub
|name|url|status|comment|
|----|----|----|----|
<br>

#### Hugging Face
|name|url|status|comment|
|----|----|----|----|
|Demo: U.S. Patent Phrase to Phrase Matching|[URL](https://huggingface.co/spaces/jungealexander/uspppm-demo)|Done|本コンペのデータセットを使ったデモ<br>モデルにはAI-Growth-Lab/PatentSBERTaが使われている|
|Value error : sentencepiece|[URL](https://discuss.huggingface.co/t/value-error-sentencepiece/4313)|Done|`pip install sentencepiece`で解決|
<br>

#### Colab Notebook
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Notebooks)
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Datasets)
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Discussion)
|name|url|status|comment|
|----|----|----|----|
<br>



***
## Diary

#### 2022-05-31  
<br>
<br>
<br>

#### 2022-06-20
結果はxxx/xxxだった. <br>
![private lb image](https://github.com/riow1983/us-patent-phrase-to-phrase-matching/blob/main/png/result.png)
<br>
<br>
**どのように取り組み, 何を反省しているか**<br>
<br>
**xxxについて**<br>
<br>
**xxxについて**<br>
<br>
<br>
<br>
Back to [Top](#us-patent-phrase-to-phrase-matching)



