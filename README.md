# recboletest
[`RecBole`](https://www.recbole.io/index.html)を使ったゼミの課題に取り組むうえで，気になったことを以下に書き記す．\
2025/12/08に作成．`sinki.py`と`sinki.yaml`は動かない．\
2025/12/22から，[`stat.ink`](https://stat.ink/)のデータを利用した推薦システムの作成を開始．


## 2025/12/08
`Anaconda Prompt` を起動し，
```
conda activate recbole
```
とコマンド上で実行．\
`cd` でファイルのある階層に移動できる．\
`python xxx.py` で`Python`を実行できる．\
`dir` コマンドで，作成した `saved` フォルダを見つける．\
`saved` フォルダ内にある`xxx.pth`を使って，新規ユーザの推薦を行う．`pth`は`PyTorch`のモデルデータである．\
`hyper parameter` の設定は，レファレンスを読んで理解する．\
`epoch`はエポック数であり，これは学習する回数である．エポック数が大きいとき，ノートパソコンだと遅すぎて終わらない．\
`xxx.yaml`にはパラメータの詳細を記す．`yaml`とは，`html`や`xml`などのデータ形式の一つ．\
学習は，何度か試して，一番良いパラメータを探す．\
`TensorBoard` で機械学習の結果を可視化，学習曲線をプロットできる．\
自分のノートパソコンだと厳しいので，サーバーを使う．\
サーバーを使う際は，以下のコマンドを実行する．
```
ssh-keygen -t ED25519
```
これで，鍵を取得できる．


## 2025/12/22
`Python`は仮想環境で行うことを推奨．以下のコマンドを実行する．
```
python -m venv rec_env
source rec_env/bin/activate
```
`recbole`のインストールは，以下のコマンドを実行する．
```
pip install recbole
```
`Python 3.13`だと，新しすぎて`PyTorch`などが対応していなく，エラーが起きる．その為，仮想環境を使ったうえで，`Python3`のライブラリを直接書き換える．\
`PyTorch`で学習していると，早期終了`Early Stopping`することがある．これは，過学習を防ぐためである．\
サーバーでは，`vim`を使ってプログラムファイルを作る．`touch test.py`と適当にファイルを作ったら，`vi test.py`としてターミナル上で`vim`を開く．`i`と打つことで，編集ができる．やめるときは，`escape`を押し，閉じるときは`:wq`とする．`:aqw`とする場合もある．サーバーを閉じるときは，`exit`とする．\
`VS Code`は，拡張機能の`Remote - SSH`で接続することができる．エクスプローラーより，任意のフォルダを開いて使う．


## 2025/12/23
`Nintendo Switch`のゲームソフト`Splatoon 3`において，ブキとルール，ステージを選択したら，おすすめのギアパワーを教えてくれる推薦システムを作りたい．\
まずは，`dataset`を作成する．`splatoon 3`の詳細なバトル結果を入手したいので，`stat.ink`よりデータをダウンロードする．[ここ](https://stat.ink/downloads)からダウンロードできる．\
`splatoon 3`では，各シーズンごとに調整が入るため，直近1週間ほどのデータからデータセットを作成する．ためしに直近1週間分のデータでデータセットを作成したら，約66万件と十分なデータを得ることができた．\
`trainer.py`で作成したデータセット`splatoon3.inter`を用いて，モデルを作成する．`train.py`と`config.yaml`を使って，サーバー上で実行してモデルを作成する．\
以下のようにディレクトリを構成した．
```
myproject/
├── dataset/
│   └── splatoon3/
│       └── splatoon3.inter  # 作成したファイル
├── config.yaml              # 設定ファイル
└── train.py                 # 学習用スクリプト
```

## 2026/01/04
`ModuleNotFoundError: No module named 'ray'`このエラーが起きたら，以下のコマンドを実行する．
```
pip install ray
```
`ModuleNotFoundError: No module named 'pyarrow'`このエラーが起きたら，以下のコマンドを実行する．
```
pip install pyarrow
```
`ModuleNotFoundError: No module named 'pydantic'`このエラーが起きたら，以下のコマンドを実行する．
```
pip install pydantic
```
仮想環境のアクティベートはこれを実行．
```
source bin/activate
```
