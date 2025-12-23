# recboletest
2025/12/08に作成。動かない。

使い方　忘れるからメモ\
`anaconda prompt` を起動し，`conda activate recbole` と打つ\
`cd` でファイルのある階層に移動\
`python xxx.py` で実行\
`dir` コマンドで，作成した `saved` フォルダを見つける\
`saved` フォルダ内にある`xxx.pth`を使って，新規ユーザの推薦を行う．`pth`は`PyTorch`のモデルデータである．\
`hyper parameter` の設定は，レファレンス読んで理解\
`epoch`はエポック数であり，これは学修する回数である．エポック数が大きいと，ノートパソコンだと遅すぎて終わらない．\
`xxx.yaml`にはパラメータの詳細を記す．`yaml`とは，`html`や`xml`などのデータ形式の一つ．\
何度か試して，一番良いパラメータを探す\
`TensorBoard` で機械学習の結果を可視化，学習曲線をプロットできる\
ノートだと厳しいので，サーバを使う
`ssh-keygen -t ED25519`で，鍵を取得


2025/12/22追記\
`Python`は仮想環境で行うことを推奨．以下のコードで実行\
`python -m venv rec_env`\
`source rec_env/bin/activate`\
`recbole`のインストールは以下のコマンド\
`pip install recbole`\
`Python 3.13`だと，新しすぎて`PyTorch`などが対応していなく，エラーが起きる．その為，仮想環境を使ったうえで，`Python3`のライブラリを直接書き換える．\
`PyTorch`で学習していると，早期終了`Early Stopping`することがある．これは，過学習を防ぐためである．\
サーバーでは，`vim`を使ってプログラムファイルを作る．`touch test.py`と適当にファイルを造ったら，`vi test.py`としてターミナル上で`vim`を開く．`i`と打つことで，編集ができる．やめるときは，`escape`を押し，閉じるときは`:wq`とする．`:aqw`とする場合もある．サーバーを閉じるときは，`exit`とする．\
`VS Code`は，拡張機能の`Remote - SSH`で接続することができる．エクスプローラーより，任意のフォルダを開いて使う．

2025/12/23\
`dataset`を作成する．`splatoon 3`のバトル結果を入手したいので，`stat.ink`よりデータをダウンロードする．リンクは[これ](https://stat.ink/)\
`splatoon 3`では，各シーズンごとに調整が入るため，直近1週間ほどのデータからデータセットを作成する．ためしに直近1週間分のデータでデータセットを作成したら，約66万件になった．\
`trainer.py`で作成したデータセット`splatoon3.inter`を用いて，モデルを作成する．`train.py`と`config.yaml`を使って，サーバー上で実行して作成．
