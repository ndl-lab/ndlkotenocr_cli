# NDL古典籍OCRアプリケーション
NDL古典籍OCRを利用してテキスト化を実行するためのアプリケーションを提供するリポジトリです。 

NDL古典籍OCRは、江戸期以前の和古書、清代以前の漢籍といった古典籍資料のデジタル化画像からテキストデータを作成するOCRです。

本プログラムは、国立国会図書館が[令和3年度OCR関連事業](https://lab.ndl.go.jp/data_set/ocr/)から得られた知見や、
[NDLラボ](https://lab.ndl.go.jp)におけるこれまでの調査研究活動、そして人文情報学分野において構築・蓄積されてきたデータ資源を活用することで独自に開発したものです。

本プログラムを開発するにあたって利用したデータセットや手法の詳細については、[古典籍資料のOCRテキスト化実験](https://lab.ndl.go.jp/data_set/r4ocr/r4_koten/)及び[NDL古典籍OCR学習用データセット（みんなで翻刻加工データ）](https://github.com/ndl-lab/ndl-minhon-ocrdataset)も参照してください。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については
[LICENSE](./LICENSE
)をご覧ください。
 
## 環境構築

### 1. リポジトリのクローン
本リポジトリは一部の機能に、NDLOCRが利用しているモジュールを再利用しています。

これらのモジュールはhttps://github.com/ndl-lab
において公開しているリポジトリであり、本リポジトリとの間をsubmoduleで紐づけています。

リポジトリをclone する際は、次のコマンドを実行すると、関連リポジトリを一度に取得することができます。
```
git clone --recursive https://github.com/ndl-lab/ndlkotenocr_cli
```

### 2. ホストマシンのNVIDIA Driverのアップデート
コンテナ内でCUDA 11.1を利用します。

ホストマシンのNVIDIA Driverが

Linuxの場合: 455.23以上 

Windowsの場合:456.38以上

のバージョンを満たさない場合は、ご利用のGPUに対応するドライバの更新を行ってください。

（参考情報）

以下の環境で動作確認を行っています。

OS: Ubuntu 18.04.5 LTS

GPU: GeForce RTX 2080Ti

NVIDIA Driver: 455.45.01


### 3. dockerのインストール
https://docs.docker.com/engine/install/
に従って、OS及びディストリビューションにあった方法でdockerをインストールしてください。

### 4. dockerコンテナのビルド
Linux:
```
cd ndlkotenocr_cli
sh ./docker/dockerbuild.sh
```

Windows:
```
cd ndlkotenocr_cli
docker\dockerbuild.bat
```

### 5. dockerコンテナの起動
Linux:
```
cd ndlkotenocr_cli
sh ./docker/run_docker.sh
```

Windows:
```
cd ndlkotenocr_cli
docker\run_docker.bat
```

### 環境構築後のディレクトリ構成（参考）
```
ndlocr_cli
├── main.py : メインとなるPythonスクリプト
├── cli : CLIコマンド的に利用するPythonスクリプトの格納されたディレクトリ
├── src : 各推論処理のソースコード用ディレクトリ
│   ├── ndl_kotenseki_layout : レイアウト抽出処理のソースコードの格納されたディレクトリ
│   └── text_kotenseki_recognition : 文字認識処理のソースコードの格納されたディレクトリ
├── config.yml : サンプルの推論設定ファイル
├── docker : Dockerによる環境作成のスクリプトの格納されたディレクトリ
├── README.md : このファイル
└── requirements.txt : 必要なPythonパッケージリスト
```


## チュートリアル
起動後は以下のような`docker exec`コマンドを利用してコンテナにログインできます。

```
docker exec -i -t --user root kotenocr_cli_runner bash
```

### 推論処理の実行
input_rootディレクトリの直下にimgディレクトリがあり、その下に資料毎の画像ディレクトリ(bookid1,bookid2,...)がある場合、
```
input_root/
  └── img
      ├── bookid1
      │   ├── page01.jpg
      │   ├── page02.jpg
      │   ・・・
      │   └── page10.jpg
      ├── bookid2
```
以下のコマンドで実行することができます。
```
python main.py infer input_root output_dir
```

実行後の出力例は次の通りです。

```
output_dir/
  ├── bookid1
  │   ├── txt
  │   │     ├── page01.txt
  │   │     ├── page02.txt
  │   │    ・・・
  │   │    
  │   └── json
  │         ├── page01.json
  │         ├── page02.json
  │        ・・・
  └── opt.json
```


重みファイルのパス等、各モジュールで利用する設定値は`config.yml`の内容を修正することで変更することができます。

#### オプション情報の保存
出力ディレクトリでは、実行時に指定したオプション情報が`opt.json`に保存されています。


## モデルの再学習について
2023年1月現在、文字列認識モデルの再学習手順を公開しています。
[train.py](/src/text_kotenseki_recognition/train.py)
を参照してください。
