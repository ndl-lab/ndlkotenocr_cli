# NDL古典籍OCRアプリケーション(ver.2)
NDL古典籍OCRを利用してテキスト化を実行するためのアプリケーションを提供するリポジトリです。 

NDL古典籍OCRは、江戸期以前の和古書、清代以前の漢籍といった古典籍資料のデジタル化画像からテキストデータを作成するOCRです。

本プログラムは、国立国会図書館が[令和3年度OCR関連事業](https://lab.ndl.go.jp/data_set/ocr/)から得られた知見や、
[NDLラボ](https://lab.ndl.go.jp)におけるこれまでの調査研究活動、そして人文情報学分野において構築・蓄積されてきたデータ資源を活用することで独自に開発したものです。

2023年1月に公開した[ver.1](https://github.com/ndl-lab/ndlkotenocr_cli/tree/ver.1)から、文字認識性能及び読み順の整序機能の性能が向上しています。

読み順の整序機能の性能改善に当たっては、[令和4年度OCR関連事業](https://lab.ndl.go.jp/data_set/r4ocr/r4_software/)から得られた知見を活用しています。

本プログラムを開発・改善するに当たって利用したデータセットや手法の詳細については、[古典籍資料のOCRテキスト化実験](https://lab.ndl.go.jp/data_set/r4ocr/r4_koten/)及び[OCR学習用データセット（みんなで翻刻）](https://github.com/ndl-lab/ndl-minhon-ocrdataset)も参照してください。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については
[LICENSE](./LICENSE
)をご覧ください。

 **2023年8月まで公開していたバージョンを継続して利用したい場合には、[ver.1](https://github.com/ndl-lab/ndlkotenocr_cli/tree/ver.1)をご利用ください。**
```
git clone --recursive https://github.com/ndl-lab/ndlkotenocr_cli -b ver.1
```
のようにソースコード取得部分を書き換えることで継続してお使いいただけます。

## 環境構築

### 1. リポジトリのクローン
下記のコマンドを実行してください。
```
git clone https://github.com/ndl-lab/ndlkotenocr_cli
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
|   ├── reading_order：読み順整序処理のソースコードの格納されたディレクトリ
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
      ├── page01.jpg
      ├── page02.jpg
      ・・・
      └── page10.jpg
```
以下のコマンドで実行することができます。
```
python main.py infer input_root output_dir
```

実行後の出力例は次の通りです。

```
output_dir/
  ├── input_root
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

### オプションについて

#### 入力形式オプション
実行時に
-s b を指定することで、次の入力形式のフォルダ構造を処理できます。

例：
```
python main.py infer input_root output_dir -s b
```

入力形式
```
input_root/
  └── img
      ├── bookid1
      │   ├── page01.jpg
      │   ├── page02.jpg
      │   ・・・
      │   └── page10.jpg
      ├── bookid2
          ├── page01.jpg
          ├── page02.jpg
          ・・・
          └── page10.jpg
```
出力形式
```
output_dir/
  ├── input_root
  |     ├──bookid1
  │     |     ├── txt
  │     |     │     ├── page01.txt
  │     |     │     ├── page02.txt
  │     |     │         ・・・
  │     |     │    
  │     |     └── json
  │     |           ├── page01.json
  │     |           ├── page02.json
  │     |               ・・・
  |     ├──bookid2
  │     |     ├── txt
  │     |     │     ├── page01.txt
  │     |     │     ├── page02.txt
  │     |     │         ・・・
  │     |     │    
  │     |     └── json
  │     |           ├── page01.json
  │     |           ├── page02.json
  │                    ・・・
  └── opt.json
```

#### 画像サイズ出力オプション
実行時に
-a を指定することで、出力jsonに画像サイズ情報を追加します。

例：
```
python main.py infer input_root output_dir -a
```

**注意**
このオプションを有効化すると出力jsonの形式が以下の構造になります。
```
{
  "contents":{
    (各文字列矩形の座標、認識文字列等)
  },
  "imginfo": {
    "img_width": (元画像の幅),
    "img_height": (元画像の高さ),
    "img_path":（元画像のディレクトリパス）,
    "img_name":（元画像名）
  }
}
```



#### オプション情報の保存
出力ディレクトリでは、実行時に指定したオプション情報が`opt.json`に保存されています。


## モデルの再学習について
2023年1月現在、文字列認識モデルの再学習手順を公開しています。
[train.py](/src/text_kotenseki_recognition/train.py)
を参照してください。
