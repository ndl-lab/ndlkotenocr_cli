# ReadingOrder
読み上げ順序推定モジュールのリポジトリです。
本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。
本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については LICENSEをご覧ください。

This is a repository for the reading order estimation module. This program was created by Morpho AI Solutions, Inc. under contract from the National Diet Library. The program is released by the National Diet Library under the CC BY 4.0 license. For details, please see the LICENSE file.

## Installation
  
To install the program, first navigate to the cloned directory and install the required packages. It is recommended that you create an appropriate environment using venv or conda before proceeding with the installation process.
``` console
$ cd reading_order
$ pip install .
```

## Reading Order Detection from XML Files

To detect reading order from XML files, organize the input data in the following folder structure, where `pid` is a unique identifier for each document:
```
|- XXXXXX (pid)
|  |- xml
|  |  |- XXXXXX.xml
|  |- img (optional)
|     |- *.jpg
|- ...
```

After organizing the input data, run the following command:
``` console
$ python tools/eval.py --skip .sorted.xml XML/DIR
```

Replace `XML/DIR` with either the unsorted XML file or the folder containing unsorted XML files. If a folder is specified, the program will search for XML files recursively.
The above command generates a `.sorted.xml` file in the same directory as the original `.xml` file.
```
|- XXXXXX (pid)
|  |- xml
|  |  |- XXXXXX.xml
|  |  |- XXXXXX.sorted.xml
|  |- img (optional)
|     |- *.jpg
|- ...
```

## Visualize the Results

You can visualize the results by running the following command:
``` console
$ python tools/visualize.py XML/DIR --only .sorted.xml
```

This generates a `.order.jpg` file in the same directory as the original `.jpg` file.
```
|- XXXXXX (pid)
|  |- xml
|  |  |- XXXXXX.xml
|  |  |- XXXXXX.sorted.xml
|  |- img (optional)
|     |- *.jpg
|     |- *.order.jpg
|- ...
```

Alternatively, you can output all the `.order.jpg` files in one place `DIR` by using the `--output` option as follows:
``` console
$ python tools/visualize.py XML/DIR --only .sorted.xml --output DIR
```
