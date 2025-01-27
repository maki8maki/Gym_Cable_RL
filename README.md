# GYM_CABLE_RL

* フラットケーブルのコネクタ挿入のための、把持姿勢へのリーチング動作の自動生成のための環境
* 強化学習を回すための環境は[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)を参考に作成

## 使い方

* conf/**/*.yamlにパラメータを書く
  * 書き方は各フォルダのdefaults.yamlを参照する
* 実行ファイルでパラメータファイルを指定し、学習を開始する
  * 特徴量抽出の学習：train_fe.py
  * 動作生成の学習：train_sb3.py
  * ドメイン適応の学習：train_da.py
* 学習後のモデルの実行は上記のパラメータファイルとモデルのパラメータファイル（、入力画像）の指定が必要
  * 特徴量抽出の実行：exec_fe.py
  * 動作生成の実行：exec_sb3.py
  * ドメイン適応の実行：exec_da.py

## ファイル構成

* agents/
  * モデルの定義
  * [agentsリポジトリ](https://github.com/maki8maki/agents)をサブモジュールとしたフォルダ
  * [CycleGANリポジトリ](https://github.com/maki8maki/pytorch-CycleGAN-and-pix2pix.git)をサブモジュールとして持つ
* conf/
  * パラメータファイルを格納する
  * 直下には動作学習に関するパラメータを格納
  * fe/
    * 特徴量抽出モデル自体の定義に関するパラメータ
  * train_da/
    * ドメイン適応の学習に関するパラメータ
  * train_fe/
    * 特徴量抽出の学習に関するパラメータ
* data/
  * 特徴量抽出やドメイン適応の学習に使用する画像データを格納
* gym_cable/
  * シミュレーション環境の定義
  * `envs/assets/*/stl`にマニピュレータなどのモデルファイルを格納
* logs/
  * 学習や実行のログを格納
* model/
  * 最新のモデルのパラメータファイルを格納
  * 新しく学習が実行されると上書きされるので注意
    * 学習を途中で中止した場合は、その時点でのパラメータが保存される
* action_collection.py
  * 動作生成実行時の行動のノルム、位置・姿勢誤差を収集する
* action_threshhold.ipynb
  * 行動のノルムなどの可視化
  * 閾値決定アルゴリズムも実装しているが、実際には利用しなかった
* data_collection.py
  * シミュレーション環境データを収集する
  * 実際には利用しなかった
* error_collection.py
  * 動作生成実行時の最終的なフラットケーブル表面との距離を収集する
* exec_*.py
  * 各モデルの実行
* playback.py
  * ログ（実環境のものを想定）を元に動作を再現する
* read_log.py
  * ログから必要なデータを抽出し、出力・描画する
* train_*.py
  * 各モデルの学習
* test_*.py
  * 主要なプログラムではないが、使用したもの
  * test_compare.py
    * 複数のデータをグラフで描画する
  * test_env.py
    * テスト用のシミュレーション環境画像を収集する
  * test_hist.py
    * ヒストグラムを表示する
  * test_trans.py
    * モデルの学習・実行時と同じ画像変換を行う
    * 別に取得した生データに対して実行する
