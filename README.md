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

* agents
  * モデルの定義
  * [agentsリポジトリ](https://github.com/maki8maki/agents)をサブモジュールとしたフォルダ
  * [CycleGANリポジトリ](https://github.com/maki8maki/pytorch-CycleGAN-and-pix2pix.git)をサブモジュールとして持つ
* conf
  * パラメータファイルを格納する
  * 直下には動作学習に関するパラメータを格納
  * fe
    * 特徴量抽出モデル自体の定義に関するパラメータ
  * train_da
    * ドメイン適応の学習に関するパラメータ
  * train_fe
    * 特徴量抽出の学習に関するパラメータ
* data
  * 特徴量抽出やドメイン適応の学習に使用する画像データを格納
* gym_cable
  * シミュレーション環境の定義
  * `envs/assets/*/stl`にマニピュレータなどのモデルファイルを格納
* logs
  * 学習や実行のログを格納
* model
  * 最新のモデルのパラメータファイルを格納
  * 新しく学習が実行されると上書きされるので注意
    * 学習を途中で中止した場合は、その時点でのパラメータが保存される
