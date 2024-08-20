# GYM_CABLE_RL

* ケーブルのコネクタ挿入動作の自動生成のための環境
* 強化学習を回すための環境は[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)を参考

* conf/*.yamlにパラメータを書き、実行ファイルで該当ファイルを指定する
  * 書き方はdefaults.yamlを参照する

## 現状

### 2024/08/20

* 自作のエージェントは使用せず、stable baselines3を利用
  * FEの実行：train_fe.py
  * RLの実行：sb3.py
