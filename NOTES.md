# 調査メモ書き

## `PyTorch`の型アノテーション

- 型アノテーションがライブラリに含まれていないのでstubする必要がある
- `torchtyping`はdeprecated
- `jaxtyping`というライブラリに移行しているが、python 3.9+ サポート
- 当ライブラリは`ubuntu20`, `ROS Noetic` を前提としているので採用しない

## Python3.8対応関連

- `Pytorch`は 2.5+ で Python3.8 サポートを終了(PyPlには対応しているように見えるが)

## GPU対応

- 現在(2024/11/01) `CAPC`はCPU対応のみであるため、当ライブラリもGPU対応を行わない
- GPU対応を行う場合は、以下のエンドポイントをGPU対応に修正する必要がある
  - `points.ml.match`
