from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


class PointNetEncoder(nn.Module):
    """CPUに最適化したPointNetベースのエンコーダー"""

    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()

        # MLPベースの特徴抽出（KPConvより軽量）
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

        # グローバル特徴抽出
        self.global_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        点群から特徴量を抽出
        :param points: 入力点群 (B, N, 3)
        :return: 特徴量 (B, N, out_channels)
        """
        batch_size, num_points, _ = points.shape

        # ポイントごとの特徴抽出
        x = points.transpose(1, 2)  # (B, 3, N)
        point_features = self.mlp1(x.reshape(-1, 3)).view(batch_size, num_points, -1)

        # グローバル特徴の抽出と結合
        global_features = torch.max(point_features, dim=1)[0]  # (B, C)
        global_features = self.global_mlp(global_features)  # (B, C)
        global_features = global_features.unsqueeze(1).expand(-1, num_points, -1)

        # ローカル特徴とグローバル特徴の結合
        features = torch.cat([point_features, global_features], dim=-1)
        return features


class CPUPointCloudMatcher:
    """CPU環境に最適化した点群マッチングクラス"""

    def __init__(self, model_path: Optional[str] = None):
        self.encoder = PointNetEncoder()
        if model_path:
            self.encoder.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.encoder.eval()

    def process_batch(self, points: NDArray, batch_size: int = 1000) -> torch.Tensor:
        """
        大規模点群をバッチ処理
        :param points: 入力点群 (N, 3)
        :param batch_size: バッチサイズ
        :return: 特徴量 (N, C)
        """
        features_list = []
        num_points = len(points)

        for i in range(0, num_points, batch_size):
            batch_points = points[i : min(i + batch_size, num_points)]
            with torch.no_grad():
                batch_tensor = torch.from_numpy(batch_points).float().unsqueeze(0)
                batch_features = self.encoder(batch_tensor)
                features_list.append(batch_features.squeeze(0))

        return torch.cat(features_list, dim=0)

    def find_matches_cpu(
        self, source_features: torch.Tensor, target_features: torch.Tensor, k: int = 1, leaf_size: int = 40
    ) -> NDArray:
        """
        CPUに最適化した最近傍点探索
        :param source_features: ソース点群の特徴量 (N, C)
        :param target_features: ターゲット点群の特徴量 (M, C)
        :param k: 探索する近傍点の数
        :param leaf_size: kd-treeのリーフサイズ
        :return: ターゲット点群に対応するソース点群のインデックス (M,)
        """
        # NumPyに変換
        source_features_np = source_features.numpy()
        target_features_np = target_features.numpy()

        # 正規化（NumPyで実行）
        source_norm = np.linalg.norm(source_features_np, axis=1, keepdims=True)
        target_norm = np.linalg.norm(target_features_np, axis=1, keepdims=True)
        source_features_normalized = source_features_np / (source_norm + 1e-8)
        target_features_normalized = target_features_np / (target_norm + 1e-8)

        # NearestNeighborsを使用した効率的な最近傍探索
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", leaf_size=leaf_size, metric="euclidean").fit(
            source_features_normalized
        )

        distances, indices = nbrs.kneighbors(target_features_normalized)
        assert type(indices) is np.ndarray
        return indices[:, 0]  # 最近傍点のみ返す


def match(source: NDArray, target: NDArray) -> NDArray:
    """
    機械学習手法を用いてソース点群からターゲット点群を探索する
    :param source: ソース点群(N, 3)
    :param target: ターゲット点群(M, 3)
    :return: ターゲット点群に対応するソース点群のインデックス(M, )
    """
    matcher = CPUPointCloudMatcher()

    # バッチ処理による特徴量抽出
    source_features = matcher.process_batch(source)
    target_features = matcher.process_batch(target)

    # CPU最適化された最近傍探索
    matching_indices = matcher.find_matches_cpu(source_features, target_features)

    return matching_indices


def train_matcher_cpu(
    train_data: List[Tuple[NDArray, NDArray, NDArray]],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "matcher_model_cpu.pth",
) -> None:
    """
    CPU環境に最適化されたマッチャーの学習
    :param train_data: 学習データ [(source, target, correspondence), ...]
    :param epochs: エポック数
    :param batch_size: バッチサイズ
    :param learning_rate: 学習率
    :param model_save_path: モデルの保存パス
    """
    encoder = PointNetEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0.0

        # データをシャッフル
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i : i + batch_size]

            # バッチデータの準備
            batch_source = torch.tensor(np.stack([d[0] for d in batch_data])).float()
            batch_target = torch.tensor(np.stack([d[1] for d in batch_data])).float()
            batch_corr = torch.tensor(np.stack([d[2] for d in batch_data])).long()

            # メモリ効率を考慮した特徴量抽出
            source_features = encoder(batch_source)
            target_features = encoder(batch_target)

            # メモリ効率の良いloss計算
            sim_matrix = torch.matmul(source_features, target_features.transpose(1, 2))
            loss = F.cross_entropy(sim_matrix.view(-1, sim_matrix.size(-1)), batch_corr.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}")

    torch.save(encoder.state_dict(), model_save_path)


# 使用例
if __name__ == "__main__":
    # サンプルデータの生成
    source = np.random.randn(1000, 3)
    target = np.random.randn(800, 3)

    # マッチング実行
    matching_indices = match(source, target)
    print(f"Matching indices shape: {matching_indices.shape}")
    print(f"First 5 matches: {matching_indices[:5]}")
