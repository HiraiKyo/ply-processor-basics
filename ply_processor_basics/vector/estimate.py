import numpy as np
from numpy.typing import NDArray


def estimate_vector(vector_samples: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    複数のベクトルから外れ値を考慮した平均ベクトルを推定する

    外れ値除去の閾値 1.5 * IQR

    :param vector_samples: ベクトルの集合(N, 3)
    :return: 推定された平均ベクトル(1, 3)
    """

    # 1. 単位ベクトル化
    vector_samples = np.array(vector_samples)
    vector_samples = vector_samples / np.linalg.norm(vector_samples, axis=1)[:, np.newaxis]

    # 2. 平均ベクトルの計算
    mean: NDArray[np.floating] = np.mean(vector_samples, axis=0)
    mean = mean / np.linalg.norm(mean)

    # 3. 外れ値の除去
    cos_similarities = np.dot(vector_samples, mean)
    q1, q3 = np.percentile(cos_similarities, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_samples = vector_samples[(cos_similarities >= lower_bound) & (cos_similarities <= upper_bound)]

    # 4. 最終的なベクトルの計算
    final: NDArray[np.floating] = np.mean(filtered_samples, axis=0)
    final = final / np.linalg.norm(final)

    return final


def ensure_consistent_direction(samples: NDArray[np.floating], reference_vector=None) -> NDArray[np.floating]:
    """
    ベクトルの方向の一貫性を確保する関数

    :param samples: ベクトルの集合(N, 3)
    :param reference_vector: 参照ベクトル(1, 3)
    :return: 一貫性が確保されたベクトルの集合(N, 3)
    """
    samples = np.array(samples)

    if reference_vector is None:
        reference_vector = samples[0]
    else:
        reference_vector = np.array(reference_vector)

    # 各ベクトルと参照ベクトルの内積を計算
    dots = np.dot(samples, reference_vector)

    # 内積が負のベクトルを反転
    samples[dots < 0] *= -1

    return samples
