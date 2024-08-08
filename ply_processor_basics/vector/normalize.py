import numpy as np
from numpy.typing import NDArray


def normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """_summary_

    Args:
        vector (NDArray[np.float32]): _description_

    Returns:
        NDArray[np.float32]: _description_
    """
    if np.linalg.norm(vector) == 0:
        raise ZeroDivisionError("The norm of the vector is zero.")

    return vector / np.linalg.norm(vector)
