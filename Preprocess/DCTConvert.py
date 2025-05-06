from .Preprocess import *
from functools import lru_cache


class DCTConvert:
    """
    Класс для выполнения прямого и обратного двумерного дискретного косинусного преобразования (DCT/IDCT)
    над изображением, разбитым на блоки.
    """
    def __init__(self):
        self.pre = Preprocess()

    @staticmethod
    def _c(u):
        if isinstance(u, int):
            return 2 ** -0.5 if u == 0 else 1.0
        u = np.asarray(u)
        return np.where(u == 0, 2 ** -0.5, 1.0)
    
    @staticmethod
    @lru_cache(maxsize=None)
    def _get_dct_matrix(n: int):
        x = np.arange(n)
        u = np.arange(n)
        A = np.cos((2 * x[:, None] + 1) * u[None, :] * np.pi / (2 * n))
        C = np.where(u == 0, 2**-0.5, 1.0) * np.sqrt(2 / n)
        return A * C

    def _apply_dct_blocks(self, blocks: np.ndarray, A: np.ndarray) -> np.ndarray:
        return A.T @ blocks @ A 

    def _apply_dct_blocks_vec(self, blocks: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.einsum("ij,xyjk,kl->xyil", A.T, blocks, A)

    def dct2d(self, img: Union[List[np.ndarray], Image.Image, np.ndarray], block_size=8) \
            -> Union[Tuple[np.ndarray], List[np.ndarray]]:
        if not isinstance(img, list):
            img = self.pre.split_by_blocks(img, block_size=block_size)
        
        A = self._get_dct_matrix(block_size)
        channels = []
        for channel in img:
            channel = channel.astype(np.float64)
            dct_blocks = np.einsum("ij,xyjk,kl->xyil", A.T, channel, A)
            channels.append(dct_blocks)
        return channels

    def idct2d(self, img: Union[List[np.ndarray], Image.Image, np.ndarray], block_size=8) \
            -> Union[Tuple[np.ndarray], List[np.ndarray]]:
        blocks = []
        if isinstance(img, (list, tuple)) and isinstance(img[0], np.ndarray):
            for channel in img:
                if channel.ndim == 3 and channel.shape[1:] == (block_size, block_size):
                    num_blocks = channel.shape[0]
                    side = int(np.sqrt(num_blocks))
                    if side * side != num_blocks:
                        raise ValueError(f"Невозможно преобразовать {num_blocks} блоков в квадратную сетку")
                    channel = channel.reshape(side, side, block_size, block_size)
                blocks.append(channel)
        else:
            blocks = self.pre.split_by_blocks(img, block_size=block_size)

        A = self._get_dct_matrix(block_size)
        channels = []
        for channel in blocks:
            channel = channel.astype(np.float64)
            idct_blocks = np.einsum("ij,xyjk,kl->xyil", A, channel, A.T)
            channels.append(idct_blocks)
        return channels
