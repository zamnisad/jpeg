from .Preprocess import *


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

    def _dct2d_for_block(self, block: np.ndarray) -> np.ndarray:
        """
        Внутренний метод для выполнения DCT над одним блоком.

        Args:
            block (np.ndarray): Блок изображения размером NxN

        Returns:
            np.ndarray: Блок частотных коэффициентов DCT
        """
        def F(u, v):
            n = block.shape[0]
            x = np.arange(n)
            y = np.arange(n)

            cos1 = np.cos((2 * x[:, None] + 1) * u * np.pi / (2 * n))
            cos2 = np.cos((2 * y[None, :] + 1) * v * np.pi / (2 * n))

            cos_matrix = cos1 @ cos2
            res = np.sum(block * cos_matrix)

            res *= self._c(u) * self._c(v) * 2 / n
            return res

        h, w = block.shape[:2]
        if h != w:
            raise ValueError(f"Матрица должна быть квадратной: h={h}, w={w}")
        n_ = h
        return np.array([[F(i, j) for j in range(n_)] for i in range(n_)])

    def _idct2d_for_block(self, block: np.ndarray) -> np.ndarray:
        """
        Внутренний метод для выполнения обратного DCT над одним блоком.

        Args:
            block (np.ndarray): Блок частотных коэффициентов DCT

        Returns:
            np.ndarray: Восстановленный блок пикселей
        """
        def F(x, y):
            n = block.shape[0]
            u = np.arange(n)
            v = np.arange(n)

            cos1 = np.cos((2 * x + 1) * u[:, None] * np.pi / (2 * n))
            cos2 = np.cos((2 * y + 1) * v[:, None] * np.pi / (2 * n))

            cos_matrix = np.outer(cos1[:, 0], cos2[:, 0])

            c_u = self._c(u).reshape(-1, 1)
            c_v = self._c(v).reshape(1, -1)
            scale = c_u * c_v

            res = np.sum(scale * block * cos_matrix)
            res *= 2 / n
            return res

        h, w = block.shape[:2]
        if h != w:
            raise ValueError(f"Матрица должна быть квадратной: h={h}, w={w}")
        n_ = h
        return np.array([[F(i, j) for j in range(n_)] for i in range(n_)])

    def dct2d(self, img: Union[List[np.ndarray], Image.Image, np.ndarray], block_size=8) \
            -> Union[Tuple[np.ndarray], List[np.ndarray]]:
        """
        Применяет двумерное DCT к изображению, разбитому на блоки.

        Args:
            img (Union[List[np.ndarray], Image.Image, np.ndarray]):
                Входное изображение в формате:
                - Списка NumPy массивов (каналы)
                - Объекта PIL.Image
                - NumPy массива (HxWxC)
            block_size (int, optional):
                Размер блоков для обработки. По умолчанию 8.

        Returns:
            Union[Tuple[np.ndarray], List[np.ndarray]]:
                Список массивов DCT-коэффициентов для каждого канала
                в формате (число_блоков_Y, число_блоков_X, block_size, block_size)
        """
        if not isinstance(img, list):
            img = self.pre.split_by_blocks(img, block_size=block_size)
        channels = [] if True else tuple()
        for channel in img:
            channel = channel.astype(np.float64)
            num1, num2, _, _ = channel.shape
            blocks = np.zeros_like(channel, dtype=np.float64)
            for i in range(num1):
                for j in range(num2):
                    blocks[i, j] = self._dct2d_for_block(channel[i, j])
            channels.append(blocks)
        return channels

    def idct2d(self, img: Union[List[np.ndarray], Image.Image, np.ndarray], block_size=8) \
            -> Union[Tuple[np.ndarray], List[np.ndarray]]:
        """
        Выполняет обратное двумерное DCT преобразование.

        Args:
            img (Union[List[np.ndarray], Image.Image, np.ndarray]):
                DCT-коэффициенты в формате:
                - Списка NumPy массивов (каналы)
                - Объекта PIL.Image
                - NumPy массива (HxWxC)
            block_size (int, optional):
                Размер блоков, использовавшихся при кодировании. По умолчанию 8.

        Returns:
            Union[Tuple[np.ndarray], List[np.ndarray]]:
                Список массивов восстановленных пикселей для каждого канала
                в формате (число_блоков_Y, число_блоков_X, block_size, block_size)
        """
        blocks = []

        if isinstance(img, (list, tuple)) and isinstance(img[0], np.ndarray):
            for channel in img:
                if channel.ndim == 3 and channel.shape[1:] == (block_size, block_size):
                    # Плоский вид: (num_blocks, 8, 8)
                    num_blocks = channel.shape[0]
                    side = int(np.sqrt(num_blocks))
                    if side * side != num_blocks:
                        raise ValueError(f"Невозможно преобразовать {num_blocks} блоков в квадратную сетку")
                    channel = channel.reshape(side, side, block_size, block_size)
                blocks.append(channel)
        else:
            blocks = self.pre.split_by_blocks(img, block_size=block_size)

        channels = []
        for channel in blocks:
            channel = channel.astype(np.float64)
            num1, num2, _, _ = channel.shape
            new_blocks = np.zeros_like(channel, dtype=np.float64)
            for i in range(num1):
                for j in range(num2):
                    new_blocks[i, j] = self._idct2d_for_block(channel[i, j])
            channels.append(new_blocks)

        return channels
