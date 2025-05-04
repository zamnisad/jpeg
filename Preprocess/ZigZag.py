from .Quantum import *

class ZigZag:
    """
    Выполняет зигзаг-сканирование блоков DCT-коэффициентов для JPEG-сжатия и восстановление блоков.

    Методы:
        forward: Преобразует блоки коэффициентов в зигзаг-последовательность
        inverse: Восстанавливает блоки коэффициентов из зигзаг-последовательности
    """
    def __init__(self):
        pass

    @staticmethod
    def _forward_for_block(block: np.ndarray) -> np.ndarray:
        """
        Внутренний метод для зигзаг-сканирования одного блока.

        Args:
            block (np.ndarray): Блок коэффициентов размером NxN

        Returns:
            np.ndarray: Плоский массив коэффициентов в зигзаг-порядке
        """
        a, n, _ = block.shape
        result = []
        for k in range(a):
            sigm = []
            for d in range(2 * n - 1):
                for i in range(d + 1):
                    j = d - i
                    if i < n and j < n:
                        if d % 2 == 0:
                            sigm.append(block[k, i, j])
                        else:
                            sigm.append(block[k, j, i])
            result.append(np.array(sigm))
        return np.array(result)

    def forward(self, channel: np.ndarray) -> np.ndarray:
        """
        Преобразует блоки DCT-коэффициентов в зигзаг-последовательность.

        Args:
            channel (np.ndarray):
                Входные данные в формате:
                - Отдельный блок (2D массив)
                - Набор блоков (3D массив [число_блоков, N, N])
                - Канал изображения (4D массив [y_blocks, x_blocks, N, N])

        Returns:
            np.ndarray: Массив коэффициентов в зигзаг-порядке:
                - Для 2D входа: 1D массив
                - Для 3D/4D входа: 2D массив [число_блоков, N*N]
        """
        if len(channel.shape) == 2:
            return self._forward_for_block(channel)

        result = []
        for block in channel:
            result.append(self._forward_for_block(block))
        return np.array(result)
    
    @staticmethod
    def _inverse_for_block(flat_blocks: np.ndarray) -> np.ndarray:
        """
        Внутренний метод для восстановления блока из зигзаг-последовательности.

        Args:
            flat_blocks (np.ndarray): Плоский массив коэффициентов в зигзаг-порядке

        Returns:
            np.ndarray: Восстановленный блок коэффициентов размером NxN
        """
        num_blocks = flat_blocks.shape[0]
        N = 8
        result = []

        zigzag_indices = []
        for d in range(2 * N - 1):
            for i in range(d + 1):
                j = d - i
                if i < N and j < N:
                    if d % 2 == 0:
                        zigzag_indices.append((i, j))
                    else:
                        zigzag_indices.append((j, i))

        for k in range(num_blocks):
            block = np.zeros((N, N), dtype=flat_blocks.dtype)
            for idx, (i, j) in enumerate(zigzag_indices):
                block[i, j] = flat_blocks[k][idx]
            result.append(block)

        return np.array(result)
    
    def inverse(self, channel: np.ndarray) -> np.ndarray:
        """
        Восстанавливает блоки DCT-коэффициентов из зигзаг-последовательности.

        Args:
            channel (np.ndarray):
                Зигзаг-последовательность в формате:
                - 1D массив (один блок)
                - 2D массив [число_блоков, N*N]
                - 3D массив [каналы, число_блоков, N*N]

        Returns:
            np.ndarray: Восстановленные блоки коэффициентов:
                - Для 1D входа: 2D массив NxN
                - Для 2D/3D входа: 4D массив [y_blocks, x_blocks, N, N]
        """
        if len(channel.shape) == 2:
            return self._inverse_for_block(channel)

        result = []
        for blocks in channel:
            result.append(self._inverse_for_block(blocks))
        return np.array(result)
