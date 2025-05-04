from Coder.Converter import *


class Preprocess:
    """
    Класс предназначен для предварительной обработки изображений:
    разбиения на блоки, сжатия цветовых каналов (subsampling), восстановления
    каналов (upsampling), а также объединения блоков обратно в изображение.
    """
    def __init__(self):
        pass

    @staticmethod
    def _downsample_channel(channel: np.ndarray, factor: int) -> np.ndarray:
        """
        Внутренний метод для сжатия одного канала.

        Args:
            channel (np.ndarray): Канал изображения в формате NumPy массива
            factor (int): Коэффициент сжатия (размер окна усреднения)

        Returns:
            np.ndarray: Сжатый канал
        """
        h, w = channel.shape
        new_h = h // factor
        new_w = w // factor

        channel = channel[:new_h * factor, :new_w * factor]

        return channel.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))

    def downsample(self, img: Union[np.ndarray, Image.Image, Tuple[np.ndarray, ...]],
                   factor: int = 2) -> Union[Tuple[np.ndarray, ...]]:
        """
        Сжимает цветовые каналы изображения.

        Args:
            img (Union[np.ndarray, Image.Image, Tuple[np.ndarray, ...]]):
                Изображение в формате NumPy массива, PIL.Image или кортежа каналов
            factor (int, optional):
                Коэффициент сжатия для хроматических каналов. По умолчанию 2

        Returns:
            Union[Tuple[np.ndarray, ...]]: Кортеж из трёх каналов (Y, Cb, Cr)
        """
        if isinstance(img, Image.Image):
            img = np.array(img).astype(np.float32)
        elif isinstance(img, np.ndarray):
            c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        else:
            c1, c2, c3 = img
        c2 = self._downsample_channel(c2, factor)
        c3 = self._downsample_channel(c3, factor)

        return c1, c2, c3

    @staticmethod
    def _split_channel_by_block(channel: np.ndarray, block_size: int):
        """
        Внутренний метод для разбиения канала на блоки.

        Args:
            channel (np.ndarray): Канал изображения
            block_size (int): Размер блока

        Returns:
            np.ndarray: Массив блоков заданного размера
        """
        h, w = channel.shape
        pad_h = int(np.ceil(h / block_size) * block_size - h)
        pad_w = int(np.ceil(w / block_size) * block_size - w)

        padded = np.pad(channel, ((0, pad_h), (0, pad_w)), constant_values=0)
        blocks = padded.reshape(
            padded.shape[0] // block_size, block_size,
            padded.shape[1] // block_size, block_size
        )
        blocks = blocks.transpose(0, 2, 1, 3) # y, x, block_size, block_size
        return blocks

    def split_by_blocks(self, img: Union[np.ndarray, Image.Image, Tuple[np.ndarray, ...]],
                        block_size: int = 8) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Разбивает изображение на блоки заданного размера.

        Args:
            img (Union[np.ndarray, Image.Image, Tuple[np.ndarray, ...]]):
                Изображение в формате NumPy массива, PIL.Image или кортежа каналов
            block_size (int, optional):
                Размер блоков. По умолчанию 8

        Returns:
            Union[np.ndarray, List[np.ndarray]]:
                Массив блоков для одного канала или список массивов для нескольких каналов
        """
        if isinstance(img, Image.Image):
            img = np.array(img).astype(np.float32)

        blocks = []
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            _, _, channels = img.shape
            for c in range(channels):
                block = self._split_channel_by_block(img[:, :, c], block_size)
                blocks.append(block)
        else:
            for channel in img:
                block = self._split_channel_by_block(channel, block_size)
                blocks.append(block)

        if len(blocks) == 1:
            return blocks[0]

        return blocks
    
    def merge_blocks(self, img: Tuple[np.ndarray, ...]) -> Union[Image.Image, np.ndarray]:
        """
        Объединяет блоки обратно в изображение.

        Args:
            img (Tuple[np.ndarray, ...]): Кортеж массивов блоков для каждого канала

        Returns:
            Union[Image.Image, np.ndarray]:
                Восстановленное изображение в формате NumPy массива
        """
        merged_channels = []
        for blocks in img:
            y_blocks, x_blocks, block_h, block_w = blocks.shape
            channel = blocks.transpose(0, 2, 1, 3).reshape(y_blocks * block_h, x_blocks * block_w)
            merged_channels.append(channel)

        merged_image = np.stack(merged_channels, axis=-1)
        return merged_image
    
    @staticmethod
    def _upsample_channel(channel: np.ndarray, factor: int) -> np.ndarray:
        """
        Внутренний метод для восстановления канала.

        Args:
            channel (np.ndarray): Сжатый канал изображения
            factor (int): Коэффициент увеличения

        Returns:
            np.ndarray: Восстановленный канал
        """
        return np.repeat(np.repeat(channel, factor, axis=0), factor, axis=1)
    
    def upsample(self, img: Union[Image.Image, np.ndarray, Tuple[np.ndarray, ...]], factor=2) \
                                                    -> Union[Image.Image, np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Восстанавливает сжатые цветовые каналы изображения.

        Args:
            img (Union[Image.Image, np.ndarray, Tuple[np.ndarray, ...]]):
                Изображение в формате NumPy массива, PIL.Image или кортежа каналов
            factor (int, optional):
                Коэффициент увеличения для хроматических каналов. По умолчанию 2

        Returns:
            Union[Image.Image, np.ndarray, Tuple[np.ndarray, ...]]:
                Восстановленное изображение в исходном формате
        """
        if isinstance(img, Image.Image):
            img = np.array(img).astype(np.float32)

        if isinstance(img, tuple):
            c1, c2, c3 = img
        else:
            c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        c2 = self._upsample_channel(c2, factor)
        c3 = self._upsample_channel(c3, factor)

        return self.merge_blocks((c1, c2, c3))
