from .ACDC import *


class Converter:
    """
    Класс предназначенный для конвертации изображений из одного формата в другой
    """
    def __init__(self):
        pass
    
    @staticmethod
    def save_raw(img_path: str) -> None:
        """
        Сохраняет изображение в формате .raw с тем же именем.

        Args:
            img_path (str): Путь к исходному изображению.
        """
        from os.path import splitext
        import os

        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.uint8)

        # Получаем имя без расширения
        root, _ = splitext(img_path)
        raw_path = root + '.raw'

        # Сохраняем байты изображения (в порядке RGB)
        with open(raw_path, 'wb') as f:
            f.write(img_array.tobytes())

    def RGB2YCbCr(self, img: Union[Image.Image, str, np.ndarray],
                  split_channels: bool = False,
                  return_type: str = 'array',
                  out_path: str = None) -> Union[Image.Image, np.ndarray,
                                                    Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
        """
        Переводит изображение из RGB в YCbCr цветовое пространство.

        Args:
            img (Union[Image.Image, str, np.ndarray]): Изображение PIL.Image, np.ndarray или путь к файлу изображения.
            split_channels (bool, optional):
                Если True, возвращает отдельные каналы Y, Cb и Cr в виде кортежа NumPy массивов.
                Если False, возвращает объединённое изображение. По умолчанию False.
            return_type (str, optional):
                Тип возвращаемого результата при объединении каналов:
                'array' (NumPy массив) или 'image' (PIL.Image). По умолчанию 'array'.
                Не используется, если separate_channels=True.
            out_path (str, optional): Путь для сохранения результата в файл. По умолчанию None (не сохранять).

        Returns:
            Union[Image.Image, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
            Преобразованное изображение в формате массива, изображения или кортежа каналов.
        """
        if isinstance(img, str):
            self.save_raw(img)
            img = Image.open(img)
        if isinstance(img, Image.Image):
            if img.mode in ('L', '1'):
                img = img.convert('RGB')
            img = np.array(img).astype(np.float32)
        
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255
        cb = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255
        cr = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255

        if split_channels:
            return y, cb, cr

        ycbcr = np.stack((y, cb, cr), axis=-1)
        ycbcr = np.clip(ycbcr, 0, 255)
        ycbcr = ycbcr.astype(np.uint8)

        ycbcr_img = Image.fromarray(ycbcr, 'RGB')
        if out_path:
            from os.path import splitext
            import os
            root, ext = splitext(out_path)
            ext = ext.lower()
            save_ext = ext if ext in ['.png', '.jpg', '.jpeg'] else '.png'
            temp_path = root + save_ext
            ycbcr_img.save(temp_path)
            if temp_path != out_path:
                os.replace(temp_path, out_path)


        if return_type.lower() == 'image':
            return ycbcr_img
        
        return ycbcr

    @staticmethod
    def YCbCr2RGB(img: Union[Image.Image, str, np.ndarray],
                  split_channels = False,
                  return_type: str = 'array',
                  out_path: str = None) -> Union[Image.Image, np.ndarray,
                                                    Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
        """
        Переводит изображение из YCbCr обратно в RGB цветовое пространство.
        Args:
            img (Union[Image.Image, str]): Изображение PIL.Image или путь к изображению.
            split_channels (bool, optional):
                Если True, возвращает отдельные каналы R, G, B в виде кортежа NumPy массивов.
                Если False, возвращает объединённое изображение. По умолчанию False.
            return_type (str, optional): Тип возвращаемого результата: 'array' (NumPy массив) или 'image' (PIL.Image).
                                         По умолчанию 'array'.
            out_path (str, optional): Путь для сохранения результата в файл. По умолчанию None (не сохранять).
        Returns:
            Union[Image.Image, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], None]: Преобразованное изображение в формате массива или изображения.
        """
        if not isinstance(img, np.ndarray):
            if isinstance(img, str):
                img = Image.open(img)
            img = np.array(img).astype(np.float32)
            
        y = img[:, :, 0]
        cb = img[:, :, 1]
        cr = img[:, :, 2]

        r = 1.164 * (y - 16) + 1.596 * (cr - 128)
        g = 1.164 * (y - 16) - 0.392 * (cb - 128) - 0.813 * (cr - 128)
        b = 1.164 * (y - 16) + 2.017 * (cb - 128)

        if split_channels:
            return r, g, b

        rgb = np.stack((r, g, b), axis=-1)
        rgb = np.clip(rgb, 0, 255)
        rgb = rgb.astype(np.uint8)
        
        rgb_img = Image.fromarray(rgb, 'RGB')
        if out_path:
            from os.path import splitext
            import os
            root, ext = splitext(out_path)
            ext = ext.lower()
            save_ext = ext if ext in ['.png', '.jpg', '.jpeg'] else '.png'
            temp_path = root + save_ext
            rgb_img.save(temp_path)
            if temp_path != out_path:
                os.replace(temp_path, out_path)


        if return_type.lower() == 'image':
            return rgb_img
        
        return rgb

    def show_images(self, img: Union[Image.Image, str, np.ndarray]) -> None:
        """
        Отображает изображение RGB и YCbCr, а также каналы R, G, B, Y, Cb, Cr в цвете.

        Args:
            img (Union[Image.Image, str, np.ndarray]): Изображение PIL.Image, путь или массив.

        Returns:
            None
        """
        if isinstance(img, str):
            img = Image.open(img)
        if isinstance(img, Image.Image):
            img = np.array(img)

        # RGB каналы
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        
        r_img = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=2).astype(np.uint8)
        g_img = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=2).astype(np.uint8)
        b_img = np.stack([np.zeros_like(b), np.zeros_like(b), b], axis=2).astype(np.uint8)

        # YCbCr каналы
        ycbcr = self.RGB2YCbCr(img)
        y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
        y_img = np.stack([y, y, y], axis=2).astype(np.uint8)
        
        # для Cb и Cr применим сдвиг + визуализацию в синих/красных оттенках
        cb_img = np.stack([np.zeros_like(cb), 255 - cb, cb], axis=2).astype(np.uint8)
        cr_img = np.stack([cr, 255 - cr, np.zeros_like(cr)], axis=2).astype(np.uint8)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        titles = [
            'RGB Image', 'R Channel (Red)', 'G Channel (Green)', 'B Channel (Blue)',
            'YCbCr Image', 'Y Channel (Luma)', 'Cb Channel (Blue diff)', 'Cr Channel (Red diff)'
        ]
        images = [
            img, r_img, g_img, b_img,
            ycbcr, y_img, cb_img, cr_img
        ]

        for ax, image, title in zip(axes.flat, images, titles):
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

