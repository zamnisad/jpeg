from .ACDC import *


class Converter:
    """
    Класс предназначенный для конвертации изображений из одного формата в другой
    """
    def __init__(self):
        pass

    @staticmethod
    def RGB2YCbCr(img: Union[Image.Image, str, np.ndarray],
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

    def show_images(self, img: Union[Image.Image, str, np.ndarray], # добавить np.ndarray
                    show_split: bool = True) -> None:
        """
        Отображает изображения, связанные с преобразованием между цветовыми пространствами RGB и YCbCr.

        Args:
            img (Union[Image.Image, str]): Изображение PIL.Image или путь к изображению, которое нужно отобразить.
            show_split (bool, optional): Если True, отображаются отдельные каналы Y, Cb и Cr. Если False, отображаются только оригинальное изображение RGB, преобразованное в YCbCr и обратно в RGB. По умолчанию True.

        Returns:
            None: Функция не возвращает значений, а только отображает изображения с помощью matplotlib.
        """
        arr_ycbcr = self.RGB2YCbCr(img)
        img_ycbcr = Image.fromarray(arr_ycbcr, 'RGB')

        arr_rgb = self.YCbCr2Rgb(img_ycbcr)
        img_rgb = Image.fromarray(arr_rgb, 'RGB')

        if show_split:
            y_channel, cb_channel, cr_channel = self.RGB2YCbCr(img, split_channels=True)
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            titles = ['Original RGB', 'Combined YCbCr', 'Recovered RGB',
                      'Y Channel (Luminance)', 'Cb Channel (Blue difference)', 'Cr Channel (Red difference)']
            images = [img, img_ycbcr, img_rgb, y_channel, cb_channel, cr_channel]
            cmaps = [None, None, None, 'gray', 'gray', 'gray']

            for ax, image, title, cmap in zip(axes.flat, images, titles, cmaps):
                ax.imshow(image, cmap=cmap)
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            titles = ['Original RGB', 'Combined YCbCr', 'Recovered RGB']
            images = [img, img_ycbcr, img_rgb]

            for ax, image, title in zip(axes, images, titles):
                ax.imshow(image)
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            plt.show()

