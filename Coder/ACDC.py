from .Huffman import *


class ACDC:
    """
    Класс для кодирования/декодирования DC и AC коэффициентов DCT-преобразования 
    с использованием алгоритма Хаффмана и сохранения в бинарный файл.

    Args:
        file (str): Путь к файлу для записи/чтения закодированных данных

    Атрибуты:
        CAT_FORM (str): Формат упаковки категорий DC коэффициентов (>BI - big-endian, unsigned char + unsigned int)
        SEQ_LEN_FORM (str): Формат длины закодированной последовательности (>I - big-endian unsigned int)
        RLE_CAT_FORM (str): Формат RLE пар для AC коэффициентов (>IBI - big-endian, unsigned int + unsigned char + unsigned int)
        INFO_FORM (str): Формат метаданных изображения (>HHB - big-endian, 3 unsigned short)

    Методы:
        encode_dc: Кодирует DC коэффициенты
        encode_ac: Кодирует AC коэффициенты с RLE
        process: Основной метод обработки канала
        decode: Декодирует данные из файла
        reprocess: Полное восстановление изображения из файла
    """
    def __init__(self, file: str):
        self.fn = file
        self.file = open(file, 'wb')
        self.file.close()
        self.Huf = Huffman()
        self.CAT_FORM = '>BI'
        self.SEQ_LEN_FORM = '>I'
        self.RLE_CAT_FORM = '>IBI'
        self.INFO_FORM = '>HHB'
        
    def close_file(self):
        self.file.close()
        
    def open_file(self):
        self.file = open(self.fn, 'ab')
    
    @staticmethod
    def merge_dc_ac(dc: np.ndarray, ac: np.ndarray) -> np.ndarray:
        """
        Объединяет DC и AC коэффициенты.

        Args:
            dc (np.ndarray): Массив DC коэффициентов формы (N, 1)
            ac (np.ndarray): Массив AC коэффициентов формы (N, 63)

        Returns:
            np.ndarray: Объединённый массив коэффициентов формы (N, 64)
        """
        return np.array([[dc[i][0]] + list(ac[i]) for i in range(len(dc))])
        
    @staticmethod
    def convert(x, cat):
        """
        Конвертирует значение коэффициента в битовую строку.

        Args:
            x (int): Значение коэффициента
            cat (int): Категория коэффициента

        Returns:
            str: Битовая строка заданной длины
        """
        if cat == 0:
            return ""
        if x < 0:
            x = ((1 << cat) - 1) ^ (abs(x))
        return bin(x)[2:].zfill(cat)
    
    @staticmethod
    def iconvert(bits, cat):
        """
        Обратное преобразование битовой строки в значение.

        Args:
            bits (str): Битовая строка
            cat (int): Категория коэффициента

        Returns:
            int: Восстановленное значение коэффициента
        """
        if cat == 0:
            return 0
        value = int(bits, 2)
        if bits[0] == '0':
            value = -((1 << cat) - 1 - value)
        return value
    
    @staticmethod
    def category(delta):
        """
        Вычисляет категорию для коэффициента.

        Args:
            delta (int): Разность коэффициентов

        Returns:
            int: Категория (0 для нуля, иначе floor(log2(|delta|)) + 1)
        """
        abs_val = abs(delta)
        if abs_val == 0:
            return 0
        return int(np.floor(np.log2(abs_val))) + 1

    def encode_dc(self, arr: np.array):
        """
    Кодирует DC-коэффициенты с использованием алгоритма Хаффмана.

    Args:
        arr (np.array): Массив DC-коэффициентов формы (N,) 
            где N - количество блоков в изображении

    Returns:
        None: Результаты записываются в файл:
            - Частотный словарь категорий
            - Закодированные данные
            - Информация о дополнении
    """
        n = len(arr)
        arr = tuple(map(int, arr))

        arr_dc = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
        categories = tuple(map(self.category, arr_dc))
        freq_dict = dict(Counter(categories))

        self.file.write(struct.pack('>H', len(freq_dict)))
        for i in sorted(freq_dict.keys()):
            self.file.write(struct.pack(self.CAT_FORM, i, freq_dict[i]))

        root = self.Huf.build_tree(freq_dict)
        codes = self.Huf.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(n):
            dc = arr_dc[i]
            cat = self.category(dc)

            huf_str += codes[cat] + self.convert(dc, cat)
            while len(huf_str) >= 8:
                encoded.append(int(huf_str[:8], 2))
                huf_str = huf_str[8:]

        padding = 0
        if len(huf_str) != 0:
            padding = 8 - len(huf_str)
            encoded.append(int(huf_str.ljust(8, '0'), 2))

        self.file.write(struct.pack(self.SEQ_LEN_FORM, len(encoded)))
        self.file.write(bytes(encoded))
        self.file.write(struct.pack('>B', padding))

    def encode_ac(self, arr: np.array):
        """
        Кодирует AC-коэффициенты с RLE и алгоритмом Хаффмана.

        Args:
            arr (np.array): Массив AC-коэффициентов формы (M,) 
                где M = (число блоков) * (размер блока^2 - 1)

        Returns:
            None: Результаты записываются в файл:
                - RLE пары (длина нулевой последовательности, категория)
                - Закодированные данные
                - Информация о дополнении
        """
        n = len(arr)
        arr = tuple(map(int, arr))
        arr_ac = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
        ac, rle_ac = [], []

        zeros_cnt = 0
        for i in range(n):
            if arr_ac[i] != 0:
                ac.append(arr_ac[i])
                rle_ac.append((zeros_cnt, self.category(arr_ac[i])))
                zeros_cnt = 0
            else:
                zeros_cnt += 1
        if zeros_cnt != 0:
            ac.append(0)
            rle_ac.append((zeros_cnt, 0))

        freq_dict = dict(Counter(rle_ac))
        self.file.write(struct.pack('>H', len(freq_dict)))
        for couple, value in sorted(freq_dict.items()):
            self.file.write(struct.pack(self.RLE_CAT_FORM, *couple, value))

        root = self.Huf.build_tree(freq_dict)
        codes = self.Huf.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(len(rle_ac)):
            cat = rle_ac[i][1]
            huf_str += codes[rle_ac[i]] + self.convert(ac[i], cat)

            while len(huf_str) >= 8:
                encoded.append(int(huf_str[:8], 2))
                huf_str = huf_str[8:]

        padding = 0
        if len(huf_str) != 0:
            padding = 8 - len(huf_str)
            encoded.append(int(huf_str.ljust(8, '0'), 2))

        self.file.write(struct.pack(self.SEQ_LEN_FORM, len(encoded)))
        self.file.write(bytes(encoded))
        self.file.write(struct.pack('>B', padding))
        
    def process(self, channel: np.ndarray):
        """
        Обрабатывает канал изображения: кодирует DC и AC коэффициенты.

        Args:
            channel (np.ndarray): 
                Канал изображения в виде 3D массива формы (число_блоков, размер_блока^2)
                Например: (100, 64) для 100 блоков 8x8
        """
        self.open_file()
        
        _, _, n = channel.shape
        blocks = channel.reshape(-1, n)
        dc = blocks[:, 0]
        ac = blocks[:, 1:].flatten()
        
        self.encode_dc(dc)
        self.encode_ac(ac)
        
        self.close_file()
        
    def decode(self, file, mode, blocks_cnt, block_size):
        """
        Декодирует данные из файла.

        Args:
            file (file object): Открытый файловый объект
            mode (str): Режим декодирования ('DC' или 'AC')
            blocks_cnt (int): Количество блоков для декодирования
            block_size (int): Размер блока (обычно 8)

        Returns:
            np.ndarray: Массив декодированных коэффициентов
        """
        assert mode == 'AC' or mode == 'DC'

        freg_dict_len = struct.unpack('>H', file.read(2))[0]
        freq_dict = dict()
        for i in range(freg_dict_len):
            form = self.CAT_FORM if mode == 'DC' else self.RLE_CAT_FORM
            s = struct.calcsize(form)
            couple = struct.unpack(form, file.read(s))

            key, value = couple if mode == 'DC' else (couple[:2], couple[2])
            freq_dict[key] = value

        len_data = struct.unpack(self.SEQ_LEN_FORM, file.read(struct.calcsize(self.SEQ_LEN_FORM)))[0]
        data = file.read(len_data)
        padding = struct.unpack('>B', file.read(1))[0]

        root = self.Huf.build_tree(freq_dict)
        return ACDC.decode_data(self, data, root=root, padding=padding, mode=mode, blocks_cnt=blocks_cnt, n=block_size)

    def decode_data(self, data: bytes, root: Huffman.Node, padding: int, mode: str, blocks_cnt: int, n):
        """
        Декодирует бинарные данные с использованием дерева Хаффмана.

        Args:
            data (bytes): Закодированные бинарные данные
            root (Huffman.Node): Корневой узел дерева Хаффмана
            padding (int): Количество бит дополнения в последнем байте
            mode (str): Режим декодирования: 'DC' или 'AC'
            blocks_cnt (int): Количество блоков для декодирования
            n (int): Размер блока (для AC - длина зигзаг-последовательности)

        Returns:
            np.ndarray: Массив декодированных коэффициентов:
                - Для DC: форма (blocks_cnt,)
                - Для AC: форма (blocks_cnt, n-1)

        Raises:
            ValueError: Если обнаружена недопустимая битовая последовательность
        """
        assert mode == 'AC' or mode == 'DC'

        decoded = []
        bits_buffer = ""
        cur_node = root

        codes = self.Huf.build_code(root)

        for b in data:
            bits_buffer += bin(b)[2:].rjust(8, '0')
        if 0 < padding:
            bits_buffer = bits_buffer[:-padding]

        while len(bits_buffer) >= 0:
            if cur_node.value is not None:
                if mode == 'DC':
                    cat = cur_node.value
                    if cat != 0:
                        dc = self.iconvert(bits_buffer[:cat], cat)
                        decoded.append(dc)
                    else:
                        decoded.append(0)
                else:
                    run_len, cat = cur_node.value
                    decoded.extend([0] * run_len)
                    if cat != 0:
                        ac = self.iconvert(bits_buffer[:cat], cat)
                        decoded.append(ac)
                    else:
                        break
                cur_node = root
                bits_buffer = bits_buffer[cat:]

            if not bits_buffer:
                break

            bit = bits_buffer[0]
            bits_buffer = bits_buffer[1:]
            if bit == '0':
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        for i in range(1, len(decoded)):
            decoded[i] += decoded[i -  1]

        return np.array(decoded)

    def reprocess(self):
        """
        Восстанавливает изображение из закодированного файла.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                Кортеж из трёх каналов (Y, Cb, Cr) в виде 2D массивов
        """
        with open(self.fn, 'rb') as file:
            h, w, block_size = struct.unpack(self.INFO_FORM, file.read(struct.calcsize(self.INFO_FORM)))

            yb_cnt = int(np.ceil(h / block_size) * np.ceil(h / block_size))
            cbcrb_cnt = int(np.ceil(h // 2 / block_size) * np.ceil(h // 2 / block_size))
            
            dc_y = np.array(ACDC.decode(self, file, 'DC', yb_cnt, block_size))  # Передаем все параметры
            dc_y.resize(yb_cnt, 1)
            ac_y = np.array(ACDC.decode(self, file, 'AC', yb_cnt, block_size))  # Передаем все параметры
            ac_y.resize(yb_cnt, block_size**2 - 1)
            
            dc_cb = np.array(ACDC.decode(self, file, 'DC', cbcrb_cnt, block_size))  # Передаем все параметры
            dc_cb.resize(cbcrb_cnt, 1)
            ac_cb = np.array(ACDC.decode(self, file, 'AC', cbcrb_cnt, block_size))  # Передаем все параметры
            ac_cb.resize(cbcrb_cnt, block_size**2 - 1)
            
            dc_cr = np.array(ACDC.decode(self, file, 'DC', cbcrb_cnt, block_size))  # Передаем все параметры
            dc_cr.resize(cbcrb_cnt, 1)
            ac_cr = np.array(ACDC.decode(self, file, 'AC', cbcrb_cnt, block_size))  # Передаем все параметры
            ac_cr.resize(cbcrb_cnt, block_size**2 - 1)
            
            y = self.merge_dc_ac(dc_y, ac_y)
            cb = self.merge_dc_ac(dc_cb, ac_cb)
            cr = self.merge_dc_ac(dc_cr, ac_cr)
            
            return y, cb, cr
