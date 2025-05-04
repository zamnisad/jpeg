from Main.imports import *


class Huffman:
    """
    Реализует алгоритм Хаффмана для построения деревьев и генерации кодов сжатия данных.

    Включает внутренний класс Node для представления узлов дерева.

    Атрибуты:
        reps_size (int): Размер блока для группового кодирования (по умолчанию 4)
        count_size (int): Размер счетчика в битах (по умолчанию 2)
        step (int): Шаг обработки данных (рассчитывается как reps_size + 1)

    Методы:
        build_tree: Строит дерево Хаффмана по частотному словарю
        build_code: Генерирует коды Хаффмана для всех символов
    """
    class Node:
        """
        Внутренний класс для представления узла дерева Хаффмана.

        Атрибуты:
            value: Значение символа (None для внутренних узлов)
            freq: Частота символа/суммарная частота поддерева
            left: Левый потомок
            right: Правый потомок
        """
        def __init__(self, value=None, freq=0, left=None, right=None):
            self.value = value
            self.freq = freq
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(self, reps_size=4):
        self.count_size = 2
        self.reps_size = reps_size
        self.step = self.reps_size + 1

    @classmethod
    def build_tree(cls, freq_dict: dict):
        """
        Строит дерево Хаффмана на основе частотного словаря.

        Args:
            freq_dict (dict): Словарь частот в формате {символ: частота}

        Returns:
            Node: Корневой узел построенного дерева
        """
        nodes = [cls.Node(value=i, freq=freq_dict[i]) for i in freq_dict.keys() if freq_dict[i]]
        nodes.sort(key=lambda x: x.value)
        nodes.sort(key=lambda x: x.freq)
        heapq.heapify(nodes)

        while len(nodes) != 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            heapq.heappush(nodes, cls.Node(value=None, freq=left.freq + right.freq, left=left, right=right))
        return heapq.heappop(nodes)

    @classmethod
    def build_code(cls, node, prefix="", code_dict=None):
        """
        Генерирует коды Хаффмана для всех символов в дереве.

        Args:
            node (Node): Текущий обрабатываемый узел дерева
            prefix (str, optional): Текущий префикс кода. По умолчанию ""
            code_dict (dict, optional): Словарь для накопления кодов. По умолчанию None

        Returns:
            dict: Словарь кодов в формате {символ: битовая_строка}
        """
        if code_dict is None:
            code_dict = dict()
        if node:
            if node.value is not None:
                code_dict[node.value] = prefix
            else:
                cls.build_code(node.left, prefix + '0', code_dict)
                cls.build_code(node.right, prefix + '1', code_dict)
        return code_dict
