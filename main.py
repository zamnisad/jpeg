from Main.Main import *


if __name__ == "__main__":
    main = Main()
    main.encode(r'Z:\prog\jpeg\.resume\test_imgs\Мальчик в пустыне с чипсами.png', 100, print_info=True)
    main.decode(print_info=True)