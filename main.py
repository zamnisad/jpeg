from Main.Main import *


if __name__ == "__main__":
    main = Main()
    main.encode(r'Z:\prog\jpeg\.resume\test_imgs\Lenna.png', 0, print_info=True)
    main.decode(print_info=True)