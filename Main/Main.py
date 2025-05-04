from Preprocess.ZigZag import *


class Main:
    def __init__(self):
        self.cv = Converter()
        self.pre = Preprocess()
        self.dctCLASS = DCTConvert()
        self.q = Quantum()
        self.zigzag = ZigZag()
        self.dif = ACDC('out.txt')
        
    
    def encode(self, img: Union[str, Image.Image, np.ndarray]) -> Union[None, str]:
        image = self.cv.RGB2YCbCr(img)
        
        h, w = image.shape[:2]
        output_file = open('out.txt', "wb")
        output_file.write(struct.pack('>HHB', h, w, 8))
        output_file.close()
        
        y, cb, cr = self.pre.downsample(image, 2)
        blocks = self.pre.split_by_blocks((y, cb, cr), 8)
        y, cb, cr = self.dctCLASS.dct2d(blocks)
        
        Q_Y = self.q.requant('y', 10)
        Q_C = self.q.requant('c', 10)
        
        y = self.q.quantile(Q_Y, y)
        cb = self.q.quantile(Q_C, cb)
        cr = self.q.quantile(Q_C, cr)

        y = self.zigzag.forward(y)
        cb = self.zigzag.forward(cb)
        cr = self.zigzag.forward(cr)
        
        y = self.dif.process(y)
        cb = self.dif.process(cb)
        cr = self.dif.process(cr)
        
        print('BOO')
    
    def decode(self, img: str) -> Union[Image.Image, np.ndarray, str]:
        print("Decoding...")
        y, cb, cr = self.dif.reprocess()

        y = self.zigzag.inverse(y)
        cb = self.zigzag.inverse(cb)
        cr = self.zigzag.inverse(cr)

        Q_Y = self.q.requant('y', 10)
        Q_C = self.q.requant('c', 10)

        y = self.q.dequantile(Q_Y, y)
        cb = self.q.dequantile(Q_C, cb)
        cr = self.q.dequantile(Q_C, cr)

        y, cb, cr = self.dctCLASS.idct2d((y, cb, cr))

        image = self.pre.upsample((y, cb, cr), 2)

        rgb_image = self.cv.YCbCr2RGB(image, out_path='SAVE.jpeg')

        return rgb_image
