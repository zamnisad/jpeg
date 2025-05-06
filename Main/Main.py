import time
from Preprocess.ZigZag import *


class Main:
    def __init__(self, output: str='compressed.zmn'):
        self.out = output
        self.cv = Converter()
        self.pre = Preprocess()
        self.dctCLASS = DCTConvert()
        self.q = Quantum()
        self.zigzag = ZigZag()
        self.dif = ACDC(output)
        
    def encode(self, img: Union[str, Image.Image, np.ndarray], quality: int=80, print_info=False) -> Union[None, str]:
        if not (0 < quality <= 100):
            print(f"Error quality: {quality}")
            quit(1488)
        if print_info:
            print("Encoding...")
            st = time.time()
        
        image = self.cv.RGB2YCbCr(img)
        
        h, w = image.shape[:2]

        output_file = open(self.out, "wb")
        output_file.write(struct.pack('>HHBB', h, w, 8, quality))
        output_file.close()
        
        y, cb, cr = self.pre.downsample(image, 2)
        
        blocks = self.pre.split_by_blocks((y, cb, cr), 8)
        y, cb, cr = self.dctCLASS.dct2d(blocks)
        
        Q_Y = self.q.requant('y', quality=quality)
        Q_C = self.q.requant('c', quality=quality)
        
        y = self.q.quantile(Q_Y, y)
        cb = self.q.quantile(Q_C, cb)
        cr = self.q.quantile(Q_C, cr)

        y = self.zigzag.forward(y)
        cb = self.zigzag.forward(cb)
        cr = self.zigzag.forward(cr)
        
        self.dif.process(y)
        self.dif.process(cb)
        self.dif.process(cr)
        
        if print_info:
            print('Time for encoding: ', round(time.time()-st, 3), ' seconds')
            print('File saved: ', self.out)
    
    def decode(self, img: str='decompressed.zmn', print_info=False) -> Union[Image.Image, np.ndarray, str]:
        if print_info:
            print("Decoding...")
            st = time.time()
            
        buf_y, buf_cb, buf_cr, bsz, quality = self.dif.reprocess()

        y  = self.zigzag.inverse(buf_y)
        cb = self.zigzag.inverse(buf_cb)
        cr = self.zigzag.inverse(buf_cr)

        Q_Y = self.q.requant('y', quality=quality)
        Q_C = self.q.requant('c', quality=quality)

        y = self.q.dequantile(Q_Y, y)
        cb = self.q.dequantile(Q_C, cb)
        cr = self.q.dequantile(Q_C, cr)

        y, cb, cr = self.dctCLASS.idct2d((y, cb, cr), bsz)

        y_plane  = self.pre.merge_blocks((y,))
        cb_plane = self.pre.merge_blocks((cb,))
        cr_plane = self.pre.merge_blocks((cr,))

        subh, subw = y_plane.shape[:2]
        cb_plane = cb_plane[:subh, :subw]
        cr_plane = cr_plane[:subh, :subw]
        image = self.pre.upsample(
            (y_plane, cb_plane, cr_plane),
            factor=2
        )

        rgb_image = self.cv.YCbCr2RGB(image, out_path=img)
        
        if print_info:
            print('Time for decoding: ', round(time.time()-st, 3), ' seconds')
            print('File decompressed: ', img)

        return rgb_image
