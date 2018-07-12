from PIL import Image, ImageDraw, ImageChops
import numpy as np


class Mask:
    """ a bit map to determine whehter a pixel should be included in
    azimuthal average of a 2D scattering pattern
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maskfile = ""
        self.map = None

    def reload(self):
        self.read_file(self.maskfile)

    def read_file(self, filename):
        self.maskfile = filename
        mask_map = Image.new('1', (self.width, self.height))
        for line in open(filename):
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            stype = fields[0]
            if stype in ['h', 'c', 'r', 'f', 'p']:
                para = [float(t) for t in fields[1:]]
                mask_map = self.add_item(mask_map, stype, para)
        self.map = np.asarray(mask_map.convert("I"), dtype=np.bool)
        del mask_map

    def fix_dead_pixels(self, d2, max_value=2097151):
        """ this is for dealing with dead pixels on Pilatus detectors
            fabio reads the value of -2 as a positive number
            the Pilatus pixels are 20-bit counters, the maximum value should be 2^21-1 = 2097151
        """
        if d2.data.d.shape != self.map.shape:
            print("mismatched data and mask shapes.")
            return
        self.map[d2.data.d>max_vlaue] = True
  
    def invert(self):
        """ invert the mask
        """
        self.map = 1 - self.map

    @staticmethod
    def add_item(mask_map, stype, para):
        tmap = Image.new('1', mask_map.size)
        draw = ImageDraw.Draw(tmap)
        if stype == 'c':
            # filled circle
            # c  x  y  r
            (x, y, r) = para
            draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
        elif stype == 'h':
            # inverse of filled circle
            # h  x  y  r
            (x, y, r) = para
            draw.rectangle(((0, 0), tmap.size), fill=1)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
        elif stype == 'r':
            # rectangle
            # r  x  y  w  h  rotation
            margin = (np.asarray(mask_map.size) / 2).astype(np.int)
            tmap = tmap.resize(np.asarray(mask_map.size) + 2 * margin)
            draw = ImageDraw.Draw(tmap)
            (x, y, w, h, rot) = para
            draw.rectangle(
                (tmap.size[0] / 2 - w / 2, tmap.size[1] / 2 - h / 2,
                 tmap.size[0] / 2 + w / 2, tmap.size[1] / 2 + h / 2),
                fill=1)
            tmap = tmap.rotate(rot)
            tmap = ImageChops.offset(tmap, int(x + margin[0] - tmap.size[0] / 2 + 0.5),
                                     int(y + margin[1] - tmap.size[1] / 2 + 0.5))
            tmap = tmap.crop((margin[0], margin[1], mask_map.size[0] + margin[0], mask_map.size[1] + margin[1]))
        elif stype == 'f':
            # fan
            # f  x  y  start  end  r1  r2  (r1<r2)
            (x, y, a_st, a_nd, r1, r2) = para
            draw.pieslice((x - r2, y - r2, x + r2, y + r2), a_st, a_nd, fill=1)
            draw.pieslice((x - r1, y - r1, x + r1, y + r1), a_st, a_nd, fill=0)
        elif stype == 'p':
            # polygon
            # p  x1  y1  x2  y1  x3  y3  ...
            draw.polygon(para, fill=1)

        mask_map = ImageChops.lighter(mask_map, tmap)
        del draw
        return mask_map

    def clear(self):
        self.map = np.zeros(self.map.shape, dtype=np.bool)

    def val(self, x, y):
        return self.map[x, y]

    def set_bit(self, x, y):
        self.map[x, y]=True
        
    def clear_bit(self, x, y):
        self.map[x, y]=False

