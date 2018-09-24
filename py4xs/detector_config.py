import copy
import pickle,bz2
import numpy as np
from py4xs.local import ExpPara, exp_attr

det_attr = ['ImageWidth', 'ImageHeight', 'extension', 'fix_scale']

def create_det_from_attrs(attrs): #, qgrid):
    det = DetectorConfig()  #qgrid=qgrid)
    det.unpack_dict(attrs)
    return det

class DetectorConfig():
    """ tis deals with things that are specific to 
    """
    def __init__(self, extension = "", exp_para = None, qgrid = None, 
                 dark = None, flat = None, dezinger = False, 
                 fix_scale = None, bad_pixels = [[], []]):
        self.extension = extension
        self.exp_para = exp_para
        if exp_para!=None:
            self.ImageWidth = exp_para.ImageWidth
            self.ImageHeight = exp_para.ImageHeight
        #self.qgrid = qgrid
        if qgrid is not None:
            print("Warning: qgrid under DectorConfig is no longer in use.")
        self.fix_scale = fix_scale
        
        self.dark = dark
        self.flat = flat
        self.dezinger = dezinger

        [bpx, bpy] = bad_pixels

        if exp_para!=None:
            # seems unnecessary
            ## make a copy in case the bad pixel list need to be revised and the original exp_para.mask need
            ## to be preserved
            #self.mask = copy.copy(exp_para.mask)
            if len(bpx)>0:
                for i in range(len(bpx)):
                    exp_para.mask.set_bit(bpx[i],bpy[i])
                    ##self.mask.set_bit(bpx[i],bpy[i])
        #else:
        #    self.mask = None
    
    def pack_dict(self):
        det_dict = {}
        exp_dict = {}
        for attr in exp_attr:
            exp_dict[attr] = self.exp_para.__getattribute__(attr)
        for attr in det_attr:
            det_dict[attr] = self.__getattribute__(attr)
        exp_dict['mask'] =  list(bz2.compress(pickle.dumps(self.exp_para.mask)))
        det_dict['exp_para'] = exp_dict
        return det_dict 
        
    def unpack_dict(self, det_dict):
        for attr in det_dict:
            self.__setattr__(attr, det_dict[attr])
        #self.qgrid = np.asarray(det_dict['qgrid'])
        self.exp_para = ExpPara(self.ImageWidth, self.ImageHeight) 
        for attr in exp_attr:
            self.exp_para.__setattr__(attr, det_dict['exp_para'][attr])
        self.exp_para.mask = pickle.loads(bz2.decompress(bytes(det_dict['exp_para']['mask'])))
        self.exp_para.calc_rot_matrix()
        self.exp_para.init_coordinates()
    
    def pre_process(self, data):
        """ this deals with flat field and dark current corrections, and dezinger
        """
        if self.dezinger:
            pass
        if self.dark is not None:
            pass
        if self.flat is not None:
            pass