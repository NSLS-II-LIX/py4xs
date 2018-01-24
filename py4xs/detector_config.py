import copy

class DetectorConfig():
    """ tis deals with things that are specific to 
    """
    def __init__(self, extension = "", exp_para = None, qgrid = None, 
                 dark = None, flat = None, dezinger = False, 
                 fix_scale = None, bad_pixels = [[], []]):
        self.extension = extension
        self.exp_para = exp_para
        self.qgrid = qgrid
        self.fix_scale = fix_scale
        
        self.dark = dark
        self.flat = flat
        self.dezinger = dezinger

        [bpx, bpy] = bad_pixels

        if exp_para.mask is not None:
            # make a copy in case the bad pixel list need to be revised and the original exp_para.mask need
            # to be preserved
            self.mask = copy.copy(exp_para.mask)
            if len(bpx)>0:
                for i in range(len(bpx)):
                    self.mask.set_bit(bpx[i],bpy[i])
        else:
            self.mask = None
                    
    def pre_process(self, data):
        """ this deals with flat field and dark current corrections, and dezinger
        """
        if self.dezinger:
            pass
        if self.dark is not None:
            pass
        if self.flat is not None:
            pass