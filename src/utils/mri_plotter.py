import matplotlib.pyplot as plt
import numpy as np

SEGMENTS = [0, 1, 2, 4]
SEGMENT_NAMES = ["ELSE", "NCR/NET", "ED", "ELSE", "ET"]

class MRIPlotter:
    
    def plot_img(self, img_data, fig, axs, row, col, title=None, cmap=None, alpha=None, auto_cbar=True, slice_idx=None, **kwargs):
        '''
        Args:
            img_data: loaded 2d or 3d nifti file
            auto_bar: automatically adds the color bar if set to true
            slice_idx: z-index if image is 3d
            **kwargs: keyword arguments to capture other parameters that can be passed to imshow (e.g. alpha, aspect, etc.)
        Returns: Generic image
        '''
        display_data = img_data
           
        # calculate slice index if image is 3d
        if img_data.ndim == 3:

            # if slice is not given, take middle slice
            if slice_idx is None:
                z_idx = img_data.shape[2]
                slice_idx = z_idx/2
            
            display_data = img_data[:, :, slice_idx]

        # axs can be 1d or 2d
        axs_element = axs
        if len(fig.axes) > 1:
            axs_element = axs[col] if len(axs.shape) == 1 else axs[row, col]

        img = axs_element.imshow(display_data, aspect='equal', cmap=cmap, alpha=alpha, **kwargs)
        
        if title:
            axs_element.set_title(title)
            
        if auto_cbar:
            fig.colorbar(img, ax=axs_element, fraction=0.05)
        return img

    def plot_struct_img(self, img_data, fig, axs, row, col, title=None, cmap="Greys_r", alpha=None, auto_cbar=True, slice_idx=None, **kwargs):
        '''
        Args:
            img_data: loaded 2d or 3d nifti file
            auto_bar: automatically adds the color bar if set to true
            slice_idx: z-index if image is 3d
            **kwargs: keyword arguments to capture other parameters that can be passed to imshow (e.g. alpha, aspect, etc.)
        Returns: Structural image
        '''
        return self.plot_img(img_data, fig, axs, row, col, title=title, 
                            cmap=cmap, alpha=alpha, auto_cbar=auto_cbar, 
                            slice_idx=slice_idx, **kwargs)


    def plot_segm_img(self, img_data, fig, axs, row, col,title=None, cmap=None, auto_cbar=False, overlay=False, 
                      segm_cbar=True, slice_idx=None, alpha=None, **kwargs):
        '''
        Args:
            img_data: loaded 2d or 3d nifti file
            auto_cbar: automatically generate cbar if true
            segm_cbar: show custom cbar with integer tickmarks based on the segments. if auto_cbar is True, this will not apply
            overlay: if true, default cmap and alpha values will be generated (if not provided) to create a segment mask 
            **kwargs: keyword arguments to capture other parameters that can be passed to imshow (e.g. alpha, aspect, etc.)
        Returns: Segmented image
        '''
        
        if overlay is True and cmap is None:
            cmap = "gnuplot"
        if overlay is True and alpha is None:
            alpha = np.where(img_data == 0, 0, 0.4)
        
        if overlay is False and cmap is None:
            cmap = "turbo"
            
        display_data = img_data
        
        # display 3d as 2d
        if img_data.ndim == 3:
            if slice_idx is None:
                z_idx = img_data.shape[2]
                slice_idx = z_idx/2
            display_data = img_data[:, :, slice_idx]
            
            if overlay is True and alpha is not None:
                alpha = alpha[:, :, slice_idx]

        # get label names only from the slice being displayed
        label_names = np.unique(display_data)
        label_cmap = plt.get_cmap(cmap, len(label_names))

        # re-use plot_struct_img but using cbar=false as default so we can manually build the discrete color bar
        img = self.plot_img(img_data, fig, axs, row, col, title=title,
                                   cmap=label_cmap, auto_cbar=(segm_cbar==False and auto_cbar), 
                                   alpha=alpha, slice_idx=slice_idx, **kwargs)

        axs_element = axs
        if len(fig.axes) > 1:
            axs_element = axs[col] if len(axs.shape) == 1 else axs[row, col]
        
        # build the color bar
        if (auto_cbar == False and segm_cbar == True):  
            # note: axs can be 1d or 2d. if axs 2d and no. of rows =1, axs shape becomes (col,). 
            # if there are more than 1 rows, axs shape becmes (row, col)
            
            cbar = fig.colorbar(img, ax=axs_element, fraction=0.05)
            ticks = np.linspace(np.min(label_names), np.max(label_names)-1, len(label_names))
            cbar.set_ticks(ticks + 0.5)
            
            # convert numerical labels to the actual names
            cbar.set_ticklabels(np.array(SEGMENT_NAMES)[label_names])
