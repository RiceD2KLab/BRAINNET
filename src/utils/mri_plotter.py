import matplotlib.pyplot as plt
import numpy as np

class MRIPlotter:
    
    def plot_struct_img(self, img_data, fig, axs, row, col, title=None, cmap="Greys_r", auto_cbar=True, alpha=None, slice_idx=None):
        '''
        Args:
            img_data: loaded 2d or 3d nifti file
            auto_bar: automatically adds the color bar if set to true
            alpha: custom alpha. make sure that this has the same shape as your image
            
        Returns: Plotted image
        '''
        display_data = img_data
        
        # calculate slice index if image is 3d
        if img_data.ndim == 3:

            # if slice is not given, find the slice with the largest tumor
            if slice_idx is None:
                slice_idx = self.get_largest_tumor_slice_idx(img_data)[0]
            
            display_data = img_data[:, :, slice_idx]

        # axs can be 1d or 2d
        axs_element = axs
        if len(fig.axes) > 1:
            axs_element = axs[col] if len(axs.shape) == 1 else axs[row, col]

        img = axs_element.imshow(display_data, aspect='equal', cmap=cmap, alpha=alpha)
        
        if title:
            axs_element.set_title(title)
            
        if auto_cbar:
            fig.colorbar(img, ax=axs_element, fraction=0.05)
        return img

    def plot_segm_img(self, img_data, fig, axs, row, col, title=None, cmap=None, auto_cbar=False, overlay=False, 
                      segm_cbar=True, slice_idx=None, alpha=None):
        '''
        Args:
            img_data: loaded 2d or 3d nifti file
            auto_cbar: automatically generate cbar if true
            segm_cbar: show custom cbar with integer tickmarks based on the segments. if auto_cbar is True, this will not apply
            overlay: if true, default cmap and alpha values will be generated (if not provided) to create a segment mask 
        Returns: Plotted image
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
                slice_idx = self.get_largest_tumor_slice_idx(img_data)[0]
            display_data = img_data[:, :, slice_idx]
            
            if overlay is True and alpha is not None:
                alpha = alpha[:, :, slice_idx]

        # get label names only from the slice being displayed
        label_names = np.unique(display_data)
        label_cmap = plt.get_cmap(cmap, len(label_names))

        # re-use plot_struct_img but using cbar=false as default so we can manually build the discrete color bar
        img = self.plot_struct_img(img_data, fig, axs, row, col, title=title,
                                   cmap=label_cmap, auto_cbar=(segm_cbar==False and auto_cbar), 
                                   alpha=alpha, slice_idx=slice_idx)

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
            cbar.set_ticklabels(label_names)
            

    def get_largest_tumor_slice_idx(self, img_data):
        nonzero_counts = np.sum(np.count_nonzero(img_data, axis=0), axis=0 )
        slice_idx = np.argmax(nonzero_counts)
        return slice_idx, nonzero_counts[slice_idx]
    
    def test(self):
        print("test")
