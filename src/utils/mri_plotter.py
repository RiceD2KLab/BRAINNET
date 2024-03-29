import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import utils.mri_common as mri
from PIL import Image
from matplotlib.font_manager import FontProperties

class MRIPlotter:
    """
    Class containing methods for plotting MRI scans and segmentation masks.

    Intialization is done by instantiating an MRIPlotter object:

    mri_plotter = MRIPlotter()

    All access is through plotting methods.
    """
    
    def set_axis_config(self, axs_element):
        """
        Helper function to set default fontsizes

        Inputs:
            axs_element - matplotlib.pyplot.axes object

        Returns None.
        """
        axs_element.tick_params(axis='x', labelsize=18)
        axs_element.tick_params(axis='y', labelsize=18)
        axs_element.title.set_fontsize(24)
        axs_element.yaxis.label.set_size(18)
        axs_element.xaxis.label.set_size(18)

    def plot_img(self, img_data, fig, axs, row, col, title=None, cmap=None, alpha=None, colorbar=False, **kwargs):
        '''
        Ensures common parameters for all MRI plots

        Args:
            img_data: image data to be displayed
            title: title of the plot
            colorbar: automatically adds the color bar if set to true
            **kwargs: keyword arguments to capture other parameters that can be passed to imshow (e.g. alpha, aspect, etc.)

        Returns: Matplotlib image
        '''
        display_data = img_data
        axs_element = self._get_subplot_axs(fig, axs, row, col)

        # plot the image
        img = axs_element.imshow(display_data, cmap=cmap, alpha=alpha, aspect='equal', **kwargs)

        if title:
            axs_element.set_title(title)

        if colorbar:
            fig.colorbar(img, ax=axs_element, fraction=0.05)

        # call common config to set size of titles/labels
        self.set_axis_config(axs_element)
        
        # turn off scale for plotting mri 
        # other plots can still use common config if they have scale
        axs_element.xaxis.set_visible(False)
        axs_element.yaxis.set_visible(False)

        return img

    def plot_struct_img(self, img_data, fig, axs, row, col, title=None, cmap="Greys_r", colorbar=True, slice_idx=None, **kwargs):
        '''
        Args:
            img_data: image data to be displayed
            title: title of the plot
            colorbar: automatically adds the color bar if set to true
            slice_idx: z-index if image is 3d but intended to be displayed as 2d
            plot_channels: will allow plotting of image with more than 2 dimensions.
                this will override assigning a default slice index
            **kwargs: keyword arguments to capture other parameters that can be passed to imshow (e.g. alpha, aspect, etc.)

        Returns: Matplolib image for structural scan
        '''

        # calculate slice index if image is 3d and if we do not intend to plot channels
        display_data = img_data
        if self._is_image_3d(img_data):
            # if slice is not given, take middle slice
            if slice_idx is None:
                z_max = img_data.shape[2]
                slice_idx = int(z_max//2)

            display_data = img_data[:, :, slice_idx]

        return self.plot_img(display_data, fig, axs, row, col, title=title,
                            cmap=cmap, colorbar=colorbar, **kwargs)


    def plot_segm_img(self, img_data, fig, axs, row, col, title=None, cmap=None, colorbar=False, segm_colorbar=True, slice_idx=None,
                      alpha=None, use_legend=False, overlay=False, loc=None, **kwargs):
        '''
        Args:
            img_data: image data to be displayed
            title: title of the plot
            colorbar: automatically adds the color bar showing the scale of the image
            segm_colorbar: automatically adds color bar showing different segments
            slice_idx: z-index if image is 3d but intended to be displayed as 2d
            plot_channels: will allow plotting of image with more than 2 dimensions.
                this will override assigning a default slice index
            **kwargs: keyword arguments to capture other parameters that can be passed to imshow (e.g. alpha, aspect, etc.)

        Returns: Segmented image
        '''
        if overlay is True and alpha is None:
            alpha = self._get_default_alpha(img_data)

        display_data = img_data

        # add a default slice if data is 3d
        if self._is_image_3d(img_data):
            if slice_idx is None:
                z_max = img_data.shape[2]
                slice_idx = int(z_max//2)

            display_data = img_data[:, :, slice_idx]

            # update alpha to accommodate 2d image
            if overlay is True and alpha is not None:
                alpha = alpha[:, :, slice_idx]

        # update labels to from 4 to 3
        display_data = np.where(display_data == 4, 3, display_data)

        # default segment info
        segment_dict = mri.SEGMENTS.copy()
        segment_colors_dict = mri.SEGMENT_COLORS.copy()
        segment_labels = np.unique(display_data)
        segment_labels.sort()

        # maskformer segmentation has -1. if data has this value, update the segment info
        if -1 in segment_labels and overlay is False:
            segment_dict[-1] = "NO LABEL"
            segment_colors_dict[-1] = "black"

        # take only the labels existing in the display image
        segment_colors = [segment_colors_dict[i] for i in segment_labels]
        segment_names = [segment_dict[i] for i in segment_labels]

        # set up the cmap
        cmap = mcolors.ListedColormap(segment_colors)

        # apply nearest interpolation to address bugs in displaying np.uint8
        kwargs["interpolation"] = "nearest"

        # re-use base plotting function
        img = self.plot_img(display_data, fig, axs, row, col, title=title,
                                cmap=cmap, colorbar=colorbar, alpha=alpha, **kwargs)

        # choose between using color bar or handles
        axs_element = self._get_subplot_axs(fig, axs, row, col)
        if segm_colorbar is True and colorbar is False and use_legend is False:
            cbar = fig.colorbar(img, ax=axs_element, fraction=0.05)
            ticks = np.linspace(np.min(segment_labels), np.max(segment_labels)-1, len(segment_labels))
            cbar.set_ticks(ticks + 0.5)
            cbar.set_ticklabels(segment_names)

        elif use_legend is True:
            handles = []
            for idx, color in enumerate(segment_colors):
                handles.append(mpatches.Patch(color=color, label=segment_names[idx]))
            
            handle_loc = loc if not None else 'lower right'
            axs_element.legend(handles=handles, loc=handle_loc)

    def plot_masks(self, masks, fig, axs, row, col, title, legends, **kwargs):
        """
        Adds annotation masks to a given matplotlib figure and axes

        Inputs:
            masks - numpy ndarray of annotation masks
            fig - matplotlib.pyplot.figure object
            axs - matplotlib.pyplot.axes object
            row - int, number of rows in the axs
            col - int, number of columns in the axs
            title - str, figure title
            legends - list of strings
            **kwargs - keyword arguments to capture other parameters that can be passed to imshow (e.g. alpha, aspect, etc.)
        Returns None.
        """
        colors = ["red", "green"]
        axs_element = self._get_subplot_axs(fig, axs, row, col)
        for mask_idx, mask in enumerate(masks):
            color = colors[mask_idx]
            cmap = mcolors.ListedColormap([color])
            alpha = self._get_default_alpha(mask)

            self.plot_img(mask, fig, axs, row, col, title=title,
                                    cmap=cmap, alpha=alpha, **kwargs)

        # legends
        handles = []
        for idx, color in enumerate(colors):
            handles.append(mpatches.Patch(color=color, label=legends[idx]))

        axs_element.legend(handles=handles, loc='upper left')
        
    def _get_default_alpha(self, img_data):
        """
        Helper function for getting the default transparency mask
        Values that are zero will be fully transparent and
        all else will have an alpha blending transparency of 0.5 (50%)

        Inputs:
            img_data - Nifti image, or numpy ndarray

        Returns numpy ndarray or None.
        """
        if isinstance(img_data, Image.Image):
            return np.where(np.array(img_data) <= 0, 0, 0.5)
        elif isinstance(img_data, np.ndarray):
            return np.where(img_data <= 0, 0, 0.5)
        else:
            return None

    def _get_subplot_axs(self, fig, axs, row, col):
        """
        Helper function to access specified subplot in a
        matplotlib.pyplot.axes object

        Inputs:
            fig - matplotlib.pyplot.figure object
            axs - matplotlib.pyplot.axes object
            row - int, specified row in subplot to access
            col - int, specified col in subplot to access

        Returns matplotlib.pyplot.axes object at specified subplot location
        """
        if len(fig.axes) > 1:
            if len(axs.shape) == 1:
                # if axis has subplot but with 1 row
                return axs[col]
            else:
                return axs[row, col]
        else:
            # if figure has no subplots
            return axs

    def _is_image_3d(self, img_data):
        """
        Helper function to determine whether the image is 3D

        Inputs:
            img_data, numpy ndarray, or other

        Returns an int or a bool
        """
        if isinstance(img_data, np.ndarray):
            return img_data.ndim == 3
        return False

