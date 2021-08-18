import os
import platform
import numpy as np
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt

def train(model, data):

    return 1

if __name__ == "__main__":
    current = os.getcwd()

    # check which system you're running on to get path
    if (platform.system() == "Windows"):
        img_path = os.path.join(current, 'BRATS_data\imagesTr\\')
        label_path = os.path.join(current, 'BRATS_data\labelsTr\\')
    else:
        img_path = os.path.join(current, 'BRATS_data/imagesTr/')
        label_path = os.path.join(current, 'BRATS_data/labelsTr/')

    img_list = os.listdir(img_path)
    label_list = os.listdir(label_path)
    img_list = [img_path + img for img in img_list]
    label_list =  [label_path + label for label in label_list]

    # pick which image to choose from
    index = 65

    img = nib.load(img_list[index]).get_fdata()
    label = nib.load(label_list[index]).get_fdata()

    whichImg = img.shape[2]//2 + 10

    # (FLAIR, T1w, T1gd,T2w)
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize = (20, 10))
    ax1.imshow(img[:,:,whichImg,0], cmap = 'gray')
    ax1.set_title('FLAIR')
    ax2.imshow(img[:,:,whichImg, 1], cmap = 'gray')
    ax2.set_title('T1w')
    ax3.imshow(img[:,:,whichImg, 2], cmap = 'gray')
    ax3.set_title('T1gd')
    ax4.imshow(img[:,:,whichImg, 3], cmap = 'gray')
    ax4.set_title('T2w')
    ax5.imshow(label[:,:,whichImg] == 0)
    ax6.imshow(label[:,:,whichImg] == 1)
    ax7.imshow(label[:,:,whichImg] == 2)
    ax8.imshow(label[:,:,whichImg] == 3)
    ax5.set_title('background')
    ax6.set_title('edema')
    ax7.set_title('non-enhancing tumor')
    ax8.set_title('enhancing tumour')


    # https://matplotlib.org/3.3.2/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
    # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    # fig = plt.figure(figsize=(20, 10))

    # gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])

    # #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])
    # flair = ax0.imshow(sample_img[:,:,65], cmap='bone')
    # ax0.set_title("FLAIR", fontsize=18, weight='bold', y=-0.2)
    # fig.colorbar(flair)

    # #  Varying density along a streamline
    # ax1 = fig.add_subplot(gs[0, 1])
    # t1 = ax1.imshow(sample_img2[:,:,65], cmap='bone')
    # ax1.set_title("T1", fontsize=18, weight='bold', y=-0.2)
    # fig.colorbar(t1)

    # #  Varying density along a streamline
    # ax2 = fig.add_subplot(gs[0, 2])
    # t2 = ax2.imshow(sample_img3[:,:,65], cmap='bone')
    # ax2.set_title("T2", fontsize=18, weight='bold', y=-0.2)
    # fig.colorbar(t2)

    # #  Varying density along a streamline
    # ax3 = fig.add_subplot(gs[0, 3])
    # t1ce = ax3.imshow(sample_img4[:,:,65], cmap='bone')
    # ax3.set_title("T1 contrast", fontsize=18, weight='bold', y=-0.2)
    # fig.colorbar(t1ce)

    # #  Varying density along a streamline
    # ax4 = fig.add_subplot(gs[1, 1:3])

    # #ax4.imshow(np.ma.masked_where(mask_WT[:,:,65]== False,  mask_WT[:,:,65]), cmap='summer', alpha=0.6)
    # l1 = ax4.imshow(mask_WT[:,:,65], cmap='summer',)
    # l2 = ax4.imshow(np.ma.masked_where(mask_TC[:,:,65]== False,  mask_TC[:,:,65]), cmap='rainbow', alpha=0.6)
    # l3 = ax4.imshow(np.ma.masked_where(mask_ET[:,:,65] == False, mask_ET[:,:,65]), cmap='winter', alpha=0.6)

    # ax4.set_title("", fontsize=20, weight='bold', y=-0.1)

    # _ = [ax.set_axis_off() for ax in [ax0,ax1,ax2,ax3, ax4]]

    # colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]
    # labels = ['Non-Enhancing tumor core', 'Peritumoral Edema ', 'GD-enhancing tumor']
    # patches = [ mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 'xx-large',
    #         title='Mask Labels', title_fontsize=18, edgecolor="black",  facecolor='#c5c6c7')

    # plt.suptitle("Multimodal Scans -  Data | Manually-segmented mask - Target", fontsize=20, weight='bold')

    # fig.savefig("data_sample.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    # fig.savefig("data_sample.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    
    plt.show()