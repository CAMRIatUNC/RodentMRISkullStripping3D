import numpy as np
from .utils import dim_2_categorical


def out_LabelHot_map_3D(img, seg_net, pre_paras, keras_paras,add_input_list=[]):
    # reset the variables
    patch_dims = pre_paras.patch_dims
    label_dims = pre_paras.patch_label_dims
    strides = pre_paras.patch_strides
    n_class = pre_paras.n_class
    meanvalue = pre_paras.meanvalue
    
    if meanvalue is None and pre_paras.issubtract:
        meanvalue = DB.mean_patch_generation(img,patch_dims,3)
    if meanvalue is not None:
        meanvalue = meanvalue[np.newaxis,:]
    
    
    # build new variables for output
    length, col, row = img.shape
    categorical_map = np.zeros((n_class, length, col, row), dtype=np.uint8)
    likelihood_map = np.zeros((length, col, row), dtype=np.float32)
    counter_map = np.zeros((length,col,row), dtype=np.float32)
    # length_step = int(patch_dims[0]/2)
    
    addinputnum = len(add_input_list)

    
    """-----predict the whole image from two directions, small to large and large to small----"""
    for i in range(0, length-patch_dims[0]+1, strides[0]):
        for j in range(0, col-patch_dims[1]+1, strides[1]):
            for k in range(0, row-patch_dims[2]+1, strides[2]):
                cur_patch=img[i:i+patch_dims[0],
                              j:j+patch_dims[1],
                              k:k+patch_dims[2]][:].reshape([1,1,
                                                             patch_dims[0],
                                                             patch_dims[1],
                                                             patch_dims[2]])
                for addidx in range(addinputnum):
                    curaddpatch = add_input_list[addidx][i:i+patch_dims[0],j:j+patch_dims[1],k:k+patch_dims[2]][:].reshape([1,1,patch_dims[0],patch_dims[1],patch_dims[2]])
                    cur_patch = np.append(cur_patch,curaddpatch,axis=1)
                
                
                if pre_paras.issubtract:
                    if meanvalue.shape!=cur_patch.shape:
                        cur_patch[:,0:meanvalue.shape[1],:,:,:] = cur_patch[:,0:meanvalue.shape[1],:,:,:] - meanvalue
                    else:
                        cur_patch = cur_patch - meanvalue              
                
                if keras_paras.img_format == 'channels_last':
                    cur_patch = np.transpose(cur_patch, (0, 2, 3, 4, 1))

                cur_patch_output = seg_net.predict(cur_patch, batch_size=1, verbose=0)

                # if there are multiple outputs
                if isinstance(cur_patch_output,list):
                    cur_patch_output = cur_patch_output[keras_paras.outID]
                cur_patch_output = np.squeeze(cur_patch_output)
                cur_patch_out_label = cur_patch_output.copy()
                cur_patch_out_label[cur_patch_out_label >= keras_paras.thd] = 1
                cur_patch_out_label[cur_patch_out_label < keras_paras.thd] = 0

                # middle = i + length_step
                cur_patch_out_label = dim_2_categorical(cur_patch_out_label,n_class)

                categorical_map[:, i:i+label_dims[0], j:j+label_dims[1], k:k+label_dims[2]] \
                    = categorical_map[:, i:i+label_dims[0], j:j+label_dims[1], k:k+label_dims[2]] + cur_patch_out_label
                likelihood_map[i:i+label_dims[0], j:j+label_dims[1], k:k+label_dims[2]] \
                    = likelihood_map[i:i+label_dims[0], j:j+label_dims[1], k:k+label_dims[2]] + cur_patch_output
                counter_map[i:i+label_dims[0], j:j+label_dims[1], k:k+label_dims[2]] += 1

    for i in range(length, patch_dims[0]-1, -strides[0]):
        for j in range(col, patch_dims[1]-1, -strides[1]):
            for k in range(row, patch_dims[2]-1, -strides[2]):

                cur_patch=img[i-patch_dims[0]:i,
                              j-patch_dims[1]:j,
                              k-patch_dims[2]:k][:].reshape([1,1, patch_dims[0], patch_dims[1], patch_dims[2]])
                
                for addidx in range(addinputnum):
                    curaddpatch = add_input_list[addidx][i:i+patch_dims[0],j:j+patch_dims[1],k:k+patch_dims[2]][:].reshape([1,1,patch_dims[0],patch_dims[1],patch_dims[2]])
                    cur_patch = np.append(cur_patch,curaddpatch,axis=1)
                
                if pre_paras.issubtract:
                    if meanvalue.shape!=cur_patch.shape:
                        cur_patch[:,0:meanvalue.shape[1],:,:,:] = cur_patch[:,0:meanvalue.shape[1],:,:,:] - meanvalue
                    else:
                        cur_patch = cur_patch - meanvalue   
                               
                
                if keras_paras.img_format == 'channels_last':
                    cur_patch = np.transpose(cur_patch, (0, 2, 3, 4, 1))

                cur_patch_output = seg_net.predict(cur_patch, batch_size=1, verbose=0)

                if isinstance(cur_patch_output,list):
                    cur_patch_output = cur_patch_output[keras_paras.outID]
                cur_patch_output = np.squeeze(cur_patch_output)

                cur_patch_out_label = cur_patch_output.copy()
                cur_patch_out_label[cur_patch_out_label >= keras_paras.thd] = 1
                cur_patch_out_label[cur_patch_out_label < keras_paras.thd] = 0

                # middle = i - patch_dims[0] + length_step
                cur_patch_out_label = dim_2_categorical(cur_patch_out_label,n_class)
                categorical_map[:, i-label_dims[0]:i, j-label_dims[1]:j, k-label_dims[2]:k] = \
                    categorical_map[:, i-label_dims[0]:i, j-label_dims[1]:j, k-label_dims[2]:k] + cur_patch_out_label
                likelihood_map[i-label_dims[0]:i, j-label_dims[1]:j, k-label_dims[2]:k] = \
                    likelihood_map[i-label_dims[0]:i, j-label_dims[1]:j, k-label_dims[2]:k] + cur_patch_output
                counter_map[i-label_dims[0]:i, j-label_dims[1]:j, k-label_dims[2]:k] += 1

    label_map = np.zeros([length,col,row],dtype=np.uint8)
    for idx in range(0,length):
        cur_slice_label = np.squeeze(categorical_map[:, idx,].argmax(axis=0))
        label_map[idx,] = cur_slice_label

    counter_map = np.maximum(counter_map, 10e-10)
    likelihood_map = np.divide(likelihood_map,counter_map)

    return label_map, likelihood_map
