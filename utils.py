
import h5py
import random
import numpy as np
from PIL import Image
from math import floor
from itertools import groupby, chain
from keras.utils.io_utils import HDF5Matrix

root = './csiq_video_database/'
random.seed(123)

def list_random(ran):                                                                                                                       #EMBARALHA A LISTA
    random.shuffle(ran)
    return ran

def keyfunc(s):                                                                                                                             #KEY PARA ORDERNAR A LISTA
    return [int(''.join(g)) if k else ''.join(g) for k, g in groupby('\0'+s, str.isdigit)]

def create_lists(txt_file, of_type, res, train_size, val_split, dataset_split):                                                                                     #CRIA LISTAS DE TREINAMENTO E DE TESTE
    f_train = open(txt_file, 'r')
    train_list = f_train.readlines()

    train_list = [x.split('	') for x in train_list]
    video_names = [x[0].split('.yuv')[0] for x in train_list[1:]]
    video_scores = [float(x[1].split('.yuv')[0])/100. for x in train_list[1:]]
    video_std = [float(x[2].split('.yuv')[0])/100. for x in train_list[1:]]

    videos = []
    for i in range(0,len(video_names),18):
        videos.append(video_names[i].split('_dst')[0])
    #videos = list_random(videos)

    ##SPLIT DATASET
    split = int(len(videos)/dataset_split)
    videos = videos[:split]

    ##TRAIN, VAL, TEST SPLIT
    videos_names_train1 = []
    videos_names_test = []
    for i in range(len(videos)):
        if i < int(train_size*len(videos)):
            videos_names_train1.append(videos[i])
        else:
            videos_names_test.append(videos[i])

    videos_names_train = []
    videos_names_validation = []
    for i in range(len(videos_names_train1)):
        if i < int(val_split*len(videos_names_train1)):
            videos_names_train.append(videos_names_train1[i])
        else:
            videos_names_validation.append(videos_names_train1[i])

    videos_names_test = list_random(videos_names_test)
    videos_names_train = list_random(videos_names_train)
    videos_names_validation = list_random(videos_names_validation)

    video_train_shuffled = []
    for video in videos_names_train:
        for i in range(18):
            if i < 9:
                video_train_shuffled.append(video+'_dst_0%d' %(i+1))
            else:
                video_train_shuffled.append(video+'_dst_%d' %(i+1))

    video_val_shuffled = []
    for video in videos_names_validation:
        for i in range(18):
            if i < 9:
                video_val_shuffled.append(video+'_dst_0%d' %(i+1))
            else:
                video_val_shuffled.append(video+'_dst_%d' %(i+1))

    video_test_shuffled = []
    for video in videos_names_test:
        for i in range(18):
            if i < 9:
                video_test_shuffled.append(video+'_dst_0%d' %(i+1))
            else:
                video_test_shuffled.append(video+'_dst_%d' %(i+1))

    ##CREATE LISTS
    folder_path_train = []
    for i in range(len(video_train_shuffled)):
        folder_path_train.append(root+video_train_shuffled[i].split('_dst')[0]+'/'+video_train_shuffled[i]+'/')

    folder_path_val = []
    for i in range(len(video_val_shuffled)):
        folder_path_val.append(root+video_val_shuffled[i].split('_dst')[0]+'/'+video_val_shuffled[i]+'/')

    folder_path_test = []
    for i in range(len(video_test_shuffled)):
        folder_path_test.append(root+video_test_shuffled[i].split('_dst')[0]+'/'+video_test_shuffled[i]+'/')

    folder_path_train = list_random(folder_path_train)
    folder_path_val = list_random(folder_path_val)
    folder_path_test = list_random(folder_path_test)

    list_train = []
    for i in range(len(folder_path_train)):
        list_train.append(folder_path_train[i]+'of_'+res+'_'+of_type+'_f')

    list_val = []
    for i in range(len(folder_path_val)):
        list_val.append(folder_path_val[i]+'of_'+res+'_'+of_type+'_f')

    list_test = []
    for i in range(len(folder_path_test)):
        list_test.append(folder_path_test[i]+'of_'+res+'_'+of_type+'_f')

    of_filelist_train = []
    for i in range(len(list_train)):
        of_filelist_train.append([x for x in os.listdir(list_train[i])])

    of_filelist_val = []
    for i in range(len(list_val)):
        of_filelist_val.append([x for x in os.listdir(list_val[i])])

    of_filelist_test = []
    for i in range(len(list_test)):
        of_filelist_test.append([x for x in os.listdir(list_test[i])])

    for i in range(len(list_train)):
        of_filelist_train[i] = sorted(of_filelist_train[i], key=keyfunc)

    for i in range(len(list_val)):
        of_filelist_val[i] = sorted(of_filelist_val[i], key=keyfunc)

    for i in range(len(list_test)):
        of_filelist_test[i] = sorted(of_filelist_test[i], key=keyfunc)

    of_filelist_train = [item for sublist in of_filelist_train for item in sublist]
    of_filelist_val = [item for sublist in of_filelist_val for item in sublist]
    of_filelist_test = [item for sublist in of_filelist_test for item in sublist]

    if of_type == 'comp':
        of_filelist_dx_train = []
        of_filelist_dy_train = []
        for name in of_filelist_train:
            if name.split('_')[6] == 'dx':
                of_filelist_dx_train.append(name)
            else:
                of_filelist_dy_train.append(name)

        of_filelist_dx_val = []
        of_filelist_dy_val = []
        for name in of_filelist_val:
            if name.split('_')[6] == 'dx':
                of_filelist_dx_val.append(name)
            else:
                of_filelist_dy_val.append(name)

        of_filelist_dx_test = []
        of_filelist_dy_test = []
        for name in of_filelist_test:
            if name.split('_')[6] == 'dx':
                of_filelist_dx_test.append(name)
            else:
                of_filelist_dy_test.append(name)

        return of_filelist_dx_train, of_filelist_dy_train, of_filelist_dx_val, of_filelist_dy_val, of_filelist_dx_test, of_filelist_dy_test, video_names, video_scores, list_random(video_train_shuffled), list_random(video_val_shuffled), list_random(video_test_shuffled)

    elif of_type == 'mod':
        return of_filelist_train, of_filelist_val, of_filelist_test, video_names, video_scores, list_random(video_train_shuffled), list_random(video_val_shuffled), list_random(video_test_shuffled)

def normalizeStacks(x):

    print('normalizing batch with shape: '+str(x.shape))

    x = np.rollaxis(x, 1, 4)
    x = x/255.0

    aux = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[3]):
            aux[i,:,:,j] = x[i,:,:,j] - np.mean(x[i,:,:,j])
            aux[i,:,:,j] = aux[i,:,:,j] / np.std(1e-8 + x[i,:,:,j])

    aux = np.rollaxis(aux, 3, 1)

    print('Batch normalized with shape: ' + str(aux.shape))

    return aux

def get_data_mod(chunk_list, video_names, video_score, size,step):

    print('mounting stacks...')                                                                        #MONTA STACK PARA MODULOS
    count = 0
    firstTime = 1
    labelVec = []
    for j in range(0,len(chunk_list),step):
        if (len(chunk_list) - j) > size:
            fmod = []
            for k in range(j,j+size):
                img_mod = Image.open(chunk_list[k])
                img_mod = np.asarray(img_mod)

                fmod.append(img_mod)

            for vidfile in video_names:
                if vidfile == chunk_list[k].split('/')[3]: ##VERIFICAR
                    idx = video_names.index(vidfile)
                    score = video_score[idx]
                    labelVec.append(score)
                    # print('With score ' + str(score))

            flow_mod = np.dstack((fmod[0],fmod[1],fmod[2],fmod[3],fmod[4]))
            flow_mod = np.expand_dims(flow_mod, axis = 0)

            if not firstTime:
                inputVec = np.concatenate((inputVec,flow_mod))

            else:
                inputVec = flow_mod
                firstTime = 0
            count += 1
            #print('stack %d mounted...' %count)

    inputVec = np.rollaxis(inputVec,3,1)
    labelVec = np.asarray(labelVec)
    labelVec = np.expand_dims(labelVec, axis = 1)
    print('stack mounted with shape: ' + str(inputVec.shape))
    print('Label shape: ' + str(labelVec.shape))
    inputVec = inputVec.astype('float32',copy=False)
    labelVec = labelVec.astype('float32',copy=False)

    return inputVec, labelVec

def store_videos_mod(txt_file, size, img_rows, img_cols, step, res, train_size):                                                            #ARMAZENA DADOS DE COMPONENTES EM DISCO*****

    train_mod, test_mod, video_names, video_scores, video_names_train, video_names_test = create_lists(txt_file, 'mod', res, train_size)

    total_size_train = 0
    step_train = []
    for videoname in video_names_train: ## A LISTA VIDEO_NAMES TEM QUE SER AQUELA QUE CONTEM SO OS VIDEOS DE TREINAMENTO
        data_list_mod = []
        for filename in train_mod:
            if videoname == filename.split('_mod')[0]:     ##BQMall_832x480_dst_02 = BQMall_832x480_dst_02
                data_list_mod.append(root+filename.split('_dst')[0]+'/'+filename.split('_mod')[0]+'/of_%dx%d_comp_f/' % (img_cols, img_rows) + filename) ##VERIFICAR

        if (len(data_list_mod) - size) % step == 0:
            nb_stacks_train = ((len(data_list_mod) - size)/step)

        else:
            nb_stacks_train = floor(((len(data_list_mod) - size)/step))

        step_train.append(int(nb_stacks_train))
        total_size_train += int(nb_stacks_train)

    total_size_test = 0
    step_test = []
    for videoname in video_names_test: ## A LISTA VIDEO_NAMES TEM QUE SER AQUELA QUE CONTEM SO OS VIDEOS DE TREINAMENTO
        data_list_mod = []
        for filename in test_mod:
            if videoname == filename.split('_mod')[0]:     ##BQMall_832x480_dst_02 = BQMall_832x480_dst_02
                data_list_mod.append(root+filename.split('_dst')[0]+'/'+filename.split('_mod')[0]+'/of_%dx%d_comp_f/' % (img_cols, img_rows) + filename) ##VERIFICAR

        if (len(data_list_mod) - size) % step == 0:
            nb_stacks_test = ((len(data_list_mod - size)/step))

        else:
            nb_stacks_test = floor(((len(data_list_mod - size)/step)))

        step_test.append(int(nb_stacks_test))
        total_size_test += int(nb_stacks_test)

    f = h5py.File('./data/all_of_chunks_mod_%dx%d.h5' % (img_cols, img_rows), 'w')

    X_dset_train = f.create_dataset('of_stacks_data_mod_train', (total_size_train, size, img_rows, img_cols))
    Y_dset_train = f.create_dataset('labels_train', (total_size_train, 1))

    X_dset_test = f.create_dataset('of_stacks_data_mod_test', (total_size_test, size, img_rows, img_cols))
    Y_dset_test = f.create_dataset('labels_test', (total_size_test, 1))

    gen1 = video_stack_mod_generator(train_mod, video_names, video_names_train, video_scores, size, step, img_cols, img_rows)
    gen2 = video_stack_mod_generator(test_mod, video_names, video_names_test, video_scores, size, step, img_cols, img_rows)

    i = 0
    for s in step_train:
        (X_dset_train[i:i+s,:,:,:], Y_dset_train[i:i+s,:]) = next(gen1)
        i += s

    j = 0
    for s in step_test:
        (X_dset_test[j:j+s,:,:,:], Y_dset_test[j:j+s,:]) = next(gen2)
        j += s

    f.close()
    return

def batch_mod_generator(list_mod, video_names, video_names_list, video_scores, size, step, img_cols, img_rows, batch_size, type):
    if type == 'train':

        for videoname in video_names_list: ## A LISTA VIDEO_NAMES TEM QUE SER AQUELA QUE CONTEM SO OS VIDEOS DE TREINAMENTO
            data_list_mod = []
            for filename in list_mod:
                if videoname == filename.split('_mod')[0]:     ##BQMall_832x480_dst_02 = BQMall_832x480_dst_02
                    data_list_mod.append(root+filename.split('_dst')[0]+'/'+filename.split('_mod')[0]+'/of_%dx%d_mod_f/' % (img_cols, img_rows) + filename) ##VERIFICAR

            X, Y = get_data_mod(data_list_mod, video_names, video_scores, size, step)

            if X.shape[0] % batch_size == 0:
                nb_iteration = int(X.shape[0]/batch_size)
            else:
                nb_iteration = int(X.shape[0]/batch_size) + 1

            index_list = range(0, nb_iteration)
            index_list = np.random.permutation(index_list)

            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            count = 0
            for i in index_list:
                if i*batch_size+batch_size < X.shape[0]:
                    x = normalizeStacks(X[i*batch_size:i*batch_size+batch_size,:,:,:])
                    yield x, Y[i*batch_size:i*batch_size+batch_size,:]
                    print('Iteracao %.0f' % (count+1) +'/' + '%.0f ' %nb_iteration+ 'do video ' + data_list_mod[0].split('/')[3].split('_mod')[0])
                    count += 1
                else:
                    x = normalizeStacks(X[i*batch_size:X.shape[0],:,:,:])
                    yield x, Y[i*batch_size:X.shape[0],:]
                    print('Iteracao %.0f' % (count+1) +'/' + '%.0f ' %nb_iteration+ 'do video ' + data_list_mod[0].split('/')[3].split('_mod')[0])
                    count += 1

    elif type == 'test' or type == 'val':
        for videoname in video_names_list: ## A LISTA VIDEO_NAMES TEM QUE SER AQUELA QUE CONTEM SO OS VIDEOS DE TREINAMENTO
            data_list_mod = []
            for filename in list_mod:
                if videoname == filename.split('_mod')[0]:     ##BQMall_832x480_dst_02 = BQMall_832x480_dst_02
                    data_list_mod.append(root+filename.split('_dst')[0]+'/'+filename.split('_mod')[0]+'/of_%dx%d_mod_f/' % (img_cols, img_rows) + filename) ##VERIFICAR

            X, Y = get_data_mod(data_list_mod, video_names, video_scores, size, step)

            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            x = normalizeStacks(X)

            yield x, Y

def block_shuffle_mod(target_list_mod, size, video_names):
    block_list_mod = []
    for vidname in video_names:
        block = []
        for nameoffile in target_list_mod:
            if vidname == nameoffile.split('_mod')[0]:
                block.append(nameoffile)
        block_list_mod.append(block)

    #block_list_mod = [target_list_mod[x:x+size] for x in range(0, len(target_list_mod), size)]

    index = range(len(block_list_mod))
    index = np.random.permutation(index)

    block_mod = []
    for idx in index:
        block_mod.append(block_list_mod[idx])

    list_mod = list(chain.from_iterable(block_mod))

    return list_mod
