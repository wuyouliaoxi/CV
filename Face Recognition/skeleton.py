import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap
import scipy.spatial.distance as spdistance


#############################################
# overview:
# Firstly we use train data set to compute the cluster center. The procedure is that:
# 1. open labels_train and read all lines to get filenames and labels
# 2. random select 500000 descriptors from the 100 train files, we limit max descriptor per file to 5000
# 3. we use MiniBatchKMeans to get 100 cluster centers
# Then we implement VLAD encoding. Here we turn to the test data set
# vlad encoding includes embedding and aggregation, the goal is to get the global descriptor of each file
# 1. we use the function assignments to get the association matrix
# 2. VLAD Embedding: compute residual of each cluster
# 3. Aggregation: aggregate them to a global descriptor of each file
# 4. normalize the global descriptor (Frequent descriptors dominate similarity achieved by power normalization)
# 5. evaluate encodings assuming using associated labels
# Finally we try to improve the precision by using SVM
# 1. To do that we need to encode the train data set
# 2. Set up labels: capture only one enc_test sample as 1 and the all enc_train as -1
# 3. Use sklearn's LinearSVC to create new global descriptors
# 4. evaluate the newly created descriptors
###############################################


def parseArgs(parser):
    # adding arguments
    parser.add_argument('--labels_test',
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    return parser


def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb', '.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p, '')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels


def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')  # matrix of an image containing all descriptors e.g. (5062, 64)

        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)  # descriptors of one image shall be max 5000
        desc = desc[indices]
        descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors  # descriptors from 100 images


def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    dummy = MiniBatchKMeans(n_clusters=n_clusters).fit(descriptors).cluster_centers_
    return dummy


def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # TODO
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors, clusters)  # T长的向量

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)))  # TxK assignment matrix
    # TODO
    for i in range(len(descriptors)):
        assignment[i, matches[i].trainIdx] = 1  # the association function/ matrix, TxK
    return assignment  # return an association matrix (TxK)


def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]  # mus: K x D, 100 x 64. K=100
    encodings = []
    # vlad encoding includes embedding and aggregation, the goal is to get the global descriptor of each file
    # we use the function assignments to get the association matrix a
    for f in tqdm(files):  # 3600 files
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)  # association matrix: descriptors of test match the cluster center of train

        T, D = desc.shape  # definitely D is also 64
        f_enc = np.zeros((D * K), dtype=np.float32)  # global descriptor: 64*100
        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select
            # VLAD Embedding: compute residual of each cluster
            des_idx = np.nonzero(a[:, k])  # Return the indices of the descriptors corresponding to cluster
            residual = desc[des_idx] - mus[k]  # compute residual of each cluster, shape: (xx, 64)
            # Aggregation: use sum pooling, aggregate them to one global descriptor
            if gmp is False:
                # sum pooling
                sum_dis = np.sum(residual, axis=0)  # (64, )
                f_enc[k * D:k * D + D] = sum_dis  # aggregate them to one global descriptor, (1,6400)
            else:
                # Bonus: generalized max pooling
                x = residual
                y = np.ones_like(des_idx)
                clf = Ridge(alpha=1.0, fit_intercept=False, max_iter=500, solver='sparse_cg')
                clf.fit(x, y)  # train the model
                f_enc[k * D:k * D + D] = clf.coef_

        # c) Frequent descriptors dominate similarity achieved by power normalization
        # normalize the global descriptors
        if powernorm:
            # TODO
            f_enc = f_enc.reshape((1, -1))  # Expected 2D array
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))  # power normalization, 幂归一化
        # l2 normalization
        # TODO
        normalize_f_enc = normalize(f_enc, norm='l2')
        # The encoding result shall be the K*D dimensional, i.g. 6400
        if (len(encodings) == 0):
            encodings = normalize_f_enc
        else:
            encodings = np.concatenate((encodings, normalize_f_enc), axis=0)

        # The encoding result shall be the K*D dimensional, i.g. 6400
        # if (len(encodings) == 0):
        #     encodings = f_enc
        # else:
        #     encodings = np.concatenate((encodings, f_enc), axis=0)
    return encodings


def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """

    # set up labels
    # TODO
    N, D = encs_test.shape
    M = encs_train.shape[0]
    # set only one enc_test sample as 1 and the all enc_train as -1
    encs_test_label = np.ones(1)
    encs_train_label = -np.ones(M)
    y_labels = np.concatenate((encs_test_label, encs_train_label), axis=0)  # (M+1,1)
    lsvc = LinearSVC(C=C, class_weight='balanced')

    def loop(i):  # Multi-process?
        # compute SVM 
        # and make feature transformation
        # TODO
        # capture one sample from test
        X = np.concatenate((encs_test[i].reshape(1, D), encs_train), axis=0)  # (M+1,D), D = 6400
        lsvc.fit(X, y_labels)  # X is Training vector, y is Target vector relative to X.
        x = normalize(lsvc.coef_, norm='l2')  # (1xD), normalize the weight vector
        return x

    # let's do that in parallel: 
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    new_encs = list(map(loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO
    dists = 1 - np.dot(encs, encs.T)

    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists


def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)  # TxT distance matrix
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:  # writer retrieval
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1  # top-1 accuracy with smallest distance
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))

    with open("result.txt", "w") as f:
        f.write("Top-1 accuracy:" + str(float(correct) / n_encs) + "\n")
        f.write("mAP:" + str(mAP))


if __name__ == '__main__':
    # creating a parser, description = 'retrieval'
    parser = argparse.ArgumentParser('retrieval')
    # adding arguments
    parser = parseArgs(parser)
    # parsing arguments
    args = parser.parse_args()
    args.__setattr__('labels_train', 'icdar17_local_features/icdar17_labels_train.txt')
    args.__setattr__('labels_test', 'icdar17_local_features/icdar17_labels_test.txt')
    args.__setattr__('in_train', 'icdar17_local_features/train')
    args.__setattr__('in_test', 'icdar17_local_features/test')
    np.random.seed(42)  # fix random seed
    gamma = 1000


    # a) dictionary
    # Firstly we use train data set to compute the cluster center. The procedure is that:
    # open labels_train and read all lines to get filenames and labels
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)  # return: absolute filenames + labels
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):  # if path 'mus.pkl.gz' doesnt exist
        # TODO
        # random select 500000 descriptors from the 100 train files, we limit max descriptor per file to 5000
        # number of descriptors: 470776, local features of dimension 64
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)  # shape: (470776, 64)
        print('> loaded {} descriptors:'.format(len(descriptors)))

        # cluster centers
        print('> compute dictionary')
        # TODO
        # we use MiniBatchKMeans to get 100 cluster centers
        mus = dictionary(descriptors, n_clusters=100)  # return cluster centers (K,D)

        with gzip.open('mus.pkl.gz', 'wb') as fOut:  # store
            cPickle.dump(mus, fOut, -1)  # protocol = -1: pickle.HIGHEST_PROTOCOL=4
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

    # b) Then we implement VLAD encoding. Here we turn to the test data set
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_test = vlad(files_test, mus, powernorm=True, gmp=False, gamma=gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)  # global descriptor: (3600, 6400)

    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_train = vlad(files_train, mus, powernorm=True, gmp=False, gamma=gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    # TODO
    enc_test = esvm(enc_test, enc_train, C=1000)

    # eval
    print('> evaluate')
    evaluate(enc_test, labels_test)
