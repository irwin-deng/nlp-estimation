import numpy as np
import random
from sklearn.metrics import accuracy_score
import pickle
from bootstrap import bootstrap_CI, weighted_bootstrap_CI
from sentence_transformers import SentenceTransformer


sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# L2 square distance between two vectorized images x and y
def distance1(x,y):
    return np.sum(np.square(x-y))
# L2 distance between two vectorized images x and y
def distance2(x,y):
    return np.sqrt(np.sum(np.square(x-y)))
# and can be coded as below
def distance3(x,y):
    return np.linalg.norm(x-y)

def distance_text(x,y):
    # {'text':text, "NER": m, "GOLD_NER": gold_ner, 'token_offset': gold_token_offset}
    embed_x = sbert_model.encode(x['text']).reshape(1, -1)
    embed_y = sbert_model.encode(y['text']).reshape(1, -1)
    cos_dist = cosine_distances(embed_x, embed_y)[0][0]
    return cos_dist

def distance_entities(x,y):
    list_entities_x = [x['NER'][key][0] for key in x['NER'].keys()]
    list_entities_y = [y['NER'][key][0] for key in y['NER'].keys()]
    return 0

def kNN(x, k, data): #, label):
    #create a list of distances between the given image and the images of the training set
#     distances =[np.linalg.norm(x-data[i]) for i in range(len(data))]
    distances = [distance1(x, data[i]) for i in range(len(data))]
    #Use "np.argpartition". It does not sort the entire array.
    #It only guarantees that the kth element is in sorted position
    # and all smaller elements will be moved before it.
    # Thus the first k elements will be the k-smallest elements.
    idx = np.argpartition(distances, k)
    return idx[:k]
    # clas, freq = np.unique(label[idx[:k]], return_counts=True)
    # return clas[np.argmax(freq)]

def kNN_NER(x, k, data):
    distances = [distance_text(x, data[i]) + distance_entities(x, data[i]) for i in range(len(data))]
    idx = np.argpartition(distances, k)
    return idx[:k]


def weighted_bootstrap(unlabeled_gold_y, gold_y, pred_y_labeled, weights, seed, alpha):
    # sample from the labeled test dataset according to weights that are determined by the similarity to the
    # unlabeled test dataset
    np.random.seed(seed)
    n_boots = 1000
    n_examples_gold = len(unlabeled_gold_y)
    n_examples = len(pred_y_labeled)
    stats = []
    # perform weighted bootstrap sampling
    for r in range(n_boots):
        unlabeled_boot_sample_inds = np.random.randint(0, n_examples_gold, n_examples)  # sample with repetitions
        boot_sample_inds = []
        for unlabeled_ind in unlabeled_boot_sample_inds:
            boot_sample_inds.append(np.argmax(np.random.multinomial(1, weights[unlabeled_ind], size=1)))
        temp_gold_y = []
        temp_pred_y = []
        for ind in boot_sample_inds:
            temp_gold_y.append(gold_y[ind])
            temp_pred_y.append(pred_y_labeled[ind])
        delta = accuracy_score(temp_gold_y, temp_pred_y)
        stats.append(delta)
    # calculate CI
    # p = ((1.0 - alpha) / 2.0) * 100
    p = alpha / 2.0
    lower = max(0.0, np.percentile(stats, p))
    # p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    p = 1.0 - alpha / 2.0
    upper = np.percentile(stats, p)

    # for normal bootstrap
    std_sample = np.std(stats)
    z = scipy.stats.norm.ppf(alpha / 2.0)
    avg_delta = np.average(stats)
    lower_normal = avg_delta - std_sample * z
    upper_normal = avg_delta + std_sample * z

    return lower, upper, lower_normal, upper_normal


def uniform_bootstrap(gold_y, pred_y_labeled, seed, alpha):
    # sample uniformly from the labeled test dataset to estimate the performance on the unlabeled dataset
    np.random.seed(seed)
    n_boots = 1000
    n_examples = len(pred_y_labeled)
    stats = []
    # perform uniform bootstrap sampling
    for r in range(n_boots):
        boot_sample_inds = np.random.randint(0, n_examples, n_examples)  # sample with repetitions
        temp_gold_y = []
        temp_pred_y = []
        for ind in boot_sample_inds:
            temp_gold_y.append(gold_y[ind])
            temp_pred_y.append(pred_y_labeled[ind])
        delta = accuracy_score(temp_gold_y, temp_pred_y)
        stats.append(delta)
    # calculate CI
    # p = ((1.0 - alpha) / 2.0) * 100
    p = alpha / 2.0
    lower = max(0.0, np.percentile(stats, p))
    # p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    p = 1.0 - alpha / 2.0
    upper = min(1.0, np.percentile(stats, p))

    # for normal bootstrap
    std_sample = np.std(stats)
    z = scipy.stats.norm.ppf(alpha / 2.0)
    avg_delta = np.average(stats)
    lower_normal = avg_delta - std_sample * z
    upper_normal = avg_delta + std_sample * z

    return lower, upper, lower_normal, upper_normal


def calc_KNN_matrix(test_set, k):
    N = len(test_set)
    knn_dict = {}
    for ind in range(N):
        knn_dict[ind] = kNN(test_set[ind], 2*k, test_set)
    with open('KNN_mat_MNIST_4000.pickle', 'wb') as handle:
        pickle.dump(knn_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # return knn_dict


def calc_KNN_matrix_NER(test_set, k):
    N = len(test_set)
    knn_dict = {}
    for ind in range(N):
        knn_dict[ind] = kNN_NER(test_set[ind], k, test_set)
    with open('KNN_mat_NER.pickle', 'wb') as handle:
        pickle.dump(knn_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_k_nearest(neighbors, allowed_indexes, k):
    added = 0
    output_neighbors = np.zeros(k)
    i = 0
    while added < k:
        if neighbors[i] in allowed_indexes:
            output_neighbors[added] = neighbors[i]
            added += 1
        i += 1
    return output_neighbors


def NER_experiment(seed):
    random.seed(seed)
    k = 4  # the number of neighbors taken in weighted bootstrap
    with open('ontonotes_ner.pkl', 'rb') as f:
        ontonotes_ner = pickle.load(f)
    all_docs_annotations = list(ontonotes_ner.values())
    calc_KNN_matrix_NER(all_docs_annotations, k)
    exit(0)
    with open('KNN_mat_NER.pickle', 'rb') as handle:
        KNN_All = pickle.load(handle)
    counter_u = 0
    counter_w = 0
    counter_overestimate_u = 0
    counter_underestimate_u = 0
    counter_overestimate_w = 0
    counter_underestimate_w = 0
    counter_u_e = 0
    counter_w_e = 0
    counter_overestimate_u_e = 0
    counter_underestimate_u_e = 0
    counter_overestimate_w_e = 0
    counter_underestimate_w_e = 0
    for i in range(100):
        indices = np.random.permutation(len(all_docs_annotations))
        labeled_test_inds = indices[:int(0.9 * len(indices))]
        unlabeled_test_inds = indices[int(0.9 * len(indices)):]
        labeled_test_gold_y = []
        labeled_test_predicted_y = []
        labeled_test_predicted_x = []
        for j in labeled_test_inds:
            labeled_test_gold_y.append(all_docs_annotations[j]['GOLD_NER'])
            labeled_test_predicted_y.append(all_docs_annotations[j]['NER'])
            labeled_test_predicted_x.append({'text': all_docs_annotations[j]['text'],
                                             'tokens_offset': all_docs_annotations[j]['token_offset']})
        unlabeled_test_gold_y = []
        unlabeled_test_predicted_y = []
        unlabeled_test_predicted_x = []
        weights_matrix = []
        for j in unlabeled_test_inds:
            unlabeled_test_gold_y.append(all_docs_annotations[j]['GOLD_NER'])
            unlabeled_test_predicted_y.append(all_docs_annotations[j]['NER'])
            unlabeled_test_predicted_x.append({'text': all_docs_annotations[j]['text'],
                                               'tokens_offset': all_docs_annotations[j]['token_offset']})


            indexes = find_k_nearest(KNN_All[j], labeled_test_inds, k)
            weights = np.zeros(len(labeled_test_predicted_y))
            for m in range(k):
                rel_ind = int(np.where(labeled_test_inds == indexes[m])[0])
                weights[rel_ind] = 2.0 * (k - m) / (k * (k + 1))  # high prob. to closest - causes overestimation
            weights_matrix.append(weights)

        # calculate true performance on unlabeled set
        labeled_accuracy = accuracy_score(labeled_test_predicted_y, labeled_test_gold_y)
        true_accuracy = accuracy_score(unlabeled_test_predicted_y, unlabeled_test_gold_y)
        uni_lower, uni_upper = bootstrap_CI(labeled_test_gold_y, labeled_test_predicted_y, 0.95, 100, 10, seed)

        empirical_uni_lower = 2 * labeled_accuracy - uni_upper
        empirical_uni_upper = 2 * labeled_accuracy - uni_lower

        weight_lower, weight_upper = weighted_bootstrap_CI(labeled_test_gold_y, labeled_test_predicted_y,
                                                           unlabeled_test_gold_y, weights_matrix, 0.95, 100, 10, seed)

        empirical_weighted_lower = 2 * labeled_accuracy - weight_upper
        empirical_weighted_upper = 2 * labeled_accuracy - weight_lower
        if true_accuracy <= uni_upper and true_accuracy >= uni_lower:
            counter_u += 1
        if true_accuracy <= weight_upper and true_accuracy >= weight_lower:
            counter_w += 1
        if true_accuracy > uni_upper:
            counter_underestimate_u += 1
        if true_accuracy < uni_lower:
            counter_overestimate_u += 1
        if true_accuracy > weight_upper:
            counter_underestimate_w += 1
        if true_accuracy < weight_lower:
            counter_overestimate_w += 1

        if true_accuracy <= empirical_uni_upper and true_accuracy >= empirical_uni_lower:
            counter_u_e += 1
        if true_accuracy <= empirical_weighted_upper and true_accuracy >= empirical_weighted_lower:
            counter_w_e += 1
        if true_accuracy > empirical_uni_upper:
            counter_underestimate_u_e += 1
        if true_accuracy < empirical_uni_lower:
            counter_overestimate_u_e += 1
        if true_accuracy > empirical_weighted_upper:
            counter_underestimate_w_e += 1
        if true_accuracy < empirical_weighted_lower:
            counter_overestimate_w_e += 1

        if i % 10 == 0:
            print(i)
            print("true accuracy: ", true_accuracy)
            print("uniform: ", (uni_lower, uni_upper))
            print("weighted: ", (weight_lower, weight_upper))
            print("empirical uniform: ", (empirical_uni_lower, empirical_uni_upper))
            print("empirical weighted: ", (empirical_weighted_lower, empirical_weighted_upper))
            print('uniform_bootstrap: ', counter_u / (i + 1))
            print('weighted_bootstrap: ', counter_w / (i + 1))
            print("uniform overestimate: ", counter_overestimate_u / (i + 1))
            print("uniform underestimate: ", counter_underestimate_u / (i + 1))
            print("weighted overestimate: ", counter_overestimate_w / (i + 1))
            print("weighted underestimate: ", counter_underestimate_w / (i + 1))
            print('uniform_bootstrap empirical: ', counter_u_e / (i + 1))
            print('weighted_bootstrap empirical: ', counter_w_e / (i + 1))
            print("uniform overestimate empirical: ", counter_overestimate_u_e / (i + 1))
            print("uniform underestimate empirical: ", counter_underestimate_u_e / (i + 1))
            print("weighted overestimate empirical: ", counter_overestimate_w_e / (i + 1))
            print("weighted underestimate empirical: ", counter_underestimate_w_e / (i + 1))

    print('uniform_bootstrap: ', counter_u / 100.0)
    print('weighted_bootstrap: ', counter_w / 100.0)
    print("uniform overestimate: ", counter_overestimate_u / 100.0)
    print("uniform underestimate: ", counter_underestimate_u / 100.0)
    print("weighted overestimate: ", counter_overestimate_w / 100.0)
    print("weighted underestimate: ", counter_underestimate_w / 100.0)
    print('empirical uniform_bootstrap: ', counter_u_e / 100.0)
    print('empirical weighted_bootstrap: ', counter_w_e / 100.0)
    print("empirical uniform overestimate: ", counter_overestimate_u_e / 100.0)
    print("empirical uniform underestimate: ", counter_underestimate_u_e / 100.0)
    print("empirical weighted overestimate: ", counter_overestimate_w_e / 100.0)
    print("empirical weighted underestimate: ", counter_underestimate_w_e / 100.0)


def MNIST_experiment(seed):
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # k = 4000  # the number of neighbors taken in weighted bootstrap
    k = 100
    # KNN_All = calc_KNN_matrix(test_X, k)
    # exit(0)
    with open('KNN_mat_MNIST_4000.pickle', 'rb') as handle:
        KNN_All = pickle.load(handle)
    test_y_preds = pickle.load(open("MNIST_test_preds.pkl", "rb"))
    random.seed(seed)
    counter_u = 0
    counter_w = 0
    counter_overestimate_u = 0
    counter_underestimate_u = 0
    counter_overestimate_w = 0
    counter_underestimate_w = 0
    counter_u_e = 0
    counter_w_e = 0
    counter_overestimate_u_e = 0
    counter_underestimate_u_e = 0
    counter_overestimate_w_e = 0
    counter_underestimate_w_e = 0
    counter_u_n = 0
    counter_w_n = 0
    counter_overestimate_u_n = 0
    counter_underestimate_u_n = 0
    counter_overestimate_w_n = 0
    counter_underestimate_w_n = 0
    for i in range(10):
        indices = np.random.permutation(len(test_y_preds))
        labeled_test_inds = indices[:int(0.9*len(indices))]
        unlabeled_test_inds = indices[int(0.9*len(indices)):]
        labeled_test_gold_y = []
        labeled_test_predicted_y = []
        labeled_test_predicted_x = []
        for j in labeled_test_inds:
            labeled_test_gold_y.append(test_y[j])
            labeled_test_predicted_y.append(test_y_preds[j])
            labeled_test_predicted_x.append(test_X[j])
        unlabeled_test_gold_y = []
        unlabeled_test_predicted_y = []
        unlabeled_test_predicted_x = []
        weights_matrix = []
        for j in unlabeled_test_inds:
            unlabeled_test_gold_y.append(test_y[j])
            unlabeled_test_predicted_y.append(test_y_preds[j])
            unlabeled_test_predicted_x.append(test_X[j])

            indexes = find_k_nearest(KNN_All[j], labeled_test_inds, k)
            weights = np.zeros(len(labeled_test_predicted_y))
            for m in range(k):
                rel_ind = int(np.where(labeled_test_inds == indexes[m])[0])
                # weights[rel_ind] = 2.0 * (k - m) / (k * (k + 1))  # high prob. to closest - causes overestimation
                # weights[rel_ind] = 2.0 * (m + 1) / (k * (k + 1))  # low prob. to closest
                weights[rel_ind] = 1/k
            weights_matrix.append(weights)

        # calculate true performance on unlabeled set
        labeled_accuracy = accuracy_score(labeled_test_predicted_y, labeled_test_gold_y)
        true_accuracy = accuracy_score(unlabeled_test_predicted_y, unlabeled_test_gold_y)
        # counter = 0
        # for l in range(len(unlabeled_test_predicted_y)):
        #     if unlabeled_test_predicted_y[l] == unlabeled_test_gold_y[l]:
        #         counter += 1
        # accuracy = counter / float(len(unlabeled_test_predicted_y))


        # uniform bootstrap CI
        # uni_lower, uni_upper = uniform_bootstrap(labeled_test_gold_y, labeled_test_predicted_y, seed, 0.95)

        uni_lower, uni_upper, uni_lower_n, uni_upper_n = bootstrap_CI(labeled_test_gold_y, labeled_test_predicted_y, 0.95, 100, 10, seed)

        # weighted bootstrap CI
        # weights_matrix = []
        #
        # for ind_x in unlabeled_test_inds:
        #     indexes = find_k_nearest(KNN_All[ind_x], unlabeled_test_inds, k)
        #     weights = np.zeros(len(labeled_test_predicted_y))
        #     for m in range(k):
        #         weights[int(indexes[m])] = 2.0 * (k - m) / (k * (k + 1))
        #     weights_matrix.append(weights)


        # for x in unlabeled_test_predicted_x:
        #     indexes = kNN(x, k, labeled_test_predicted_x)  # the first index is the closest
        #     weights = np.zeros(len(labeled_test_predicted_y))
        #     for m in range(k):
        #         weights[indexes[m]] = 2.0*(k-m)/(k*(k+1))  # (k-i)/S_k
        #         # weights[indexes[m]] = 2.0 * (k - m) / (k * (k + 1)) #trial - gives low probability to the more similar ones
        #     Weights.append(weights)

        # weight_lower, weight_upper = weighted_bootstrap(unlabeled_test_gold_y, labeled_test_gold_y, labeled_test_predicted_y, Weights, seed, 0.95)

        # empirical_uni_lower = 2*labeled_accuracy - uni_upper
        # empirical_uni_upper = 2*labeled_accuracy - uni_lower

        weight_lower, weight_upper, weight_lower_n, weight_upper_n = weighted_bootstrap_CI(labeled_test_gold_y, labeled_test_predicted_y,
                                                           unlabeled_test_gold_y, weights_matrix, 0.95, 100, 10, seed)

        # empirical_weighted_lower = 2*labeled_accuracy - weight_upper
        # empirical_weighted_upper = 2*labeled_accuracy - weight_lower
        # if true_accuracy <= uni_upper and true_accuracy >= uni_lower:
        #     counter_u += 1
        # if true_accuracy <= weight_upper and true_accuracy >= weight_lower:
        #     counter_w += 1
        # if true_accuracy > uni_upper:
        #     counter_underestimate_u += 1
        # if true_accuracy < uni_lower:
        #     counter_overestimate_u += 1
        # if true_accuracy > weight_upper:
        #     counter_underestimate_w += 1
        # if true_accuracy < weight_lower:
        #     counter_overestimate_w += 1

        # if true_accuracy <= empirical_uni_upper and true_accuracy >= empirical_uni_lower:
        #     counter_u_e += 1
        # if true_accuracy <= empirical_weighted_upper and true_accuracy >= empirical_weighted_lower:
        #     counter_w_e += 1
        # if true_accuracy > empirical_uni_upper:
        #     counter_underestimate_u_e += 1
        # if true_accuracy < empirical_uni_lower:
        #     counter_overestimate_u_e += 1
        # if true_accuracy > empirical_weighted_upper:
        #     counter_underestimate_w_e += 1
        # if true_accuracy < empirical_weighted_lower:
        #     counter_overestimate_w_e += 1

        if true_accuracy <= uni_upper_n and true_accuracy >= uni_lower_n:
            counter_u_n += 1
        if true_accuracy <= weight_upper_n and true_accuracy >= weight_lower_n:
            counter_w_n += 1
        if true_accuracy > uni_upper_n:
            counter_underestimate_u_n += 1
        if true_accuracy < uni_lower_n:
            counter_overestimate_u_n += 1
        if true_accuracy > weight_upper_n:
            counter_underestimate_w_n += 1
        if true_accuracy < weight_lower_n:
            counter_overestimate_w_n += 1



        # if i % 10 == 0:
        # print(i)
        print("true accuracy: ", true_accuracy)
        # print("uniform: ", (uni_lower, uni_upper))
        # print("weighted: ", (weight_lower, weight_upper))
        # print("empirical uniform: ", (empirical_uni_lower, empirical_uni_upper))
        # print("empirical weighted: ", (empirical_weighted_lower, empirical_weighted_upper))
        print("normal uniform: ", (uni_lower_n, uni_upper_n))
        print("normal weighted: ", (weight_lower_n, weight_upper_n))

        # print('uniform_bootstrap: ', counter_u / (i+1))
        # print('weighted_bootstrap: ', counter_w / (i+1))
        # print("uniform overestimate: ", counter_overestimate_u / (i+1))
        # print("uniform underestimate: ", counter_underestimate_u / (i+1))
        # print("weighted overestimate: ", counter_overestimate_w / (i+1))
        # print("weighted underestimate: ", counter_underestimate_w / (i+1))
        # print('uniform_bootstrap empirical: ', counter_u_e / (i + 1))
        # print('weighted_bootstrap empirical: ', counter_w_e / (i + 1))
        # print("uniform overestimate empirical: ", counter_overestimate_u_e / (i + 1))
        # print("uniform underestimate empirical: ", counter_underestimate_u_e / (i + 1))
        # print("weighted overestimate empirical: ", counter_overestimate_w_e / (i + 1))
        # print("weighted underestimate empirical: ", counter_underestimate_w_e / (i + 1))

    print("\n-------------------------")
    # print('uniform_bootstrap: ', counter_u / 10.0)
    # print('weighted_bootstrap: ', counter_w / 10.0)
    # print("uniform overestimate: ", counter_overestimate_u / 10.0)
    # print("uniform underestimate: ", counter_underestimate_u / 10.0)
    # print("weighted overestimate: ", counter_overestimate_w / 10.0)
    # print("weighted underestimate: ", counter_underestimate_w / 10.0)
    # print('empirical uniform_bootstrap: ', counter_u_e / 10.0)
    # print('empirical weighted_bootstrap: ', counter_w_e / 10.0)
    # print("empirical uniform overestimate: ", counter_overestimate_u_e / 10.0)
    # print("empirical uniform underestimate: ", counter_underestimate_u_e / 10.0)
    # print("empirical weighted overestimate: ", counter_overestimate_w_e / 10.0)
    # print("empirical weighted underestimate: ", counter_underestimate_w_e / 10.0)
    print('normal uniform_bootstrap: ', counter_u_n / 30.0)
    print('normal weighted_bootstrap: ', counter_w_n / 30.0)
    print("normal uniform overestimate: ", counter_overestimate_u_n / (30.0 - counter_u_n))
    print("normal uniform underestimate: ", counter_underestimate_u_n / (30.0 - counter_u_n))
    print("normal weighted overestimate: ", counter_overestimate_w_n / (30.0 - counter_w_n))
    print("normal weighted underestimate: ", counter_underestimate_w_n / (30.0 - counter_w_n))


def vision_experiment(test_y_preds, test_X, test_y, seed):
    random.seed(seed)
    counter_u = 0
    counter_w = 0
    for i in range(1000):
        indices = np.random.permutation(len(test_y_preds))
        labeled_test_inds = indices[:int(0.9 * len(indices))]
        unlabeled_test_inds = indices[int(0.9 * len(indices)):]
        labeled_test_gold_y = []
        labeled_test_predicted_y = []
        labeled_test_predicted_x = []
        for j in labeled_test_inds:
            labeled_test_gold_y.append(test_y[j])
            labeled_test_predicted_y.append(test_y_preds[i])
            labeled_test_predicted_x.append(test_X[i])
        unlabeled_test_gold_y = []
        unlabeled_test_predicted_y = []
        unlabeled_test_predicted_x = []
        for j in unlabeled_test_inds:
            unlabeled_test_gold_y.append(test_y[j])
            unlabeled_test_predicted_y.append(test_y_preds[i])
            unlabeled_test_predicted_x.append(test_X[i])

        # calculate true performance on unlabeled set
        true_accuracy = accuracy_score(unlabeled_test_predicted_y, unlabeled_test_gold_y)

        # uniform bootstrap CI
        uni_lower, uni_upper = uniform_bootstrap(labeled_test_gold_y, labeled_test_predicted_y, seed, 0.95)

        # weighted bootstrap CI
        Weights = []
        k = 4
        for x in unlabeled_test_predicted_x:
            indexes = kNN(x, k, labeled_test_predicted_x)  # the first index is the closest
            weights = np.zeros(len(labeled_test_predicted_y))
            for m in range(k):
                weights[indexes[m]] = 2.0 * (k - m) / (k * (k + 1))  # (k-i)/S_k
            Weights.append(weights)

        weight_lower, weight_upper = weighted_bootstrap(unlabeled_test_gold_y, labeled_test_gold_y,
                                                        labeled_test_predicted_y, Weights, seed, 0.95)

        if true_accuracy <= uni_upper and true_accuracy >= uni_lower:
            counter_u += 1
        if true_accuracy <= weight_upper and true_accuracy >= weight_lower:
            counter_w += 1

    print('uniform_bootstrap: ', counter_u / 1000.0)
    print('weighted_bootstrap: ', counter_w / 1000.0)



def main():
    # NER_experiment(2020)
    MNIST_experiment(73)

    # seed = 2014
    #
    # from keras import cifar10
    # (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    # test_y_preds = pickle.load(open("CIFAR10_test_preds.pkl", "rb"))
    # vision_experiment(test_y_preds, test_X, test_y, seed)
    #
    # from keras import cifar100
    # (train_X, train_y), (test_X, test_y) = cifar100.load_data(label_mode="fine")
    # test_y_preds = []
    # vision_experiment(test_y_preds, test_X, test_y, seed)
    #
    # from keras import fashion_mnist
    # (train_X, train_y), (test_X, test_y) = cifar100.load_data(label_mode="fine")
    # test_y_preds = []
    # vision_experiment(test_y_preds, test_X, test_y, seed)






    # Goal: estimate the models performance on the unlabeled test set
    # Method: calculate CI for the performance on the unlabeled examples based on the labeled test set





    return 0





if __name__ == '__main__':
    main()