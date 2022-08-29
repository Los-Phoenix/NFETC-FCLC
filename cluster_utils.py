import sklearn
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
import numpy as np


def normalize(labels):
    return np.divide(labels, np.sum(labels, axis=1).reshape(labels.shape[0], 1))


def fix(labels):
    for i in labels:
        if sum(i) == 0:
            i[0] = 1


def one_hot_to_index(labels):
    return np.stack([np.where(r == 1)[0][0] for r in labels])

def map_cluster_label_ori(noisy_labels, pred) -> dict:
    """要把聚类的标签映射到聚类的簇中多数元素的噪声标签，否则计算得到的互信息是错误的（极低）"""
    n = len(noisy_labels)
    pred_max_idx = max(pred)
    ans = {-1: 0}

    def most_common(lst):
        return max(set(lst), key=lst.count)

    for pred_i in range(pred_max_idx + 1):
        elem = [noisy_labels[each] for each in range(n) if pred[each] == pred_i]
        mapped_label = most_common(elem)
        ans[pred_i] = mapped_label
    return ans

def map_cluster_label(noisy_labels, pred) -> dict:
    """要把聚类的标签映射到聚类的簇中多数元素的噪声标签，否则计算得到的互信息是错误的（极低）"""
    n = len(noisy_labels)
    pred_max_idx = max(pred)
    ans = {-1: 0}

    def most_common(lst):
        ans = max(set(lst), key=lst.count)
        if lst.count(ans) / len(lst) < 0.8:
            ans = -1
        return ans

    def keep_ori(lst, ori):
        ans = ori
        if lst.count(ans) / len(lst) < 0.8:
            ans = -1
        return ans

    for pred_i in range(pred_max_idx + 1):
        elem = [noisy_labels[each] for each in range(n) if pred[each] == pred_i]
        # mapped_label = most_common(elem)
        mapped_label = keep_ori(elem, pred_i)
        ans[pred_i] = mapped_label
    return ans

def map_cluster_onehotlabel(noisy_labels, pred, n_labels, n_clusters, thres) -> dict:
    """要把聚类的标签根据簇中最多数标签占簇的大小决定是否保留"""
    # n = len(noisy_labels)
    # pred_max_idx = max(pred)
    ans = {-1: 0}

    def keep_ori(lst, ori, thres = 0.8):
        total = len(lst)
        label_count = lst.sum(0)
        maxlabelcount = max(label_count)
        ans = ori
        # print('label_count', label_count.shape, label_count)
        # print('map_cluster_onehotlabel', ori, maxlabelcount, total)
        if maxlabelcount / total < thres:
            ans = -1
        return ans

    for pred_i in range(n_clusters + 1):
        elem = noisy_labels[pred == pred_i]
        # mapped_label = most_common(elem)
        mapped_label = keep_ori(elem, pred_i, thres=thres)
        ans[pred_i] = mapped_label
    return ans

def map_cluster_need_walk(noisy_labels, pred, n_labels, n_clusters, thres, min_walk_cluster_count=2) -> dict:
    """要把聚类的标签根据簇中最多数标签占簇的大小决定是否保留"""
    # n = len(noisy_labels)
    # pred_max_idx = max(pred)
    ans_map = {-1: 0}
    count_map = {}

    def most_common(lst, ori, thres = 0.8):

        # most_common = max(set(lst), key=lst.count)
        total = len(lst)
        label_count = lst.sum(0)
        maxlabelcount = max(label_count)
        maxlabelidx = np.argmax(label_count)

        ans = maxlabelidx
        # print('label_count', label_count.shape, label_count)
        # print('map_cluster_onehotlabel', ori, maxlabelcount, total)
        if maxlabelcount / total < thres:
            ans = -1
        return ans

    for pred_i in range(n_clusters + 1):
        elem = noisy_labels[pred == pred_i]
        # mapped_label = most_common(elem)
        mapped_label = most_common(elem, pred_i, thres=thres)
        ans_map[pred_i] = mapped_label
        count_map[mapped_label] = count_map.get(mapped_label, 0) + 1
        # print(ans_map)
    for k in ans_map:
        if count_map.get(ans_map[k], 0) < min_walk_cluster_count:
            ans_map[k] = -1
    return ans_map

def map_cluster_2sets(pred, noisy_labels1, noisy_labels2, n_labels, n_clusters, thres, min_walk_cluster_count=2) -> dict:
    """要把聚类的标签根据簇中最多数标签占簇的大小决定是否保留"""
    # n = len(noisy_labels)
    # pred_max_idx = max(pred)
    def most_common(lst, ori, thres = 0.8):

        # most_common = max(set(lst), key=lst.count)
        total = len(lst)
        label_count = lst.sum(0)
        maxlabelcount = max(label_count)
        maxlabelidx = np.argmax(label_count)

        ans = maxlabelidx
        # print('label_count', label_count.shape, label_count)
        # print('map_cluster_onehotlabel', ori, maxlabelcount, total)
        if maxlabelcount / total < thres:
            ans = -1
        return int(ans)

    ans_map = {}
    count_map = {}

    num1 = len(noisy_labels1)
    pred1 = pred[:num1]
    print(len(pred1), num1)
    pred2 = pred[num1:]

    for pred_i in range(n_clusters + 1):
        ans_item = {}
        elem1 = noisy_labels1[pred1 == pred_i]
        counts1 = elem1.sum(0)
        m_label1 = int(np.argmax(counts1))
        m_label_count1 = int(max(counts1))

        elem2 = noisy_labels2[pred2 == pred_i]
        counts2 = elem2.sum(0)
        m_label2 = int(np.argmax(counts2))
        m_label_count2 = int(max(counts2))

        ans_item['counts1'] =counts1.tolist()
        ans_item['all_counts1'] =len(elem1)

        ans_item['m_label1'] =m_label1
        ans_item['m_label_count1'] =m_label_count1

        ans_item['counts2'] = counts2.tolist()
        ans_item['all_counts2'] =len(elem2)
        ans_item['m_label2'] = m_label2
        ans_item['m_label_count2'] = m_label_count2

        # mapped_label = most_common(elem)
        mapped_label = most_common(elem2, m_label2, thres=thres)
        ans_item['m_label2_mapped'] = mapped_label
        # ans_map[pred_i] = mapped_label
        count_map[mapped_label] = count_map.get(mapped_label, 0) + 1

        ans_map[pred_i] = ans_item

    for k in ans_map:
            ans_map[k]['m_label2_mapped_count'] = count_map.get(ans_map[k]['m_label2_mapped'], 0)

    return ans_map

def map_cluster_2sets2(pred, noisy_labels1, noisy_labels2, n_labels, n_clusters, thres, min_walk_cluster_count=2) -> dict:
    """要把聚类的标签根据簇中最多数标签占簇的大小决定是否保留"""
    # n = len(noisy_labels)
    # pred_max_idx = max(pred)
    def most_common(lst, ori, thres = 0.8):

        # most_common = max(set(lst), key=lst.count)
        total = len(lst)
        label_count = lst.sum(0)
        maxlabelcount = max(label_count)
        maxlabelidx = np.argmax(label_count)

        ans = maxlabelidx
        # print('label_count', label_count.shape, label_count)
        # print('map_cluster_onehotlabel', ori, maxlabelcount, total)
        if maxlabelcount / total < thres:
            ans = -1
        return int(ans)

    ans_map = {}
    count_map = {}

    num1 = len(noisy_labels1)
    pred1 = pred[:num1]
    print(len(pred1), num1)
    pred2 = pred[num1:]

    for pred_i in range(n_clusters + 1):
        ans_item = {}
        elem1 = noisy_labels1[pred1 == pred_i]
        counts1 = elem1.sum(0)
        m_label1 = int(np.argmax(counts1))
        m_label_count1 = int(max(counts1))

        elem2 = noisy_labels2[pred2 == pred_i]
        counts2 = elem2.sum(0)
        m_label2 = int(np.argmax(counts2))
        m_label_count2 = int(max(counts2))

        ans_item['counts1'] =counts1.tolist()
        ans_item['all_counts1'] =len(elem1)

        ans_item['m_label1'] =m_label1
        ans_item['m_label_count1'] =m_label_count1

        ans_item['counts2'] = counts2.tolist()
        ans_item['all_counts2'] =len(elem2)
        ans_item['m_label2'] = m_label2
        ans_item['m_label_count2'] = m_label_count2

        # mapped_label = most_common(elem)
        mapped_label = most_common(elem2, m_label2, thres=thres)
        ans_item['m_label2_mapped'] = mapped_label
        # ans_map[pred_i] = mapped_label
        count_map[mapped_label] = count_map.get(mapped_label, 0) + 1

        ans_map[pred_i] = ans_item

    for k in ans_map:
        ans_map[k]['m_label2_mapped_count'] = count_map.get(ans_map[k]['m_label2_mapped'], 0)

    for k in ans_map:
        ans_map[k]['final_label'] = -1
        if ans_map[k]['all_counts2'] == 0:#no train
            print(f"Cluster {k} no train.")
            continue
        if ans_map[k]['m_label2_mapped_count'] <= 1:#single
            print(f"Cluster {k} single.")
            continue
        if ans_map[k]['m_label2_mapped'] == -1:#mixed
            print(f"Cluster {k} has mixed labels.")
            continue
        if ans_map[k]['all_counts1'] == 0:#notest
            print(f"Cluster {k} no valid label.")
            continue
        if ans_map[k]['m_label2'] == ans_map[k]['m_label1']: #no change
            print(f"Cluster {k} not changed.")
            continue
        print(f"Cluster {k} mapped from {ans_map[k]['m_label2']} to {ans_map[k]['m_label1']}.")
        ans_map[k]['final_label'] = ans_map[k]['m_label1']

    return ans_map

def map_cluster_2sets3(pred, centers, noisy_labels1, noisy_labels2, n_labels, n_clusters, thres, min_walk_cluster_count=2) -> dict:
    """要把聚类的标签根据簇中最多数标签占簇的大小决定是否保留"""
    # n = len(noisy_labels)
    # pred_max_idx = max(pred)
    def most_common(lst, ori, thres = 0.8):

        # most_common = max(set(lst), key=lst.count)
        total = len(lst)
        label_count = lst.sum(0)
        maxlabelcount = max(label_count)
        maxlabelidx = np.argmax(label_count)

        ans = maxlabelidx
        # print('label_count', label_count.shape, label_count)
        # print('map_cluster_onehotlabel', ori, maxlabelcount, total)
        if maxlabelcount / total < thres:
            ans = -1
        return int(ans)

    ans_map = {}
    count_map = {}

    num1 = len(noisy_labels1)
    pred1 = pred[:num1]
    print(len(pred1), num1)
    pred2 = pred[num1:]

    for pred_i in range(n_clusters + 1):
        ans_item = {}
        elem1 = noisy_labels1[pred1 == pred_i]
        counts1 = elem1.sum(0)
        m_label1 = int(np.argmax(counts1))
        m_label_count1 = int(max(counts1))

        elem2 = noisy_labels2[pred2 == pred_i]
        counts2 = elem2.sum(0)
        m_label2 = int(np.argmax(counts2))
        m_label_count2 = int(max(counts2))

        ans_item['counts1'] =counts1.tolist()
        ans_item['all_counts1'] =len(elem1)

        ans_item['m_label1'] =m_label1
        ans_item['m_label_count1'] =m_label_count1

        ans_item['counts2'] = counts2.tolist()
        ans_item['all_counts2'] =len(elem2)
        ans_item['m_label2'] = m_label2
        ans_item['m_label_count2'] = m_label_count2

        # mapped_label = most_common(elem)
        mapped_label = most_common(elem2, m_label2, thres=thres)

        ans_item['m_label2_mapped'] = mapped_label
        # ans_map[pred_i] = mapped_label
        count_map[mapped_label] = count_map.get(mapped_label, 0) + 1

        def max_array(labels):
            count_map = {}
            for i in labels:
                k = tuple(i)
                count_map[k] = count_map.get(k, 0) + 1

            # print(count_map)

            kk = max(count_map.keys(), key=lambda k: count_map[k])
            return np.array(kk), count_map[kk]

        most_freq_train_label, freq = max_array(elem2)
        counts3 = counts2-most_freq_train_label*freq
        m_label3 = int(np.argmax(counts3))
        m_label_count3 = int(max(counts3))
        ans_item['counts3'] = counts3.tolist()
        ans_item['all_counts3'] =len(elem2)#实在是跟2相同
        ans_item['m_label3'] = m_label3
        ans_item['m_label_count3'] = m_label_count3

        ans_map[pred_i] = ans_item

    for k in ans_map:
        ans_map[k]['m_label2_mapped_count'] = count_map.get(ans_map[k]['m_label2_mapped'], 0)

    for k in ans_map:
        ans_map[k]['final_label'] = -1
        if ans_map[k]['all_counts2'] == 0:#no train
            print(f"Cluster {k} no train.")
            continue
        if ans_map[k]['m_label2_mapped_count'] <= 1:#single
            print(f"Cluster {k} single.")
            continue
        if ans_map[k]['m_label2_mapped'] == -1:#mixed
            print(f"Cluster {k} has mixed labels.")
            continue
        if ans_map[k]['all_counts1'] == 0:#notest
            print(f"Cluster {k} no valid label.")
            continue
        if ans_map[k]['m_label2'] == ans_map[k]['m_label1']: #no change
            print(f"Cluster {k} not changed.")
            continue
        # print(f"Cluster {k} mapped from {ans_map[k]['m_label2']} to {ans_map[k]['m_label1']}.")
        ans_map[k]['final_label'] = ans_map[k]['m_label1']

    distance_map = {}
    for k in range(len(noisy_labels1[0])):
        distance_map[k] = []

    for k in ans_map:
        ans_map[k]['distances'] = []
        for k2 in ans_map:
            if k == k2:
                ans_map[k]['distances'].append(float(0))
            elif ans_map[k]['m_label2'] != ans_map[k2]['m_label2']:
                ans_map[k]['distances'].append(float(0))
            else:
                c1 = centers[k]
                c2 = centers[k2]
                distance = np.linalg.norm(c1-c2)
                ans_map[k]['distances'].append(float(distance))

        distances = ans_map[k]['distances']
        # distances.sort(reverse=True)

        labels2 = [(i, ans_map[k]['counts2'][i]) for i in range(len(ans_map[k]['counts2'])) if
                   ans_map[k]['counts2'][i] != 0]
        labels2.sort(key=lambda k: k[1], reverse=True)

        distances_nn = [d for d in distances if d != 0]
        # print(distances_nn)
        if len(distances_nn) > 0:
            avg_distance = np.average(distances_nn)
            distance_map[ans_map[k]['m_label2']].append(avg_distance)
        else:
            avg_distance = -1
        ans_map[k]['isolate_distance'] = avg_distance
        if avg_distance > 7.0 and len(labels2) > 1 and ans_map[k]['m_label2_mapped'] != -1:
            ans_map[k]['final_label2'] = labels2[1][0]
        else:
            ans_map[k]['final_label2'] = -1
        # print(labels2)
    for k in ans_map:
        m_label2 = ans_map[k]['m_label2']
        distances_label2 = distance_map.get(m_label2, [])
        if len(distances_label2) > 2:
            ans_map[k]['most_isolate'] = bool(ans_map[k]['isolate_distance'] == max(distances_label2))
        else:
            ans_map[k]['most_isolate'] = False
    return ans_map, distance_map

def feature_cluster(features, labels, logger, aux_clusters = 0, transfer=True, is_train=True, verbose=0):
    labels = one_hot_to_index(labels)  # 然后转换为index，训练集会损失一些信息
    n_class = max(labels)+aux_clusters
    print(f'Clustering {"train" if is_train else "test"} data features...')
    cluster = KMeans(n_clusters=n_class + 1, n_init=n_class + 1, verbose=verbose).fit(features)
    # cluster = KMeans(n_clusters=n_class + 1, n_init=n_class + 1).fit(features)
    pred = cluster.labels_
    # labels = normalize(labels)  # 首先normalize
    if transfer:
        map_ = map_cluster_label(labels, pred)
        # print(map_)
        pred = [map_[pred_i] for pred_i in pred]
        mutual_info = sklearn.metrics.adjusted_mutual_info_score(labels, pred)
        print(f'Mutual info with ground-truth is: {mutual_info} on {"train" if is_train else "test"}.')
        logger.info(f'Mutual info with ground-truth is: {mutual_info} on {"train" if is_train else "test"}.')
    return np.array(pred)

def reconsider_cluster(features, onehot_labels, logger, aux_clusters = 0, is_train=True, thres=0.8, verbose=0):
    # labels = one_hot_to_index(labels)  # 然后转换为index，训练集会损失一些信息
    n_class_ori = len(onehot_labels[0])
    n_class = n_class_ori + aux_clusters
    print(f'Clustering {"train" if is_train else "test"} data features...')
    cluster = KMeans(n_clusters=n_class + 1, n_init=n_class+1, verbose=verbose).fit(features)
    pred = cluster.labels_
    # labels = normalize(labels)  # 首先normalize
    # if transfer:
    map_ = map_cluster_onehotlabel(onehot_labels, pred, n_class_ori, n_class, thres)
    # print(map_)
    pred_mapped = [map_[pred_i] for pred_i in pred]

    return np.array(pred), np.array(pred_mapped)

def reconsider_cluster2(features, onehot_labels1, onehot_labels2,logger, aux_clusters = 0, is_train=True, thres=0.8, verbose=0):
    '''onehot_labels1, 前一半样本的标签，通常为test。onehot_labels2， 后一半样本的标签，通常为train'''
    # labels = one_hot_to_index(labels)  # 然后转换为index，训练集会损失一些信息
    n_class_ori = len(onehot_labels1[0])
    n_class = n_class_ori + aux_clusters
    print(f'Clustering {"train" if is_train else "test"} data features...')
    cluster = KMeans(n_clusters=n_class + 1, n_init=n_class+1, verbose=verbose).fit(features)
    pred = cluster.labels_
    # labels = normalize(labels)  # 首先normalize
    # if transfer:
    map_ = map_cluster_2sets(pred, onehot_labels1, onehot_labels2, n_class_ori, n_class, thres)
    # print(map_)
    # pred_mapped = [map_[pred_i] for pred_i in pred]

    return np.array(pred), map_

def reconsider_cluster3(features, onehot_labels1, onehot_labels2,logger, n_runs = 20, aux_clusters = 0, batch_size=0,
                        is_train=True, thres=0.8, verbose=0):
    '''onehot_labels1, 前一半样本的标签，通常为test。onehot_labels2， 后一半样本的标签，通常为train'''

    n_class_ori = len(onehot_labels1[0])
    n_class = n_class_ori + aux_clusters
    print(f'Clustering {"train" if is_train else "test"} data features into {n_class} classes...')

    # cluster = KMeans(n_clusters=n_class + 1, n_init=n_class + 1, max_iter=800, verbose=verbose).fit(
    #     features)
    if batch_size > 0:
        cluster = MiniBatchKMeans(n_clusters=n_class, n_init=n_runs, max_iter=300, batch_size=batch_size, verbose=verbose).fit(
            features)
    else:
        cluster = KMeans(n_clusters=n_class, n_init=n_runs, max_iter=300,
                                  verbose=verbose).fit(features)

    pred = cluster.labels_
    centers = cluster.cluster_centers_
    map_, map_d = map_cluster_2sets3(pred, centers, onehot_labels1, onehot_labels2, n_class_ori, n_class, thres)

    return np.array(pred), map_, map_d

def reconsider_cluster4(features, onehot_labels1, onehot_labels2,logger, n_runs = 20, aux_clusters = 0, batch_size=0,
                        is_train=True, thres=0.8, verbose=0):
    '''onehot_labels1, 前一半样本的标签，通常为test。onehot_labels2， 后一半样本的标签，通常为train'''

    n_class_ori = len(onehot_labels1[0])
    n_class = n_class_ori + aux_clusters
    print(f'Clustering {"train" if is_train else "test"} data features into {n_class} classes...')

    # cluster = KMeans(n_clusters=n_class + 1, n_init=n_class + 1, max_iter=800, verbose=verbose).fit(
    #     features)
    if batch_size > 0:
        cluster = MiniBatchKMeans(n_clusters=n_class, n_init=n_runs, max_iter=300, batch_size=batch_size, verbose=verbose).fit(
            features)
    else:
        cluster = KMeans(n_clusters=n_class, n_init=n_runs, max_iter=300,
                                  verbose=verbose).fit(features)

    pred = cluster.labels_
    centers = cluster.cluster_centers_
    # map_, map_d = map_cluster_2sets3(pred, centers, onehot_labels1, onehot_labels2, n_class_ori, n_class, thres)

    return np.array(pred)


def reconsider_cluster_need_walk(features, onehot_labels, logger, aux_clusters = 0, is_train=True, thres=0.8, verbose=0):
    # labels = one_hot_to_index(labels)  # 然后转换为index，训练集会损失一些信息
    n_class_ori = len(onehot_labels[0])
    n_class = n_class_ori + aux_clusters
    print(f'Clustering {"train" if is_train else "test"} data features...')
    cluster = KMeans(n_clusters=n_class + 1, n_init=n_class+1, verbose=verbose).fit(features)
    pred = cluster.labels_
    # labels = normalize(labels)  # 首先normalize
    # if transfer:
    map_ = map_cluster_need_walk(onehot_labels, pred, n_class_ori, n_class, thres)
    # print(map_)
    pred_mapped = [map_[pred_i] for pred_i in pred]

    return np.array(pred), np.array(pred_mapped), map_