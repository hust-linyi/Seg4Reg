import math
import numpy as np

def compute_mse(pred, gt):
    mse_distance = np.mean(np.square(pred - gt))
    return mse_distance


def compute_score(truth, observe):
    alpha0 = 21/50.0
    alpha1 = 21/20.0
    alpha2 = 21*3/10
    assert len(truth) == len(observe)
    score = 0
    nb0 = 0
    nb1 = 0
    nb2 = 0
    for i in range(len(truth)):
        try:
            assert len(truth[i])==6
            assert len(observe[i])==5
        except:
            print(len(truth[i]))
            print(len(observe[i]))
            print(i)
            print("ERROR: the format of the file is wrong")
            return 1000000000000
        cur_truth = [int(truth[i][1]), float(truth[i][2]), float(truth[i][3]), float(truth[i][4]), float(truth[i][5])]
        cur_observe = [int(observe[i][0]), float(observe[i][1]), float(observe[i][2]), float(observe[i][3]), float(observe[i][4])]
        if cur_truth[0] == 0:
            score += alpha0 * f0(cur_truth, cur_observe)
            nb0 += 1
        elif cur_truth[0] == 1:
            score += alpha1 * f1(cur_truth, cur_observe)
            nb1 += 1
        elif cur_truth[0] == 2:
            score += alpha2 * f2(cur_truth, cur_observe)
            nb2 += 1
        else :
            score+=1
    return score/ (nb0 * alpha0 + nb1 * alpha1 + nb2 * alpha2)


# function to evaluate the score for image with zero spot
def f0(t, o):
    # print o[0]
    if o[0] == 0:
        return 0
    return 1


# function to evaluate the score for image with one spot
def f1(t, o):
    if o[0] == 0:
        return 1
    if o[0] == 1:
        return d(o[1],o[2],t[1],t[2])
    return ((1 + min(d(o[1], o[2], t[1], t[2]), d(o[3],o[4],t[1],t[2])))/ 2.0)


# function to evaluate the score for image with two spot
def f2(t,o):
    if o[0]==0:
        return 1
    if o[0] == 1:
        return ((1 + min(d(o[1], o[2], t[1], t[2]), d(o[1],o[2],t[3],t[4])))/ 2.0)
    return (min( d(o[1], o[2], t[1], t[2]) + d(o[3], o[4], t[3], t[4]) , d(o[1], o[2], t[3], t[4]) + d(o[3], o[4], t[1], t[2]) )/ 2.0)


# function of distance, evalue the quality of the detection
def d(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1) * (x2-x1) + (y2-y1)*(y2-y1))
    if dist < 2:
        return ( (dist)/2.0)
    return 1

