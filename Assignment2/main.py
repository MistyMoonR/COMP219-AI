# coding=UTF-8<code>
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras.optimizers import RMSprop, Adam
from sklearn.metrics import roc_curve, auc
import itertools
from keras.models import load_model
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize


def dnn():
    model = keras.Sequential(
        [
            layers.Dense(512, activation='relu', input_shape=(64,)),
            layers.Dropout(0.25),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(10, activation='softmax')
        ]
    )

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    print('DNN model construction')
    print(model.summary())
    # Save model
    model.save('DNN_model.h5')


def cnn():
    model = keras.Sequential(
        [
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]
    )

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    print('DNN model construction')
    print(model.summary())
    # Save model
    model.save('CNN_model.h5')


def sk_gaussian_nb(tra_x, tes_x, tra_y, tes_y):
    # Scikit-learn Gaussian algorithm
    sk_naive_bayes = GaussianNB()
    sk_naive_bayes.fit(tra_x, tra_y)
    pre_y = sk_naive_bayes.predict(tes_x)
    score = sk_naive_bayes.score(tes_x, tes_y)
    return pre_y, score


def my_naive_bayes(tra_x, tes_x, tra_y, tes_y):
    # Naive bayes algorithm
    all_x_nb = np.concatenate((tra_x, tes_x), axis=0)
    all_y_nb = np.concatenate((tra_y, tes_y), axis=0)

    mean = np.zeros(all_x_nb.shape)
    variance = np.zeros(all_x_nb.shape)

    for i in range(10):
        tra_x_c = tra_x[tra_y == i]
        mean[int(i), :] = tra_x_c.mean(axis=0)
        variance[int(i), :] = tra_x_c.var(axis=0)

    x = tes_x
    y = tes_y
    classes = np.unique(all_y_nb)
    cm = np.zeros([len(classes), len(classes)])
    my_d_accuracy = []
    variance = variance + 1.5
    my_t_count = 0
    pre_y = []

    for i in range(x.shape[0]):
        lists = []
        for j in range(len(classes)):
            numerator = np.exp(-((x[i] - mean[j]) ** 2) / (2 * variance[j]))
            denominator = np.sqrt(2 * np.pi * (variance[j]))
            prob_xc = numerator / denominator
            ratio = np.sum(np.log(prob_xc))
            lists.append(ratio)
        pred = lists.index(max(lists))
        pre_y.append(pred)

        if pred == y[i]:
            my_t_count = my_t_count + 1
            cm[int(y[i])][int(y[i])] = cm[int(y[i])][int(y[i])] + 1
        else:
            for k in range(len(classes)):
                if pred == k:
                    cm[int(y[k])][int(y[i])] = cm[int(y[k])][int(y[i])] + 1

    for f in classes:
        check = x[y == f]
        a = (cm[int(f)][int(f)]) / check.shape[0]
        my_d_accuracy.append(a)

    score = my_t_count / (tes_y.shape[0])
    pre_y = np.array(pre_y).reshape(tes_y.shape)

    return pre_y, score


def draw_confusion_matrix(tes_y, pre_y, title):
    # Confusion matrix
    cm = np.zeros([10, 10], dtype=int)
    pred = pre_y.tolist()
    y = tes_y.tolist()

    for i in range(tes_y.shape[0]):
        if pred[i] == y[i]:
            cm[int(y[i])][int(y[i])] = cm[int(y[i])][int(y[i])] + 1
        else:
            for j in range(10):
                if pred[i] == j:
                    cm[int(y[j])][int(y[i])] = cm[int(y[j])][int(y[i])] + 1

    # Confusion matrix graph
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.ylim(-0.5, 9.5)
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    plt.gca().invert_yaxis()

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.savefig(title + " CM.png")
    plt.show()


def draw_roc(pre_y, tes_y, title):
    # ROC and AUC
    tes_y_array = label_binarize(tes_y, classes=[i for i in range(10)])
    pre_y_array = label_binarize(pre_y, classes=[i for i in range(10)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(tes_y_array[:, i], pre_y_array[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(tes_y_array.ravel(), pre_y_array.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 10

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ROC curves
    lw = 2
    plt.figure()
    plt.figure(figsize=(6.5, 5.5))

    colors = itertools.cycle(['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                              '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5'])

    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], '-', color=color, lw=lw,
                 label='class: ' + str(i) + ' accuracy = %.2f' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(title) + ' ROC curves')
    plt.legend(loc="lower right")
    plt.savefig(str(title) + ' ROC.png')
    plt.show()


def dnn_result(data, graph=True):
    tra_x, tes_x, tra_y, tes_y = data
    tes_y_c = tes_y
    # one-hot
    tra_x = tra_x.astype('float32')
    tes_x = tes_x.astype('float32')
    tra_x /= 128
    tes_x /= 128
    tra_y = keras.utils.to_categorical(tra_y, 10)
    tes_y = keras.utils.to_categorical(tes_y, 10)
    # load model
    dnn_model = load_model('DNN_model.h5')
    # model fit
    dnn_model.fit(tra_x, tra_y,
                  batch_size=128,
                  epochs=15,
                  verbose=0,
                  validation_data=(tes_x, tes_y),
                  )

    pre_y = np.argmax(dnn_model.predict(tes_x), axis=1)
    score = dnn_model.evaluate(tes_x, tes_y, verbose=0)
    print('DNN test loss: %.2f ' % score[0])
    print('DNN test accuracy: %.2f ' % score[1])
    print(' ')
    # empty variable
    del dnn_model
    keras.backend.clear_session()

    if graph:
        draw_confusion_matrix(tes_y_c, pre_y, 'DNN Confusion matrix')
        draw_roc(pre_y, tes_y_c, "DNN")


def cnn_result(data, graph=True):
    # one-hot
    tra_x, tes_x, tra_y, tes_y = data
    tes_y_c = tes_y
    tra_x = tra_x.reshape(tra_x.shape[0], 8, 8, 1)
    tes_x = tes_x.reshape(tes_x.shape[0], 8, 8, 1)

    tra_x /= 128
    tes_x /= 128

    tra_y = keras.utils.to_categorical(tra_y, 10)
    tes_y = keras.utils.to_categorical(tes_y, 10)
    # load model
    cnn_model = load_model('CNN_model.h5')
    # model fit
    cnn_model.fit(tra_x, tra_y,
                  verbose=0,
                  batch_size=16,
                  epochs=15,
                  validation_data=(tes_x, tes_y))

    pre_y = np.argmax(cnn_model.predict(tes_x), axis=1)
    score = cnn_model.evaluate(tes_x, tes_y, verbose=0)
    print('CNN test loss: %.2f ' % score[0])
    print('CNN test accuracy: %.2f ' % score[1])
    print(' ')
    # empty variable
    del cnn_model
    keras.backend.clear_session()

    if graph:
        draw_confusion_matrix(tes_y_c, pre_y, 'DNN Confusion matrix')
        draw_roc(pre_y, tes_y_c, "DNN")


def sk_nb_result(data, graph=True):
    tra_x, tes_x, tra_y, tes_y = data
    sk_model = sk_gaussian_nb(tra_x, tes_x, tra_y, tes_y)
    pre_y = sk_model[0]
    score = sk_model[1]
    print('GaussianNB test accuracy: %.2f' % score)
    print('')

    if graph:
        draw_confusion_matrix(tes_y, pre_y, 'GaussianNB Confusion matrix')
        draw_roc(pre_y, tes_y, 'GaussianNB')


def my_nb_result(data, graph=True):
    tra_x, tes_x, tra_y, tes_y = data
    sk_model = my_naive_bayes(tra_x, tes_x, tra_y, tes_y)
    pre_y = sk_model[0]
    score = sk_model[1]
    print('My Naive Bayes test accuracy: %.2f' % score)
    print('')

    if graph:
        draw_confusion_matrix(tes_y, pre_y, 'My Naive Bayes Confusion matrix')
        draw_roc(pre_y, tes_y, 'My Naive Bayes')


def cross_validation():
    # Cross validation
    data_hd_cv = datasets.load_digits()
    all_x_cv = data_hd_cv.data
    all_y_cv = data_hd_cv.target

    x = all_x_cv
    y = all_y_cv
    len_x = x.shape[0]
    len_y = y.shape[0]

    # Split 5 sub-samples
    x0 = x[0:int(len_x / 5)]
    x1 = x[int(len_x / 5): int(len_x / 5 * 2)]
    x2 = x[int(len_x / 5 * 2): int(len_x / 5 * 3)]
    x3 = x[int(len_x / 5 * 3): int(len_x / 5 * 4)]
    x4 = x[int(len_x / 5 * 4): len_x]

    y0 = y[0:int(len_y / 5)]
    y1 = y[int(len_y / 5): int(len_y / 5 * 2)]
    y2 = y[int(len_y / 5 * 2): int(len_y / 5 * 3)]
    y3 = y[int(len_y / 5 * 3): int(len_y / 5 * 4)]
    y4 = y[int(len_y / 5 * 4): len_y]

    # Merge dataset
    data_cv_s1 = np.vstack((x1, x2, x3, x4)), x0, np.concatenate((y1, y2, y3, y4)), y0
    data_cv_s2 = np.vstack((x2, x3, x4, x0)), x1, np.concatenate((y2, y3, y4, y0)), y1
    data_cv_s3 = np.vstack((x1, x3, x4, x0)), x2, np.concatenate((y1, y3, y4, y0)), y2
    data_cv_s4 = np.vstack((x1, x2, x4, x0)), x3, np.concatenate((y1, y2, y4, y0)), y3
    data_cv_s5 = np.vstack((x0, x1, x2, x3)), x4, np.concatenate((y0, y1, y2, y3)), y4

    # loop call variable
    for i in range(1, 5):
        print(str(i) + ' times cross validation' + ': S' + str(i) + ' is test dataset')
        print(' ')
        tmp_data = 'data_cv_s' + '%d' % i
        tmp_data_si = eval(tmp_data)
        dnn_result(tmp_data_si, graph=False)
        cnn_result(tmp_data_si, graph=False)
        sk_nb_result(tmp_data_si, graph=False)
        my_nb_result(tmp_data_si, graph=False)
        del tmp_data, tmp_data_si

    # An example of the above loop call variable
    print('5 times cross validation' + ': S5 is test dataset')
    print(' ')
    dnn_result(data_cv_s5, graph=False)
    cnn_result(data_cv_s5, graph=False)
    sk_nb_result(data_cv_s5, graph=False)
    my_nb_result(data_cv_s5, graph=False)

    print('End cross-validation of 5 sub-samples\n')


def inp():
    print('1. F1 Load two model and see model construction\n'
          '2. F2 Model evaluation\n'
          '3. End')
    while True:
        try:
            input_num = int(input('Enter number from 1 to 3: '))
            if input_num < 1 or input_num > 3:
                raise ValueError  # this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid integer. The number must be in the range of 1 - 3")
    if input_num == 1:
        print('Show model construction and save model')
        dnn()
        cnn()
        inp()

    elif input_num == 2:
        print('Start cross-validation of 5 sub-samples')
        cross_validation()
        print('Output the results of the four algorithms')

        data_hd_n = datasets.load_digits()
        all_x_n = data_hd_n.data
        all_y_n = data_hd_n.target
        data = train_test_split(all_x_n, all_y_n, test_size=1 / 4, random_state=None)
        dnn_result(data, graph=True)
        cnn_result(data, graph=True)
        sk_nb_result(data, graph=True)
        my_nb_result(data, graph=True)

        print('Discussion on the discovery\n')
        print(
            '\tBy comparing the results for Assessment 1 and Assessment 2. In the same dataset. Traditional machine\n '
            '\tlearning takes the least time and does not require high performance but need to require more human\n'
            '\tresources to achieve ideal accuracy. Deep neural network only needs a simple setting to obtain a very\n'
            '\tideal accuracy rate. However, the number of parameters in the neural network is very large,\n'
            '\tand many optimizers, such as RMSProp, SGD, ADAM, etc. are needed to find the local optimal solution.\n'
            '\tThis algorithm is the most time-consuming and most dependent on computer performance.\n'
            '\tThe appropriate algorithm can be selected according to the requirements of the scene.\n'
            '\tTraditional machine learning is suitable for scenes with high real-time requirements and low data volume.'
            '\n\tNeural network is suitable for the scene with very large amount of data and high accuracy.\n')
        inp()
    else:
        print('The end :)')


if __name__ == '__main__':
    inp()
