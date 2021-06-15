# coding=UTF-8<code>

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data_hd = datasets.load_digits()
all_x = data_hd.data
all_y = data_hd.target
tra_x, tes_x, tra_y, tes_y = train_test_split(all_x, all_y, test_size=0.7, random_state=0)


def my_naive_bayes(x, y, mean, variance):
    classes = np.unique(all_y)
    confusion_matrix = np.zeros([len(classes), len(classes)])
    my_d_accuracy = []
    variance = variance + 1.5
    my_t_count = 0

    for i in range(x.shape[0]):
        lists = []
        for j in range(len(classes)):
            numerator = np.exp(-((x[i] - mean[j]) ** 2) / (2 * variance[j]))
            denominator = np.sqrt(2 * np.pi * (variance[j]))
            prob_xc = numerator / denominator
            ratio = np.sum(np.log(prob_xc))
            lists.append(ratio)
        pred = lists.index(max(lists))

        if pred == y[i]:
            my_t_count = my_t_count + 1
            confusion_matrix[int(y[i])][int(y[i])] = confusion_matrix[int(y[i])][int(y[i])] + 1
        else:
            for k in range(len(classes)):
                if pred == k:
                    confusion_matrix[int(y[k])][int(y[i])] = confusion_matrix[int(y[k])][int(y[i])] + 1

    for f in classes:
        check = x[y == f]
        a = (confusion_matrix[int(f)][int(f)]) / check.shape[0]
        my_d_accuracy.append(a)

    return my_d_accuracy, my_t_count


def f1():
    print('The number of data entries:', all_x.shape[0])
    print('The number of classes:', data_hd.target_names.size)
    print('The number of train dataset:', tra_x.shape[0])
    print('The number of test dataset:', tes_x.shape[0], '\n')
    print('class \t number \t percentage')
    num = all_y.shape[0]
    for i in range(0, 10):
        count = 0
        for j in all_y:
            if j == all_y[i]:
                count = count + 1
        f = (count / num) * 100
        print(i, '\t', count, '\t\t', '%.2f' % f, '%')

    print('\nThe maximum value for each feature')
    print(np.max(all_x, axis=0))
    print('\nThe minimum value for each feature')
    print(np.min(all_x, axis=0), '\n')
    print('F1 is done\n')


# sklearn.naive_bayes:GaussianNB
def f2():
    tem_tra_x = tra_x
    tem_tra_y = tra_y
    tem_tes_x = tes_x
    sk_nb = GaussianNB()
    sk_nb.fit(tem_tra_x, tem_tra_y)
    sk_nb_tra_y = (sk_nb.predict(tem_tra_x))
    sk_nb_tes_y = (sk_nb.predict(tem_tes_x))
    sk_tra_digit_accuracy = []
    sk_tes_digit_accuracy = []
    sk_tra_true_count = 0
    sk_tes_true_count = 0

    for i in range(0, 10):
        count_a = 0
        count_b = 0
        for j in range(sk_nb_tra_y.shape[0]):
            if sk_nb_tra_y[j] == i:
                count_a = count_a + 1
                if sk_nb_tra_y[j] == tra_y[j]:
                    count_b = count_b + 1
                    sk_tra_true_count = sk_tra_true_count + 1
        sk_tra_digit_accuracy.append(count_b / count_a)

    for i in range(0, 10):
        count_a = 0
        count_b = 0
        for j in range(sk_nb_tes_y.shape[0]):
            if sk_nb_tes_y[j] == i:
                count_a = count_a + 1
                if sk_nb_tes_y[j] == tes_y[j]:
                    count_b = count_b + 1
                    sk_tes_true_count = sk_tes_true_count + 1
        sk_tes_digit_accuracy.append(count_b / count_a)
    print('\nF2 is done\n')

    return sk_tra_digit_accuracy, sk_tra_true_count, sk_tes_digit_accuracy, sk_tes_true_count


def f3():
    classes = np.unique(all_y)
    mean = np.zeros(all_x.shape)
    variance = np.zeros(all_x.shape)

    for i in classes:
        tra_x_c = tra_x[tra_y == i]
        mean[int(i), :] = tra_x_c.mean(axis=0)
        variance[int(i), :] = tra_x_c.var(axis=0)

    my_tra_digit_accuracy, my_tra_true_count = my_naive_bayes(tra_x, tra_y, mean, variance)
    my_tes_digit_accuracy, my_tes_true_count = my_naive_bayes(tes_x, tes_y, mean, variance)
    print('\nF3 is done\n')

    return my_tra_digit_accuracy, my_tra_true_count, my_tes_digit_accuracy, my_tes_true_count


def f4():
    print('F4 Compare the train error and test error of the two arithmetic')
    sk_tra_digit_accuracy, sk_tra_true_count, sk_tes_digit_accuracy, sk_tes_true_count = f2()
    my_tra_digit_accuracy, my_tra_true_count, my_tes_digit_accuracy, my_tes_true_count = f3()

    sk_tra_overall_accuracy = sk_tra_true_count / tra_y.shape[0] * 100
    sk_tes_overall_accuracy = sk_tes_true_count / tes_y.shape[0] * 100
    my_tra_overall_accuracy = my_tra_true_count / tra_y.shape[0] * 100
    my_tes_overall_accuracy = my_tes_true_count / tes_y.shape[0] * 100

    print('My naive bayes training accuracy: ' + '%.2f' % my_tra_overall_accuracy, "%  and the error number is ",
          tra_x.shape[0] - my_tra_true_count)
    print('My naive bayes testing accuracy: ' + '%.2f' % my_tes_overall_accuracy, "%  and the error number is ",
          tes_x.shape[0] - my_tes_true_count)
    print('Scikit-learn naive bayes training accuracy: ' + '%.2f' % sk_tra_overall_accuracy,
          "%  and the error number is ", tra_x.shape[0] - sk_tra_true_count)
    print('Scikit-learn naive bayes testing accuracy: ' + '%.2f' % sk_tes_overall_accuracy,
          "%  and the error number is ", tes_x.shape[0] - sk_tes_true_count)

    print('Please to close the picture window to continue.')
    digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index_test = np.arange(len(digit))
    fig, axs = plt.subplots(1, 2)

    axs[0].bar(index_test - 0.15, height=my_tra_digit_accuracy, width=0.25, color='dodgerblue')
    axs[0].bar(index_test + 0.15, height=my_tes_digit_accuracy, width=0.25, color='darkorange')
    axs[1].bar(index_test - 0.15, height=sk_tra_digit_accuracy, width=0.25, color='dodgerblue')
    axs[1].bar(index_test + 0.15, height=sk_tes_digit_accuracy, width=0.25, color='darkorange')

    axs[0].set_title('My naive bayes accuracy')
    axs[1].set_title('Scikit-learn naive bayes accuracy')

    plt.xlabel('Blue = Train dataset\n  Orange = Test dataset')

    fig.tight_layout()
    plt.show()
    print('\nF4 is done\n')


def f5():
    print('Query the mode')
    while True:
        try:
            i = int(input('Input integer in the range of 0 - 539 to query the train dataset: '))
            if i < 0 or i > 539:
                raise ValueError  # this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid integer. The number must be in the range of 0 - 539")

    print('\nThe image label is ', all_y[int(i)])
    print('The image data is ', all_x[int(i)], '\n')

    img = all_x[int(i)].reshape(8, 8)
    print('Please to close the picture window to continue.')
    plt.imshow(img, cmap="Greys")
    plt.show()

    while True:
        try:
            i = int(input('Input integer in the range of 0 - 1257 to query the test dataset: '))
            if i < 0 or i > 1257:
                raise ValueError  # this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid integer. The number must be in the range of 0 - 1257")
    i = i + 539
    print('\nThe image label is ', all_y[int(i)])
    print('The image data is ', all_x[int(i)], '\n')

    img = all_x[int(i)].reshape(8, 8)
    print('Please to close the picture window to continue.')
    plt.imshow(img, cmap="Greys")
    plt.show()

    print('\nF5 is done\n')


def inp():
    print('1. F1 The details of the dataset\n'
          '2. F2 Implement Scikit-learn navie GaussianNB algorithm\n'
          '3. F3 Implement my navie bayes algorithm\n'
          '4. F4 Compare the train error and test error of the two algorithm\n'
          '5. F5 Query two datasets\n'
          '6. End')
    while True:
        try:
            input_num = int(input('Enter number from 1 to 6: '))
            if input_num < 1 or input_num > 6:
                raise ValueError  # this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid integer. The number must be in the range of 1 - 6")
    if input_num == 1:
        f1()
        inp()
    elif input_num == 2:
        f2()
        inp()
    elif input_num == 3:
        f3()
        inp()
    elif input_num == 4:
        f4()
        inp()
    elif input_num == 5:
        f5()
        inp()
    else:
        print('The end :)')


inp()
