import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import gradio as gr
from PIL import Image
from net import *
from DealDataset import *

#
class_correct = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def show_wrong_picture(predictions, dataset, class_correct):
    """
    该函数是用来展示部分错误识别的图像
    :param predictions:预测结果
    :param dataset:训练集
    :param class_correct:正确的图像类别
    :return:展示四行四列十六个识别错误的图片
    """
    incorrect_images = []
    incorrect_labels = []
    incorrect_predictions = []

    for i in range(len(predictions)):
        if predictions[i] != dataset[i][1]:  # 检查预测是否正确
            incorrect_images.append(dataset[i][0])  # 添加不正确图像
            incorrect_labels.append(dataset[i][1])  # 添加不正确图像标签
            incorrect_predictions.append(predictions[i])  # 添加不正确的预测

    # 创建子图网格 四行四列用于展示部分识别错误的图像，标注图像标签及识别的错误标签
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(incorrect_images[i].reshape(3, 32, 32), (1, 2, 0)))  # 调整图像形状并转置通道顺序
        ax.set_title(
            f"True: {class_correct[incorrect_labels[i]]}, Predicted: {class_correct[incorrect_predictions[i]]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_correct_picture(predictions, dataset, class_correct):
    """
    该函数是用来展示部分识别正确的图像
    :param predictions:预测结果
    :param dataset:训练集
    :param class_correct:正确的图像类别数组
    :return:展示四行四列十六个识别正确的图片
    """
    correct_images = []
    correct_labels = []
    correct_predictions = []

    for i in range(len(predictions)):
        if predictions[i] == dataset[i][1]:  # 检查预测是否正确
            correct_images.append(dataset[i][0])  # 添加正确图像
            correct_labels.append(dataset[i][1])  # 添加正确图像标签
            correct_predictions.append(predictions[i])  # 添加正确的预测

    # 创建子图网格 四行四列用于展示部分识别正确的图像，标注图像标签及识别标签
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(correct_images[i].reshape(3, 32, 32), (1, 2, 0)))  # 调整图像形状并转置通道顺序
        ax.set_title(f"True: {class_correct[correct_labels[i]]}, Predicted: {class_correct[correct_predictions[i]]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show(loss_hist, loss_hist_val, acc_hist, acc_hist_val):
    """
    该函数是用来展示训练集和预测集的损失值和精度
    """
    legend = ['Train', 'Validation']
    plt.plot(loss_hist)
    plt.plot(loss_hist_val)
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(legend, loc='upper left')
    plt.savefig('result/loss_curve.png')  # 将Loss曲线保存为文件
    plt.show()

    legend = ['Train', 'Validation']
    plt.plot(acc_hist)
    plt.plot(acc_hist_val)
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(legend, loc='upper left')
    plt.savefig('result/accuracy_curve.png')  # 将Accuracy曲线保存为文件
    plt.show()


def unpickle(filepath):
    with open(filepath, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


def train(epoch_num, train_loader, val_loader, class_names, batch_size):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    model = ResNet(ResBlock).to(device)
    # model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    loss_hist, acc_hist = [], []
    loss_hist_val, acc_hist_val = [], []

    min_loss = np.Inf

    for epoch in range(epoch_num):
        running_loss = 0.0
        correct = 0

        # 遍历训练集的每个批次
        for data in train_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练统计数据（准确率和损失）
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # 如果当前epoch的损失小于最小损失，则保存模型
        if running_loss < min_loss:
            min_loss = running_loss
            PATH = './cifar_net.pth'
            torch.save(model.state_dict(), PATH)

        avg_loss = running_loss / 50000  # 计算平均训练损失
        avg_acc = correct / 50000  # 计算平均训练准确率
        loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)
        pre = []
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        # 进行验证
        model.eval()
        with torch.no_grad():
            loss_val = 0.0
            correct_val = 0
            # 遍历验证集的每个批次
            for data in val_loader:
                batch, labels = data
                batch, labels = batch.to(device), labels.to(device)
                outputs = model(batch.float())
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct_tensor = predicted.__eq__(labels.data.view_as(predicted))
                correct = np.squeeze(correct_tensor.cpu().numpy())
                correct_val += (predicted == labels).sum().item()
                loss_val += loss.item()
                pre.append(predicted)
                for i in range(batch_size):
                    temp_label = labels.data[i]
                    class_correct[temp_label] += correct[i].item()
                    class_total[temp_label] += 1
            for i in range(10):
                if class_total[i] > 0:
                    print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        class_names[i], 100 * class_correct[i] / class_total[i],
                        np.sum(class_correct[i]), np.sum(class_total[i])))
                else:
                    print('Test Accuracy of %5s: N/A (no training examples)' % (class_names[i]))
            pre = torch.cat(pre)
            avg_loss_val = loss_val / 10000  # 计算平均验证损失
            avg_acc_val = correct_val / 10000  # 计算平均验证准确率
            loss_hist_val.append(avg_loss_val)
            acc_hist_val.append(avg_acc_val)

        model.train()
        print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f' % (
            epoch + 1, avg_loss, avg_acc, avg_loss_val, avg_acc_val))
    show(loss_hist, loss_hist_val, acc_hist, acc_hist_val)
    print("完成训练")
    print(f"平均训练损失值为：{avg_loss}")
    print(f"平均训练准确率为：{avg_acc}")
    print(f"平均验证损失值为：{avg_loss_val}")
    print(f"平均验证准确率：{avg_acc_val}")

    return avg_loss, avg_acc, avg_loss_val, avg_acc_val, class_names, pre


if __name__ == "__main__":

    data_batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    data = np.empty((0, 3072), dtype=np.uint8)
    label = []
    path_prefix = "datamemory/cifar-10-batches-py/"
    for data_batch in data_batches:
        filepath = path_prefix + data_batch
        my_dict = unpickle(filepath)
        data_ = my_dict[b"data"]
        label_ = [int(element) for element in my_dict[b"labels"]]
        data = np.vstack((data, data_))
        label.extend(label_)

    # get testing data and label
    filepath = "datamemory/cifar-10-batches-py/test_batch"
    my_dict = unpickle(filepath)
    validation_data = my_dict[b"data"]
    validation_label = [int(element) for element in my_dict[b"labels"]]

    # # get class names
    result = unpickle("datamemory/cifar-10-batches-py/batches.meta")
    class_names = [byte_string.decode() for byte_string in result[b"label_names"]]
    print(class_names)
    train_data = torch.from_numpy(data)
    val_data = torch.from_numpy(validation_data)
    train_label = torch.Tensor(label).long()
    val_label = torch.Tensor(validation_label).long()

    # # datasets
    train_dataset = DealDataset(train_data, train_label)
    val_dataset = DealDataset(val_data, val_label)

    batch_size = 64

    tra_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    epoch_num = 50

    avg_loss, avg_acc, avg_loss_val, avg_acc_val, class_names, pre = train(epoch_num, tra_loader, val_loader,
                                                                           class_names, 16)
    prec_vec = pre
    # print(class_names[int(val_dataset[0][1])])
    # print(class_names[prec_vec[0]])

    # def funcTion(index):
    #     filepath1 = "result/accuracy_curve.png"
    #     fig1 = Image.open(filepath1)
    #     filepath2 = "result/loss_curve.png"
    #     fig2 = Image.open(filepath2)
    #     val_class = class_names[int(val_dataset[index][1])]
    #     prec_class = class_names[prec_vec[index]]
    #     show_image = np.transpose(val_dataset[index].reshape(3, 32, 32), (1, 2, 0))
    #     return fig1, fig2, val_class, prec_class, show_image
    #
    # demo = gr.Interface(fn=funcTion,
    #                     inputs=gr.Textbox(label="Input an index(0-9999)"),
    #                     outputs=[gr.outputs.Image(type="pil", label="accuracy_curve"),
    #                              gr.outputs.Image(type="pil", label="loss_curve"),
    #                              gr.Textbox(label="val_class"),
    #                              gr.Textbox(label="prec_class"),
    #                              gr.outputs.Image(type="pil", label="show_image")],
    #                     title="Image Classification with Loss & Accuracy Visualization")

    show_wrong_picture(prec_vec, val_dataset, class_names)
    show_correct_picture(prec_vec, val_dataset, class_names)

