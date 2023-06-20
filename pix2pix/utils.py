import matplotlib.pyplot as plt

class Plotter():
    def __init__(self):
        # plt.ion()
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        self.epoch_count = 0

    def plot(self, train_loss_values: list, test_loss_values: list, train_acc_values: list = None, test_acc_values: list = None):
        self.epoch_count += 1
        self.ax1.clear()
        self.ax2.clear()      
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_title("Loss and Accuracy")

        self.ax1.plot(range(self.epoch_count), train_loss_values, label="train_loss", color='red')
        self.ax1.plot(range(self.epoch_count), test_loss_values, label="test_loss", color='darkred')
        self.ax1.tick_params(axis='y', colors='red')
        self.ax1.legend(loc='lower left')
        if train_acc_values is not None and test_acc_values is not None:
            self.ax2.plot(range(self.epoch_count), train_acc_values, label="train_acc", color='green')
            self.ax2.plot(range(self.epoch_count), test_acc_values, label="test_acc", color='darkgreen')
            self.ax2.tick_params(axis='y', colors='green')
            self.ax2.legend(loc='upper left')
        plt.draw()
        plt.pause(0.1)
