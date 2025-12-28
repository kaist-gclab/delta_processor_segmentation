import os
import time

# try:
#     from tensorboardX import SummaryWriter
# except ImportError as error:
    # print('tensorboard X not installed, visualizing wont be available')
# SummaryWriter = None

class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0
        #
        # if opt.is_train and not opt.no_vis and SummaryWriter is not None:
        #     self.display = SummaryWriter(comment=opt.name)
        # else:
        # self.display = None

    def start_logs(self):
        """_summary_: creates test / train log files"""
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """_summary_: prints train loss to terminal / file"""
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                  % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_acc(self, epoch, acc):
        """_summary_: prints test accuracy to terminal / file"""
        message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
            .format(epoch, acc * 100)
        if epoch == -1:
            message = 'TEST ACC: [{:.5} %]' \
            .format(acc * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_f1(self, epoch, f1):
        """_summary_: prints test f1 to terminal"""
        message = 'epoch: {}, TEST F1: [{:.5} %]\n' \
            .format(epoch, f1 * 100)
        if epoch == -1:
            message = 'TEST F1: [{:.5} %]' \
            .format(f1 * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def reset_counter(self):
        """_summary: counts # of correct examples"""
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
