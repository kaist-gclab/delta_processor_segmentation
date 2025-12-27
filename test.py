from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    pred_labels = []
    gt_labels = []
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, pred_label, gt_label = model.test()
        pred_labels.append(pred_label)
        gt_labels.append(gt_label)
        writer.update_counter(ncorrect, nexamples)
    f1 = model.calculate_f1(pred_labels, gt_labels)
    writer.print_acc(epoch, writer.acc)
    writer.print_f1(epoch, f1)
    
    return writer.acc, f1


if __name__ == '__main__':
    run_test()