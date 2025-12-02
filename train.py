import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test
from tqdm import tqdm

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    # best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                tqdm.write('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            tqdm.write('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        tqdm.write('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc, f1 = run_test(epoch)
            ## added part ##
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                tqdm.write('saving the best f1 model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('best')
            ## added part ##
            writer.plot_acc(acc, epoch)
            writer.plot_f1(f1, epoch)
            if epoch - best_epoch > 50 and epoch > 99:
                break          

    writer.close()
