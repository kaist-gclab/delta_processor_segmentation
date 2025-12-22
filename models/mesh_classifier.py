import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()


##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True) # changed
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class), pred_class, label_class

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])

    def calculate_f1(self, pred_labels, gt_labels):
        all_mesh_f1s = []

        for pred, gt in zip(pred_labels, gt_labels):
            # If batched: (B, E) -> iterate meshes
            if pred.dim() == 2:
                preds = [pred[i].view(-1) for i in range(pred.size(0))]
                gts   = [gt[i].view(-1)   for i in range(gt.size(0))]
            else:
                preds = [pred.view(-1)]
                gts   = [gt.view(-1)]

            for p, g in zip(preds, gts):
                classes = torch.unique(g)
                f1s = []

                for c in classes:
                    tp = ((p == c) & (g == c)).sum().float()
                    fp = ((p == c) & (g != c)).sum().float()
                    fn = ((p != c) & (g == c)).sum().float()

                    denom_prec = tp + fp
                    denom_recl = tp + fn
                    if denom_prec == 0 or denom_recl == 0:
                        continue

                    precision = tp / (denom_prec + 1e-8)
                    recall    = tp / (denom_recl + 1e-8)
                    f1        = 2 * precision * recall / (precision + recall + 1e-8)
                    f1s.append(f1)

                if len(f1s) == 0:
                    all_mesh_f1s.append(torch.tensor(0.0, device=p.device))
                else:
                    all_mesh_f1s.append(torch.stack(f1s).mean())

        return torch.stack(all_mesh_f1s).mean()