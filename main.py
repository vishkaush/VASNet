__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

import torch
from torchvision import transforms
import numpy as np
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init

from config import  *
from sys_utils import *
from vsum_tools import  *
from vasnet_model import  *


def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)

    return weights_filename, splits_file


class AONet:

    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose


    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out


    def load_datasets(self, datasets = None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            # dataset_name = base_filename.split('_')[2]
            # print("\tDataset name:", dataset_name)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        self.datasets = datasets_dict
        return datasets_dict


    def load_split_file(self, splits_file):

        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ",splits_file)

        return n_folds


    def select_split(self, split_id):
        print("Selecting split: ",split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)

        split = self.splits[self.split_id]
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return



    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return


    def initialize(self, cuda_device=None):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        self.model = VASNet(apperture=self.hps.apperture)
        self.model.eval()
        self.model.apply(weights_init)
        print(self.model)

        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ",cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()

        return


    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

        return weights_filename, splits_file


    def train(self, output_dir='EX-0'):

        print("Initializing VASNet model and optimizer...")
        self.model.train()

        criterion = nn.MSELoss()

        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr[0], weight_decay=self.hps.l2_req)

        print("Starting training...")

        max_val_fscore_avg = 0
        max_val_fscore_avg_epoch = 0
        max_val_fscore_max = 0
        max_val_fscore_max_epoch = 0
        train_keys = self.train_keys[:]

        lr = self.hps.lr[0]
        for epoch in range(self.hps.epochs_max):

            print("Epoch: {0:6}".format(str(epoch)+"/"+str(self.hps.epochs_max)), end='')
            self.model.train()
            avg_loss = []

            random.shuffle(train_keys)

            for i, key in enumerate(train_keys):
                dataset = self.get_data(key)
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.float().cuda(), target.float().cuda()

                seq_len = seq.shape[1]
                y, _ = self.model(seq,seq_len)
                loss_att = 0

                loss = criterion(y, target)
                # loss2 = y.sum()/seq_len
                loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss.append([float(loss), float(loss_att)])

            # Evaluate test dataset
            epoch_result_file_path = os.path.join(output_dir, "results_temp", str(epoch) + ".h5")
            val_fscore_avg, val_fscore_max, video_scores = self.eval(self.test_keys, results_filename=epoch_result_file_path)
            if max_val_fscore_avg < val_fscore_avg:
                max_val_fscore_avg = val_fscore_avg
                max_val_fscore_avg_epoch = epoch
            if max_val_fscore_max < val_fscore_max:
                max_val_fscore_max = val_fscore_max
                max_val_fscore_max_epoch = epoch
            

            avg_loss = np.array(avg_loss)
            print("   Train loss: {0:.05f}".format(np.mean(avg_loss[:, 0])), end='')
            print('   Test FAvg-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore_avg, max_val_fscore_avg))
            print('   Test FMax-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore_max, max_val_fscore_max))

            if self.verbose:
                video_scores = [["No", "Video", "Budget", "F-score(Avg)", "F-Score(Max)"]] + video_scores
                print_table(video_scores, cell_width=[3,40,8])

            # Save model weights
            #path, filename = os.path.split(self.split_file)
            #base_filename, _ = os.path.splitext(filename)
            #path = os.path.join(output_dir, 'models_temp', base_filename+'_'#+str(self.split_id))
            #os.makedirs(path, exist_ok=True)
            path = os.path.join(output_dir, "models_temp")
            filename = str(epoch)+'_'+str(round(val_fscore_avg*100,3))+'_'+str(round(val_fscore_max*100,3)) + '.pth.tar'
            torch.save(self.model.state_dict(), os.path.join(path, filename))

        return max_val_fscore_avg, max_val_fscore_avg_epoch, max_val_fscore_max, max_val_fscore_max_epoch


    def eval(self, keys, results_filename=None):

        self.model.eval()
        summary = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                data = self.get_data(key)
                # seq = self.dataset[key]['features'][...]
                seq = data['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                y, att_vec = self.model(seq, seq.shape[1])
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()

        f_score_avg, f_score_max, video_scores = self.eval_summary(summary, keys, 
                    results_filename=results_filename, att_vecs=att_vecs)

        return f_score_avg, f_score_max, video_scores


    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, att_vecs=None):

        #eval_metric = 'avg' if metric == 'tvsum' else 'max'

        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')

        fms_avg = []
        fms_max = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probs = machine_summary_activations[key]

            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ",key)

            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]
            video_fps = d['video_fps'][()]

            budgets = [60, 90, 120, 150, 180]
            for budget in budgets:
                machine_summary, selected_segments, segment_scores, frame_scores = generate_summary(probs, cps, num_frames, nfps, positions, budget=budget, fps=video_fps)

                selected_segments = [x + 1 for x in selected_segments]

                fm_avg, fm_max = evaluate_summary(machine_summary, user_summary)
                fms_avg.append(fm_avg)
                fms_max.append(fm_max)

                # Reporting & logging
                video_scores.append([key_idx + 1, key, budget, "{:.1%}".format(fm_avg), "{:.1%}".format(fm_max)])

                if results_filename:
                    #gt = d['gtscore'][...]
                    h5_res.create_dataset(key + '/' + str(budget) + '/score', data=frame_scores)
                    h5_res.create_dataset(key + '/' + str(budget) + '/machine_summary', data=machine_summary)
                    h5_res.create_dataset(
                    key + '/' + str(budget) + '/selected_segments', data=selected_segments)
                    h5_res.create_dataset(
                    key + '/' + str(budget) + '/segment_scores', data=segment_scores)
                    h5_res.create_dataset(key + '/' + str(budget) + '/gtscore', data=d['gtscore'][...])
                    h5_res.create_dataset(key + '/' + str(budget) + '/fm_avg', data=fm_avg)
                    h5_res.create_dataset(key + '/' + str(budget) + '/fm_max', data=fm_max)
                    #h5_res.create_dataset(key + '/picks', data=positions)

                    # video_name = key.split('/')[1]
                    # if 'video_name' in d:
                    #     video_name = d['video_name'][...]
                    # h5_res.create_dataset(key + '/video_name', data=video_name)

                    # if att_vecs is not None:
                    #     h5_res.create_dataset(key + '/att', data=att_vecs[key])

        #mean_fm = np.mean(fms)
        mean_fm_avg = np.mean(fms_avg)
        mean_fm_max = np.mean(fms_max)
        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        #return mean_fm, video_scores
        return mean_fm_avg, mean_fm_max, video_scores


#==============================================================================================



def eval_split(hps, splits_filename, data_dir='test'):

    print("\n")
    ao = AONet(hps)
    ao.initialize()
    ao.load_datasets()
    ao.load_split_file(splits_filename)

    val_fscores = []
    for split_id in range(len(ao.splits)):
        ao.select_split(split_id)
        weights_filename, _ = ao.lookup_weights_file(data_dir)
        print("Loading model:", weights_filename)
        ao.load_model(weights_filename)
        val_fscore, video_scores = ao.eval(ao.test_keys)
        val_fscores.append(val_fscore)

        val_fscore_avg = np.mean(val_fscores)

        if hps.verbose:
            video_scores = [["No.", "Video", "F-score"]] + video_scores
            print_table(video_scores, cell_width=[4,45,5])

        print("Avg F-score: ", val_fscore)
        print("")

    print("Total AVG F-score: ", val_fscore_avg)
    return val_fscore_avg


def train(hps):
    os.makedirs(hps.output_dir, exist_ok=True)
    #os.makedirs(os.path.join(hps.output_dir, 'splits'), exist_ok=True)
    #os.makedirs(os.path.join(hps.output_dir, 'code'), exist_ok=True)
    #os.makedirs(os.path.join(hps.output_dir, 'models'), exist_ok=True)
    #os.system('cp -f splits/*.json  ' + hps.output_dir + '/splits/')
    #os.system('cp *.py ' + hps.output_dir + '/code/')

    # Create a file to collect results from all splits
    f = open(hps.output_dir + '/results.txt', 'wt')

    for split_filename in hps.splits:
        dataset_name, dataset_type, splits = parse_splits_filename(split_filename)

        # For no augmentation use only a dataset corresponding to the split file
        datasets = None
        if dataset_type == '':
            datasets = hps.get_dataset_by_name(dataset_name)

        if datasets is None:
            datasets = hps.datasets

        favg_avg = 0
        fmax_avg = 0
        n_folds = len(splits)
        for split_id in range(n_folds):
            ao = AONet(hps)
            ao.initialize()
            ao.load_datasets(datasets=datasets)
            ao.load_split_file(splits_file=split_filename)
            ao.select_split(split_id=split_id)
            split_id_outpath = os.path.join(hps.output_dir, str(split_id))
            if not os.path.exists(split_id_outpath):
                os.makedirs(split_id_outpath)
            fscore_avg, fscore_avg_epoch, fscore_max, fscore_max_epoch = ao.train(output_dir=split_id_outpath)
            favg_avg += fscore_avg
            fmax_avg += fscore_max

            # Log F-score for this split_id
            f.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore_avg) + ', ' + str(fscore_avg_epoch) + ', ' + str(fscore_max) + ', ' + str(fscore_max_epoch) + '\n')
            f.flush()

            # Save model with the highest F score
            #_, log_file = os.path.split(split_filename)
            #log_dir, _ = os.path.splitext(log_file)
            #log_dir += '_' + str(split_id)
            #log_file_avg = os.path.join(split_id_outpath, str(fscore_avg) + '.tar.pth'
            #log_file_max = os.path.join(hps.output_dir, 'models', log_dir) + '_' + str(fscore_max) + '.tar.pth'

            #os.makedirs(os.path.join(hps.output_dir, 'models', ), exist_ok=True)
            os.system('mv ' + split_id_outpath + '/models_temp/' + str(fscore_avg_epoch) + '_*.pth.tar ' + split_id_outpath)
            if fscore_max_epoch != fscore_avg_epoch:
                os.system('mv ' + split_id_outpath + '/models_temp/' + str(fscore_max_epoch) + '_*.pth.tar ' + split_id_outpath)
            #os.system('rm -rf ' + hps.output_dir + '/models_temp/' + log_dir)
            os.system('cp ' + split_id_outpath + '/results_temp/' + str(fscore_avg_epoch) + '.h5 ' + split_id_outpath + '/favg_best.h5')
            if fscore_max_epoch != fscore_avg_epoch:
                os.system('cp ' + split_id_outpath + '/results_temp/' + str(fscore_max_epoch) + '.h5 ' + split_id_outpath + '/fmax_best.h5')
            print("Split: {0:}   Best FAvg-score: {1:0.5f}   Best FMax-score: {2:0.5f}   ResultAvgBest: {3:}   ResultMaxBest: {4:}".format(str(split_id), fscore_avg, fscore_max, split_id_outpath + '/favg_best.h5', split_id_outpath + '/fmax_best.h5'))

        # Write average F-score for all splits to the results.txt file
        favg_avg /= n_folds
        fmax_avg /= n_folds
        f.write(str(split_id) + ', ' + str('avg') + ', ' + str(favg_avg) + ', ' + str('max') + ', ' + str(fmax_avg) + '\n')
        f.flush()

    f.close()


if __name__ == "__main__":
    print_pkg_versions()

    parser = argparse.ArgumentParser("PyTorch implementation of paper \"Summarizing Videos with Attention\"")
    parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits', type=str, help="Comma separated list of split files.")
    parser.add_argument('-t', '--train', action='store_true', help="Train")
    parser.add_argument('-w', '--window', type=int, help="Window in seconds for local attention, -1 for global attention")
    parser.add_argument('-c', '--cuda-device', type=int, help="GPU id")
    parser.add_argument('-p', '--subsampling', type=int, help="Subsampling used for this dataset")
    parser.add_argument('-e', '--epochs-max', type=int, help="Max no. of epochs")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    parser.add_argument('-o', '--output-dir', type=str, default='data', help="Experiment name")
    #parser.add_argument('-m', '--domain', type=str, default='data', help="Experiment name")
    args = parser.parse_args()

    # MAIN
    #======================
    hps = HParameters()
    hps.load_from_args(args.__dict__)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)

    if hps.train:
        train(hps)
    else:
        results=[['No', 'Split', 'Mean F-score']]
        for i, split_filename in enumerate(hps.splits):
            f_score = eval_split(hps, split_filename, data_dir=hps.output_dir)
            results.append([i+1, split_filename, str(round(f_score * 100.0, 3))+"%"])

        print("\nFinal Results:")
        print_table(results)


    sys.exit(0)

