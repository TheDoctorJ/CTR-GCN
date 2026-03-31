# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os.path as osp
import os
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import pickle
import logging

from torch.distributed.rpc import new_method


class SkeletonProcessor:

    save_path = './'
    data_path = './'
    stat_path = ''
    skes_name_file = ''
    save_data_pkl = ''
    frames_drop_pkl = ''
    skes_path = ''

    def __init__(self):
        self.frames_drop_logger = None

    def get_raw_bodies_data(self, skes_path, ske_name, frames_drop_skes, frames_drop_logger):
        """
        Get raw bodies data from a skeleton sequence.

        Each body's data is a dict that contains the following keys:
          - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
          - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
          - interval: a list which stores the frame indices of this body.
          - motion: motion amount (only for the sequence with 2 or more bodyIDs).

        Return:
          a dict for a skeleton sequence with 3 key-value pairs:
            - name: the skeleton filename.
            - data: a dict which stores raw data of each body.
            - num_frames: the number of valid frames.
        """
        if int(ske_name[1:4]) >= 18:
            skes_path = osp.join(self.data_path, '../nturgbd_raw/nturgb+d_skeletons120/')
        ske_file = osp.join(skes_path, ske_name + '.skeleton')
        assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
        # Read all data from .skeleton file into a list (in string format)
        # print('Reading data from %s' % skes_path+ske_file[-29:])
        with open(ske_file, 'r') as fr:
            str_data = fr.readlines()

        num_frames = int(str_data[0].strip('\r\n'))
        frames_drop = []
        bodies_data = dict()
        valid_frames = -1  # 0-based index
        current_line = 1

        for f in range(num_frames):
            num_bodies = int(str_data[current_line].strip('\r\n'))
            current_line += 1

            if num_bodies == 0:  # no data in this frame, drop it
                frames_drop.append(f)  # 0-based index
                continue

            valid_frames += 1
            joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
            colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

            for b in range(num_bodies):
                bodyID = str_data[current_line].strip('\r\n').split()[0]
                current_line += 1
                num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
                current_line += 1

                for j in range(num_joints):
                    temp_str = str_data[current_line].strip('\r\n').split()
                    joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                    colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                    current_line += 1

                if bodyID not in bodies_data:  # Add a new body's data
                    body_data = dict()
                    body_data['joints'] = joints[b]  # ndarray: (25, 3)
                    body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                    body_data['interval'] = [valid_frames]  # the index of the first frame
                else:  # Update an already existed body's data
                    body_data = bodies_data[bodyID]
                    # Stack each body's data of each frame along the frame order
                    body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                    body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                    pre_frame_idx = body_data['interval'][-1]
                    body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index

                bodies_data[bodyID] = body_data  # Update bodies_data

        num_frames_drop = len(frames_drop)
        assert num_frames_drop < num_frames, \
            'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
        if num_frames_drop > 0:
            frames_drop_skes[ske_name] = np.array(frames_drop, dtype=int)
            frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                        frames_drop))

        # Calculate motion (only for the sequence with 2 or more bodyIDs)
        if len(bodies_data) > 1:
            for body_data in bodies_data.values():
                body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

        return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


    def batch_process(self, skes_batch, indices, batch_num):
        raw_skes_data = []
        frames_cnt = np.zeros(skes_batch.size)
        frames_drop_skes = dict()

        for (idx, skes_name) in enumerate(skes_batch):
            bodies_data = self.get_raw_bodies_data(self.skes_path, skes_name, frames_drop_skes, self.frames_drop_logger)
            raw_skes_data.append(bodies_data)
            frames_cnt[idx] = bodies_data['num_frames']

        batch_file_name = "{}.pkl".format(batch_num)
        with open(osp.join(self.save_path, 'raw_data', 'raw_skes_batches', batch_file_name), 'wb') as fw:
            pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
        with open(osp.join(self.save_path, 'raw_data', 'frames_drop_skes_batches', batch_file_name), 'wb') as fw:
            pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)
        with open(osp.join(self.save_path, 'raw_data', 'frame_batches', str(batch_num) + '.txt'), 'wb') as fw:
            np.savetxt(fw, frames_cnt, fmt='%d')

        return batch_num, skes_batch.size


    def get_raw_skes_data(self, num_workers=192, num_files=None):
        # # save_path = './data'
        # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
        # stat_path = osp.join(save_path, 'statistics')
        #
        # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
        # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')
        # frames_drop_pkl = osp.join(save_path, 'frames_drop_skes.pkl')
        #
        # frames_drop_logger = logging.getLogger('frames_drop')
        # frames_drop_logger.setLevel(logging.INFO)
        # frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
        # frames_drop_skes = dict()

        skes_name = np.loadtxt(self.skes_name_file, dtype=str)

        if not num_files:
            num_files = skes_name.size
        print('Found %d available skeleton files.' % num_files)

        # 192 threads per core
        # Maybe we can use hd5py and some multithreading
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Batch size is num of skes_name/num_of_threads
            batch_size = int(np.ceil(num_files / num_workers))
            batch_num = 0
            threads = []
            for idx in range(0, num_files, batch_size):
                batch_num = batch_num + 1
                batch_end = min(idx + batch_size, num_files)
                skes_batch = skes_name[idx:batch_end]
                indices = np.arange(idx, batch_end)
                new_thread = executor.submit(self.batch_process, skes_batch, indices, batch_num)
                threads.append(new_thread)
            for threadWork in threads:
                batch_num, count = threadWork.result()
                print(f'Batch {batch_num} processed: {count} skeletons')
        final = self.serialize_pickles()
        return final

    def serialize_pickles(self):
        final = dict()

        frames_cnt = []
        print(os.getcwd())
        folder = os.path.join(self.save_path, 'raw_data', 'frame_batches')
        files = os.listdir(folder)
        files.sort(key=lambda x: int(x.split('.')[0]) )

        for filename in files:
            # make sure not to parse the placeholder file
            if not filename.endswith('.txt'): pass
            with open(osp.join(folder, filename), 'r') as fr:
                frames_cnt.extend(np.atleast_1d(np.loadtxt(fr, dtype=int)))
            os.remove(osp.join(folder, filename))

        np.savetxt(osp.join(self.save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

        folder = os.path.join(self.save_path, 'raw_data', 'frames_drop_skes_batches')
        files = os.listdir(folder)
        files.sort(key=lambda x: int(x.split('.')[0]) )
        frames_drop_skes_pickle_files = dict()
        for filename in files:
            if not filename.endswith('.pkl'): pass
            with open(osp.join(folder, filename), 'rb') as fr:
                loaded_pickle = pickle.load(fr)
                frames_drop_skes_pickle_files.update(loaded_pickle)
            os.remove(osp.join(folder, filename))
        pickle.dump(frames_drop_skes_pickle_files, open(osp.join(self.save_path, 'raw_data', 'frames_drop_skes.pkl'), 'wb'))

        raw_skes_data_pickle_files = []
        folder = os.path.join(self.save_path, 'raw_data', 'raw_skes_batches')
        files = os.listdir(folder)
        files.sort(key=lambda x: int(x.split('.')[0]) )
        for filename in files:
            if not filename.endswith('.pkl'): pass
            with open(osp.join(folder, filename), 'rb') as fr:
                loaded_pickle = pickle.load(fr)
                raw_skes_data_pickle_files.extend(loaded_pickle)
            os.remove(osp.join(folder, filename))
        pickle.dump(raw_skes_data_pickle_files, open(osp.join(self.save_path, 'raw_data', 'raw_skes_data.pkl'), 'wb'))

        final['frames_cnt'] = frames_cnt
        final['frames_drop_skes_pickle_files'] = frames_drop_skes_pickle_files
        final['raw_skes_data_pickle_files'] = raw_skes_data_pickle_files


    def setup_paths(self, save_path, data_path):
        self.save_path = save_path
        self.data_path = data_path
        self.stat_path = osp.join(self.data_path, 'statistics')
        self.skes_name_file = osp.join(self.stat_path, 'skes_available_name.txt')
        self.skes_path = osp.join(self.data_path, '../nturgbd_raw/nturgb+d_skeletons')

        if not osp.exists(osp.join(self.save_path, 'raw_data')):
            os.makedirs(osp.join(self.save_path, 'raw_data'))

        self.save_data_pkl = osp.join(self.save_path, 'raw_data', 'raw_skes_data.pkl')
        self.frames_drop_pkl = osp.join(self.save_path, 'raw_data', 'frames_drop_skes.pkl')

        self.frames_drop_logger = logging.getLogger('frames_drop')
        self.frames_drop_logger.setLevel(logging.INFO)
        self.frames_drop_logger.addHandler(logging.FileHandler(osp.join(self.save_path, 'raw_data', 'frames_drop.log')))

        self.setup_directories()

    def setup_directories(self):
        os.makedirs(osp.join(self.save_path, 'raw_data', 'raw_skes_batches'), exist_ok=True)
        os.makedirs(osp.join(self.save_path, 'raw_data', 'frames_drop_skes_batches'), exist_ok=True)
        os.makedirs(osp.join(self.save_path, 'raw_data', 'frame_batches'), exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./')
    parser.parse_args()

    save_path = parser.parse_args().save_path
    data_path = parser.parse_args().data_path

    skelly = SkeletonProcessor()

    skelly.setup_paths(save_path, data_path)

    skelly.get_raw_skes_data()
