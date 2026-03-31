# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging

class OriginalSkellyProcessor:
    save_path = './'
    data_path = './'
    stat_path = ''
    skes_name_file = ''
    save_data_pkl = ''
    frames_drop_pkl = ''
    skes_path = ''
    frames_drop_skes = dict()

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


    def get_raw_skes_data(self, num_files=None):
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

        raw_skes_data = []
        frames_cnt = np.zeros(num_files, dtype=int)

        for (idx, ske_name) in enumerate(skes_name):
            bodies_data = self.get_raw_bodies_data(self.skes_path, ske_name, self.frames_drop_skes, self.frames_drop_logger)
            raw_skes_data.append(bodies_data)
            frames_cnt[idx] = bodies_data['num_frames']
            if (idx + 1) % 10 == 0:
                print('Processed: %.2f%% (%d / %d)' % \
                      (100.0 * (idx + 1) / num_files, idx + 1, num_files))
            if idx + 1 == num_files:
                break

        with open(self.save_data_pkl, 'wb') as fw:
            pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
        np.savetxt(osp.join(self.save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

        print('Saved raw bodies data into %s' % self.save_data_pkl)
        print('Total frames: %d' % np.sum(frames_cnt))

        with open(self.frames_drop_pkl, 'wb') as fw:
            pickle.dump(self.frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

        final = dict()
        final['frames_cnt'] = frames_cnt
        final['raw_skes_data'] = raw_skes_data
        final['frames_drop_skes'] = self.frames_drop_skes

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



if __name__ == '__main__':
    save_path = './'

    skes_path = '../nturgbd_raw/nturgb+d_skeletons/'
    stat_path = osp.join(save_path, 'statistics')
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log')))
    frames_drop_skes = dict()


    skelly = OriginalSkellyProcessor()
    skelly.setup_paths(save_path, skes_path)


    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)
