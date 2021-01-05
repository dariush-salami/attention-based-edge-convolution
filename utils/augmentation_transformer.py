from utils import provider
import numpy as np
import torch


class AugmentationTransformer(object):
    def __init__(self, normal_channel, batch_size):
        self.normal_channel = normal_channel
        self.batch_size = batch_size

    def __call__(self, data):
        batch_data = data.pos.cpu().detach().numpy().reshape((self.batch_size, -1, 3))
        seq_number = data.x[:, 0].cpu().detach().numpy().reshape((self.batch_size, -1))
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        data_after_augmentation = rotated_data
        # data_after_augmentation, shuffled_indices = provider.shuffle_points(rotated_data)
        # seq_number = seq_number[:, shuffled_indices]
        data.pos = torch.from_numpy(data_after_augmentation.reshape(-1, 3)).to(data.pos.device)
        data.x = torch.from_numpy(
            np.insert(data_after_augmentation.reshape(-1, 3), 0, seq_number.reshape(-1), axis=1)
        ).to(data.x.device)
        return data


class ModelNetAugmentationTransformer(object):
    def __init__(self, normal_channel, batch_size):
        self.normal_channel = normal_channel
        self.batch_size = batch_size

    def __call__(self, data):
        batch_data = data.pos.cpu().detach().numpy().reshape((self.batch_size, -1, 3))
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        data_after_augmentation, shuffled_indices = provider.shuffle_points(rotated_data)
        data.pos = torch.from_numpy(data_after_augmentation.reshape(-1, 3)).to(data.pos.device)
        return data


def random_trajectory(image_size, seq_length, step_length):
    canvas_size = image_size - 28

    # Initial position uniform random inside the box.
    y = np.random.rand()
    x = np.random.rand()

    # Choose a random velocity.
    theta = np.random.rand() * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros(seq_length)
    start_x = np.zeros(seq_length)
    for i in range(seq_length):
        # Take a step along velocity.
        y += v_y * step_length
        x += v_x * step_length

        # Bounce off edges.
        if x <= 0:
            x = 0
            v_x = -v_x
        if x >= 1.0:
            x = 1.0
            v_x = -v_x
        if y <= 0:
            y = 0
            v_y = -v_y
        if y >= 1.0:
            y = 1.0
            v_y = -v_y
        start_y[i] = y
        start_x[i] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x


class MMNISTTransformer(object):
    def __init__(self, batch_size, seq_length=20, num_digits=1,
                 image_size=64, step_length=0.1, num_points=128, pixel_threshold=16):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_digits = num_digits
        self.image_size = image_size
        self.num_points = num_points
        self.step_length = step_length
        self.pixel_threshold = pixel_threshold

    def __call__(self, data):
        batch_data = data.x.cpu().detach().numpy().reshape((self.batch_size, 28, 28))
        x_data = []
        y_data = []
        for (digit_index, digit_image) in enumerate(batch_data):
            #print('Converting digit {}/{}'.format(digit_index + 1, len(batch_data)))
            fail = True
            fetch_random_item = False
            while fail:
                video = np.zeros([self.seq_length, self.image_size, self.image_size])
                cloud_sequence = []
                for i in range(self.num_digits):
                    ty, tx = random_trajectory(self.image_size, self.seq_length, self.step_length)
                    if not fetch_random_item:
                        digit = digit_image
                    else:
                        digit = batch_data[np.random.randint(self.batch_size)]
                    for j in range(self.seq_length):
                        top = ty[j]
                        left = tx[j]
                        bottom = top + 28
                        right = left + 28
                        video[j, top:bottom, left:right] += digit

                for i in range(self.seq_length):
                    image = video[i]
                    cloud = np.column_stack(np.where(image >= self.pixel_threshold))
                    if cloud.shape[0] < self.num_points:
                        fail = True
                        fetch_random_item = True
                        #print('Failed to convert digit {}/{}'.format(digit_index + 1, len(batch_data)))
                        #print('Will try with a random item in the batch!')
                        break
                    else:
                        fail = False
                    random_selection = np.random.choice(cloud.shape[0], size=self.num_points, replace=False)
                    cloud = cloud[random_selection]
                    cloud_sequence.append(cloud)

            cloud_sequence_2d = np.stack(cloud_sequence, axis=0)
            cloud_sequence_3d = np.concatenate(
                (cloud_sequence_2d, np.zeros((self.seq_length, self.num_points, 1), dtype=cloud_sequence_2d.dtype)), 2)
            x_point_cloud_with_seq_num = None
            y_point_cloud_with_seq_num = None
            for (frame_index, frame) in enumerate(cloud_sequence_3d):
                current_frame_with_seq_num = np.hstack((np.array([frame_index + 1] * self.num_points).reshape(-1, 1),
                                                        frame))

                # Should go to labels
                if frame_index + 1 > self.seq_length // 2:
                    if y_point_cloud_with_seq_num is None:
                        y_point_cloud_with_seq_num = current_frame_with_seq_num
                    else:
                        y_point_cloud_with_seq_num = np.concatenate(
                            (y_point_cloud_with_seq_num, current_frame_with_seq_num), 0)
                else:
                    if x_point_cloud_with_seq_num is None:
                        x_point_cloud_with_seq_num = current_frame_with_seq_num
                    else:
                        x_point_cloud_with_seq_num = np.concatenate(
                            (x_point_cloud_with_seq_num, current_frame_with_seq_num), 0)
            x_data.append(x_point_cloud_with_seq_num)
            y_data.append(y_point_cloud_with_seq_num)
        x_data = torch.tensor(np.array(x_data).reshape(-1, 4)).to(data.x.device)
        y_data = torch.tensor(np.array(y_data).reshape(-1, 4)).to(data.x.device)
        batch = torch.tensor(
            np.array(list(range(32))).reshape(-1, 1).repeat((self.seq_length // 2) * self.num_points)).to(data.x.device)
        data.x = x_data
        data.y = y_data
        data.pos = x_data[:, 1:4]
        data.batch = batch
        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
