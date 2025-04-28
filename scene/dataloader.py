from torch.utils.data import Dataset, DataLoader
from utils.camera_utils import cameraList_from_camInfos
from scene import Scene

class CameraDataset(Dataset):
    def __init__(self, scene):
        # self.cam_infos = cam_infos
        # self.resolution_scale = resolution_scale
        # self.args = args
        # Camera list load
        # self.camera_list = cameraList_from_camInfos(self.cam_infos, self.resolution_scale, self.args)
        # self.prompt_ids = None
        # self.source_prompt = args.source_prompt

        # scene = Scene(args, gaussians)
        self.camera_list = scene.getTrainCameras().copy()
    
    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        camera = self.camera_list[idx]
        
        return {
            'idx': idx, # data index
            'gt_image': camera.gt_image,  # 이미지
            # 'warping_candidates': camera.warping_candidates,  # warping candidate
            # 'world_view_flatten': camera.world_view_flatten, # 월드 좌표계
            # 'depth': camera.depth,  # depth map: npy
            # 'mask': camera.mask        # SAM mask (optional)
            # 'prompt_ids': self.prompt_ids,
            # 'camera_view_flatten': camera.camera_view_flatten  # 카메라 좌표계
            # 'R': camera.R,             # 회전 행렬
            # 'T': camera.T,             # 변환 벡터
            # 'FoVx': camera.FoVx,       # FOV x축
            # 'FoVy': camera.FoVy,       # FOV y축
        }

# # Dataloader 생성
# def camera_dataloader(args, batch_size=4, shuffle=False, num_workers=4):
#     dataset = CameraDataset(args)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
#     return dataloader

# # Example usage
# if __name__ == '__main__':
#     cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, mask_folder)
#     dataloader = create_dataloader(cam_infos, resolution_scale=1.0, args=args, batch_size=8)

#     for batch in dataloader:
#         images = batch['image']
#         Rs = batch['R']
#         Ts = batch['T']
#         print(images.shape, Rs.shape, Ts.shape)
