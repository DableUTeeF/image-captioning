from mmengine.config import Config
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output
    return hook

class CachedDINOEncoder(nn.Module):
    def __init__(self, 
                 feature_dir='/project/lt200060-capgen/palm/imagecaptioning/features', 
                 max_per_img=50, 
                 regen=False,
                 image_dir='/project/lt200060-capgen/palm/flickr8k/Images'
                 ):
        super().__init__()
        self.feature_dir = feature_dir
        self.max_per_img = max_per_img
        if regen:
            config = Config.fromfile('/home/nhongcha/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py')
            model = init_detector(
                config,
                '/project/lt200060-capgen/palm/parasites/cp/pretrained/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth',
                device='cuda'
            )
            model.hooks = {}
            model.bbox_head.reg_branches[-1][3].register_forward_hook(get_activation(f'reg_features', model.hooks))
            model.bbox_head.cls_branches[-1].register_forward_hook(get_activation(f'cls_features', model.hooks))
            config.test_dataloader.dataset = config.train_dataloader.dataset
            transform = Compose(get_test_pipeline_cfg(config))
            with torch.no_grad():
                for file in os.listdir(image_dir):
                    data_ = dict(img_path=os.path.join(image_dir, file), img_id=0)
                    data_ = transform(data_)
                    data_['inputs'] = [data_['inputs']]
                    data_['data_samples'] = [data_['data_samples']]
                    model.test_step(data_)
                    reg = model.hooks[f'reg_features']
                    cls_score = model.hooks[f'cls_features']
                    torch.save(
                        {'reg': reg, 'cls': cls_score},
                        os.path.join(feature_dir, file+'.pth')
                    )

    def forward(self, images):
        outputs = []
        for image in images:
            # print(self.feature_dir, image+'.pth')
            data = torch.load(os.path.join(self.feature_dir, os.path.basename(image)+'.pth'))
            reg = data['reg']
            cls_score = data['cls']
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(self.max_per_img)
            output = torch.gather(reg, 1, bbox_index.unsqueeze(-1).expand(-1, -1, 256))
            outputs.append(output)
        return torch.cat(outputs, 0)


class DINOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = Config.fromfile('/home/nhongcha/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py')
        model = init_detector(
            config,
            '/project/lt200060-capgen/palm/parasites/cp/pretrained/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth',
            device='cuda'
        )
        self.max_per_img = 50

        model.hooks = {}
        model.bbox_head.reg_branches[-1][3].register_forward_hook(get_activation(f'reg_features', model.hooks))
        model.bbox_head.cls_branches[-1].register_forward_hook(get_activation(f'cls_features', model.hooks))
        self.model = model
        config.test_dataloader.dataset = config.train_dataloader.dataset
        self.transform = Compose(get_test_pipeline_cfg(config))

    def forward(self, images):
        data = {
            'inputs': [],
            'data_samples': []
        }
        for i in range(len(images)):
            data_ = dict(img_path=images[i], img_id=i)
            data_ = self.transform(data_)
            data['inputs'].append(data_['inputs'])
            data['data_samples'].append(data_['data_samples'])

        with torch.no_grad():
            self.model.test_step(data)
        
        reg = self.model.hooks[f'reg_features']
        cls_score = self.model.hooks[f'cls_features']
        # print(f'reg: {reg.size()} - cls_score: {cls_score.size()}')
        scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(self.max_per_img)
        # print(f'scores: {scores.size()} - bbox_index: {bbox_index.size()}')
        output = torch.gather(reg, 1, bbox_index.unsqueeze(-1).expand(-1, -1, 256))
        return output
        # return reg[bbox_index.unsqueeze(-1).cpu()].cuda()
