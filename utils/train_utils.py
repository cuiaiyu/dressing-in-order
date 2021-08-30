import math
from datasets import create_dataset, create_visual_ds
import torch

def torch_transform(img, flow_field):
        #if False:
        #    self.extractor = BlockExtractor(kernel_size=4)
        #    warp = self.extractor(img, flow_field)
        #    return warp
        #else:
        [b,_,h,w] = img.size()
        flow_field = torch.nn.functional.interpolate(flow_field, (h,w), mode='bilinear')
        source_copy = img
        x = torch.arange(w).view(1, -1).expand(h, -1).float()
        y = torch.arange(h).view(-1, 1).expand(-1, w).float()
        x = 2*x/(w-1)-1
        y = 2*y/(h-1)-1
        grid = torch.stack([x,y], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        flow_x = (2*flow_field[:,0,:,:]/(w-1)).view(b,1,h,w)
        flow_y = (2*flow_field[:,1,:,:]/(h-1)).view(b,1,h,w)
        flow = torch.cat((flow_x,flow_y), 1)

        grid = (grid+flow).permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source_copy, grid)
        return  warp

def get_progressive_training_policy(opt):
    BATCH_SIZE = opt.batch_size
    CROP_SIZE = opt.crop_size[0]
    progress_steps = {}
    print("---progressive!---")
    N_steps = int(math.log(opt.crop_size[0] // 64, 2))
    epoch_jump = opt.lr_update_unit #opt.n_epochs // (N_steps + 3)
    assert epoch_jump > 0
    for i in range(N_steps + 1):
        factor = int(2 ** (N_steps - i))
        ep = int((i * epoch_jump) + 1)
        progress_steps[ep] = (min(opt.max_batch_size,  BATCH_SIZE *  factor), 64 * (2 ** i), 2)
        
    progress_steps[(N_steps + 1) * epoch_jump + 1] = (BATCH_SIZE, CROP_SIZE, 1)

    # print scheudle
    for ep in progress_steps:
        print(ep, "bs=%d, crop_size=%d, lr_factor=%d" % progress_steps[ep])
    return progress_steps

def progressive_adjust(model, opt, bs, cs, coe, square=True):
    if square or cs < 127: #255:
        opt.crop_size = (cs, cs)
    else:
        opt.crop_size = (cs, max(1,int(cs*1.0/256*176)))
    opt.batch_size = bs
    dataset = create_dataset(opt)
    visual_ds = create_visual_ds(opt)
    for name in model.loss_coe:
        model.loss_coe[name] = getattr(opt, "loss_coe_%s"%name) * coe
        print(name, model.loss_coe[name])
    return model, dataset, visual_ds