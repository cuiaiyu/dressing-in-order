from .dior_model import *
from utils.util import StoreList, StoreDictKeyPair
from models.networks.block_extractor.block_extractor import BlockExtractor

class FlowModel(DIORModel):
    def __init__(self, opt):
        opt.frozen_flownet = False
        DIORModel.__init__(self, opt)
        self.netE_opt = opt.netE
        self.visual_names = ['from_img', 'to_img', 'fake_B']

    def _init_models(self, opt):
        self.model_names += ["Flow"]
        self.netFlow = networks.define_tool_networks(tool='flownet', load_ckpt_path=opt.flownet_path, gpu_ids=opt.gpu_ids)
        self.extractor = BlockExtractor(kernel_size=1)
         

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.flow_fields, _ = self.netFlow(self.from_img, self.from_kpt, self.to_kpt)
        _, _, H, W = self.flow_fields[-1].size()
        from_img = F.interpolate(self.from_img, (H,W))
        self.fake_B = self.extractor(from_img, self.flow_fields[-1])
        _, _, H, W = self.to_img.size()
        self.fake_B = F.interpolate(self.fake_B, (H,W))
        
    def backward_G(self):
        self.loss_G = 0
        flow_feilds = self.flow_fields
        self.loss_flow_cor = 0.0
        if self.loss_coe['flow_cor'] > 0:
            self.loss_flow_cor = self.Correctness(self.to_img, self.from_img, flow_feilds, [2,3])  * self.loss_coe['flow_cor']
            self.loss_G = self.loss_G + self.loss_flow_cor
        self.loss_flow_reg = 0.0
        if self.loss_coe['flow_reg'] > 0:
            # import pdb; pdb.set_trace()
            self.loss_flow_reg = self.Regularization(flow_feilds) * self.loss_coe['flow_reg']
            self.loss_G = self.loss_G + self.loss_flow_reg

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.loss_G.backward()
        self.optimizer_G.step()             # udpate G's weights
        self.log_loss_update()
    