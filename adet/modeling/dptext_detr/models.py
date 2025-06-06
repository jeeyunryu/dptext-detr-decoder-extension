import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from adet.layers.deformable_transformer import DeformableTransformer_Det
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from .utils import MLP
from .poolers import ROIPooler
from detectron2.structures import Boxes
from adet.utils.misc import box_cxcywh_to_xyxy
from .decoder_dig import TFDecoder
from .modeling_finetune import PatchEmbed



class DPText_DETR(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone

        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.max_len = 25
        self.num_classes = 1  # only text
        self.sigmoid_offset = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.epqm = cfg.MODEL.TRANSFORMER.EPQM
        self.efsa = cfg.MODEL.TRANSFORMER.EFSA
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model) # c초기 생성해둠
        # self.char_point_embed = nn.Embedding(self.max_len, self.d_model) # c초기 생성해둠

        self.box_pooler = ROIPooler(
            output_size=(32, 128),
            scales = (0.125, 0.0625, 0.03125, 0.015625),
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )  

        self.decoder_rec = TFDecoder()

        self.patch_embed = PatchEmbed(
            img_size=(32,128), patch_size=4, in_chans=256, embed_dim=512)

        
        self.transformer = DeformableTransformer_Det(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            num_proposals=self.num_proposals,
            num_ctrl_points=self.num_ctrl_points,
            epqm=self.epqm,
            efsa=self.efsa
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model,kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            ])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList([self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList([self.ctrl_point_coord for _ in range(num_pred)])
        if self.epqm:
            self.transformer.decoder.ctrl_point_coord = self.ctrl_point_coord
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)

    def forward(self, samples: NestedTensor, tgts=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # for sam in samples:
        #     if torch.isnan(sam).any():
        #         print(f'kkk: {sam}')
        
        
        features, pos = self.backbone(samples)
        

        if self.num_feature_levels == 1:
            raise NotImplementedError

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            
            src, mask = feat.decompose()
            
           
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)


        # n_pts, embed_dim --> n_q, n_pts, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        # char_point_embed = self.char_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        # for i in range(len(srcs)):
        #     H = srcs[i].size(2)
        #     W = srcs[i].size(3)



        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory, level_start_index = self.transformer(
        #     srcs, masks, pos, ctrl_point_embed
        # )
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory, level_start_index = self.transformer(
            srcs, masks, pos, ctrl_point_embed
        )


      
        
        
        
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl]) # linear layer
            tmp = self.ctrl_point_coord[lvl](hs[lvl]) # mlp # 이 좌표를 어디에 어떻게 찔을 수 있을까?
            if reference.shape[-1] == 2:
                if self.epqm:
                    tmp += reference
                else:
                    tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                if self.epqm:
                    tmp += reference[..., :2]
                else:
                    tmp += reference[:, :, None, :2]
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_ctrl_points': outputs_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        ## 여기다가 추가
        # _, det_cntrl_pnts, det_gt_texts = det_criterion(out, targets)
        # tgts[0]['boxes']: 0번째 배치 boxes [N, 4]
        # srcs[0]: 0번째 feature map -> [2, 256, 74, 200] [B, N, H, W]
        #  srcs[0][0].shape: 0번째 feature map, 0번째 배치 

        

        if self.training:
            enc_outs = []
            for i in range(len(level_start_index)):
                start = level_start_index[i]
                if i == len(level_start_index) - 1:
                    enc_outs.append(memory[:, start:, :])
                else:
                    end = level_start_index[i+1]
                    enc_outs.append(memory[:, start:end, :])
            
            for lvl, src in enumerate(srcs):
                bs, c, h, w = src.shape
                enc_outs[lvl] = enc_outs[lvl].view(bs, h, w, c).permute(0, 3, 1, 2)

            bs = len(tgts)
            targets = []
            for i in range(bs):
                tgts[i]['boxes'] = box_cxcywh_to_xyxy(tgts[i]['boxes'])
                h = samples[i].size(1)
                w = samples[i].size(2)
                tgts[i]['boxes'][:, ::2] *= w
                tgts[i]['boxes'][:, 1::2] *= h
                targets.append(Boxes(tgts[i]['boxes']))

            # if torch.isnan(srcs).any():
            #     print(f'srcs234 has nan!! : {srcs}')

            
            text_roi = self.box_pooler(enc_outs, targets)

            

            out_enc = self.patch_embed(text_roi) # dim = 512

            concat_texts = torch.cat([tgt['texts'] for tgt in tgts], dim=0)

            tgt_lens = torch.full((len(concat_texts),), -1, dtype=torch.long)

            for i, tgt in enumerate(concat_texts):
                indices = torch.where(tgt == 96)[0]
                if len(indices) > 0:
                    first_index = indices[0]
                    tgt[first_index] = 95
                    tgt_lens[i] = first_index

            dec_output, _ = self.decoder_rec(feat=out_enc, out_enc=out_enc, targets=concat_texts, tgt_lens=tgt_lens, train_mode=self.training)
            

            out['dec_outputs'] = dec_output


            return out, concat_texts, tgt_lens
            # return out
        else:
            return out
    
        # return out
            
        

        

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {'pred_logits': a, 'pred_ctrl_points': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]