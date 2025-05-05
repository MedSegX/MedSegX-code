import math
import os
join = os.path.join
from typing import Type, List

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything.modeling.common import MLPBlock
from data.datainfo import *


def get_index(map_idx):
    idx = map_idx
    for i in range(max(map_idx.keys())+1):
        idx[i] = map_idx.get(i, -1) + 1
    idx = dict(sorted(idx.items(), key=lambda x: x[0]))
    return torch.LongTensor(list(idx.values()))

class MoEAdaptMLPBlock(nn.Module):
    def __init__(
        self,
        mlp: MLPBlock,
        expert_num: int,
        embedding_num: int,
        embedding_dim: int,
        adapter_bn: int = 64,
        adapter_act: Type[nn.Module] = nn.GELU,
        adapter_dropout: float = 0.1,
        adapter_scalar: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.expert_num = expert_num
        
        dim = mlp.embedding_dim
        self.adapter_input_embed = nn.Linear(dim, embedding_dim)
        self.adapter_gate = nn.Linear(embedding_dim * embedding_num, expert_num)
        
        self.adapter_down = nn.ModuleList([
            nn.Linear(dim, adapter_bn) for _ in range(expert_num)
        ])
        self.adapter_up = nn.ModuleList([
            nn.Linear(adapter_bn, dim) for _ in range(expert_num)
        ])
        self.adapter_act = adapter_act()
        self.adapter_dropout = adapter_dropout
        if adapter_scalar is None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = adapter_scalar
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for i in range(self.expert_num):
            nn.init.kaiming_uniform_(self.adapter_down[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down[i].bias)
            nn.init.zeros_(self.adapter_up[i].weight)
            nn.init.zeros_(self.adapter_up[i].bias)

    def forward(self, x: torch.Tensor,
                modal: torch.Tensor, organ: torch.Tensor) -> torch.Tensor:
        adpt = []
        for i in range(self.expert_num):
            adpt.append(self.adapter_up[i](nn.functional.dropout(
                self.adapter_act(self.adapter_down[i](x)),
                p=self.adapter_dropout, training=self.training)))
        adpt = torch.stack(adpt, dim=1)
        
        input = x.mean(dim=1).mean(dim=1)
        input = self.adapter_input_embed(input)
        organ_0, organ_1, organ_2, organ_3, task = organ
        gate = torch.cat([organ_0, organ_1, organ_2, organ_3, task, modal, input], dim=-1)
        gate = self.adapter_gate(gate)
        gate = torch.softmax(gate, dim=1)
        
        adpt = torch.einsum('bemnc,be->bemnc', adpt, gate).sum(dim=1)
        return self.mlp(x) + adpt * self.adapter_scale, gate

class MedSegX(nn.Module):
    """Applies Tree MoE Adapter to a SAM's image encoder.

    Args:
        sam: segment anything model, see 'segment_anything' dir
        bottleneck_dim: bottleneck dimension of adapter
        embedding_dim: modal and organ embedding dimension
        expert_num: number of experts in MoE adapter
        pos: which layer to apply adapter
    """

    def __init__(self, sam: Sam, bottleneck_dim: int, embedding_dim: int, 
                 expert_num: int, pos: list = None):
        super(MedSegX, self).__init__()

        assert bottleneck_dim > 0
        assert embedding_dim > 0
        assert expert_num > 0
        
        # assign Adapter layer position (all layers by default)
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(sam.image_encoder.blocks)))
        
        # freeze SAM image and prompt encoder
        # if sam.image_encoder.img_size != 1024:
        #     for n, p in sam.image_encoder.named_parameters():
        #         if 'pos' not in n:
        #             p.requires_grad = False
        # else:
        #     for param in sam.image_encoder.parameters():
        #         param.requires_grad = False
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        # modality and organ embedding index
        modal_index = get_index(modal_map_idx)
        organ_index_1 = get_index(organ_level_1_map_idx)
        organ_index_2 = get_index(organ_level_2_map_idx)
        organ_index_3 = get_index(organ_level_3_map_idx)
        
        sam.image_encoder.register_buffer('modal_index', modal_index)
        sam.image_encoder.register_buffer('organ_index_1', organ_index_1)
        sam.image_encoder.register_buffer('organ_index_2', organ_index_2)
        sam.image_encoder.register_buffer('organ_index_3', organ_index_3)
        
        modal_embed = nn.Embedding(modal_index.max()+1, embedding_dim)
        organ_embed_0 = nn.Embedding(1, embedding_dim)
        organ_embed_1 = nn.Embedding(organ_index_1.max()+1, embedding_dim)
        organ_embed_2 = nn.Embedding(organ_index_2.max()+1, embedding_dim)
        organ_embed_3 = nn.Embedding(organ_index_3.max()+1, embedding_dim)
        organ_embed_4 = nn.Embedding(len(task_list), embedding_dim)
        nn.init.zeros_(modal_embed.weight)
        nn.init.zeros_(organ_embed_0.weight)
        nn.init.zeros_(organ_embed_1.weight)
        nn.init.zeros_(organ_embed_2.weight)
        nn.init.zeros_(organ_embed_3.weight)
        nn.init.zeros_(organ_embed_4.weight)
        
        sam.image_encoder.modal_embed = modal_embed
        sam.image_encoder.organ_embed = nn.ModuleList([
            organ_embed_0, organ_embed_1, 
            organ_embed_2, organ_embed_3, 
            organ_embed_4, 
        ])
        
        # apply Adapter to SAM image encoder
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue
            
            # create adapter layers
            blk.mlp = MoEAdaptMLPBlock(
                blk.mlp,
                expert_num=expert_num,
                embedding_num=7,
                embedding_dim=embedding_dim,
                adapter_bn=bottleneck_dim,
            )
        
        self.sam = sam

    def save_parameters(self) -> dict:
        r"""save both adapter and mask decoder parameters.
        """
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        
        # save adapter parameters
        adapter_tensors = {}
        for key, value in state_dict.items():
            if 'adapter' in key:
                adapter_tensors[key] = value
        
        # save image encoder parameters
        image_encoder_tensors = {}
        for key, value in state_dict.items():
            if 'modal_embed' in key or 'organ_embed' in key:
                image_encoder_tensors[key] = value
        
        # save prompt encoder parameters
        prompt_encoder_tensors = {}
        # for key, value in state_dict.items():
        #     if 'prompt_encoder' in key:
        #         prompt_encoder_tensors[key] = value
        
        # save mask decoder parameters
        mask_decoder_tensors = {}
        for key, value in state_dict.items():
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**adapter_tensors, **image_encoder_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        return merged_dict

    def load_parameters(self, state_dict) -> None:
        r"""load both adapter and mask decoder parameters.
        """
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load adapter parameters
        adapter_keys = [k for k in sam_keys if 'adapter' in k]
        adapter_values = [state_dict[k] for k in adapter_keys]
        adapter_new_state_dict = {k: v for k, v in zip(adapter_keys, adapter_values)}
        sam_dict.update(adapter_new_state_dict)
        
        # load image encoder parameters
        image_encoder_keys = [k for k in sam_keys if 'modal_embed' in k or 'organ_embed' in k]
        image_encoder_values = [state_dict[k] for k in image_encoder_keys]
        image_encoder_new_state_dict = {k: v for k, v in zip(image_encoder_keys, image_encoder_values)}
        sam_dict.update(image_encoder_new_state_dict)
        
        # load prompt encoder parameters
        # prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        # prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        # prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        # sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder parameters
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        
        self.sam.load_state_dict(sam_dict)

    def forward(self, data):
        img, box = data['img'], data['box']
        modal, organ = data['modal'], data['organ']
        
        # modal and organ embedding
        B = img.shape[0]
        
        modal_index = self.sam.image_encoder.modal_index[modal]
        modal_embed = self.sam.image_encoder.modal_embed(modal_index)
        # modal_embed = None
        
        organ_1, organ_2, organ_3, organ_4 = organ
        organ_index_0 = torch.zeros(B, dtype=torch.long, device=img.device)
        organ_embed_0 = self.sam.image_encoder.organ_embed[0](organ_index_0)
        
        organ_index_1 = self.sam.image_encoder.organ_index_1[organ_1]
        organ_embed_1 = self.sam.image_encoder.organ_embed[1](organ_index_1)
        
        organ_index_2 = self.sam.image_encoder.organ_index_2[organ_2]
        organ_embed_2 = self.sam.image_encoder.organ_embed[2](organ_index_2)
        
        organ_index_3 = self.sam.image_encoder.organ_index_3[organ_3]
        organ_embed_3 = self.sam.image_encoder.organ_embed[3](organ_index_3)
        
        organ_embed_4 = self.sam.image_encoder.organ_embed[4](organ_4.to(img.device))
        organ_embed = (organ_embed_0, organ_embed_1, organ_embed_2, organ_embed_3, organ_embed_4)
        
        # prompt encoder
        if len(box.shape) == 2:
            box = box[:, None, :]  # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )
        
        # Aadapter image encoder
        input_image = self.sam.preprocess(img) # (B, 3, 1024, 1024)
        image_embedding, expert_activation = self.sam.image_encoder(input_image, modal_embed, organ_embed) # (B, 256, 64, 64)
        
        # predicted masks
        mask_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (B, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
          )
        
        expert_activation = torch.stack(expert_activation, dim=1)
        
        return mask_predictions#, expert_activation
