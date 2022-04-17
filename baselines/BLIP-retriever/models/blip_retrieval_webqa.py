import pdb
from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import dot, nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint

def dot_score(a, b):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

class BLIP_Retrieval_WebQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 embed_dim = 256,
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 max_question_length= 128,
                 max_caption_length= 128,
                 max_passage_length= 128
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        self.max_question_length = max_question_length
        self.max_caption_length = max_caption_length
        self.max_passage_length = max_passage_length

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 

        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))    # temperature
        
        self.negative_all_rank = negative_all_rank
        
    def image_forward(self, question, positive_image, positive_caption, negative_image, negative_caption, alpha, device):
        # pdb.set_trace()
        question = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation= True, max_length=self.max_question_length)
        positive_caption = self.tokenizer(positive_caption, return_tensors='pt', padding='max_length', truncation= True, max_length=self.max_caption_length)
        negative_caption = self.tokenizer(negative_caption, return_tensors='pt', padding='max_length', truncation= True, max_length=self.max_caption_length)
        
        positive_caption.input_ids[:,0] = self.tokenizer.enc_token_id
        negative_caption.input_ids[:,0] = self.tokenizer.enc_token_id

        question = question.to(device)
        positive_image = positive_image.to(device)
        positive_caption = positive_caption.to(device)
        negative_image = negative_image.to(device)
        negative_caption = negative_caption.to(device)

        positive_image_embeds = self.visual_encoder(positive_image)
        negative_image_embeds = self.visual_encoder(negative_image)
        
        positive_image_atts = torch.ones(positive_image_embeds.size()[:-1],dtype=torch.long).to(positive_image_embeds.device)
        negative_image_atts = torch.ones(negative_image_embeds.size()[:-1],dtype=torch.long).to(negative_image_embeds.device)

        question_output = self.text_encoder(question.input_ids,
                                       attention_mask = question.attention_mask,
                                       return_dict = True,
                                       mode = 'text'
                                      )

        positive_outputs = self.text_encoder(positive_caption.input_ids,
                                       attention_mask = positive_caption.attention_mask,
                                       encoder_hidden_states = positive_image_embeds,
                                       encoder_attention_mask = positive_image_atts,      
                                       return_dict = True,
                                      )
        
        negative_outputs = self.text_encoder(negative_caption.input_ids,
                                       attention_mask = negative_caption.attention_mask,
                                       encoder_hidden_states = negative_image_embeds,
                                       encoder_attention_mask = negative_image_atts,      
                                       return_dict = True,
                                      )

        question_output = F.normalize(self.text_proj(question_output.last_hidden_state[:,0,:]),dim=-1)        
        positive_outputs = F.normalize(self.text_proj(positive_outputs.last_hidden_state[:,0,:]),dim=-1)        
        negative_outputs = F.normalize(self.text_proj(negative_outputs.last_hidden_state[:,0,:]),dim=-1)

        image_outputs = torch.cat([positive_outputs, negative_outputs])
        sim_scores = dot_score(question_output, image_outputs)
        
        with torch.no_grad():
            positive_image_embeds_m = self.visual_encoder_m(positive_image)
            negative_image_embeds_m = self.visual_encoder_m(negative_image)

            positive_image_atts_m = torch.ones(positive_image_embeds_m.size()[:-1],dtype=torch.long).to(positive_image_embeds_m.device)
            negative_image_atts_m = torch.ones(negative_image_embeds_m.size()[:-1],dtype=torch.long).to(negative_image_embeds_m.device)
                                                  
            question_output_m = self.text_encoder_m(question.input_ids, attention_mask = question.attention_mask,                      
                                            return_dict = True, mode = 'text')    

            positive_outputs_m = self.text_encoder_m(positive_caption.input_ids,
                                        attention_mask = positive_caption.attention_mask,
                                        encoder_hidden_states = positive_image_embeds_m,
                                        encoder_attention_mask = positive_image_atts_m,      
                                        return_dict = True,
                                        )
        
            negative_outputs_m = self.text_encoder_m(negative_caption.input_ids,
                                        attention_mask = negative_caption.attention_mask,
                                        encoder_hidden_states = negative_image_embeds_m,
                                        encoder_attention_mask = negative_image_atts_m,      
                                        return_dict = True,
                                        )            
            
            question_output_m = F.normalize(self.text_proj(question_output_m.last_hidden_state[:,0,:]),dim=-1)        
            positive_outputs_m = F.normalize(self.text_proj(positive_outputs_m.last_hidden_state[:,0,:]),dim=-1)        
            negative_outputs_m = F.normalize(self.text_proj(negative_outputs_m.last_hidden_state[:,0,:]),dim=-1)

            image_output_m = torch.cat([positive_outputs_m, negative_outputs_m])

            sim_soft_targets = dot_score(question_output_m, image_output_m)/self.temp

            # sim_hard_targets = torch.zeros_like(sim_soft_targets)
            # sim_hard_targets.fill_diagonal_(1)
            # sim_targets = alpha*F.softmax(sim_soft_targets, dim=1) + (1-alpha)*sim_hard_targets
        # loss_ita = -torch.sum(F.log_softmax(sim_scores, dim=1)*sim_targets,dim=1).mean()
        # loss_ita = F.kl_div(sim_scores, sim_targets)

        sim_hard_targets = torch.tensor(range(len(sim_soft_targets)), dtype=torch.long, device=sim_soft_targets.device)

        loss_itc = F.nll_loss(F.log_softmax(sim_scores, dim=1), sim_hard_targets)
        loss_momentum = F.kl_div(F.log_softmax(sim_scores, dim=1), F.softmax(sim_soft_targets, dim=1))
        loss_ita = alpha*loss_momentum + (1-alpha)*loss_itc
        return loss_ita

    def text_forward(self, question, positive_passage, negative_passage, alpha, device):
        # pdb.set_trace()
        question = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation= True, max_length=self.max_question_length)
        positive_passage = self.tokenizer(positive_passage, return_tensors='pt', padding='max_length', truncation= True, max_length=self.max_passage_length)
        negative_passage = self.tokenizer(negative_passage, return_tensors='pt', padding='max_length', truncation= True, max_length=self.max_passage_length)
        
        question = question.to(device)
        positive_passage = positive_passage.to(device)
        negative_passage = negative_passage.to(device)


        question_output = self.text_encoder(question.input_ids, attention_mask = question.attention_mask,                      
                                        return_dict = True, mode = 'text')
        positive_passage_output = self.text_encoder(positive_passage.input_ids, attention_mask = positive_passage.attention_mask,                      
                                        return_dict = True, mode = 'text')
        negative_passage_output = self.text_encoder(negative_passage.input_ids, attention_mask = negative_passage.attention_mask,                      
                                        return_dict = True, mode = 'text')
        
        question_output = F.normalize(self.text_proj(question_output.last_hidden_state[:,0,:]),dim=-1)        
        positive_passage_output = F.normalize(self.text_proj(positive_passage_output.last_hidden_state[:,0,:]),dim=-1)        
        negative_passage_output = F.normalize(self.text_proj(negative_passage_output.last_hidden_state[:,0,:]),dim=-1)  

        passages = torch.cat([positive_passage_output, negative_passage_output])

        sim_scores = dot_score(question_output, passages)
    
        with torch.no_grad():
            question_output_m = self.text_encoder(question.input_ids, attention_mask = question.attention_mask,                      
                                            return_dict = True, mode = 'text')
            positive_passage_output_m = self.text_encoder(positive_passage.input_ids, attention_mask = positive_passage.attention_mask,                      
                                            return_dict = True, mode = 'text')
            negative_passage_output_m = self.text_encoder(negative_passage.input_ids, attention_mask = negative_passage.attention_mask,                      
                                            return_dict = True, mode = 'text')
            
            question_output_m = F.normalize(self.text_proj(question_output_m.last_hidden_state[:,0,:]),dim=-1)        
            positive_passage_output_m = F.normalize(self.text_proj(positive_passage_output_m.last_hidden_state[:,0,:]),dim=-1)        
            negative_passage_output_m = F.normalize(self.text_proj(negative_passage_output_m.last_hidden_state[:,0,:]),dim=-1)        

            passages_m = torch.cat([positive_passage_output, negative_passage_output])

            sim_soft_targets = dot_score(question_output_m, passages_m)/self.temp


        sim_hard_targets = torch.tensor(range(len(sim_soft_targets)), dtype=torch.long, device=sim_soft_targets.device)
        
        loss_itc = F.nll_loss(F.log_softmax(sim_scores, dim=1), sim_hard_targets)
        loss_momentum = F.kl_div(F.log_softmax(sim_scores, dim=1), F.softmax(sim_soft_targets))

        loss_ita = alpha*loss_momentum + (1-alpha)*loss_itc
        return loss_ita
        # sim_targets = alpha*F.softmax(sim_soft_targets, dim=1) + (1-alpha)*sim_hard_targets
        # loss_ita = -torch.sum(F.log_softmax(sim_scores, dim=1)*sim_targets,dim=1).mean()
        # return loss_ita
        # loss_ita = F.kl_div(sim_scores, sim_targets)

    
    def forward(self, image_question, image_pos_image, image_pos_caption, image_neg_image, image_neg_caption, text_question, text_pos_txt, text_neg_text, alpha, device):
        # pdb.set_trace()
        self._momentum_update()
        if self.negative_all_rank:
            raise ValueError("need to implement this!")

        text_loss_ita = self.text_forward(text_question, text_pos_txt, text_neg_text, alpha, device)
        image_loss_ita = self.image_forward(image_question, image_pos_image, image_pos_caption, image_neg_image, image_neg_caption, alpha, device)
        loss = image_loss_ita + text_loss_ita
        
        return loss

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data.clone() * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  


def blip_retrieval_webqa(pretrained='',**kwargs):
    model = BLIP_Retrieval_WebQA(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
