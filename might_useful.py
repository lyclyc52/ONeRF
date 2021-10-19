
#  For EM


# slots = slots.T
# position = position.T

# with torch.no_grad():
#     for i in range(20):
#         z = torch.matmul(f, slots)      # NxK
#         z_p = torch.matmul(f_p, position) 
#         z = z + w * z_p
#         z = F.softmax(z, dim=-1)                 # NxK
#         z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
#         slots = torch.matmul(f.T, z_)       # CxK
#         slots = _l2norm(slots, dim=0) 
#         position = torch.matmul(f_p.T, z_)
#         position = _l2norm(position, dim=0) 


# attn_logits = torch.matmul(f, slots)
# attn = attn_logits.softmax(dim=-1)
# print(attn.shape)
# attn = attn.reshape([B,H,W,num_slots])
# attn = attn.permute([0,3,1,2])